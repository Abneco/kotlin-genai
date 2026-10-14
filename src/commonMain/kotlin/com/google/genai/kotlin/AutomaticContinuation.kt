/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.genai.kotlin

import com.google.genai.kotlin.types.FinishReason
import com.google.genai.kotlin.types.GenerateContentConfig
import com.google.genai.kotlin.types.GenerateContentResponse
import com.google.genai.kotlin.types.GenerateContentResponseUsageMetadata
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull

/**
 * Sends a generateContent request through [send], then sends the same request again with each
 * response's continuation token until the model finishes. Returns the responses merged into one.
 *
 * The caller has already decided that automatic continuation is on. The first request goes out with
 * [config] as given, null included, so it is the request the caller would have sent anyway.
 */
internal suspend fun generateWithAutomaticContinuation(
  config: GenerateContentConfig?,
  send: suspend (GenerateContentConfig?) -> GenerateContentResponse,
): GenerateContentResponse {
  val responses = mutableListOf<GenerateContentResponse>()
  var requestConfig = config
  while (true) {
    val response = send(requestConfig)
    responses += response
    val candidate = response.candidates?.firstOrNull()
    val token = candidate?.continuationToken?.takeIf { it.isNotEmpty() }
    if (token == null || !isContinuable(candidate.finishReason)) {
      break
    }
    requestConfig = withContinuationToken(config, token)
  }
  return mergeContinuationResponses(responses)
}

/**
 * Maximum number of consecutive streaming reconnects that do not receive a new checkpoint
 * continuationToken before the error is surfaced to the caller.
 */
internal const val MAX_STREAM_RECONNECTS_WITHOUT_PROGRESS = 3

/**
 * The streaming counterpart of [generateWithAutomaticContinuation]. The chunks of every request are
 * emitted in the order they arrive, with usage metadata accumulated across requests (see
 * [StreamContinuation]).
 */
internal fun streamWithAutomaticContinuation(
  config: GenerateContentConfig?,
  send: (GenerateContentConfig?) -> Flow<GenerateContentResponse>,
): Flow<GenerateContentResponse> {
  return flow {
    val state = StreamContinuation(config)
    while (true) {
      val requestConfig = state.startHop()
      var error: Throwable? = null
      try {
        send(requestConfig).collect { chunk -> emit(state.observe(chunk)) }
      } catch (e: Throwable) {
        if (e is CancellationException) {
          throw e
        }
        error = e
      }
      if (!state.endHop(error)) {
        break
      }
    }
  }
}

/**
 * State of one [streamWithAutomaticContinuation] call.
 *
 * A stream is made of one or more hops (HTTP requests). Each hop after the first resumes from a
 * continuationToken. The caller drives the loop:
 * ```
 * val state = StreamContinuation(config)
 * while (true) {
 *   val requestConfig = state.startHop()
 *   var error: Throwable? = null
 *   try {
 *     send(requestConfig).collect { emit(state.observe(it)) }
 *   } catch (e: Throwable) {
 *     if (e is CancellationException) throw e
 *     error = e
 *   }
 *   if (!state.endHop(error)) break
 * }
 * ```
 *
 * Loop state (the usage metadata accumulated across hops, the token to resume from, the count of
 * reconnects without progress, and the latest error since the stream last made progress) carries
 * across hops.
 * See [endHop] for how errors are handled: a request rejected with a [GenAiApiException] before any
 * chunk is rethrown at once (retrying it is left to the HTTP client's retryOptions), while
 * transport drops and clean closes are retried and the latest such error is kept so it can be
 * rethrown if the stream gives up. Per-hop state lives in a [Hop] that [startHop] replaces
 * wholesale, so it can't leak from one hop into the next.
 *
 * Usage metadata: within a hop, each chunk's usage metadata is the server's running total for that
 * hop, so the latest one is the hop's usage (chunks are not summed). Across hops, the hops' usage
 * is summed, since each hop is a separate request. Each emitted chunk that has usage metadata
 * carries the usage of all previous hops plus its own.
 */
internal class StreamContinuation(private val config: GenerateContentConfig?) {
  /** What one hop observed on the chunks it streamed. */
  internal class Hop(
    /** Token the hop was started from (null for the initial request). */
    val startToken: ByteArray? = null,
    /** Latest continuationToken received during the hop. */
    var token: ByteArray? = null,
    var finishReason: FinishReason? = null,
    var hasPromptFeedback: Boolean = false,
    var hasYieldedContent: Boolean = false,
    /** Whether the hop received any chunk at all. */
    var receivedAnyChunk: Boolean = false,
    /**
     * Usage for the whole stream so far: the usage of all previous hops plus this hop's latest
     * usage metadata. Within a hop, each chunk's usage metadata is the server's running total for
     * the hop, so it replaces the previous value (it is not summed); a chunk without usage metadata
     * leaves it unchanged.
     */
    var usage: GenerateContentResponseUsageMetadata? = null,
  )

  /**
   * Usage of all finished hops, summed. Each hop is a separate request whose usage the server
   * reports independently, so hop totals are added (unlike chunks within a hop; see [Hop.usage]).
   */
  private var accumulatedUsageAcrossHops: GenerateContentResponseUsageMetadata? = null

  /** Token the next hop resumes from (null for the initial request). */
  private var resumeToken: ByteArray? = null

  /** Latest error since the stream last made progress (new content or a new checkpoint). */
  private var lastError: Throwable? = null

  // reconnectsWithoutProgress and hop are only settable from outside so that tests can drive
  // nextResumeToken directly.

  /** Number of consecutive hops that ended without a new continuationToken. */
  internal var reconnectsWithoutProgress = 0

  /** State of the current hop. */
  internal var hop = Hop()

  /**
   * Starts a new hop and returns the config to request it with. The initial request goes out with
   * the caller's config as given, null included.
   */
  fun startHop(): GenerateContentConfig? {
    hop = Hop(startToken = resumeToken)
    return resumeToken?.let { withContinuationToken(config, it) } ?: config
  }

  /**
   * Records one streamed chunk and returns the chunk to emit, with its intermediate finishReason
   * cleared when the stream will auto-continue and its usage metadata (if any) set to this hop's
   * latest usage plus the usage of all previous hops.
   */
  fun observe(chunk: GenerateContentResponse): GenerateContentResponse {
    val hop = this.hop
    hop.receivedAnyChunk = true
    if (chunk.promptFeedback != null) {
      hop.hasPromptFeedback = true
    }
    val candidate = chunk.candidates?.firstOrNull()
    if (!candidate?.content?.parts.isNullOrEmpty()) {
      hop.hasYieldedContent = true
    }
    candidate?.finishReason?.let { hop.finishReason = it }
    // A long stream also puts checkpoint tokens on chunks before the last one, so the token to
    // resume from is the last one seen unless the hop ends with a non-resumable reason.
    val chunkToken = candidate?.continuationToken?.takeIf { it.isNotEmpty() }
    if (chunkToken != null && chunk.promptFeedback == null) {
      hop.token = chunkToken
    } else if (
      chunk.promptFeedback != null ||
        (candidate?.finishReason != null && !isContinuable(candidate.finishReason))
    ) {
      hop.token = null
    }
    var outChunk = clearIntermediateFinishReason(chunk, hop.token)
    val chunkUsage = outChunk.usageMetadata
    if (chunkUsage != null) {
      val mergedUsage =
        accumulatedUsageAcrossHops?.let { mergeUsageMetadata(it, chunkUsage) } ?: chunkUsage
      hop.usage = mergedUsage
      if (mergedUsage !== chunkUsage) {
        outChunk = outChunk.copy(usageMetadata = mergedUsage)
      }
    }
    return outChunk
  }

  /**
   * Ends the current hop and returns true if another hop should be started.
   *
   * [error] is the exception thrown while streaming the hop, if any. A [GenAiApiException] (an
   * HTTP error status, or an error event in the stream) thrown before the hop received any chunk
   * means the request was rejected; it is rethrown at once without a continuation retry, since
   * retrying a rejected request is left to the HTTP client's retryOptions, which back off and can
   * be turned off. Transport failures (dropped connections, network errors) and clean closes are
   * still retried as below.
   *
   * If the stream fails or ends without a terminal finishReason, the next hop resumes from the
   * latest checkpoint token. If the hop failed before emitting any content or a checkpoint of its
   * own, it is retried from the token it started with, at most
   * [MAX_STREAM_RECONNECTS_WITHOUT_PROGRESS] times in a row. When the stream gives up, [error] is
   * rethrown if there is one; otherwise, if the hop also had no finishReason or promptFeedback, the
   * latest error since the stream last made progress is rethrown, so a dropped connection followed
   * by clean empty closes is not lost. If no error was seen, the stream ends normally, even if the
   * last hop had no finishReason.
   */
  fun endHop(error: Throwable?): Boolean {
    val hop = this.hop
    hop.usage?.let { accumulatedUsageAcrossHops = it }
    if (error is GenAiApiException && !hop.receivedAnyChunk) {
      // The request was rejected (an HTTP error status, or an error event before any chunk).
      // Retrying that is left to the HTTP client's retryOptions, which back off and which the
      // caller can turn off.
      throw error
    }
    val hopToken = hop.token
    val madeProgress =
      hop.hasYieldedContent ||
        (hopToken != null && hopToken.isNotEmpty() && !hopToken.contentEquals(hop.startToken))
    if (error != null) {
      lastError = error
    } else if (madeProgress) {
      lastError = null
    }
    val (nextToken, nextReconnects) = nextResumeToken()
    reconnectsWithoutProgress = nextReconnects
    resumeToken = nextToken
    // nextResumeToken already applies these checks, so this currently always holds for a non-null
    // token.
    if (
      nextToken != null && !hop.hasPromptFeedback && canResumeStream(nextToken, hop.finishReason)
    ) {
      return true
    }
    error?.let { throw it }
    // A hop the API ended with a finishReason or promptFeedback is a real ending, so an earlier
    // error is not raised over it.
    val gaveUp = hop.finishReason == null && !hop.hasPromptFeedback
    if (gaveUp) {
      lastError?.let { throw it }
    }
    return false
  }

  /**
   * Returns the token to resume from (or null to stop) and the updated reconnect count.
   *
   * Long-decode checkpoints are only emitted part way through each hop, so a connection can drop
   * during prefill before the hop has produced a checkpoint of its own. In that case, if no
   * candidate content has been emitted yet, the hop can be retried from the token it started with.
   */
  internal fun nextResumeToken(): Pair<ByteArray?, Int> {
    val hop = this.hop
    val hopToken = hop.token
    if (hop.finishReason != null || hop.hasPromptFeedback) {
      // When the server explicitly ends the hop with a finishReason or promptFeedback, only continue
      // if it is CONTINUATION without promptFeedback and produced a new token.
      return if (
        hop.finishReason == FinishReason.CONTINUATION &&
          !hop.hasPromptFeedback &&
          hopToken != null &&
          hopToken.isNotEmpty() &&
          !hopToken.contentEquals(hop.startToken)
      ) {
        hopToken to 0
      } else {
        null to 0
      }
    }
    if (hopToken != null && hopToken.isNotEmpty() && !hopToken.contentEquals(hop.startToken)) {
      return hopToken to 0
    }
    if (
      hop.hasYieldedContent || reconnectsWithoutProgress >= MAX_STREAM_RECONNECTS_WITHOUT_PROGRESS
    ) {
      return null to reconnectsWithoutProgress
    }
    return hop.startToken to (reconnectsWithoutProgress + 1)
  }
}

internal fun canResumeStream(token: ByteArray?, finishReason: FinishReason?): Boolean =
  token != null && token.isNotEmpty() && isContinuable(finishReason)

internal fun clearIntermediateFinishReason(
  chunk: GenerateContentResponse,
  token: ByteArray?,
): GenerateContentResponse {
  val candidate = chunk.candidates?.firstOrNull() ?: return chunk
  if (candidate.finishReason != null && canResumeStream(token, candidate.finishReason)) {
    val updatedCandidates = chunk.candidates.toMutableList()
    updatedCandidates[0] = candidate.copy(finishReason = null)
    return chunk.copy(candidates = updatedCandidates)
  }
  return chunk
}

/**
 * Merges the responses of one continued generation into a single response. Lists such as the parts
 * are concatenated in order, each candidate is merged with the one of the same index, token counts
 * are summed, and end of generation values such as the finish reason come from the last response.
 */
internal fun mergeContinuationResponses(
  responses: List<GenerateContentResponse>
): GenerateContentResponse {
  if (responses.size == 1) {
    return responses.single()
  }
  // sdkHttpResponse holds the raw body, continuation token and all, so it is kept out of the merge.
  val merged =
    responses
      .map { response ->
        Common.JSON.encodeToJsonElement(
            GenerateContentResponse.serializer(),
            response.copy(sdkHttpResponse = null),
          )
          .jsonObject
      }
      .reduce { prev, curr -> mergeObjects(prev, curr) }
  return Common.JSON.decodeFromJsonElement(GenerateContentResponse.serializer(), merged)
    .copy(sdkHttpResponse = responses.lastOrNull { it.sdkHttpResponse != null }?.sdkHttpResponse)
    .also {
      it.usageMetadata?.firstHopPromptTokenCount = responses.firstNotNullOfOrNull { response ->
        response.usageMetadata?.initialPromptTokenCount
      }
    }
}

internal fun mergeUsageMetadata(
  prev: GenerateContentResponseUsageMetadata,
  curr: GenerateContentResponseUsageMetadata,
): GenerateContentResponseUsageMetadata {
  val prevJson =
    Common.JSON.encodeToJsonElement(GenerateContentResponseUsageMetadata.serializer(), prev)
      .jsonObject
  val currJson =
    Common.JSON.encodeToJsonElement(GenerateContentResponseUsageMetadata.serializer(), curr)
      .jsonObject
  return Common.JSON.decodeFromJsonElement(
      GenerateContentResponseUsageMetadata.serializer(),
      mergeObjects(prevJson, currJson, sumAllNumbers = true),
    )
    .also {
      it.firstHopPromptTokenCount = prev.initialPromptTokenCount ?: curr.initialPromptTokenCount
    }
}

// Per the continuation protocol, when a continuationToken is present the backend only sets
// finishReason to null (intermediate checkpoint chunk) or CONTINUATION (stream end). MAX_TOKENS
// means the caller's maxOutputTokens, which the server counts across all the requests, is spent.
private fun isContinuable(finishReason: FinishReason?): Boolean =
  finishReason == null || finishReason == FinishReason.CONTINUATION

private fun withContinuationToken(config: GenerateContentConfig?, token: ByteArray) =
  (config ?: GenerateContentConfig()).copy(continuationToken = token)

private val END_OF_GENERATION_FIELDS = setOf("continuationToken", "finishReason", "finishMessage")

private val MODALITY_TOKEN_COUNT_FIELDS = setOf("modality", "tokenCount")

private fun mergeObjects(
  prev: JsonObject,
  curr: JsonObject,
  sumAllNumbers: Boolean = false,
  latestOnly: Set<String> = emptySet(),
): JsonObject {
  val merged = LinkedHashMap<String, JsonElement>()
  for (key in prev.keys + curr.keys) {
    val value =
      if (key in latestOnly) curr[key] else mergeValues(key, prev[key], curr[key], sumAllNumbers)
    if (value != null && value !is JsonNull) {
      merged[key] = value
    }
  }
  return JsonObject(merged)
}

private fun mergeValues(
  key: String,
  prev: JsonElement?,
  curr: JsonElement?,
  sumAllNumbers: Boolean,
): JsonElement? {
  val previous = prev?.takeUnless { it is JsonNull }
  val current = curr?.takeUnless { it is JsonNull }
  // Route candidates through mergeCandidates even when curr.candidates is null
  // so preserved candidates clear terminal continuation state.
  if (key == "candidates" && previous is JsonArray) {
    return mergeCandidates(previous, current as? JsonArray ?: JsonArray(emptyList()))
  }
  return when {
    current == null -> previous
    previous == null -> current
    previous is JsonObject && current is JsonObject ->
      mergeObjects(previous, current, sumAllNumbers = sumAllNumbers || key == "usageMetadata")
    previous is JsonArray && current is JsonArray -> mergeArrays(key, previous, current)
    previous is JsonPrimitive &&
      current is JsonPrimitive &&
      (sumAllNumbers || key.endsWith("Count") || key.endsWith("Sum")) ->
      sumNumbers(previous, current) ?: current
    else -> current
  }
}

private fun mergeArrays(key: String, prev: JsonArray, curr: JsonArray): JsonArray {
  val first = prev.firstOrNull() ?: curr.firstOrNull()
  return when {
    key == "candidates" -> mergeCandidates(prev, curr)
    first != null && isModalityTokenCount(first) -> mergeModalityTokenCounts(prev, curr)
    key == "safetyRatings" -> latestPerCategory(prev, curr)
    else -> JsonArray(prev + curr)
  }
}

// Each candidate is merged with the one that has the same index in the next response. A candidate
// that only one response has is kept.
private fun mergeCandidates(prev: JsonArray, curr: JsonArray): JsonArray {
  if (curr.isEmpty()) {
    return JsonArray(
      prev.map {
        mergeObjects(it.jsonObject, JsonObject(emptyMap()), latestOnly = END_OF_GENERATION_FIELDS)
      }
    )
  }
  val merged = LinkedHashMap<Int, JsonObject>()
  prev.forEachIndexed { position, candidate ->
    merged[candidateIndex(candidate.jsonObject, position)] = candidate.jsonObject
  }
  curr.forEachIndexed { position, candidate ->
    val index = candidateIndex(candidate.jsonObject, position)
    val previous = merged[index]
    merged[index] =
      if (previous == null) {
        candidate.jsonObject
      } else {
        mergeObjects(
          previous,
          shiftGroundingSupports(previous, candidate.jsonObject),
          latestOnly = END_OF_GENERATION_FIELDS,
        )
      }
  }
  return JsonArray(merged.values.toList())
}

private fun shiftGroundingSupports(previous: JsonObject, current: JsonObject): JsonObject {
  val currGm = current["groundingMetadata"] as? JsonObject ?: return current
  val currSupports = currGm["groundingSupports"] as? JsonArray ?: return current
  if (currSupports.isEmpty()) return current

  val partOffset = ((previous["content"] as? JsonObject)?.get("parts") as? JsonArray)?.size ?: 0
  val chunkOffset =
    ((previous["groundingMetadata"] as? JsonObject)?.get("groundingChunks") as? JsonArray)?.size
      ?: 0
  if (partOffset == 0 && chunkOffset == 0) return current

  val shiftedSupports =
    JsonArray(
      currSupports.map { supportElem ->
        val supportObj = supportElem as? JsonObject ?: return@map supportElem
        val updatedSupport = supportObj.toMutableMap()
        if (chunkOffset > 0) {
          val indices = supportObj["groundingChunkIndices"] as? JsonArray
          if (indices != null && indices.isNotEmpty()) {
            updatedSupport["groundingChunkIndices"] =
              JsonArray(
                indices.map { idxElem ->
                  val idx = idxElem.jsonPrimitive.intOrNull
                  if (idx != null) JsonPrimitive(idx + chunkOffset) else idxElem
                }
              )
          }
        }
        if (partOffset > 0) {
          val segmentObj = supportObj["segment"] as? JsonObject
          if (segmentObj != null) {
            val prevPartIdx = segmentObj["partIndex"]?.jsonPrimitive?.intOrNull ?: 0
            val updatedSegment = segmentObj.toMutableMap()
            updatedSegment["partIndex"] = JsonPrimitive(prevPartIdx + partOffset)
            updatedSupport["segment"] = JsonObject(updatedSegment)
          }
        }
        JsonObject(updatedSupport)
      }
    )

  val updatedGm = currGm.toMutableMap()
  updatedGm["groundingSupports"] = shiftedSupports
  val updatedCandidate = current.toMutableMap()
  updatedCandidate["groundingMetadata"] = JsonObject(updatedGm)
  return JsonObject(updatedCandidate)
}

// A candidate without an index is matched by its position.
private fun candidateIndex(candidate: JsonObject, position: Int): Int =
  candidate["index"]?.jsonPrimitive?.intOrNull ?: position

private fun isModalityTokenCount(element: JsonElement): Boolean =
  element is JsonObject &&
    element.isNotEmpty() &&
    MODALITY_TOKEN_COUNT_FIELDS.containsAll(element.keys)

private fun mergeModalityTokenCounts(prev: JsonArray, curr: JsonArray): JsonArray {
  val totals = LinkedHashMap<JsonElement?, Long>()
  for (item in prev + curr) {
    val count = item.jsonObject["tokenCount"]?.jsonPrimitive?.longOrNull ?: 0L
    val modality = item.jsonObject["modality"]
    totals[modality] = (totals[modality] ?: 0L) + count
  }
  return JsonArray(
    totals.map { (modality, count) ->
      JsonObject(
        buildMap<String, JsonElement> {
          if (modality != null) {
            put("modality", modality)
          }
          put("tokenCount", JsonPrimitive(count))
        }
      )
    }
  )
}

private fun latestPerCategory(prev: JsonArray, curr: JsonArray): JsonArray {
  val latest = LinkedHashMap<JsonElement?, JsonElement>()
  for (rating in prev + curr) {
    latest[(rating as? JsonObject)?.get("category")] = rating
  }
  return JsonArray(latest.values.toList())
}

private fun sumNumbers(a: JsonPrimitive, b: JsonPrimitive): JsonPrimitive? {
  if (a.isString || b.isString) {
    return null
  }
  val longA = a.longOrNull
  val longB = b.longOrNull
  if (longA != null && longB != null) {
    return JsonPrimitive(longA + longB)
  }
  val doubleA = a.doubleOrNull ?: return null
  val doubleB = b.doubleOrNull ?: return null
  return JsonPrimitive(doubleA + doubleB)
}
