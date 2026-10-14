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

import com.google.genai.kotlin.types.BlockedReason
import com.google.genai.kotlin.types.Candidate
import com.google.genai.kotlin.types.Citation
import com.google.genai.kotlin.types.CitationMetadata
import com.google.genai.kotlin.types.Content
import com.google.genai.kotlin.types.FinishReason
import com.google.genai.kotlin.types.GenerateContentConfig
import com.google.genai.kotlin.types.GenerateContentResponse
import com.google.genai.kotlin.types.GenerateContentResponsePromptFeedback
import com.google.genai.kotlin.types.GenerateContentResponseUsageMetadata
import com.google.genai.kotlin.types.GroundingChunk
import com.google.genai.kotlin.types.GroundingChunkImage
import com.google.genai.kotlin.types.GroundingChunkWeb
import com.google.genai.kotlin.types.GroundingMetadata
import com.google.genai.kotlin.types.GroundingSupport
import com.google.genai.kotlin.types.HarmCategory
import com.google.genai.kotlin.types.HarmProbability
import com.google.genai.kotlin.types.HttpResponse
import com.google.genai.kotlin.types.MediaModality
import com.google.genai.kotlin.types.ModalityTokenCount
import com.google.genai.kotlin.types.Part
import com.google.genai.kotlin.types.SafetyRating
import com.google.genai.kotlin.types.Segment
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.emptyFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest

private val TOKEN_1 = "token-1".encodeToByteArray()
private val TOKEN_2 = "token-2".encodeToByteArray()

private val CONFIG = GenerateContentConfig(automaticContinuation = true, temperature = 0.5)

private fun response(
  text: String,
  finishReason: FinishReason? = null,
  token: ByteArray? = null,
): GenerateContentResponse =
  GenerateContentResponse(
    candidates =
      listOf(
        Candidate(
          content = Content(role = "model", parts = listOf(Part(text = text))),
          finishReason = finishReason,
          continuationToken = token,
        )
      )
  )

/** A chunk whose prompt was blocked: no candidates, only promptFeedback. */
private fun blockedResponse(): GenerateContentResponse =
  GenerateContentResponse(
    promptFeedback = GenerateContentResponsePromptFeedback(blockReason = BlockedReason.SAFETY)
  )

/** Hands out [responses] in order and records the config of every request. */
private class FakeModel(vararg responses: GenerateContentResponse) {
  private val pending = ArrayDeque(responses.toList())
  val configs = mutableListOf<GenerateContentConfig?>()

  fun send(config: GenerateContentConfig?): GenerateContentResponse {
    configs += config
    return pending.removeFirst()
  }
}

/** Hands out one stream per request, in order, and records the config of every request. */
private class FakeStreamingModel(vararg streams: Flow<GenerateContentResponse>) {
  private val pending = ArrayDeque(streams.toList())
  val configs = mutableListOf<GenerateContentConfig?>()

  fun send(config: GenerateContentConfig?): Flow<GenerateContentResponse> {
    configs += config
    return pending.removeFirst()
  }
}

class AutomaticContinuationTest {

  @Test
  fun testContinuesWithEachTokenUntilTheModelFinishes() = runTest {
    val model =
      FakeModel(
        response("Hello ", FinishReason.CONTINUATION, TOKEN_1),
        response("big ", FinishReason.CONTINUATION, TOKEN_2),
        response("world", FinishReason.STOP),
      )

    val merged = generateWithAutomaticContinuation(CONFIG) { model.send(it) }

    assertEquals("Hello big world", merged.text)
    assertEquals(3, model.configs.size)
    assertEquals(CONFIG, model.configs[0])
    assertContentEquals(TOKEN_1, model.configs[1]?.continuationToken)
    assertContentEquals(TOKEN_2, model.configs[2]?.continuationToken)
    // Apart from the token, every request is the original one.
    assertEquals(CONFIG, model.configs[1]?.copy(continuationToken = null))
    assertEquals(CONFIG, model.configs[2]?.copy(continuationToken = null))
  }

  @Test
  fun testDoesNotContinueOnMaxTokens() = runTest {
    val only = response("Hello", FinishReason.MAX_TOKENS, TOKEN_1)
    val model = FakeModel(only)

    assertSame(only, generateWithAutomaticContinuation(CONFIG) { model.send(it) })
    assertEquals(1, model.configs.size)
  }

  @Test
  fun testStopsWhenTheResponseHasNoToken() = runTest {
    val model = FakeModel(response("Hello", FinishReason.CONTINUATION))

    val response = generateWithAutomaticContinuation(CONFIG) { model.send(it) }

    assertEquals(FinishReason.CONTINUATION, response.finishReason)
    assertEquals(1, model.configs.size)

    val streamingModel = FakeStreamingModel(flowOf(response("Hello", FinishReason.CONTINUATION)))
    val chunks = streamWithAutomaticContinuation(CONFIG) { streamingModel.send(it) }.toList()

    assertEquals(listOf<FinishReason?>(FinishReason.CONTINUATION), chunks.map { it.finishReason })
    assertEquals(1, streamingModel.configs.size)
  }

  @Test
  fun testStopsOnOtherFinishReasons() = runTest {
    val model = FakeModel(response("Hello", FinishReason.SAFETY, TOKEN_1))

    generateWithAutomaticContinuation(CONFIG) { model.send(it) }

    assertEquals(1, model.configs.size)
  }

  @Test
  fun testSendsTheFirstRequestWithoutAConfigWhenGivenNone() = runTest {
    val model =
      FakeModel(
        response("Hello ", FinishReason.CONTINUATION, TOKEN_1),
        response("world", FinishReason.STOP),
      )

    val merged = generateWithAutomaticContinuation(null) { model.send(it) }

    assertEquals("Hello world", merged.text)
    assertNull(model.configs[0])
    assertContentEquals(TOKEN_1, model.configs[1]?.continuationToken)
    assertEquals(GenerateContentConfig(), model.configs[1]?.copy(continuationToken = null))
  }

  @Test
  fun testSendsMaxOutputTokensWithEveryRequest() = runTest {
    val config = CONFIG.copy(maxOutputTokens = 100)
    val model =
      FakeModel(
        response("Hello ", FinishReason.CONTINUATION, TOKEN_1),
        response("world", FinishReason.STOP),
      )

    generateWithAutomaticContinuation(config) { model.send(it) }

    assertEquals(listOf<Int?>(100, 100), model.configs.map { it?.maxOutputTokens })
  }

  @Test
  fun testStreamEmitsTheChunksOfEveryRequestInOrder() = runTest {
    val model =
      FakeStreamingModel(
        flowOf(
          response("a"),
          // A checkpoint in the middle of a long stream.
          response("b", token = TOKEN_1),
          response("c", FinishReason.CONTINUATION, TOKEN_2),
        ),
        flowOf(response("d", FinishReason.STOP)),
      )

    val chunks = streamWithAutomaticContinuation(CONFIG) { model.send(it) }.toList()

    assertEquals(listOf<String?>("a", "b", "c", "d"), chunks.map { it.text })
    assertEquals(listOf(null, null, null, FinishReason.STOP), chunks.map { it.finishReason })
    assertEquals(2, model.configs.size)
    assertContentEquals(TOKEN_2, model.configs[1]?.continuationToken)
    assertEquals(CONFIG, model.configs[1]?.copy(continuationToken = null))
  }

  @Test
  fun testStreamResumesFromATokenSentBeforeTheFinishReason() = runTest {
    val model =
      FakeStreamingModel(
        flowOf(response("a", token = TOKEN_1), response("b", FinishReason.CONTINUATION)),
        flowOf(response("c", FinishReason.STOP)),
      )

    val chunks = streamWithAutomaticContinuation(CONFIG) { model.send(it) }.toList()

    assertEquals(listOf<String?>("a", "b", "c"), chunks.map { it.text })
    assertEquals(listOf(null, null, FinishReason.STOP), chunks.map { it.finishReason })
    assertEquals(2, model.configs.size)
    assertContentEquals(TOKEN_1, model.configs[1]?.continuationToken)
  }

  @Test
  fun testStreamResumesFromCheckpointTokenOnMidStreamCutoffOrError() = runTest {
    val model =
      FakeStreamingModel(
        // Hop 1: intermediate checkpoint chunk (finishReason == null), then stream cuts off early
        flowOf(response("a", token = TOKEN_1)),
        // Hop 2: intermediate checkpoint chunk, then mid-stream exception
        flow {
          emit(response("b", token = TOKEN_2))
          throw IllegalStateException("stream dropped")
        },
        // Hop 3: completes with STOP
        flowOf(response("c", FinishReason.STOP)),
      )

    val chunks = streamWithAutomaticContinuation(CONFIG) { model.send(it) }.toList()

    assertEquals(listOf<String?>("a", "b", "c"), chunks.map { it.text })
    assertEquals(3, model.configs.size)
    assertContentEquals(TOKEN_1, model.configs[1]?.continuationToken)
    assertContentEquals(TOKEN_2, model.configs[2]?.continuationToken)
  }

  @Test
  fun testStreamRethrowsMidStreamErrorWhenNoCheckpointTokenWasReceived() = runTest {
    val model =
      FakeStreamingModel(
        flow {
          emit(response("a"))
          throw IllegalStateException("unrecoverable error")
        }
      )

    assertFailsWith<IllegalStateException> {
      streamWithAutomaticContinuation(CONFIG) { model.send(it) }.toList()
    }
    assertEquals(1, model.configs.size)
  }

  @Test
  fun testStreamRetriesFromHopStartTokenWhenResumedHopDropsBeforeFirstCheckpoint() = runTest {
    val model =
      FakeStreamingModel(
        // Hop 1: finishes normally with CONTINUATION and TOKEN_1
        flowOf(response("a", FinishReason.CONTINUATION, TOKEN_1)),
        // Hop 2 (attempt 1): drops with an exception before emitting any content or checkpoint
        flow { throw IllegalStateException("hop 2 dropped early") },
        // Hop 2 (attempt 2): cuts off without finishReason before emitting any content or
        // checkpoint
        emptyFlow(),
        // Hop 2 (attempt 3): emits a new checkpoint TOKEN_2, then drops
        flow {
          emit(response("b3", token = TOKEN_2))
          throw IllegalStateException("hop 2 dropped after checkpoint")
        },
        // Hop 3: completes with STOP
        flowOf(response("c", FinishReason.STOP)),
      )

    val chunks = streamWithAutomaticContinuation(CONFIG) { model.send(it) }.toList()

    assertEquals(listOf<String?>("a", "b3", "c"), chunks.map { it.text })
    assertEquals(5, model.configs.size)
    assertEquals(CONFIG, model.configs[0])
    assertContentEquals(TOKEN_1, model.configs[1]?.continuationToken)
    assertContentEquals(TOKEN_1, model.configs[2]?.continuationToken)
    assertContentEquals(TOKEN_1, model.configs[3]?.continuationToken)
    assertContentEquals(TOKEN_2, model.configs[4]?.continuationToken)
  }

  @Test
  fun testStreamDoesNotRetryFromHopStartTokenAfterYieldingContent() = runTest {
    val errorModel =
      FakeStreamingModel(
        // Hop 1: finishes normally with CONTINUATION and TOKEN_1
        flowOf(response("a", FinishReason.CONTINUATION, TOKEN_1)),
        // Hop 2: emits content parts without a new checkpoint token, then fails
        flow {
          emit(response("b"))
          throw IllegalStateException("hop 2 dropped after content")
        },
      )

    val emittedBeforeError = mutableListOf<GenerateContentResponse>()
    val error =
      assertFailsWith<IllegalStateException> {
        streamWithAutomaticContinuation(CONFIG) { errorModel.send(it) }
          .collect { emittedBeforeError += it }
      }
    assertEquals("hop 2 dropped after content", error.message)
    assertEquals(listOf<String?>("a", "b"), emittedBeforeError.map { it.text })
    assertEquals(2, errorModel.configs.size)
    assertEquals(CONFIG, errorModel.configs[0])
    assertContentEquals(TOKEN_1, errorModel.configs[1]?.continuationToken)

    val cutoffModel =
      FakeStreamingModel(
        // Hop 1: finishes normally with CONTINUATION and TOKEN_1
        flowOf(response("a", FinishReason.CONTINUATION, TOKEN_1)),
        // Hop 2: emits content parts without a new checkpoint token or finishReason, then closes
        flowOf(response("b")),
      )

    // The server reported no error, so the stream ends normally with the chunks already emitted.
    val emittedBeforeCutoff =
      streamWithAutomaticContinuation(CONFIG) { cutoffModel.send(it) }.toList()
    assertEquals(listOf<String?>("a", "b"), emittedBeforeCutoff.map { it.text })
    assertEquals(2, cutoffModel.configs.size)
    assertEquals(CONFIG, cutoffModel.configs[0])
    assertContentEquals(TOKEN_1, cutoffModel.configs[1]?.continuationToken)
  }

  @Test
  fun testStreamRethrowsAfterMaxReconnectsWithoutProgressOnResumedHop() = runTest {
    val model =
      FakeStreamingModel(
        // Hop 1: finishes with CONTINUATION and TOKEN_1
        flowOf(response("a", FinishReason.CONTINUATION, TOKEN_1)),
        // Initial Hop 2 attempt + MAX_STREAM_RECONNECTS_WITHOUT_PROGRESS retries all fail
        // before emitting a new checkpoint token.
        flow { throw IllegalStateException("attempt 1 failed") },
        flow { throw IllegalStateException("retry 1 failed") },
        flow { throw IllegalStateException("retry 2 failed") },
        flow { throw IllegalStateException("retry 3 failed") },
      )

    val error =
      assertFailsWith<IllegalStateException> {
        streamWithAutomaticContinuation(CONFIG) { model.send(it) }.toList()
      }
    assertEquals("retry 3 failed", error.message)
    assertEquals(MAX_STREAM_RECONNECTS_WITHOUT_PROGRESS + 2, model.configs.size)
    assertEquals(CONFIG, model.configs[0])
    for (i in 1..MAX_STREAM_RECONNECTS_WITHOUT_PROGRESS + 1) {
      assertContentEquals(TOKEN_1, model.configs[i]?.continuationToken)
    }
  }

  @Test
  fun testStreamDoesNotRetryAResumedHopRejectedBeforeAnyChunk() = runTest {
    val model =
      FakeStreamingModel(
        // Hop 1: finishes with CONTINUATION and TOKEN_1
        flowOf(response("a", FinishReason.CONTINUATION, TOKEN_1)),
        // Hop 2: the request is rejected with an HTTP 503 before any chunk. Retrying that is
        // left to the HTTP client's retryOptions.
        flow { throw ServerException(503, "UNAVAILABLE", "overloaded") },
        // Only reached if the loop wrongly retries the rejected request.
        flowOf(response("unexpected", FinishReason.STOP)),
      )
    val chunks = mutableListOf<GenerateContentResponse>()

    val error =
      assertFailsWith<ServerException> {
        streamWithAutomaticContinuation(CONFIG) { model.send(it) }.toList(chunks)
      }
    assertEquals(503, error.code)
    assertEquals(listOf<String?>("a"), chunks.map { it.text })
    assertEquals(2, model.configs.size)
    assertContentEquals(TOKEN_1, model.configs[1]?.continuationToken)
  }

  @Test
  fun testStreamRethrowsADropFollowedByCleanEmptyClosesAfterMaxReconnects() = runTest {
    val model =
      FakeStreamingModel(
        // Hop 1: finishes with CONTINUATION and TOKEN_1
        flowOf(response("a", FinishReason.CONTINUATION, TOKEN_1)),
        // Hop 2 (attempt 1): the connection drops before any chunk.
        flow { throw IllegalStateException("hop 2 dropped") },
        // Every retry closes cleanly without a chunk until the retries run out.
        *Array(MAX_STREAM_RECONNECTS_WITHOUT_PROGRESS) { emptyFlow() },
      )

    val error =
      assertFailsWith<IllegalStateException> {
        streamWithAutomaticContinuation(CONFIG) { model.send(it) }.toList()
      }
    // The drop is not lost just because the later attempts closed cleanly.
    assertEquals("hop 2 dropped", error.message)
    assertEquals(MAX_STREAM_RECONNECTS_WITHOUT_PROGRESS + 2, model.configs.size)
    for (i in 1..MAX_STREAM_RECONNECTS_WITHOUT_PROGRESS + 1) {
      assertContentEquals(TOKEN_1, model.configs[i]?.continuationToken)
    }
  }

  @Test
  fun testStreamStopsWhenResumedHopEndsWithContinuationButNoNewToken() = runTest {
    val model =
      FakeStreamingModel(
        flowOf(response("a", FinishReason.CONTINUATION, TOKEN_1)),
        flowOf(response("b", FinishReason.CONTINUATION)),
      )

    val chunks = streamWithAutomaticContinuation(CONFIG) { model.send(it) }.toList()

    assertEquals(listOf<String?>("a", "b"), chunks.map { it.text })
    assertEquals(listOf(null, FinishReason.CONTINUATION), chunks.map { it.finishReason })
    assertEquals(2, model.configs.size)
    assertContentEquals(TOKEN_1, model.configs[1]?.continuationToken)
  }

  /** Runs [StreamContinuation.nextResumeToken] on a hop in the given state. */
  private fun nextStreamContinuationToken(
    hopStartToken: ByteArray?,
    hopToken: ByteArray?,
    finishReason: FinishReason?,
    reconnectsWithoutProgress: Int,
    hasPromptFeedback: Boolean = false,
    hopHasYieldedContent: Boolean = false,
  ): Pair<ByteArray?, Int> {
    val state = StreamContinuation(CONFIG)
    state.hop =
      StreamContinuation.Hop(
        startToken = hopStartToken,
        token = hopToken,
        finishReason = finishReason,
        hasPromptFeedback = hasPromptFeedback,
        hasYieldedContent = hopHasYieldedContent,
      )
    state.reconnectsWithoutProgress = reconnectsWithoutProgress
    return state.nextResumeToken()
  }

  @Test
  fun testNextStreamContinuationToken() {
    // New checkpoint token received during hop -> resumes from it and resets counter.
    val (token1, count1) =
      nextStreamContinuationToken(
        hopStartToken = TOKEN_1,
        hopToken = TOKEN_2,
        finishReason = null,
        reconnectsWithoutProgress = 2,
      )
    assertContentEquals(TOKEN_2, token1)
    assertEquals(0, count1)

    // Hop ended without finishReason and without a new token -> retries from hopStartToken.
    val (token2, count2) =
      nextStreamContinuationToken(
        hopStartToken = TOKEN_1,
        hopToken = null,
        finishReason = null,
        reconnectsWithoutProgress = 0,
      )
    assertContentEquals(TOKEN_1, token2)
    assertEquals(1, count2)

    // Server echoed the same hopStartToken without advancing -> counts as no progress.
    val (token3, count3) =
      nextStreamContinuationToken(
        hopStartToken = TOKEN_1,
        hopToken = TOKEN_1.copyOf(),
        finishReason = null,
        reconnectsWithoutProgress = 1,
      )
    assertContentEquals(TOKEN_1, token3)
    assertEquals(2, count3)

    // Retry budget exhausted -> returns null token.
    val (token4, count4) =
      nextStreamContinuationToken(
        hopStartToken = TOKEN_1,
        hopToken = null,
        finishReason = null,
        reconnectsWithoutProgress = MAX_STREAM_RECONNECTS_WITHOUT_PROGRESS,
      )
    assertNull(token4)
    assertEquals(MAX_STREAM_RECONNECTS_WITHOUT_PROGRESS, count4)

    // Hop ended with a finishReason and no new token -> stops instead of replaying hopStartToken.
    val (token5, count5) =
      nextStreamContinuationToken(
        hopStartToken = TOKEN_1,
        hopToken = null,
        finishReason = FinishReason.CONTINUATION,
        reconnectsWithoutProgress = 0,
      )
    assertNull(token5)
    assertEquals(0, count5)

    // Initial hop (hopStartToken == null) fails before any checkpoint -> returns null token.
    val (token6, count6) =
      nextStreamContinuationToken(
        hopStartToken = null,
        hopToken = null,
        finishReason = null,
        reconnectsWithoutProgress = 0,
      )
    assertNull(token6)
    assertEquals(1, count6)

    // Hop ended with promptFeedback -> stops instead of replaying hopStartToken.
    val (token7, count7) =
      nextStreamContinuationToken(
        hopStartToken = TOKEN_1,
        hopToken = null,
        finishReason = null,
        reconnectsWithoutProgress = 1,
        hasPromptFeedback = true,
      )
    assertNull(token7)
    assertEquals(0, count7)

    // Hop yielded content parts before dropping without a new token -> does not replay
    // hopStartToken.
    val (token8, count8) =
      nextStreamContinuationToken(
        hopStartToken = TOKEN_1,
        hopToken = null,
        finishReason = null,
        reconnectsWithoutProgress = 1,
        hopHasYieldedContent = true,
      )
    assertNull(token8)
    assertEquals(1, count8)
  }

  @Test
  fun testStreamDoesNotRetryHopStartTokenWhenHopReturnsPromptFeedback() = runTest {
    val promptFeedback = GenerateContentResponsePromptFeedback(blockReason = BlockedReason.SAFETY)
    val model =
      FakeStreamingModel(
        flowOf(response("a", FinishReason.CONTINUATION, TOKEN_1)),
        flowOf(GenerateContentResponse(candidates = null, promptFeedback = promptFeedback)),
      )

    val chunks = streamWithAutomaticContinuation(CONFIG) { model.send(it) }.toList()

    assertEquals(2, model.configs.size)
    assertEquals(CONFIG, model.configs[0])
    assertContentEquals(TOKEN_1, model.configs[1]?.continuationToken)
    assertEquals(2, chunks.size)
    assertEquals("a", chunks[0].text)
    assertNull(chunks[0].finishReason)
    assertEquals(BlockedReason.SAFETY, chunks[1].promptFeedback?.blockReason)
  }

  @Test
  fun testStreamEndsNormallyWhenResumedHopEndsWithoutFinishReasonAfterMaxReconnects() = runTest {
    val model =
      FakeStreamingModel(
        // Hop 1: finishes with CONTINUATION and TOKEN_1
        flowOf(response("a", FinishReason.CONTINUATION, TOKEN_1)),
        // Initial Hop 2 attempt + MAX_STREAM_RECONNECTS_WITHOUT_PROGRESS retries all close
        // cleanly with 0 chunks (no finishReason, no promptFeedback, no continuationToken).
        emptyFlow(),
        emptyFlow(),
        emptyFlow(),
        emptyFlow(),
      )

    // The server reported no error, so the stream ends normally with the chunks already emitted.
    val chunks = streamWithAutomaticContinuation(CONFIG) { model.send(it) }.toList()
    assertEquals(listOf<String?>("a"), chunks.map { it.text })
    assertEquals(MAX_STREAM_RECONNECTS_WITHOUT_PROGRESS + 2, model.configs.size)
    assertEquals(CONFIG, model.configs[0])
    for (i in 1..MAX_STREAM_RECONNECTS_WITHOUT_PROGRESS + 1) {
      assertContentEquals(TOKEN_1, model.configs[i]?.continuationToken)
    }
  }

  @Test
  fun testStreamStopsAtACheckpointWhenTheModelFinishes() = runTest {
    val model =
      FakeStreamingModel(flowOf(response("a", token = TOKEN_1), response("b", FinishReason.STOP)))

    val chunks = streamWithAutomaticContinuation(CONFIG) { model.send(it) }.toList()

    assertEquals(2, chunks.size)
    assertEquals(1, model.configs.size)
  }

  @Test
  fun testStreamSendsTheFirstRequestWithoutAConfigWhenGivenNone() = runTest {
    val model =
      FakeStreamingModel(
        flowOf(response("a", FinishReason.CONTINUATION, TOKEN_1)),
        flowOf(response("b", FinishReason.STOP)),
      )

    val chunks = streamWithAutomaticContinuation(null) { model.send(it) }.toList()

    assertEquals(listOf<String?>("a", "b"), chunks.map { it.text })
    assertNull(model.configs[0])
    assertContentEquals(TOKEN_1, model.configs[1]?.continuationToken)
  }

  @Test
  fun testStreamDoesNotContinueOnMaxTokens() = runTest {
    val model = FakeStreamingModel(flowOf(response("a", FinishReason.MAX_TOKENS, TOKEN_1)))

    val chunks = streamWithAutomaticContinuation(CONFIG) { model.send(it) }.toList()

    assertEquals(listOf<FinishReason?>(FinishReason.MAX_TOKENS), chunks.map { it.finishReason })
    assertEquals(1, model.configs.size)
  }

  @Test
  fun testStreamDoesNotResumeFromACheckpointAfterABlockedChunk() = runTest {
    val blocked = blockedResponse()
    val model =
      FakeStreamingModel(
        flowOf(response("a"), response("b", token = TOKEN_1), blocked),
        // Only reached if the loop wrongly resumes from the checkpoint.
        flowOf(response("unexpected", FinishReason.STOP)),
      )

    val chunks = streamWithAutomaticContinuation(CONFIG) { model.send(it) }.toList()

    assertEquals(listOf<String?>("a", "b", null), chunks.map { it.text })
    assertSame(blocked, chunks.last())
    assertEquals(1, model.configs.size)
  }

  @Test
  fun testStreamRethrowsErrorAfterABlockedChunkInsteadOfResuming() = runTest {
    val model =
      FakeStreamingModel(
        flow {
          emit(response("a", token = TOKEN_1))
          emit(blockedResponse())
          throw IllegalStateException("stream dropped")
        },
        // Only reached if the loop wrongly resumes from the checkpoint.
        flowOf(response("unexpected", FinishReason.STOP)),
      )
    val chunks = mutableListOf<GenerateContentResponse>()

    assertFailsWith<IllegalStateException> {
      streamWithAutomaticContinuation(CONFIG) { model.send(it) }.toList(chunks)
    }
    assertEquals(2, chunks.size)
    assertEquals(BlockedReason.SAFETY, chunks.last().promptFeedback?.blockReason)
    assertEquals(1, model.configs.size)
  }

  @Test
  fun testStreamAccumulatesUsageMetadataAcrossHops() = runTest {
    val model =
      FakeStreamingModel(
        // Hop 1: checkpoint chunk with usage, then stream drops mid-hop
        flow {
          emit(
            response("a", token = TOKEN_1)
              .copy(
                usageMetadata =
                  GenerateContentResponseUsageMetadata(
                    promptTokenCount = 10,
                    cachedContentTokenCount = 4,
                    candidatesTokenCount = 40,
                    thoughtsTokenCount = 5,
                    totalTokenCount = 55,
                    promptTokensDetails = listOf(ModalityTokenCount(MediaModality.TEXT, 10)),
                    candidatesTokensDetails = listOf(ModalityTokenCount(MediaModality.TEXT, 40)),
                  )
              )
          )
          throw IllegalStateException("stream dropped")
        },
        // Hop 2: finishes with CONTINUATION and TOKEN_2
        flowOf(
          response("b"),
          response("c", FinishReason.CONTINUATION, TOKEN_2)
            .copy(
              usageMetadata =
                GenerateContentResponseUsageMetadata(
                  promptTokenCount = 55,
                  cachedContentTokenCount = 50,
                  candidatesTokenCount = 60,
                  thoughtsTokenCount = 3,
                  totalTokenCount = 118,
                  promptTokensDetails = listOf(ModalityTokenCount(MediaModality.TEXT, 55)),
                  candidatesTokensDetails = listOf(ModalityTokenCount(MediaModality.TEXT, 60)),
                )
            ),
        ),
        // Hop 3: finishes with STOP
        flowOf(
          response("d", FinishReason.STOP)
            .copy(
              usageMetadata =
                GenerateContentResponseUsageMetadata(
                  promptTokenCount = 118,
                  cachedContentTokenCount = 110,
                  candidatesTokenCount = 25,
                  thoughtsTokenCount = 2,
                  totalTokenCount = 145,
                  promptTokensDetails = listOf(ModalityTokenCount(MediaModality.TEXT, 118)),
                  candidatesTokensDetails = listOf(ModalityTokenCount(MediaModality.TEXT, 25)),
                )
            )
        ),
      )

    val chunks = streamWithAutomaticContinuation(CONFIG) { model.send(it) }.toList()

    assertEquals(listOf<String?>("a", "b", "c", "d"), chunks.map { it.text })
    // Chunk 0 (Hop 1): unchanged initial hop usage
    assertEquals(10, chunks[0].usageMetadata?.promptTokenCount)
    assertEquals(10, chunks[0].usageMetadata?.initialPromptTokenCount)
    assertEquals(10, chunks[0].initialPromptTokenCount)
    assertEquals(4, chunks[0].usageMetadata?.cachedContentTokenCount)
    assertEquals(40, chunks[0].usageMetadata?.candidatesTokenCount)
    assertEquals(5, chunks[0].usageMetadata?.thoughtsTokenCount)
    assertEquals(55, chunks[0].usageMetadata?.totalTokenCount)
    // Chunk 1 (Hop 2 first chunk): no usageMetadata
    assertNull(chunks[1].usageMetadata)
    assertNull(chunks[1].initialPromptTokenCount)
    // Chunk 2 (Hop 2 final chunk): accumulated across Hop 1 + Hop 2
    assertEquals(65, chunks[2].usageMetadata?.promptTokenCount)
    assertEquals(10, chunks[2].usageMetadata?.initialPromptTokenCount)
    assertEquals(10, chunks[2].initialPromptTokenCount)
    assertEquals(54, chunks[2].usageMetadata?.cachedContentTokenCount)
    assertEquals(100, chunks[2].usageMetadata?.candidatesTokenCount)
    assertEquals(8, chunks[2].usageMetadata?.thoughtsTokenCount)
    assertEquals(173, chunks[2].usageMetadata?.totalTokenCount)
    // Chunk 3 (Hop 3 final chunk): accumulated across Hop 1 + Hop 2 + Hop 3
    val finalUsage = chunks[3].usageMetadata
    assertEquals(183, finalUsage?.promptTokenCount)
    assertEquals(10, finalUsage?.initialPromptTokenCount)
    assertEquals(10, chunks[3].initialPromptTokenCount)
    assertEquals(164, finalUsage?.cachedContentTokenCount)
    assertEquals(125, finalUsage?.candidatesTokenCount)
    assertEquals(10, finalUsage?.thoughtsTokenCount)
    assertEquals(318, finalUsage?.totalTokenCount)
    assertEquals(
      listOf(ModalityTokenCount(MediaModality.TEXT, 183)),
      finalUsage?.promptTokensDetails,
    )
    assertEquals(
      listOf(ModalityTokenCount(MediaModality.TEXT, 125)),
      finalUsage?.candidatesTokensDetails,
    )
  }

  @Test
  fun testMergeReturnsASingleResponseAsIs() {
    val only = response("Hello", FinishReason.STOP)

    assertSame(only, mergeContinuationResponses(listOf(only)))
  }

  @Test
  fun testMergeConcatenatesPartsAndOtherLists() {
    val first =
      response("Hello ", FinishReason.CONTINUATION, TOKEN_1).withFirstCandidate {
        copy(citationMetadata = CitationMetadata(citations = listOf(Citation(uri = "a"))))
      }
    val second =
      response("world", FinishReason.STOP).withFirstCandidate {
        copy(citationMetadata = CitationMetadata(citations = listOf(Citation(uri = "b"))))
      }

    val merged = mergeContinuationResponses(listOf(first, second))

    assertEquals(listOf<String?>("Hello ", "world"), merged.parts?.map { it.text })
    assertEquals(
      listOf<String?>("a", "b"),
      merged.candidates?.first()?.citationMetadata?.citations?.map { it.uri },
    )
  }

  @Test
  fun testMergeShiftsGroundingSupportChunkAndPartIndices() {
    val first =
      GenerateContentResponse(
        candidates =
          listOf(
            Candidate(
              content =
                Content(
                  role = "model",
                  parts = listOf(Part(text = "Thinking...", thought = true), Part(text = "part 1 ")),
                ),
              finishReason = FinishReason.CONTINUATION,
              continuationToken = TOKEN_1,
              groundingMetadata =
                GroundingMetadata(
                  groundingChunks =
                    listOf(
                      GroundingChunk(
                        image =
                          GroundingChunkImage(
                            imageUri = "https://img1.example/a.png",
                            sourceUri = "https://source1.example/img",
                            title = "Image 1",
                            domain = "source1.example",
                          )
                      ),
                      GroundingChunk(
                        web = GroundingChunkWeb(uri = "https://source1.example", title = "Source 1")
                      ),
                    ),
                  groundingSupports =
                    listOf(
                      GroundingSupport(
                        groundingChunkIndices = listOf(0, 1),
                        confidenceScores = listOf(0.9, 0.8),
                        segment =
                          Segment(partIndex = 1, startIndex = 0, endIndex = 7, text = "part 1 "),
                      )
                    ),
                ),
            )
          )
      )
    val second =
      GenerateContentResponse(
        candidates =
          listOf(
            Candidate(
              content = Content(role = "model", parts = listOf(Part(text = "part 2 "))),
              finishReason = FinishReason.STOP,
              groundingMetadata =
                GroundingMetadata(
                  groundingChunks =
                    listOf(
                      GroundingChunk(
                        web = GroundingChunkWeb(uri = "https://source2.example", title = "Source 2")
                      )
                    ),
                  groundingSupports =
                    listOf(
                      GroundingSupport(
                        groundingChunkIndices = listOf(0),
                        confidenceScores = listOf(0.95),
                        segment =
                          Segment(partIndex = 0, startIndex = 0, endIndex = 7, text = "part 2 "),
                      ),
                      GroundingSupport(
                        groundingChunkIndices = listOf(0),
                        segment = Segment(startIndex = 0, endIndex = 7, text = "part 2 "),
                      ),
                    ),
                ),
            )
          )
      )

    val merged = mergeContinuationResponses(listOf(first, second))
    val cand = merged.candidates?.first()
    val gm = cand?.groundingMetadata

    assertEquals(3, gm?.groundingChunks?.size)
    assertEquals(3, gm?.groundingSupports?.size)

    val s0 = gm?.groundingSupports?.get(0)
    assertEquals(listOf(0, 1), s0?.groundingChunkIndices)
    assertEquals(1, s0?.segment?.partIndex)

    val s1 = gm?.groundingSupports?.get(1)
    assertEquals(listOf(2), s1?.groundingChunkIndices)
    assertEquals(2, s1?.segment?.partIndex)
    assertEquals(
      "https://source2.example",
      gm?.groundingChunks?.get(s1!!.groundingChunkIndices!!.first())?.web?.uri,
    )
    assertEquals("part 2 ", cand?.content?.parts?.get(s1!!.segment!!.partIndex!!)?.text)

    val s2 = gm?.groundingSupports?.get(2)
    assertEquals(listOf(2), s2?.groundingChunkIndices)
    assertEquals(2, s2?.segment?.partIndex)
  }

  @Test
  fun testMergeGroundingSupportsCornerCases() = runTest {
    val hop1 =
      GenerateContentResponse(
        candidates =
          listOf(
            Candidate(
              index = 0,
              content =
                Content(
                  role = "model",
                  parts = listOf(Part(text = "c0-h1-p0 "), Part(text = "c0-h1-p1 ")),
                ),
              groundingMetadata = null,
              finishReason = FinishReason.CONTINUATION,
              continuationToken = TOKEN_1,
            ),
            Candidate(
              index = 1,
              content = Content(role = "model", parts = listOf(Part(text = "c1-h1-p0 "))),
              groundingMetadata =
                GroundingMetadata(
                  groundingChunks =
                    listOf(GroundingChunk(web = GroundingChunkWeb(uri = "https://c1-s0.example")))
                ),
              finishReason = FinishReason.CONTINUATION,
              continuationToken = TOKEN_1,
            ),
          )
      )

    val hop2 =
      GenerateContentResponse(
        candidates =
          listOf(
            Candidate(
              index = 0,
              content = Content(role = "model", parts = listOf(Part(text = "c0-h2-p0 "))),
              groundingMetadata =
                GroundingMetadata(
                  groundingChunks =
                    listOf(
                      GroundingChunk(web = GroundingChunkWeb(uri = "https://c0-s0.example")),
                      GroundingChunk(web = GroundingChunkWeb(uri = "https://c0-s1.example")),
                    )
                ),
              finishReason = FinishReason.CONTINUATION,
              continuationToken = TOKEN_2,
            ),
            Candidate(
              index = 1,
              content = Content(role = "model", parts = listOf(Part(text = "c1-h2-p0 "))),
              groundingMetadata =
                GroundingMetadata(
                  groundingChunks =
                    listOf(GroundingChunk(web = GroundingChunkWeb(uri = "https://c1-s1.example"))),
                  groundingSupports =
                    listOf(
                      GroundingSupport(
                        groundingChunkIndices = listOf(0),
                        segment = Segment(partIndex = 0, text = "c1-h2-p0 "),
                      )
                    ),
                ),
              finishReason = FinishReason.STOP,
            ),
          )
      )

    val hop3 =
      GenerateContentResponse(
        candidates =
          listOf(
            Candidate(
              index = 0,
              content = Content(role = "model", parts = listOf(Part(text = "c0-h3-p0."))),
              groundingMetadata =
                GroundingMetadata(
                  groundingChunks =
                    listOf(GroundingChunk(web = GroundingChunkWeb(uri = "https://c0-s2.example"))),
                  groundingSupports =
                    listOf(
                      GroundingSupport(groundingChunkIndices = listOf(0), segment = null),
                      GroundingSupport(
                        groundingChunkIndices = emptyList(),
                        segment = Segment(partIndex = 0, text = "c0-h3-p0."),
                      ),
                      GroundingSupport(
                        groundingChunkIndices = listOf(0),
                        segment = Segment(partIndex = 0, text = "c0-h3-p0."),
                      ),
                    ),
                ),
              finishReason = FinishReason.STOP,
            )
          )
      )

    val merged = mergeContinuationResponses(listOf(hop1, hop2, hop3))

    val cand0 = merged.candidates?.get(0)
    val gm0 = cand0?.groundingMetadata
    assertEquals(4, cand0?.content?.parts?.size)
    assertEquals(3, gm0?.groundingChunks?.size)
    assertEquals(3, gm0?.groundingSupports?.size)
    assertNull(gm0?.groundingSupports?.get(0)?.segment)
    assertEquals(listOf(2), gm0?.groundingSupports?.get(0)?.groundingChunkIndices)
    assertEquals(emptyList(), gm0?.groundingSupports?.get(1)?.groundingChunkIndices)
    assertEquals(3, gm0?.groundingSupports?.get(1)?.segment?.partIndex)
    val s2 = gm0?.groundingSupports?.get(2)
    assertEquals(
      "https://c0-s2.example",
      gm0?.groundingChunks?.get(s2!!.groundingChunkIndices!!.first())?.web?.uri,
    )
    assertEquals("c0-h3-p0.", cand0?.content?.parts?.get(s2!!.segment!!.partIndex!!)?.text)

    val cand1 = merged.candidates?.get(1)
    val gm1 = cand1?.groundingMetadata
    assertEquals(listOf(1), gm1?.groundingSupports?.get(0)?.groundingChunkIndices)
    assertEquals(1, gm1?.groundingSupports?.get(0)?.segment?.partIndex)
    assertEquals(
      "https://c1-s1.example",
      gm1
        ?.groundingChunks
        ?.get(gm1.groundingSupports!!.first().groundingChunkIndices!!.first())
        ?.web
        ?.uri,
    )
    assertEquals(
      "c1-h2-p0 ",
      cand1?.content?.parts?.get(gm1!!.groundingSupports!!.first().segment!!.partIndex!!)?.text,
    )
  }

  @Test
  fun testGenerateWithAutomaticContinuationMergesGroundingSupportsAcrossHops() = runTest {
    val hop1 =
      GenerateContentResponse(
        candidates =
          listOf(
            Candidate(
              content = Content(role = "model", parts = listOf(Part(text = "part 1 "))),
              finishReason = FinishReason.CONTINUATION,
              continuationToken = TOKEN_1,
              groundingMetadata =
                GroundingMetadata(
                  groundingChunks =
                    listOf(
                      GroundingChunk(web = GroundingChunkWeb(uri = "https://source1.example"))
                    ),
                  groundingSupports =
                    listOf(
                      GroundingSupport(
                        groundingChunkIndices = listOf(0),
                        segment = Segment(partIndex = 0, text = "part 1 "),
                      )
                    ),
                ),
            )
          )
      )
    val hop2 =
      GenerateContentResponse(
        candidates =
          listOf(
            Candidate(
              content = Content(role = "model", parts = listOf(Part(text = "part 2 "))),
              finishReason = FinishReason.STOP,
              groundingMetadata =
                GroundingMetadata(
                  groundingChunks =
                    listOf(
                      GroundingChunk(web = GroundingChunkWeb(uri = "https://source2.example"))
                    ),
                  groundingSupports =
                    listOf(
                      GroundingSupport(
                        groundingChunkIndices = listOf(0),
                        segment = Segment(partIndex = 0, text = "part 2 "),
                      )
                    ),
                ),
            )
          )
      )

    val model = FakeModel(hop1, hop2)
    val merged = generateWithAutomaticContinuation(CONFIG) { model.send(it) }

    val cand = merged.candidates?.first()
    val gm = cand?.groundingMetadata
    val s = gm?.groundingSupports?.last()
    assertEquals(
      "https://source2.example",
      gm?.groundingChunks?.get(s!!.groundingChunkIndices!!.first())?.web?.uri,
    )
    assertEquals("part 2 ", cand?.content?.parts?.get(s!!.segment!!.partIndex!!)?.text)
  }

  @Test
  fun testStreamWithAutomaticContinuationPreservesPerChunkGroundingAndMergesAcrossHops() = runTest {
    val hop1Chunk1 = response("hop 1 intro, ")
    val hop1Chunk2 =
      GenerateContentResponse(
        candidates =
          listOf(
            Candidate(
              content = Content(role = "model", parts = listOf(Part(text = "hop 1 grounded. "))),
              finishReason = FinishReason.CONTINUATION,
              continuationToken = TOKEN_1,
              groundingMetadata =
                GroundingMetadata(
                  groundingChunks =
                    listOf(
                      GroundingChunk(web = GroundingChunkWeb(uri = "https://source1.example"))
                    ),
                  groundingSupports =
                    listOf(
                      GroundingSupport(
                        groundingChunkIndices = listOf(0),
                        segment = Segment(partIndex = 0, text = "hop 1 grounded. "),
                      )
                    ),
                ),
            )
          )
      )
    val hop2Chunk1 = response("hop 2 intro, ")
    val hop2Chunk2 =
      GenerateContentResponse(
        candidates =
          listOf(
            Candidate(
              content = Content(role = "model", parts = listOf(Part(text = "hop 2 grounded."))),
              finishReason = FinishReason.STOP,
              groundingMetadata =
                GroundingMetadata(
                  groundingChunks =
                    listOf(
                      GroundingChunk(web = GroundingChunkWeb(uri = "https://source2.example"))
                    ),
                  groundingSupports =
                    listOf(
                      GroundingSupport(
                        groundingChunkIndices = listOf(0),
                        segment = Segment(partIndex = 0, text = "hop 2 grounded."),
                      )
                    ),
                ),
            )
          )
      )

    val model = FakeStreamingModel(flowOf(hop1Chunk1, hop1Chunk2), flowOf(hop2Chunk1, hop2Chunk2))
    val chunks = streamWithAutomaticContinuation(CONFIG) { model.send(it) }.toList()

    assertEquals(4, chunks.size)
    val sHop1 = chunks[1].candidates?.first()?.groundingMetadata?.groundingSupports?.first()
    assertEquals(
      "https://source1.example",
      chunks[1]
        .candidates
        ?.first()
        ?.groundingMetadata
        ?.groundingChunks
        ?.get(sHop1!!.groundingChunkIndices!!.first())
        ?.web
        ?.uri,
    )
    assertNull(chunks[1].finishReason)

    val sHop2 = chunks[3].candidates?.first()?.groundingMetadata?.groundingSupports?.first()
    assertEquals(
      "https://source2.example",
      chunks[3]
        .candidates
        ?.first()
        ?.groundingMetadata
        ?.groundingChunks
        ?.get(sHop2!!.groundingChunkIndices!!.first())
        ?.web
        ?.uri,
    )
    assertEquals(FinishReason.STOP, chunks[3].finishReason)

    val merged = mergeContinuationResponses(chunks)
    val cand = merged.candidates?.first()
    val gm = cand?.groundingMetadata
    assertEquals(2, gm?.groundingSupports?.size)
    assertEquals(
      "https://source1.example",
      gm?.groundingChunks?.get(gm.groundingSupports!![0].groundingChunkIndices!!.first())?.web?.uri,
    )
    assertEquals(
      "hop 1 grounded. ",
      cand?.content?.parts?.get(gm!!.groundingSupports!![0].segment!!.partIndex!!)?.text,
    )
    assertEquals(
      "https://source2.example",
      gm?.groundingChunks?.get(gm.groundingSupports!![1].groundingChunkIndices!!.first())?.web?.uri,
    )
    assertEquals(
      "hop 2 grounded.",
      cand?.content?.parts?.get(gm!!.groundingSupports!![1].segment!!.partIndex!!)?.text,
    )
  }

  @Test
  fun testKeepsAnEmptyTextPartFromARequestSpentThinking() = runTest {
    val signature = "signature".encodeToByteArray()
    val thinking =
      GenerateContentResponse(
        candidates =
          listOf(
            Candidate(
              content =
                Content(
                  role = "model",
                  parts = listOf(Part(text = "", thoughtSignature = signature)),
                ),
              finishReason = FinishReason.CONTINUATION,
              continuationToken = TOKEN_1,
            )
          )
      )
    val model = FakeModel(thinking, response("Final answer.", FinishReason.STOP))

    val merged = generateWithAutomaticContinuation(CONFIG) { model.send(it) }

    assertEquals("Final answer.", merged.text)
    val parts = merged.parts!!
    assertEquals(2, parts.size)
    assertEquals("", parts[0].text)
    assertContentEquals(signature, parts[0].thoughtSignature)
    assertEquals("Final answer.", parts[1].text)
  }

  @Test
  fun testMergeSumsTokenCounts() {
    val first =
      response("Hello ", FinishReason.CONTINUATION, TOKEN_1)
        .copy(
          usageMetadata =
            GenerateContentResponseUsageMetadata(
              promptTokenCount = 10,
              cachedContentTokenCount = 4,
              candidatesTokenCount = 100,
              thoughtsTokenCount = 5,
              totalTokenCount = 115,
              promptTokensDetails = listOf(ModalityTokenCount(MediaModality.TEXT, 10)),
              cacheTokensDetails = listOf(ModalityTokenCount(MediaModality.TEXT, 4)),
              candidatesTokensDetails = listOf(ModalityTokenCount(MediaModality.TEXT, 100)),
            )
        )
    val second =
      response("world", FinishReason.STOP)
        .copy(
          usageMetadata =
            GenerateContentResponseUsageMetadata(
              promptTokenCount = 110,
              cachedContentTokenCount = 100,
              candidatesTokenCount = 50,
              totalTokenCount = 160,
              promptTokensDetails = listOf(ModalityTokenCount(MediaModality.TEXT, 110)),
              cacheTokensDetails = listOf(ModalityTokenCount(MediaModality.TEXT, 100)),
              candidatesTokensDetails =
                listOf(
                  ModalityTokenCount(MediaModality.TEXT, 50),
                  ModalityTokenCount(MediaModality.IMAGE, 3),
                ),
            )
        )

    val merged = mergeContinuationResponses(listOf(first, second))
    val usage = merged.usageMetadata

    assertEquals(120, usage?.promptTokenCount)
    assertEquals(10, merged.initialPromptTokenCount)
    assertEquals(10, usage?.initialPromptTokenCount)
    assertEquals(104, usage?.cachedContentTokenCount)
    assertEquals(150, usage?.candidatesTokenCount)
    assertEquals(5, usage?.thoughtsTokenCount)
    assertEquals(275, usage?.totalTokenCount)
    assertEquals(listOf(ModalityTokenCount(MediaModality.TEXT, 120)), usage?.promptTokensDetails)
    assertEquals(listOf(ModalityTokenCount(MediaModality.TEXT, 104)), usage?.cacheTokensDetails)
    assertEquals(
      listOf(
        ModalityTokenCount(MediaModality.TEXT, 150),
        ModalityTokenCount(MediaModality.IMAGE, 3),
      ),
      usage?.candidatesTokensDetails,
    )
  }

  @Test
  fun testMergeTakesHowTheGenerationEndedFromTheLastResponse() {
    val first =
      response("Hello ", FinishReason.CONTINUATION, TOKEN_1).withFirstCandidate {
        copy(finishMessage = "Stopped early.")
      }
    val second = response("world", FinishReason.STOP)

    val candidate = mergeContinuationResponses(listOf(first, second)).candidates?.first()

    assertEquals(FinishReason.STOP, candidate?.finishReason)
    assertNull(candidate?.continuationToken)
    assertNull(candidate?.finishMessage)
  }

  @Test
  fun testMergeKeepsTheLatestSafetyRatingPerCategory() {
    val first =
      response("Hello ", FinishReason.CONTINUATION, TOKEN_1).withFirstCandidate {
        copy(
          safetyRatings =
            listOf(
              SafetyRating(
                category = HarmCategory.HARM_CATEGORY_HARASSMENT,
                probability = HarmProbability.LOW,
              ),
              SafetyRating(
                category = HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                probability = HarmProbability.LOW,
              ),
            )
        )
      }
    val second =
      response("world", FinishReason.STOP).withFirstCandidate {
        copy(
          safetyRatings =
            listOf(
              SafetyRating(
                category = HarmCategory.HARM_CATEGORY_HARASSMENT,
                probability = HarmProbability.MEDIUM,
              )
            )
        )
      }

    val ratings =
      mergeContinuationResponses(listOf(first, second)).candidates?.first()?.safetyRatings

    assertEquals(
      listOf(
        SafetyRating(
          category = HarmCategory.HARM_CATEGORY_HARASSMENT,
          probability = HarmProbability.MEDIUM,
        ),
        SafetyRating(
          category = HarmCategory.HARM_CATEGORY_HATE_SPEECH,
          probability = HarmProbability.LOW,
        ),
      ),
      ratings,
    )
  }

  @Test
  fun testMergeTakesOtherValuesFromTheLastResponseUnlessUnset() {
    val first =
      response("Hello ", FinishReason.CONTINUATION, TOKEN_1)
        .copy(
          modelVersion = "model-1",
          responseId = "first",
          sdkHttpResponse = HttpResponse(body = "first"),
        )
    val second =
      response("world", FinishReason.STOP)
        .copy(responseId = "second", sdkHttpResponse = HttpResponse(body = "second"))

    val merged = mergeContinuationResponses(listOf(first, second))

    assertEquals("model-1", merged.modelVersion)
    assertEquals("second", merged.responseId)
    assertEquals(HttpResponse(body = "second"), merged.sdkHttpResponse)
  }

  @Test
  fun testMergeMergesEachCandidateWithTheOneOfTheSameIndex() {
    val first =
      GenerateContentResponse(
        candidates =
          listOf(
            candidate(0, "Hello ", FinishReason.CONTINUATION),
            candidate(1, "Bonjour ", FinishReason.CONTINUATION),
            candidate(2, "Hola", FinishReason.STOP),
          )
      )
    // Listed out of order, so the candidates can only be matched by index.
    val second =
      GenerateContentResponse(
        candidates =
          listOf(
            candidate(1, "le monde", FinishReason.STOP),
            candidate(0, "world", FinishReason.STOP),
          )
      )

    val candidates = mergeContinuationResponses(listOf(first, second)).candidates

    assertEquals(
      listOf("Hello world", "Bonjour le monde", "Hola"),
      candidates?.map { candidate -> candidate.content?.parts?.joinToString("") { it.text!! } },
    )
    assertEquals(FinishReason.STOP, candidates?.get(1)?.finishReason)
  }

  @Test
  fun testMergeClearsFinishReasonAndPreservesPromptFeedbackWhenHopHasNoCandidates() {
    val first =
      response("Partial output before block.", FinishReason.CONTINUATION, TOKEN_1)
        .withFirstCandidate { copy(finishMessage = "Continuing") }
        .copy(
          usageMetadata =
            GenerateContentResponseUsageMetadata(
              promptTokenCount = 10,
              candidatesTokenCount = 100,
              totalTokenCount = 110,
            )
        )

    for (emptyCandidates in listOf<List<Candidate>?>(null, emptyList())) {
      val second =
        GenerateContentResponse(
          candidates = emptyCandidates,
          promptFeedback =
            GenerateContentResponsePromptFeedback(
              blockReason = BlockedReason.OTHER,
              blockReasonMessage = "Blocked on continuation hop",
            ),
          usageMetadata =
            GenerateContentResponseUsageMetadata(promptTokenCount = 110, totalTokenCount = 110),
        )

      val merged = mergeContinuationResponses(listOf(first, second))

      assertEquals("Partial output before block.", merged.text)
      val candidate = merged.candidates?.single()
      assertNull(candidate?.finishReason)
      assertNull(candidate?.continuationToken)
      assertNull(candidate?.finishMessage)
      assertEquals(BlockedReason.OTHER, merged.promptFeedback?.blockReason)
      assertEquals("Blocked on continuation hop", merged.promptFeedback?.blockReasonMessage)
      assertEquals(120, merged.usageMetadata?.promptTokenCount)
      assertEquals(10, merged.usageMetadata?.initialPromptTokenCount)
    }
  }
}

private fun candidate(index: Int, text: String, finishReason: FinishReason): Candidate =
  Candidate(
    index = index,
    content = Content(role = "model", parts = listOf(Part(text = text))),
    finishReason = finishReason,
  )

private fun GenerateContentResponse.withFirstCandidate(
  change: Candidate.() -> Candidate
): GenerateContentResponse = copy(candidates = listOf(candidates!!.first().change()))
