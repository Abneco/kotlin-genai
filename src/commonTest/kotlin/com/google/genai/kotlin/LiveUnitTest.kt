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

import com.google.genai.kotlin.types.Content
import com.google.genai.kotlin.types.FunctionDeclaration
import com.google.genai.kotlin.types.LiveClientContextUpdate
import com.google.genai.kotlin.types.LiveClientContextUpdateTools
import com.google.genai.kotlin.types.Part
import com.google.genai.kotlin.types.Tool
import io.ktor.client.plugins.websocket.DefaultClientWebSocketSession
import io.ktor.websocket.Frame
import io.ktor.websocket.readText
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import io.mockk.spyk
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlinx.coroutines.Job
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

private const val TEST_MODEL = "gemini-3.1-flash-live-preview"

/**
 * Tests for [Live] that need no server, using a mocked WebSocket session. See [LiveTest] for the
 * record/replay tests.
 */
class LiveUnitTest {

  private val job = Job()

  private val socket =
    mockk<DefaultClientWebSocketSession>().also { every { it.coroutineContext } returns job }

  private fun live(): Live {
    val apiClient = spyk(ApiClient(apiKey = "test-api-key"))
    coEvery { apiClient.openWebSocketSession(any()) } returns socket
    return Live(apiClient)
  }

  @Test
  fun testConnect_setupFailureClosesTheSocket() = runTest {
    coEvery { socket.send(any<Frame>()) } throws IllegalStateException("setup send failed")

    val thrown = assertFailsWith<IllegalStateException> { live().connect(TEST_MODEL) }

    assertEquals("setup send failed", thrown.message, "the original failure must not be masked")
    assertTrue(job.isCancelled, "the open socket must not be left behind")
  }

  @Test
  fun testConnect_successLeavesTheSocketOpen() = runTest {
    coEvery { socket.send(any<Frame>()) } returns Unit

    live().connect(TEST_MODEL)

    assertFalse(job.isCancelled)
  }

  @Test
  fun testSendContextUpdate_systemInstruction() = runTest {
    listOf(false, true).forEach { enterprise ->
      val sentFrames = mutableListOf<Frame>()
      coEvery { socket.send(capture(sentFrames)) } returns Unit
      val apiClient =
        ApiClient(
          apiKey = "test-api-key",
          enterprise = enterprise,
          project = if (enterprise) "test-project" else null,
          location = if (enterprise) "us-central1" else null,
        )
      val session = LiveSession(socket, apiClient)

      session.sendContextUpdate(
        LiveClientContextUpdate(
          systemInstruction = Content(parts = listOf(Part(text = "You are a helpful assistant.")))
        )
      )

      assertEquals(1, sentFrames.size)
      val json = Json.parseToJsonElement((sentFrames.single() as Frame.Text).readText()).jsonObject
      assertEquals(
        "You are a helpful assistant.",
        json["contextUpdate"]!!
          .jsonObject["systemInstruction"]!!
          .jsonObject["parts"]!!
          .jsonArray[0]
          .jsonObject["text"]!!
          .jsonPrimitive
          .content,
      )
    }
  }

  @Test
  fun testSendContextUpdate_clearTools() = runTest {
    listOf(false, true).forEach { enterprise ->
      val sentFrames = mutableListOf<Frame>()
      coEvery { socket.send(capture(sentFrames)) } returns Unit
      val apiClient =
        ApiClient(
          apiKey = "test-api-key",
          enterprise = enterprise,
          project = if (enterprise) "test-project" else null,
          location = if (enterprise) "us-central1" else null,
        )
      val session = LiveSession(socket, apiClient)

      session.sendContextUpdate(
        LiveClientContextUpdate(tools = LiveClientContextUpdateTools(tools = emptyList()))
      )

      assertEquals(1, sentFrames.size)
      val json = Json.parseToJsonElement((sentFrames.single() as Frame.Text).readText()).jsonObject
      assertEquals(
        0,
        json["contextUpdate"]!!.jsonObject["tools"]!!.jsonObject["tools"]!!.jsonArray.size,
      )
    }
  }

  @Test
  fun testSendContextUpdate_withFunctionDeclaration() = runTest {
    listOf(false, true).forEach { enterprise ->
      val sentFrames = mutableListOf<Frame>()
      coEvery { socket.send(capture(sentFrames)) } returns Unit
      val apiClient =
        ApiClient(
          apiKey = "test-api-key",
          enterprise = enterprise,
          project = if (enterprise) "test-project" else null,
          location = if (enterprise) "us-central1" else null,
        )
      val session = LiveSession(socket, apiClient)

      session.sendContextUpdate(
        LiveClientContextUpdate(
          tools =
            LiveClientContextUpdateTools(
              tools =
                listOf(
                  Tool(
                    functionDeclarations =
                      listOf(
                        FunctionDeclaration(
                          name = "get_current_weather",
                          description = "Gets the weather for a location.",
                        )
                      )
                  )
                )
            )
        )
      )

      assertEquals(1, sentFrames.size)
      val json = Json.parseToJsonElement((sentFrames.single() as Frame.Text).readText()).jsonObject
      val toolsArray = json["contextUpdate"]!!.jsonObject["tools"]!!.jsonObject["tools"]!!.jsonArray
      assertEquals(1, toolsArray.size)
      assertEquals(
        "get_current_weather",
        toolsArray[0]
          .jsonObject["functionDeclarations"]!!
          .jsonArray[0]
          .jsonObject["name"]!!
          .jsonPrimitive
          .content,
      )
    }
  }

  @Test
  fun testSendContextUpdate_bothSystemInstructionAndTools() = runTest {
    listOf(false, true).forEach { enterprise ->
      val sentFrames = mutableListOf<Frame>()
      coEvery { socket.send(capture(sentFrames)) } returns Unit
      val apiClient =
        ApiClient(
          apiKey = "test-api-key",
          enterprise = enterprise,
          project = if (enterprise) "test-project" else null,
          location = if (enterprise) "us-central1" else null,
        )
      val session = LiveSession(socket, apiClient)

      session.sendContextUpdate(
        LiveClientContextUpdate(
          systemInstruction = Content(parts = listOf(Part(text = "Updated instruction."))),
          tools =
            LiveClientContextUpdateTools(
              tools =
                listOf(
                  Tool(
                    functionDeclarations =
                      listOf(
                        FunctionDeclaration(
                          name = "get_current_weather",
                          description = "Gets the weather for a location.",
                        )
                      )
                  )
                )
            ),
        )
      )

      assertEquals(1, sentFrames.size)
      val json = Json.parseToJsonElement((sentFrames.single() as Frame.Text).readText()).jsonObject
      assertEquals(
        "Updated instruction.",
        json["contextUpdate"]!!
          .jsonObject["systemInstruction"]!!
          .jsonObject["parts"]!!
          .jsonArray[0]
          .jsonObject["text"]!!
          .jsonPrimitive
          .content,
      )
      val toolsArray = json["contextUpdate"]!!.jsonObject["tools"]!!.jsonObject["tools"]!!.jsonArray
      assertEquals(1, toolsArray.size)
      assertEquals(
        "get_current_weather",
        toolsArray[0]
          .jsonObject["functionDeclarations"]!!
          .jsonArray[0]
          .jsonObject["name"]!!
          .jsonPrimitive
          .content,
      )
    }
  }
}
