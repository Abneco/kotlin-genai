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

// Auto-generated code. Do not edit.

package com.google.genai.kotlin.types

import kotlinx.serialization.Serializable

/** A tool that enables the model to interact directly with a computer environment. */
@Serializable
data class ComputerUse(

  /** Required. The target environment where the computer use tool operates. */
  val environment: Environment? = null,

  /**
   * Optional. A list of predefined functions to explicitly exclude from the model call. By default,
   * [predefined functions](https://cloud.google.com/vertex-ai/generative-ai/docs/computer-use#supported-actions)
   * are included. Excluding functions allows for a more restricted action space or custom
   * definitions for predefined functions.
   */
  val excludedPredefinedFunctions: List<String>? = null,

  /**
   * Optional. Whether to enable the prompt injection detection check on the computer use request.
   */
  val enablePromptInjectionDetection: Boolean? = null,

  /** Optional. A list of safety policies to disable for the computer use tool. */
  val disabledSafetyPolicies: List<SafetyPolicy>? = null,
)
