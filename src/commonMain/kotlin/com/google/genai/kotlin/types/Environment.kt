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

import kotlin.jvm.JvmInline
import kotlinx.serialization.Serializable

/** The target environment where the computer use tool operates. */
@Serializable
@JvmInline
value class Environment(val value: String) {
  override fun toString(): String = value

  companion object {

    /** The environment is unspecified. */
    val ENVIRONMENT_UNSPECIFIED = Environment("ENVIRONMENT_UNSPECIFIED")

    /** The tool operates in a web browser. */
    val ENVIRONMENT_BROWSER = Environment("ENVIRONMENT_BROWSER")

    /** The tool operates in a mobile environment. */
    val ENVIRONMENT_MOBILE = Environment("ENVIRONMENT_MOBILE")

    /** The tool operates in a desktop environment. */
    val ENVIRONMENT_DESKTOP = Environment("ENVIRONMENT_DESKTOP")
  }
}
