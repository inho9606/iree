// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_HAL_LOCAL_TASK_COMMAND_BUFFER_H_
#define IREE_HAL_LOCAL_TASK_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/arena.h"
#include "iree/task/scope.h"
#include "iree/task/submission.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_status_t iree_hal_task_command_buffer_create(
    iree_hal_device_t* device, iree_task_scope_t* scope,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_arena_block_pool_t* block_pool,
    iree_hal_command_buffer_t** out_command_buffer);

// Stitches together all of the command buffers in the |batch| and wires up the
// semaphore dependencies. A submission ready to be made to an executor is
// produced into |out_submission|.
iree_status_t iree_hal_task_command_buffer_stitch_batch(
    const iree_hal_submission_batch_t* batch,
    iree_task_submission_t* out_submission);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_TASK_COMMAND_BUFFER_H_
