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

#ifndef IREE_HAL_LOCAL_TASK_QUEUE_H_
#define IREE_HAL_LOCAL_TASK_QUEUE_H_

#include "iree/base/api.h"
#include "iree/base/synchronization.h"
#include "iree/hal/api.h"
#include "iree/hal/local/arena.h"
#include "iree/task/executor.h"
#include "iree/task/scope.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct {
  iree_task_executor_t* executor;
  iree_task_scope_t scope;

  iree_arena_block_pool_t* block_pool;

  iree_slim_mutex_t mutex;
  // DAG builder?
} iree_hal_task_queue_t;

void iree_hal_task_queue_initialize(iree_string_view_t identifier,
                                    iree_task_executor_t* executor,
                                    iree_arena_block_pool_t* block_pool,
                                    iree_hal_task_queue_t* out_queue);

void iree_hal_task_queue_deinitialize(iree_hal_task_queue_t* queue);

iree_status_t iree_hal_task_queue_submit(
    iree_hal_task_queue_t* queue, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches);

iree_status_t iree_hal_task_queue_wait_idle_with_deadline(
    iree_hal_task_queue_t* queue, iree_time_t deadline_ns);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_TASK_QUEUE_H_
