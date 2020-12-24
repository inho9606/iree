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

#include "iree/hal/local/task_queue.h"

#include "iree/base/tracing.h"
#include "iree/task/submission.h"

void iree_hal_task_queue_initialize(iree_string_view_t identifier,
                                    iree_task_executor_t* executor,
                                    iree_arena_block_pool_t* block_pool,
                                    iree_hal_task_queue_t* out_queue) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, identifier.data, identifier.size);

  out_queue->executor = executor;
  iree_task_executor_retain(out_queue->executor);
  iree_task_scope_initialize(identifier, &out_queue->scope);
  out_queue->block_pool = block_pool;
  iree_slim_mutex_initialize(&out_queue->mutex);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_task_queue_deinitialize(iree_hal_task_queue_t* queue) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_deinitialize(&queue->mutex);
  iree_task_scope_deinitialize(&queue->scope);
  iree_task_executor_release(queue->executor);

  IREE_TRACE_ZONE_END(z0);
}

typedef struct {
  // Call to iree_hal_cmd_task_queue_issue.
  iree_task_call_t task;

  // Command buffers to be issued in the order the appeared in the submission.
  iree_host_size_t command_buffer_count;
  iree_hal_command_buffer_t* command_buffers[];
} iree_hal_task_queue_issue_cmd_t;

static iree_status_t iree_hal_cmd_task_queue_issue(uintptr_t user_context,
                                                   uintptr_t task_context) {
  const iree_hal_task_queue_issue_cmd_t* cmd =
      (const iree_hal_task_queue_issue_cmd_t*)user_context;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  // TODO(benvanik): prevent needing to do this scope reassignment by
  // reworking scopes or keeping a fixup list?
  // DO NOT SUBMIT reassign to cmd->task.scope

  // - issue cmd bufs

  // - lock
  // - clear queue tail if tail == this
  // - unlock

  IREE_TRACE_ZONE_END(z0);
  return status;
}

typedef struct {
  // Call to iree_hal_cmd_task_queue_retire.
  iree_task_call_t task;

  // Original arena used for all transient allocations required for the
  // submission. All queue-related commands are allocated from this, **including
  // this retire command**.
  // TODO(#4026): this arena needs to be discarded on scope failure.
  iree_arena_allocator_t arena;

  // signal semas
} iree_hal_task_queue_retire_cmd_t;

static iree_status_t iree_hal_cmd_task_queue_retire(uintptr_t user_context,
                                                    uintptr_t task_context) {
  iree_hal_task_queue_retire_cmd_t* cmd =
      (iree_hal_task_queue_retire_cmd_t*)user_context;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Signal all semaphores to their new values.
  // DO NOT SUBMIT
  iree_status_t status = iree_ok_status();

  // Drop all memory used by the submission (**including cmd**).
  iree_arena_deinitialize(&cmd->arena);
  cmd = NULL;

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_task_queue_submit_batch(
    iree_hal_task_queue_t* queue, const iree_hal_submission_batch_t* batch) {
  // Setup the arena used to allocate submission-related tasks.
  // We'll stash this arena on the retire command such that it is discarded once
  // the submission has completed.
  iree_arena_allocator_t arena;
  iree_arena_initialize(queue->block_pool, &arena);

  iree_status_t status = iree_ok_status();

  // waits

  // Task to issue all the command buffers in the batch.
  // After this task completes the commands have been issued but have not yet
  // completed and the issued commands may complete in any order.
  iree_hal_task_queue_issue_cmd_t* issue_cmd = NULL;
  iree_host_size_t total_issue_cmd_size =
      sizeof(*issue_cmd) +
      batch->command_buffer_count * sizeof(*issue_cmd->command_buffers);
  status =
      iree_arena_allocate(&arena, total_issue_cmd_size, (void**)&issue_cmd);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_arena_deinitialize(&arena);
    return status;
  }
  iree_task_call_initialize(
      &queue->scope,
      iree_task_make_closure(iree_hal_cmd_task_queue_issue,
                             (uintptr_t)issue_cmd),
      &issue_cmd->task);
  issue_cmd->command_buffer_count = batch->command_buffer_count;
  memcpy(issue_cmd->command_buffers, batch->command_buffers,
         issue_cmd->command_buffer_count * sizeof(*issue_cmd->command_buffers));

  // Task to retire the submission and free the transient memory allocated for
  // it. The task is issued only once all commands from all command buffers in
  // the submission complete.
  iree_hal_task_queue_retire_cmd_t* retire_cmd = NULL;
  status =
      iree_arena_allocate(&arena, sizeof(*retire_cmd), (void**)&retire_cmd);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_arena_deinitialize(&arena);
    return status;
  }
  iree_task_call_initialize(
      &queue->scope,
      iree_task_make_closure(iree_hal_cmd_task_queue_retire,
                             (uintptr_t)retire_cmd),
      &retire_cmd->task);
  retire_cmd->arena = arena;

  // IREE_RETURN_IF_ERROR(
  //     iree_hal_task_command_buffer_stitch_batch(batch, &submission));

  iree_task_submission_t submission;
  iree_task_submission_initialize(&submission);

  // preserving queue issue order (but may retire in any order)
  // lock
  // chain previous tail to issue
  // set issue cmd as tail
  // unlock

  // Submit the tasks immediately. The executor may queue them up until we
  // force the flush after all batches have been processed.
  status = iree_task_executor_submit(queue->executor, &submission);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_task_submission_discard(&submission);
  }
  return status;
}

iree_status_t iree_hal_task_queue_submit(
    iree_hal_task_queue_t* queue, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches) {
  // For now we process each batch independently. To elide additional semaphore
  // work and prevent unneeded coordinator scheduling logic we could instead
  // build the whole DAG prior to submitting.
  for (iree_host_size_t i = 0; i < batch_count; ++i) {
    const iree_hal_submission_batch_t* batch = &batches[i];
    IREE_RETURN_IF_ERROR(iree_hal_task_queue_submit_batch(queue, batch));
  }
  return iree_task_executor_flush(queue->executor);
}

iree_status_t iree_hal_task_queue_wait_idle_with_deadline(
    iree_hal_task_queue_t* queue, iree_time_t deadline_ns) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_task_scope_wait_idle(&queue->scope, deadline_ns);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
