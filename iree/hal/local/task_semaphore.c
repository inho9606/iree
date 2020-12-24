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

#include "iree/hal/local/task_semaphore.h"

#include "iree/base/tracing.h"
#include "iree/hal/device.h"

// Sentinel used the semaphore has failed and an error status is set.
#define IREE_HAL_TASK_SEMAPHORE_FAILURE_VALUE UINT64_MAX

typedef struct {
  iree_hal_resource_t resource;
  iree_hal_device_t* device;

  iree_atomic_int64_t value;
  iree_notification_t notification;

  iree_status_t failure_status;
} iree_hal_task_semaphore_t;

static const iree_hal_semaphore_vtable_t iree_hal_task_semaphore_vtable;

static iree_hal_task_semaphore_t* iree_hal_task_semaphore_cast(
    iree_hal_semaphore_t* base_semaphore) {
  return (iree_hal_task_semaphore_t*)base_semaphore;
}

iree_status_t iree_hal_task_semaphore_create(
    iree_hal_device_t* device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_task_semaphore_t* semaphore = NULL;
  iree_status_t status =
      iree_allocator_malloc(iree_hal_device_host_allocator(device),
                            sizeof(*semaphore), (void**)&semaphore);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_task_semaphore_vtable,
                                 &semaphore->resource);
    semaphore->device = device;

    iree_atomic_store_int64(&semaphore->value, initial_value,
                            iree_memory_order_release);
    iree_notification_initialize(&semaphore->notification);

    semaphore->failure_status = iree_ok_status();

    *out_semaphore = (iree_hal_semaphore_t*)semaphore;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_task_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);
  iree_allocator_t host_allocator =
      iree_hal_device_host_allocator(semaphore->device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_free(semaphore->failure_status);
  iree_notification_deinitialize(&semaphore->notification);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_task_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);
  *out_value = (uint64_t)iree_atomic_load_int64(&semaphore->value,
                                                iree_memory_order_acquire);
  return iree_ok_status();
}

static iree_status_t iree_hal_task_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);

  // exchange - if previous is failure then return status
  // if previous >= current return invalid argument
  iree_atomic_store_int64(&semaphore->value, new_value,
                          iree_memory_order_release);

  iree_notification_post(&semaphore->notification, IREE_ALL_WAITERS);
  return iree_ok_status();
}

static void iree_hal_task_semaphore_fail(iree_hal_semaphore_t* base_semaphore,
                                         iree_status_t status) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);
  // DO NOT SUBMIT
  // compare exchange ok with new status, release
  // signal to new value UINT64_MAX
}

static iree_status_t iree_hal_task_semaphore_wait_with_deadline(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_time_t deadline_ns) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);
  iree_hal_semaphore_list_t semaphore_list = {
      .count = 1,
      .semaphores = &base_semaphore,
      .payload_values = &value,
  };
  return iree_hal_device_wait_semaphores_with_deadline(
      semaphore->device, IREE_HAL_WAIT_MODE_ALL, &semaphore_list, deadline_ns);
}

static iree_status_t iree_hal_task_semaphore_wait_with_timeout(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_duration_t timeout_ns) {
  iree_hal_task_semaphore_t* semaphore =
      iree_hal_task_semaphore_cast(base_semaphore);
  iree_hal_semaphore_list_t semaphore_list = {
      .count = 1,
      .semaphores = &base_semaphore,
      .payload_values = &value,
  };
  return iree_hal_device_wait_semaphores_with_timeout(
      semaphore->device, IREE_HAL_WAIT_MODE_ALL, &semaphore_list, timeout_ns);
}

// DO NOT SUBMIT
// Returns a wait handle in |out_wait_handle| that will be signaled when the
// semaphore value reaches or exceeds |minimum_value|.
iree_status_t iree_hal_task_semaphore_acquire_wait_handle(
    iree_hal_semaphore_t* semaphore, uint64_t minimum_value,
    iree_wait_handle_t* out_wait_handle);

static const iree_hal_semaphore_vtable_t iree_hal_task_semaphore_vtable = {
    .destroy = iree_hal_task_semaphore_destroy,
    .query = iree_hal_task_semaphore_query,
    .signal = iree_hal_task_semaphore_signal,
    .fail = iree_hal_task_semaphore_fail,
    .wait_with_deadline = iree_hal_task_semaphore_wait_with_deadline,
    .wait_with_timeout = iree_hal_task_semaphore_wait_with_timeout,
};
