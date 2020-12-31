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

#ifndef IREE_HAL_VULKAN_ENUMLATED_SEMAPHORE_H_
#define IREE_HAL_VULKAN_ENUMLATED_SEMAPHORE_H_

#include "iree/hal/api.h"
#include "iree/hal/vulkan/command_queue.h"
#include "iree/hal/vulkan/handle_util.h"
#include "iree/hal/vulkan/timepoint_util.h"

iree_status_t iree_hal_vulkan_emulated_semaphore_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    absl::Span<iree::hal::vulkan::CommandQueue*> command_queues,
    iree::hal::vulkan::TimePointSemaphorePool* semaphore_pool,
    uint64_t initial_value, iree_hal_semaphore_t** out_semaphore);

// Acquires a binary semaphore for waiting on the timeline to advance to the
// given |value|. The semaphore returned won't be waited by anyone else.
// |wait_fence| is the fence associated with the queue submission that waiting
// on this semaphore.
//
// Returns VK_NULL_HANDLE if there are no available semaphores for the given
// |value|.
iree_status_t iree_hal_vulkan_emulated_semaphore_acquire_wait_handle(
    iree_hal_semaphore_t* semaphore, uint64_t value,
    const iree::ref_ptr<iree::hal::vulkan::TimePointFence>& wait_fence,
    VkSemaphore* out_handle);

// Cancels the waiting attempt on the given binary |semaphore|. This allows
// the |semaphore| to be waited by others.
iree_status_t iree_hal_vulkan_emulated_semaphore_cancel_wait_handle(
    iree_hal_semaphore_t* semaphore, VkSemaphore handle);

// Acquires a binary semaphore for signaling the timeline to the given |value|.
// |value| must be smaller than the current timeline value. |signal_fence| is
// the fence associated with the queue submission that signals this semaphore.
iree_status_t iree_hal_vulkan_emulated_semaphore_acquire_signal_handle(
    iree_hal_semaphore_t* semaphore, uint64_t value,
    const iree::ref_ptr<iree::hal::vulkan::TimePointFence>& signal_fence,
    VkSemaphore* out_handle);

iree_status_t iree_hal_vulkan_emulated_semaphore_wait(
    iree_hal_semaphore_t* semaphore, uint64_t value, iree_time_t deadline_ns);

iree_status_t iree_hal_vulkan_emulated_semaphore_multi_wait(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    const iree_hal_semaphore_list_t* semaphore_list, iree_time_t deadline_ns,
    VkSemaphoreWaitFlags wait_flags);

// // Triggers necessary processing on all queues due to new values gotten
// // signaled for the given timeline |semaphore|.
// // Different clang-format versions disagree about asterisk placement.
// // clang-format off
// [this](Semaphore* /*semaphore*/) -> Status {
//   // clang-format on
//   IREE_TRACE_SCOPE0("<lambda>::OnSemaphoreSignal");
//   for (const auto& queue : command_queues_) {
//     IREE_RETURN_IF_ERROR(
//         static_cast<SerializingCommandQueue*>(queue.get())
//             ->AdvanceQueueSubmission());
//   }
//   return OkStatus();
// },
// // Triggers necessary processing on all queues due to failures for the
// // given timeline |semaphore|.
// [this](Semaphore* /*semaphore*/) {
//   IREE_TRACE_SCOPE0("<lambda>::OnSemaphoreFailure");
//   for (const auto& queue : command_queues_) {
//     static_cast<SerializingCommandQueue*>(queue.get())
//         ->AbortQueueSubmission();
//   }
// },
// // Triggers necessary processing on all queues due to the given |fence|
// // being signaled. This allows the queue to drop the fence ref it holds
// // even when we are not waiting on the queue directly.
// [this](absl::Span<VkFence> fences) {
//   IREE_TRACE_SCOPE0("<lambda>::OnFenceSignal");
//   for (const auto& queue : command_queues_) {
//     static_cast<SerializingCommandQueue*>(queue.get())
//         ->SignalFences(fences);
//   }
// },

iree_status_t iree_hal_vulkan_emulated_semaphore_wait(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    const iree_hal_semaphore_list_t* semaphore_list, iree_time_t deadline_ns,
    VkSemaphoreWaitFlags wait_flags) {
  // TODO(antiagainst): We actually should get the fences associated with the
  // emulated timeline semaphores so that we can wait them in a bunch. This
  // implementation is problematic if we wait to wait any and we have the
  // first semaphore taking extra long time but the following ones signal
  // quickly.
  for (iree_host_size_t i = 0; i < semaphore_list->count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_emulated_semaphore_wait(
        semaphore_list->semaphores[i], semaphore_list->payload_values[i],
        deadline_ns));
    if (wait_flags & VK_SEMAPHORE_WAIT_ANY_BIT) return iree_ok_status();
  }
  return iree_ok_status();
}

#endif  // IREE_HAL_VULKAN_ENUMLATED_SEMAPHORE_H_
