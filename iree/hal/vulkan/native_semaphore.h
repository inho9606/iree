// Copyright 2019 Google LLC
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

#ifndef IREE_HAL_VULKAN_NATIVE_SEMAPHORE_H_
#define IREE_HAL_VULKAN_NATIVE_SEMAPHORE_H_

#include "iree/hal/api.h"
#include "iree/hal/vulkan/handle_util.h"

iree_status_t iree_hal_vulkan_native_semaphore_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore);

VkSemaphore iree_hal_vulkan_native_semaphore_handle(
    iree_hal_semaphore_t* semaphore);

iree_status_t iree_hal_vulkan_native_semaphore_multi_wait(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    const iree_hal_semaphore_list_t* semaphore_list, iree_time_t deadline_ns,
    VkSemaphoreWaitFlags wait_flags);

#endif  // IREE_HAL_VULKAN_NATIVE_SEMAPHORE_H_
