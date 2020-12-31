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

#ifndef IREE_HAL_VULKAN_VMA_ALLOCATOR_H_
#define IREE_HAL_VULKAN_VMA_ALLOCATOR_H_

#include "iree/hal/api.h"
#include "iree/hal/vulkan/handle_util.h"
#include "iree/hal/vulkan/internal_vk_mem_alloc.h"

iree_status_t iree_hal_vulkan_vma_allocator_create(
    VkInstance instance, VkPhysicalDevice physical_device,
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    VmaRecordSettings record_settings, iree_hal_allocator_t** out_allocator);

#endif  // IREE_HAL_VULKAN_VMA_ALLOCATOR_H_
