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

#ifndef IREE_HAL_VULKAN_VULKAN_DEVICE_H_
#define IREE_HAL_VULKAN_VULKAN_DEVICE_H_

#include "iree/hal/api.h"
#include "iree/hal/vulkan/api.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/extensibility_util.h"

// Creates a device that owns and manages its own VkDevice.
iree_status_t iree_hal_vulkan_device_create(
    iree_string_view_t identifier,
    const iree_hal_vulkan_device_options_t* options,
    iree_hal_vulkan_syms_t* instance_syms, VkInstance instance,
    VkPhysicalDevice physical_device, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device);

#endif  // IREE_HAL_VULKAN_VULKAN_DEVICE_H_
