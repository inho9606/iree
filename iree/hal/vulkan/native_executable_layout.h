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

#ifndef IREE_HAL_VULKAN_NATIVE_EXECUTABLE_LAYOUT_H_
#define IREE_HAL_VULKAN_NATIVE_EXECUTABLE_LAYOUT_H_

// clang-format off: Must be included before all other headers:
#include "iree/hal/vulkan/vulkan_headers.h"
// clang-format on

#include "iree/hal/api.h"
#include "iree/hal/vulkan/handle_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_status_t iree_hal_vulkan_native_executable_layout_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_host_size_t push_constant_count,
    iree_hal_executable_layout_t** out_executable_layout);

VkPipelineLayout iree_hal_vulkan_native_executable_layout_handle(
    iree_hal_executable_layout_t* base_executable_layout);

iree_host_size_t iree_hal_vulkan_native_executable_layout_set_count(
    iree_hal_executable_layout_t* base_executable_layout);

iree_hal_descriptor_set_layout_t* iree_hal_vulkan_native_executable_layout_set(
    iree_hal_executable_layout_t* base_executable_layout, iree_host_size_t i);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_VULKAN_NATIVE_EXECUTABLE_LAYOUT_H_
