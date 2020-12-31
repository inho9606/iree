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

#include "iree/hal/vulkan/api.h"

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/extensibility_util.h"
#include "iree/hal/vulkan/vulkan_device.h"
#include "iree/hal/vulkan/vulkan_driver.h"

namespace iree {
namespace hal {
namespace vulkan {

//===----------------------------------------------------------------------===//
// iree::hal::vulkan::DynamicSymbols
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_vulkan_syms_create(
    void* vkGetInstanceProcAddr_fn, iree_allocator_t host_allocator,
    iree_hal_vulkan_syms_t** out_syms) {
  IREE_TRACE_SCOPE0("iree_hal_vulkan_syms_create");
  IREE_ASSERT_ARGUMENT(out_syms);
  *out_syms = nullptr;

  IREE_ASSIGN_OR_RETURN(
      auto syms, DynamicSymbols::Create([&vkGetInstanceProcAddr_fn](
                                            const char* function_name) {
        // Only resolve vkGetInstanceProcAddr, rely on syms->LoadFromInstance()
        // and/or syms->LoadFromDevice() for further loading.
        std::string fn = "vkGetInstanceProcAddr";
        if (strncmp(function_name, fn.data(), fn.size()) == 0) {
          return reinterpret_cast<PFN_vkVoidFunction>(vkGetInstanceProcAddr_fn);
        }
        return reinterpret_cast<PFN_vkVoidFunction>(NULL);
      }));

  *out_syms = reinterpret_cast<iree_hal_vulkan_syms_t*>(syms.release());
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_syms_create_from_system_loader(
    iree_allocator_t host_allocator, iree_hal_vulkan_syms_t** out_syms) {
  IREE_TRACE_SCOPE0("iree_hal_vulkan_syms_create_from_system_loader");
  IREE_ASSERT_ARGUMENT(out_syms);
  *out_syms = nullptr;

  IREE_ASSIGN_OR_RETURN(auto syms, DynamicSymbols::CreateFromSystemLoader());
  *out_syms = reinterpret_cast<iree_hal_vulkan_syms_t*>(syms.release());
  return iree_ok_status();
}

IREE_API_EXPORT void IREE_API_CALL
iree_hal_vulkan_syms_retain(iree_hal_vulkan_syms_t* syms) {
  IREE_ASSERT_ARGUMENT(syms);
  auto* handle = reinterpret_cast<DynamicSymbols*>(syms);
  if (handle) {
    handle->AddReference();
  }
}

IREE_API_EXPORT void IREE_API_CALL
iree_hal_vulkan_syms_release(iree_hal_vulkan_syms_t* syms) {
  IREE_ASSERT_ARGUMENT(syms);
  auto* handle = reinterpret_cast<DynamicSymbols*>(syms);
  if (handle) {
    handle->ReleaseReference();
  }
}

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_device_t extensibility util
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_query_extensibility_set(
    iree_hal_vulkan_features_t requested_features,
    iree_hal_vulkan_extensibility_set_t set, iree_host_size_t string_capacity,
    const char** out_string_values, iree_host_size_t* out_string_count) {
  *out_string_count = 0;

  iree_status_t status = iree_ok_status();
  iree_host_size_t string_count = 0;
#define ADD_EXT(target_set, name_literal)                       \
  if (iree_status_is_ok(status) && set == (target_set)) {       \
    if (string_count >= string_capacity && out_string_values) { \
      status = iree_status_from_code(IREE_STATUS_OUT_OF_RANGE); \
    } else if (out_string_values) {                             \
      out_string_values[string_count] = (name_literal);         \
    }                                                           \
    ++string_count;                                             \
  }

  //===--------------------------------------------------------------------===//
  // Baseline IREE requirements
  //===--------------------------------------------------------------------===//
  // Using IREE at all requires these extensions unconditionally. Adding things
  // here changes our minimum requirements and should be done carefully.
  // Optional extensions here are feature detected by the runtime.

  // VK_KHR_storage_buffer_storage_class:
  // Our generated SPIR-V kernels use storage buffers for all their data access.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_REQUIRED,
          VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME);

  // VK_KHR_get_physical_device_properties2:
  // Multiple extensions depend on VK_KHR_get_physical_device_properties2.
  // This extension was deprecated in Vulkan 1.1 as its functionality was
  // promoted to core so we list it as optional even though we require it.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_OPTIONAL,
          VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

  // VK_KHR_push_descriptor:
  // We can avoid a lot of additional Vulkan descriptor set manipulation
  // overhead when this extension is present. Android is a holdout, though, and
  // we have a fallback for when it's not available.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

  //===--------------------------------------------------------------------===//
  // Vulkan forward-compatibility shims
  //===--------------------------------------------------------------------===//
  // These are shims or extensions that are made core later in the spec and can
  // be removed once we require the core version that contains them.

  // VK_KHR_timeline_semaphore:
  // timeline semaphore support is optional and will be emulated if necessary.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);

  // VK_LAYER_KHRONOS_timeline_semaphore:
  // polyfill layer - enable if present instead of our custom emulation. Ignored
  // if timeline semaphores are supported natively (Vulkan 1.2+).
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_OPTIONAL,
          "VK_LAYER_KHRONOS_timeline_semaphore");

  //===--------------------------------------------------------------------===//
  // Optional debugging features
  //===--------------------------------------------------------------------===//
  // Used only when explicitly requested as they drastically change the
  // performance behavior of Vulkan.

  // VK_LAYER_KHRONOS_standard_validation:
  // only enabled if validation is desired. Since validation in Vulkan is just a
  // API correctness check it can't be used as a security mechanism and is fine
  // to ignore.
  if (iree_all_bits_set(requested_features,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_VALIDATION_LAYERS)) {
    ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_OPTIONAL,
            "VK_LAYER_KHRONOS_standard_validation");
  }

  // VK_EXT_debug_utils:
  // only enabled if debugging is desired to route Vulkan debug messages through
  // our logging sinks. Note that this adds a non-trivial runtime overhead and
  // we may want to disable it even in debug builds.
  if (iree_all_bits_set(requested_features,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_DEBUG_UTILS)) {
    ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_OPTIONAL,
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  *out_string_count = string_count;
  return status;
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
