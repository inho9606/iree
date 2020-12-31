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

#include "iree/hal/vulkan/vulkan_driver.h"

#include <memory>

#include "iree/base/memory.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/debug_reporter.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/extensibility_util.h"
#include "iree/hal/vulkan/status_util.h"
#include "iree/hal/vulkan/vulkan_device.h"

using namespace iree::hal::vulkan;

typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  // Identifier used for the driver in the IREE driver registry.
  // We allow overriding so that multiple Vulkan versions can be exposed in the
  // same process.
  iree_string_view_t identifier;

  iree_hal_vulkan_device_options_t device_options;
  int default_device_index;

  iree_hal_vulkan_instance_extensions_t instance_extensions;

  // (Partial) loaded Vulkan symbols. Devices created within the driver may have
  // different function pointers for device-specific functions that change
  // behavior with enabled layers/extensions.
  iree::ref_ptr<DynamicSymbols> syms;

  // The Vulkan instance that all devices created from the driver will share.
  VkInstance instance;
  bool owns_instance;

  // Optional debug reporter: may be disabled or unavailable (no debug layers).
  iree_hal_vulkan_debug_reporter_t* debug_reporter;
} iree_hal_vulkan_driver_t;

extern const iree_hal_driver_vtable_t iree_hal_vulkan_driver_vtable;

IREE_API_EXPORT void IREE_API_CALL iree_hal_vulkan_driver_options_initialize(
    iree_hal_vulkan_driver_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->api_version = VK_API_VERSION_1_2;
  out_options->requested_features = 0;
  iree_hal_vulkan_device_options_initialize(&out_options->device_options);
  out_options->default_device_index = 0;
}

// Returns a VkApplicationInfo struct populated with the default app info.
// We may allow hosting applications to override this via weak-linkage if it's
// useful, otherwise this is enough to create the application.
static void iree_hal_vulkan_driver_populate_default_app_info(
    const iree_hal_vulkan_driver_options_t* options,
    VkApplicationInfo* out_app_info) {
  memset(out_app_info, 0, sizeof(*out_app_info));
  out_app_info->sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  out_app_info->pNext = NULL;
  out_app_info->pApplicationName = "IREE-ML";
  out_app_info->applicationVersion = 0;
  out_app_info->pEngineName = "IREE";
  out_app_info->engineVersion = 0;
  out_app_info->apiVersion = options->api_version;
}

// NOTE: takes ownership of |instance|.
static iree_status_t iree_hal_vulkan_driver_create_internal(
    iree_string_view_t identifier,
    const iree_hal_vulkan_driver_options_t* options,
    const iree_hal_vulkan_string_list_t* enabled_extensions,
    iree_hal_vulkan_syms_t* opaque_syms, VkInstance instance,
    bool owns_instance, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  auto* instance_syms = (DynamicSymbols*)opaque_syms;

  iree_hal_vulkan_instance_extensions_t instance_extensions =
      iree_hal_vulkan_populate_enabled_instance_extensions(enabled_extensions);

  // The real debug messenger (not just the static one used above) can now be
  // created as we've loaded all the required symbols.
  // TODO(benvanik): strip in min-size release builds.
  iree_hal_vulkan_debug_reporter_t* debug_reporter = NULL;
  if (instance_extensions.debug_utils) {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_debug_reporter_allocate(
        instance, instance_syms, /*allocation_callbacks=*/NULL, host_allocator,
        &debug_reporter));
  }

  iree_hal_vulkan_driver_t* driver = NULL;
  iree_host_size_t total_size = sizeof(*driver) + identifier.size;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&driver);
  if (!iree_status_is_ok(status)) {
    // Need to clean up if we fail (as we own these).
    iree_hal_vulkan_debug_reporter_free(debug_reporter);
    if (driver->owns_instance) {
      driver->syms->vkDestroyInstance(driver->instance, /*pAllocator=*/NULL);
    }
    return status;
  }
  iree_hal_resource_initialize(&iree_hal_vulkan_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + total_size - identifier.size);
  memcpy(&driver->device_options, &options->device_options,
         sizeof(driver->device_options));
  driver->default_device_index = options->default_device_index;
  driver->syms = iree::add_ref(instance_syms);
  driver->instance = instance;
  driver->owns_instance = owns_instance;
  driver->debug_reporter = debug_reporter;
  *out_driver = (iree_hal_driver_t*)driver;
  return status;
}

static void iree_hal_vulkan_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_vulkan_driver_t* driver = (iree_hal_vulkan_driver_t*)base_driver;
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_debug_reporter_free(driver->debug_reporter);
  if (driver->owns_instance) {
    driver->syms->vkDestroyInstance(driver->instance, /*pAllocator=*/NULL);
  }
  driver->syms.reset();
  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_vulkan_driver_query_extensibility_set(
    iree_hal_vulkan_features_t requested_features,
    iree_hal_vulkan_extensibility_set_t set, iree::Arena* arena,
    iree_hal_vulkan_string_list_t* out_string_list) {
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_query_extensibility_set(
      requested_features, set, 0, NULL, &out_string_list->count));
  out_string_list->values = (const char**)arena->AllocateBytes(
      out_string_list->count * sizeof(out_string_list->values[0]));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_query_extensibility_set(
      requested_features, set, out_string_list->count, out_string_list->values,
      &out_string_list->count));
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_driver_compute_enabled_extensibility_sets(
    iree::hal::vulkan::DynamicSymbols* syms,
    iree_hal_vulkan_features_t requested_features, iree::Arena* arena,
    iree_hal_vulkan_string_list_t* out_enabled_layers,
    iree_hal_vulkan_string_list_t* out_enabled_extensions) {
  // Query our required and optional layers and extensions based on the IREE
  // features the user requested.
  iree_hal_vulkan_string_list_t required_layers;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_driver_query_extensibility_set(
      requested_features,
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_REQUIRED, arena,
      &required_layers));
  iree_hal_vulkan_string_list_t optional_layers;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_driver_query_extensibility_set(
      requested_features,
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_OPTIONAL, arena,
      &optional_layers));
  iree_hal_vulkan_string_list_t required_extensions;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_driver_query_extensibility_set(
      requested_features,
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_REQUIRED, arena,
      &required_extensions));
  iree_hal_vulkan_string_list_t optional_extensions;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_driver_query_extensibility_set(
      requested_features,
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_OPTIONAL, arena,
      &optional_extensions));

  // Find the layers and extensions we need (or want) that are also available
  // on the instance. This will fail when required ones are not present.
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_match_available_instance_layers(
      syms, &required_layers, &optional_layers, arena, out_enabled_layers));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_match_available_instance_extensions(
      syms, &required_extensions, &optional_extensions, arena,
      out_enabled_extensions));

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_vulkan_driver_create(
    iree_string_view_t identifier,
    const iree_hal_vulkan_driver_options_t* options,
    iree_hal_vulkan_syms_t* opaque_syms, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(opaque_syms);
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_SCOPE();

  auto* instance_syms = (DynamicSymbols*)opaque_syms;

  // Query required and optional instance layers/extensions for the requested
  // features.
  iree::Arena arena;
  iree_hal_vulkan_string_list_t enabled_layers;
  iree_hal_vulkan_string_list_t enabled_extensions;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_driver_compute_enabled_extensibility_sets(
          instance_syms, options->requested_features, &arena, &enabled_layers,
          &enabled_extensions));

  // Create the instance this driver will use for all requests.
  VkApplicationInfo app_info;
  iree_hal_vulkan_driver_populate_default_app_info(options, &app_info);
  VkInstanceCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pNext = NULL;
  create_info.flags = 0;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledLayerCount = enabled_layers.count;
  create_info.ppEnabledLayerNames = enabled_layers.values;
  create_info.enabledExtensionCount = enabled_extensions.count;
  create_info.ppEnabledExtensionNames = enabled_extensions.values;

  // Some ICDs appear to leak in here, out of our control.
  // Warning: leak checks remain disabled if an error is returned.
  IREE_DISABLE_LEAK_CHECKS();
  VkInstance instance = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(instance_syms->vkCreateInstance(
      &create_info, /*pAllocator=*/NULL, &instance))
      << "Unable to create Vulkan instance";
  IREE_ENABLE_LEAK_CHECKS();

  // TODO(benvanik): enable validation layers if needed.

  // Now that the instance has been created we can fetch all of the instance
  // symbols.
  IREE_RETURN_IF_ERROR(instance_syms->LoadFromInstance(instance));

  iree_status_t status = iree_hal_vulkan_driver_create_internal(
      identifier, options, &enabled_extensions, opaque_syms, instance,
      /*owns_instance=*/true, host_allocator, out_driver);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_driver_create_using_instance(
    iree_string_view_t identifier,
    const iree_hal_vulkan_driver_options_t* options,
    iree_hal_vulkan_syms_t* opaque_syms, VkInstance instance,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(opaque_syms);
  IREE_ASSERT_ARGUMENT(out_driver);
  if (instance == VK_NULL_HANDLE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "a non-NULL VkInstance must be provided");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  // May be a no-op but don't rely on that so we can be sure we have the right
  // function pointers.
  auto* instance_syms = (DynamicSymbols*)opaque_syms;
  IREE_RETURN_IF_ERROR(instance_syms->LoadFromInstance(instance));

  // Since the instance is already created we can't actually enable any
  // extensions or even query if they are really enabled - we just have to trust
  // that the caller already enabled them for us (or we may fail later).
  iree::Arena arena;
  iree_hal_vulkan_string_list_t enabled_layers;
  iree_hal_vulkan_string_list_t enabled_extensions;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_driver_compute_enabled_extensibility_sets(
          instance_syms, options->requested_features, &arena, &enabled_layers,
          &enabled_extensions));
  iree_hal_vulkan_instance_extensions_t instance_extensions =
      iree_hal_vulkan_populate_enabled_instance_extensions(&enabled_extensions);

  iree_status_t status = iree_hal_vulkan_driver_create_internal(
      identifier, options, &enabled_extensions, opaque_syms, instance,
      /*owns_instance=*/true, host_allocator, out_driver);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Returns the size, in bytes, of the iree_hal_device_info_t storage required
// for holding the given |physical_device|.
static iree_host_size_t iree_hal_vulkan_calculate_device_info_size(
    VkPhysicalDevice physical_device, DynamicSymbols* syms) {
  VkPhysicalDeviceProperties physical_device_properties;
  syms->vkGetPhysicalDeviceProperties(physical_device,
                                      &physical_device_properties);
  return sizeof(iree_hal_device_info_t) +
         strlen(physical_device_properties.deviceName) + 1;
}

// Populates device information from the given Vulkan physical device handle.
// |out_device_info| must point to valid memory and additional data will be
// appended to |buffer_ptr| and the new pointer is returned.
static uint8_t* iree_hal_vulkan_populate_device_info(
    VkPhysicalDevice physical_device, DynamicSymbols* syms, uint8_t* buffer_ptr,
    iree_hal_device_info_t* out_device_info) {
  memset(out_device_info, 0, sizeof(*out_device_info));
  out_device_info->device_id = (iree_hal_device_id_t)physical_device;

  VkPhysicalDeviceFeatures physical_device_features;
  syms->vkGetPhysicalDeviceFeatures(physical_device, &physical_device_features);
  // TODO(benvanik): check and optionally require these features:
  // - physical_device_features.robustBufferAccess
  // - physical_device_features.shaderInt16
  // - physical_device_features.shaderInt64
  // - physical_device_features.shaderFloat64

  VkPhysicalDeviceProperties physical_device_properties;
  syms->vkGetPhysicalDeviceProperties(physical_device,
                                      &physical_device_properties);
  // TODO(benvanik): check and optionally require reasonable limits.

  // TODO(benvanik): more clever/sanitized device naming.
  iree_host_size_t name_length = strlen(physical_device_properties.deviceName);
  out_device_info->name =
      iree_make_string_view((const char*)buffer_ptr, name_length);
  memcpy(buffer_ptr, physical_device_properties.deviceName, name_length + 1);
  buffer_ptr += name_length + 1;

  return buffer_ptr;
}

// // Query all available devices (at this moment, note that this may change!).
// uint32_t physical_device_count = 0;
// VK_RETURN_IF_ERROR(syms()->vkEnumeratePhysicalDevices(
//     driver->instance, &physical_device_count, NULL));
// absl::InlinedVector<VkPhysicalDevice, 2> physical_devices(
//     physical_device_count);
// VK_RETURN_IF_ERROR(syms()->vkEnumeratePhysicalDevices(
//     driver->instance, &physical_device_count, physical_devices.data()));
// // Convert to our HAL structure.
// std::vector<DeviceInfo> device_infos;
// device_infos.reserve(physical_device_count);
// for (auto physical_device : physical_devices) {
//   // TODO(benvanik): if we fail should we just ignore the device in the list?
//   IREE_ASSIGN_OR_RETURN(auto device_info,
//                         PopulateDeviceInfo(physical_device, syms()));
//   device_infos.push_back(std::move(device_info));
// }
// return device_infos;
static iree_status_t iree_hal_vulkan_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t allocator,
    iree_hal_device_info_t** out_device_infos,
    iree_host_size_t* out_device_info_count) {
  // DO NOT SUBMIT
  // static const iree_hal_device_info_t device_infos[1] = {
  //     {
  //         .device_id = IREE_HAL_TASK_DEVICE_ID_DEFAULT,
  //         .name = iree_string_view_literal("default"),
  //     },
  // };
  // *out_device_info_count = IREE_ARRAYSIZE(device_infos);
  // return iree_allocator_clone(
  //     allocator, iree_make_const_byte_span(device_infos,
  //     sizeof(device_infos)), (void**)out_device_infos);
}

// StatusOr<ref_ptr<Device>> VulkanDriver::CreateDefaultDevice() {
//   IREE_TRACE_SCOPE0("VulkanDriver::CreateDefaultDevice");
//   // Query available devices.
//   IREE_ASSIGN_OR_RETURN(auto available_devices, EnumerateAvailableDevices());
//   if (default_device_index < 0 ||
//       default_device_index >= available_devices.size()) {
//     return NotFoundErrorBuilder(IREE_LOC)
//            << "Device index " << default_device_index << " not found "
//            << "(of " << available_devices.size() << ")";
//   }
//   // Just create the first one we find.
//   return CreateDevice(available_devices[default_device_index].device_id());
// }
static iree_status_t iree_hal_vulkan_driver_create_device(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_allocator_t allocator, iree_hal_device_t** out_device) {
  iree_hal_vulkan_driver_t* driver = (iree_hal_vulkan_driver_t*)base_driver;

  if (device_id = IREE_HAL_DEVICE_ID_INVALID) {
    //
  }

  auto physical_device = reinterpret_cast<VkPhysicalDevice>(device_id);
  IREE_ASSIGN_OR_RETURN(auto device_info,
                        PopulateDeviceInfo(physical_device, syms()));

  // Attempt to create the device.
  // This may fail if the device was enumerated but is in exclusive use,
  // disabled by the system, or permission is denied.
  IREE_ASSIGN_OR_RETURN(
      auto device,
      VulkanDevice::Create(add_ref(this), instance(), device_info,
                           physical_device, &driver->device_options, syms()));

  IREE_LOG(INFO) << "Created Vulkan Device: " << device->info().name();

  return device;

  return iree_hal_vulkan_device_create(
      driver->identifier, &driver->default_params, driver->executor,
      driver->loader_count, driver->loaders, allocator, out_device);
}

const iree_hal_driver_vtable_t iree_hal_vulkan_driver_vtable = {
    /* .destroy = */ iree_hal_vulkan_driver_destroy,
    /* .query_available_devices = */
    iree_hal_vulkan_driver_query_available_devices,
    /* .create_device = */ iree_hal_vulkan_driver_create_device,
};
