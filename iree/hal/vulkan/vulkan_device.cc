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

#include "iree/hal/vulkan/vulkan_device.h"

#include <functional>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "iree/base/math.h"
#include "iree/base/status.h"
#include "iree/base/time.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/command_queue.h"
#include "iree/hal/vulkan/descriptor_pool_cache.h"
#include "iree/hal/vulkan/direct_command_buffer.h"
#include "iree/hal/vulkan/direct_command_queue.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/emulated_semaphore.h"
#include "iree/hal/vulkan/extensibility_util.h"
#include "iree/hal/vulkan/handle_util.h"
#include "iree/hal/vulkan/native_descriptor_set.h"
#include "iree/hal/vulkan/native_descriptor_set_layout.h"
#include "iree/hal/vulkan/native_event.h"
#include "iree/hal/vulkan/native_executable_layout.h"
#include "iree/hal/vulkan/native_semaphore.h"
#include "iree/hal/vulkan/nop_executable_cache.h"
#include "iree/hal/vulkan/serializing_command_queue.h"
#include "iree/hal/vulkan/status_util.h"
#include "iree/hal/vulkan/vma_allocator.h"

// DO NOT SUBMIT
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_vulkan_wrap_device(
    iree_string_view_t identifier, iree_hal_vulkan_syms_t* instance_syms,
    VkInstance instance, VkPhysicalDevice physical_device,
    VkDevice logical_device, iree_hal_vulkan_queue_set_t compute_queue_set,
    iree_hal_vulkan_queue_set_t transfer_queue_set,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(instance_syms);
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(out_device);
}

namespace iree {
namespace hal {
namespace vulkan {

namespace {

constexpr uint32_t kInvalidQueueFamilyIndex = -1;

struct QueueFamilyInfo {
  uint32_t dispatch_index = kInvalidQueueFamilyIndex;
  uint32_t dispatch_queue_count = 0;
  uint32_t transfer_index = kInvalidQueueFamilyIndex;
  uint32_t transfer_queue_count = 0;
};

// Finds the first queue in the listing (which is usually the
// driver-preferred) that has all of the |required_queue_flags| and none of
// the |excluded_queue_flags|. Returns kInvalidQueueFamilyIndex if no matching
// queue is found.
uint32_t FindFirstQueueFamilyWithFlags(
    absl::Span<const VkQueueFamilyProperties> queue_family_properties,
    uint32_t required_queue_flags, uint32_t excluded_queue_flags) {
  for (int queue_family_index = 0;
       queue_family_index < queue_family_properties.size();
       ++queue_family_index) {
    const auto& properties = queue_family_properties[queue_family_index];
    if ((properties.queueFlags & required_queue_flags) ==
            required_queue_flags &&
        (properties.queueFlags & excluded_queue_flags) == 0) {
      return queue_family_index;
    }
  }
  return kInvalidQueueFamilyIndex;
}

// Selects queue family indices for compute and transfer queues.
// Note that both queue families may be the same if there is only one family
// available.
StatusOr<QueueFamilyInfo> SelectQueueFamilies(
    VkPhysicalDevice physical_device, const ref_ptr<DynamicSymbols>& syms) {
  // Enumerate queue families available on the device.
  uint32_t queue_family_count = 0;
  syms->vkGetPhysicalDeviceQueueFamilyProperties(physical_device,
                                                 &queue_family_count, nullptr);
  absl::InlinedVector<VkQueueFamilyProperties, 4> queue_family_properties(
      queue_family_count);
  syms->vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device, &queue_family_count, queue_family_properties.data());

  QueueFamilyInfo queue_family_info;

  // Try to find a dedicated compute queue (no graphics caps).
  // Some may support both transfer and compute. If that fails then fallback
  // to any queue that supports compute.
  queue_family_info.dispatch_index = FindFirstQueueFamilyWithFlags(
      queue_family_properties, VK_QUEUE_COMPUTE_BIT, VK_QUEUE_GRAPHICS_BIT);
  if (queue_family_info.dispatch_index == kInvalidQueueFamilyIndex) {
    queue_family_info.dispatch_index = FindFirstQueueFamilyWithFlags(
        queue_family_properties, VK_QUEUE_COMPUTE_BIT, 0);
  }
  if (queue_family_info.dispatch_index == kInvalidQueueFamilyIndex) {
    return NotFoundErrorBuilder(IREE_LOC)
           << "Unable to find any queue family support compute operations";
  }
  queue_family_info.dispatch_queue_count =
      queue_family_properties[queue_family_info.dispatch_index].queueCount;

  // Try to find a dedicated transfer queue (no compute or graphics caps).
  // Not all devices have one, and some have only a queue family for
  // everything and possibly a queue family just for compute/etc. If that
  // fails then fallback to any queue that supports transfer. Finally, if
  // /that/ fails then we just won't create a transfer queue and instead use
  // the compute queue for all operations.
  queue_family_info.transfer_index = FindFirstQueueFamilyWithFlags(
      queue_family_properties, VK_QUEUE_TRANSFER_BIT,
      VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT);
  if (queue_family_info.transfer_index == kInvalidQueueFamilyIndex) {
    queue_family_info.transfer_index = FindFirstQueueFamilyWithFlags(
        queue_family_properties, VK_QUEUE_TRANSFER_BIT, VK_QUEUE_GRAPHICS_BIT);
  }
  if (queue_family_info.transfer_index == kInvalidQueueFamilyIndex) {
    queue_family_info.transfer_index = FindFirstQueueFamilyWithFlags(
        queue_family_properties, VK_QUEUE_TRANSFER_BIT, 0);
  }
  if (queue_family_info.transfer_index != kInvalidQueueFamilyIndex) {
    queue_family_info.transfer_queue_count =
        queue_family_properties[queue_family_info.transfer_index].queueCount;
  }

  // Ensure that we don't share the dispatch queues with transfer queues if
  // that would put us over the queue count.
  if (queue_family_info.dispatch_index == queue_family_info.transfer_index) {
    queue_family_info.transfer_queue_count = std::min(
        queue_family_properties[queue_family_info.dispatch_index].queueCount -
            queue_family_info.dispatch_queue_count,
        queue_family_info.transfer_queue_count);
  }

  return queue_family_info;
}

// Creates a transient command pool for the given queue family.
// Command buffers allocated from the pool must only be issued on queues
// belonging to the specified family.
StatusOr<ref_ptr<VkCommandPoolHandle>> CreateTransientCommandPool(
    const ref_ptr<VkDeviceHandle>& logical_device,
    uint32_t queue_family_index) {
  VkCommandPoolCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  create_info.queueFamilyIndex = queue_family_index;

  auto command_pool = make_ref<VkCommandPoolHandle>(logical_device);
  VK_RETURN_IF_ERROR(logical_device->syms()->vkCreateCommandPool(
      *logical_device, &create_info, logical_device->allocator(),
      command_pool->mutable_value()));
  return command_pool;
}

// Creates command queues for the given sets of queues.
absl::InlinedVector<std::unique_ptr<CommandQueue>, 4> CreateCommandQueues(
    const DeviceInfo& device_info,
    const ref_ptr<VkDeviceHandle>& logical_device,
    const iree_hal_vulkan_queue_set_t& compute_queue_set,
    const iree_hal_vulkan_queue_set_t& transfer_queue_set,
    const ref_ptr<TimePointFencePool>& fence_pool,
    const ref_ptr<DynamicSymbols>& syms) {
  absl::InlinedVector<std::unique_ptr<CommandQueue>, 4> command_queues;

  uint64_t compute_queue_count =
      iree_math_count_ones_u64(compute_queue_set.queue_indices);
  for (uint32_t i = 0; i < compute_queue_count; ++i) {
    if (!(compute_queue_set.queue_indices & (1ull << i))) continue;

    VkQueue queue = VK_NULL_HANDLE;
    syms->vkGetDeviceQueue(*logical_device,
                           compute_queue_set.queue_family_index, i, &queue);
    std::string queue_name = absl::StrCat(device_info.name(), ":d", i);

    if (fence_pool != nullptr) {
      command_queues.push_back(absl::make_unique<SerializingCommandQueue>(
          logical_device.get(), std::move(queue_name),
          IREE_HAL_COMMAND_CATEGORY_ANY, queue, fence_pool.get()));
    } else {
      command_queues.push_back(absl::make_unique<DirectCommandQueue>(
          logical_device.get(), std::move(queue_name),
          IREE_HAL_COMMAND_CATEGORY_ANY, queue));
    }
  }

  uint64_t transfer_queue_count =
      iree_math_count_ones_u64(transfer_queue_set.queue_indices);
  for (uint32_t i = 0; i < transfer_queue_count; ++i) {
    if (!(transfer_queue_set.queue_indices & (1ull << i))) continue;

    VkQueue queue = VK_NULL_HANDLE;
    syms->vkGetDeviceQueue(*logical_device,
                           transfer_queue_set.queue_family_index, i, &queue);
    std::string queue_name = absl::StrCat(device_info.name(), ":t", i);
    if (fence_pool != nullptr) {
      command_queues.push_back(absl::make_unique<SerializingCommandQueue>(
          logical_device.get(), std::move(queue_name),
          IREE_HAL_COMMAND_CATEGORY_TRANSFER, queue, fence_pool.get()));
    } else {
      command_queues.push_back(absl::make_unique<DirectCommandQueue>(
          logical_device.get(), std::move(queue_name),
          IREE_HAL_COMMAND_CATEGORY_TRANSFER, queue));
    }
  }

  return command_queues;
}

class VulkanDevice final : public DeviceBase {
 public:
  // Creates a device that manages its own VkDevice.
  static StatusOr<ref_ptr<VulkanDevice>> Create(
      ref_ptr<Driver> driver, VkInstance instance,
      const DeviceInfo& device_info, VkPhysicalDevice physical_device,
      iree_hal_vulkan_device_options_t options,
      const ref_ptr<DynamicSymbols>& syms);

  // Creates a device that wraps an externally managed VkDevice.
  static StatusOr<ref_ptr<VulkanDevice>> Wrap(
      ref_ptr<Driver> driver, VkInstance instance,
      const DeviceInfo& device_info, VkPhysicalDevice physical_device,
      VkDevice logical_device, iree_hal_vulkan_device_options_t options,
      const iree_hal_vulkan_queue_set_t& compute_queue_set,
      const iree_hal_vulkan_queue_set_t& transfer_queue_set,
      const ref_ptr<DynamicSymbols>& syms);

  VulkanDevice(
      ref_ptr<Driver> driver, const DeviceInfo& device_info,
      VkPhysicalDevice physical_device, ref_ptr<VkDeviceHandle> logical_device,
      iree_hal_allocator_t* device_allocator,
      absl::InlinedVector<std::unique_ptr<CommandQueue>, 4> command_queues,
      ref_ptr<VkCommandPoolHandle> dispatch_command_pool,
      ref_ptr<VkCommandPoolHandle> transfer_command_pool,
      ref_ptr<TimePointSemaphorePool> semaphore_pool,
      ref_ptr<TimePointFencePool> fence_pool);
  ~VulkanDevice() override;

  const ref_ptr<DynamicSymbols>& syms() const {
    return logical_device_->syms();
  }

  absl::string_view id() const override { return "vulkan"; }

  Status CreateCommandBuffer(
      iree_hal_command_buffer_mode_t mode,
      iree_hal_command_category_t command_categories,
      iree_hal_command_buffer_t** out_command_buffer) override;

  Status CreateDescriptorSet(
      iree_hal_descriptor_set_layout_t* set_layout,
      absl::Span<const iree_hal_descriptor_set_binding_t> bindings,
      iree_hal_descriptor_set_t** out_descriptor_set) override;

  Status CreateDescriptorSetLayout(
      iree_hal_descriptor_set_layout_usage_type_t usage_type,
      absl::Span<const iree_hal_descriptor_set_layout_binding_t> bindings,
      iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) override;

  Status CreateEvent(iree_hal_event_t** out_event) override;

  Status CreateExecutableCache(
      iree_string_view_t identifier,
      iree_hal_executable_cache_t** out_executable_cache) override;

  Status CreateExecutableLayout(
      absl::Span<iree_hal_descriptor_set_layout_t*> set_layouts,
      size_t push_constants,
      iree_hal_executable_layout_t** out_executable_layout) override;

  Status QueueSubmit(iree_hal_command_category_t command_categories,
                     uint64_t queue_affinity, iree_host_size_t batch_count,
                     const iree_hal_submission_batch_t* batches) override;

  Status CreateSemaphore(uint64_t initial_value,
                         iree_hal_semaphore_t** out_semaphore) override;

  Status WaitSemaphores(iree_hal_wait_mode_t wait_mode,
                        const iree_hal_semaphore_list_t* semaphore_list,
                        iree_time_t deadline_ns) override;

  Status WaitIdle(iree_time_t deadline_ns) override;

 private:
  Status WaitSemaphores(const iree_hal_semaphore_list_t* semaphore_list,
                        iree_time_t deadline_ns,
                        VkSemaphoreWaitFlags wait_flags);

  bool emulating_timeline_semaphores() const {
    return semaphore_pool_ != nullptr;
  }

  ref_ptr<Driver> driver_;
  VkPhysicalDevice physical_device_;
  ref_ptr<VkDeviceHandle> logical_device_;

  iree_hal_allocator_t* allocator_;

  mutable absl::InlinedVector<std::unique_ptr<CommandQueue>, 4> command_queues_;
  mutable absl::InlinedVector<CommandQueue*, 4> dispatch_queues_;
  mutable absl::InlinedVector<CommandQueue*, 4> transfer_queues_;

  ref_ptr<DescriptorPoolCache> descriptor_pool_cache_;

  ref_ptr<VkCommandPoolHandle> dispatch_command_pool_;
  ref_ptr<VkCommandPoolHandle> transfer_command_pool_;

  // Fields used for emulated timeline semaphores.
  ref_ptr<TimePointSemaphorePool> semaphore_pool_;
  ref_ptr<TimePointFencePool> fence_pool_;
};

static StatusOr<ref_ptr<VulkanDevice>> CreateVulkanDevice(
    ref_ptr<Driver> driver, const DeviceInfo& device_info, VkInstance instance,
    VkPhysicalDevice physical_device, ref_ptr<VkDeviceHandle> logical_device,
    iree_hal_vulkan_device_options_t options,
    const iree_hal_vulkan_queue_set_t& compute_queue_set,
    const iree_hal_vulkan_queue_set_t& transfer_queue_set) {
  auto& syms = logical_device->syms();

  // Create command pools for each queue family. If we don't have a transfer
  // queue then we'll ignore that one and just use the dispatch pool.
  // If we wanted to expose the pools through the HAL to allow the VM to more
  // effectively manage them (pool per fiber, etc) we could, however I doubt
  // the overhead of locking the pool will be even a blip.
  IREE_ASSIGN_OR_RETURN(
      auto dispatch_command_pool,
      CreateTransientCommandPool(logical_device,
                                 compute_queue_set.queue_family_index));
  ref_ptr<VkCommandPoolHandle> transfer_command_pool;
  if (transfer_queue_set.queue_indices != 0) {
    IREE_ASSIGN_OR_RETURN(
        transfer_command_pool,
        CreateTransientCommandPool(logical_device,
                                   transfer_queue_set.queue_family_index));
  }

  // Emulate timeline semaphores if associated functions are not defined.
  ref_ptr<TimePointSemaphorePool> semaphore_pool = nullptr;
  ref_ptr<TimePointFencePool> fence_pool = nullptr;
  if (syms->vkGetSemaphoreCounterValue == nullptr ||
      options.force_timeline_semaphore_emulation) {
    IREE_ASSIGN_OR_RETURN(semaphore_pool, TimePointSemaphorePool::Create(
                                              add_ref(logical_device)));
    IREE_ASSIGN_OR_RETURN(fence_pool,
                          TimePointFencePool::Create(add_ref(logical_device)));
  }

  auto command_queues =
      CreateCommandQueues(device_info, logical_device, compute_queue_set,
                          transfer_queue_set, fence_pool, syms);

  // Create the device memory allocator.
  iree_hal_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_vma_allocator_create(
      instance, physical_device, logical_device.get(),
      options.vma_record_settings, &allocator));

  // DO NOT SUBMIT
  // host_allocator from somewhere
  // placement new into the host allocator

  return assign_ref(new VulkanDevice(
      std::move(driver), device_info, physical_device,
      std::move(logical_device), allocator, std::move(command_queues),
      std::move(dispatch_command_pool), std::move(transfer_command_pool),
      std::move(semaphore_pool), std::move(fence_pool)));
}

}  // namespace

// static
StatusOr<ref_ptr<VulkanDevice>> VulkanDevice::Create(
    ref_ptr<Driver> driver, VkInstance instance, const DeviceInfo& device_info,
    VkPhysicalDevice physical_device, iree_hal_vulkan_device_options_t options,
    const ref_ptr<DynamicSymbols>& syms) {
  IREE_TRACE_SCOPE0("VulkanDevice::Create");

  if (!options.extensibility_spec.optional_layers.empty() ||
      !options.extensibility_spec.required_layers.empty()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Device layers are deprecated and unsupported by IREE";
  }

  // Find the extensions we need (or want) that are also available
  // on the device. This will fail when required ones are not present.
  IREE_ASSIGN_OR_RETURN(
      auto enabled_extension_names,
      MatchAvailableDeviceExtensions(physical_device,
                                     options.extensibility_spec, *syms));
  iree_hal_vulkan_device_extensions_t enabled_device_extensions =
      iree_hal_vulkan_populate_enabled_device_extensions(
          enabled_extension_names);

  // Find queue families we will expose as HAL queues.
  IREE_ASSIGN_OR_RETURN(auto queue_family_info,
                        SelectQueueFamilies(physical_device, syms));

  // Limit the number of queues we create (for now).
  // We may want to allow this to grow, but each queue adds overhead and we
  // need to measure to make sure we can effectively use them all.
  queue_family_info.dispatch_queue_count =
      std::min(2u, queue_family_info.dispatch_queue_count);
  queue_family_info.transfer_queue_count =
      std::min(1u, queue_family_info.transfer_queue_count);
  bool has_dedicated_transfer_queues =
      queue_family_info.transfer_queue_count > 0;

  // Setup the queue info we'll be using.
  // Each queue here (created from within a family) will map to a HAL queue.
  //
  // Note that we need to handle the case where we have transfer queues that
  // are of the same queue family as the dispatch queues: Vulkan requires that
  // all queues created from the same family are done in the same
  // VkDeviceQueueCreateInfo struct.
  IREE_DVLOG(1) << "Creating " << queue_family_info.dispatch_queue_count
                << " dispatch queue(s) in queue family "
                << queue_family_info.dispatch_index;
  absl::InlinedVector<VkDeviceQueueCreateInfo, 2> queue_create_info;
  absl::InlinedVector<float, 4> dispatch_queue_priorities;
  absl::InlinedVector<float, 4> transfer_queue_priorities;
  queue_create_info.push_back({});
  auto& dispatch_queue_info = queue_create_info.back();
  dispatch_queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  dispatch_queue_info.pNext = nullptr;
  dispatch_queue_info.flags = 0;
  dispatch_queue_info.queueFamilyIndex = queue_family_info.dispatch_index;
  dispatch_queue_info.queueCount = queue_family_info.dispatch_queue_count;
  if (has_dedicated_transfer_queues) {
    if (queue_family_info.dispatch_index == queue_family_info.transfer_index) {
      IREE_DVLOG(1) << "Creating " << queue_family_info.transfer_queue_count
                    << " dedicated transfer queue(s) in shared queue family "
                    << queue_family_info.transfer_index;
      dispatch_queue_info.queueCount += queue_family_info.transfer_queue_count;
    } else {
      IREE_DVLOG(1)
          << "Creating " << queue_family_info.transfer_queue_count
          << " dedicated transfer queue(s) in independent queue family "
          << queue_family_info.transfer_index;
      queue_create_info.push_back({});
      auto& transfer_queue_info = queue_create_info.back();
      transfer_queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      transfer_queue_info.pNext = nullptr;
      transfer_queue_info.queueFamilyIndex = queue_family_info.transfer_index;
      transfer_queue_info.queueCount = queue_family_info.transfer_queue_count;
      transfer_queue_info.flags = 0;
      transfer_queue_priorities.resize(transfer_queue_info.queueCount);
      transfer_queue_info.pQueuePriorities = transfer_queue_priorities.data();
    }
  }
  dispatch_queue_priorities.resize(dispatch_queue_info.queueCount);
  dispatch_queue_info.pQueuePriorities = dispatch_queue_priorities.data();

  // Create device and its queues.
  VkDeviceCreateInfo device_create_info = {};
  device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_create_info.enabledLayerCount = 0;
  device_create_info.ppEnabledLayerNames = nullptr;
  device_create_info.enabledExtensionCount = enabled_extension_names.size();
  device_create_info.ppEnabledExtensionNames = enabled_extension_names.data();
  device_create_info.queueCreateInfoCount = queue_create_info.size();
  device_create_info.pQueueCreateInfos = queue_create_info.data();
  device_create_info.pEnabledFeatures = nullptr;

  VkPhysicalDeviceTimelineSemaphoreFeatures semaphore_features;
  std::memset(&semaphore_features, 0, sizeof(semaphore_features));
  semaphore_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
  semaphore_features.timelineSemaphore = VK_TRUE;
  VkPhysicalDeviceFeatures2 features2;
  std::memset(&features2, 0, sizeof(features2));
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = &semaphore_features;

  if (!enabled_device_extensions.timeline_semaphore ||
      options.force_timeline_semaphore_emulation) {
    device_create_info.pNext = nullptr;
  } else {
    device_create_info.pNext = &features2;
  }

  auto logical_device = make_ref<VkDeviceHandle>(
      syms, enabled_device_extensions,
      /*owns_device=*/true, iree_allocator_system(), /*allocator=*/nullptr);
  // The Vulkan loader can leak here, depending on which features are enabled.
  // This is out of our control, so disable leak checks.
  IREE_DISABLE_LEAK_CHECKS();
  VK_RETURN_IF_ERROR(syms->vkCreateDevice(physical_device, &device_create_info,
                                          logical_device->allocator(),
                                          logical_device->mutable_value()));
  IREE_RETURN_IF_ERROR(logical_device->syms()->LoadFromDevice(
      instance, logical_device->value()));
  IREE_ENABLE_LEAK_CHECKS();

  // Select queue indices and create command queues with them.
  iree_hal_vulkan_queue_set_t compute_queue_set = {};
  compute_queue_set.queue_family_index = queue_family_info.dispatch_index;
  for (uint32_t i = 0; i < queue_family_info.dispatch_queue_count; ++i) {
    compute_queue_set.queue_indices |= 1ull << i;
  }
  iree_hal_vulkan_queue_set_t transfer_queue_set = {};
  transfer_queue_set.queue_family_index = queue_family_info.transfer_index;
  uint32_t base_queue_index = 0;
  if (queue_family_info.dispatch_index == queue_family_info.transfer_index) {
    // Sharing a family, so transfer queues follow compute queues.
    base_queue_index = queue_family_info.dispatch_index;
  }
  for (uint32_t i = 0; i < queue_family_info.transfer_queue_count; ++i) {
    transfer_queue_set.queue_indices |= 1ull << (i + base_queue_index);
  }

  return CreateVulkanDevice(std::move(driver), device_info, instance,
                            physical_device, std::move(logical_device), options,
                            compute_queue_set, transfer_queue_set);
}

// static
StatusOr<ref_ptr<VulkanDevice>> VulkanDevice::Wrap(
    ref_ptr<Driver> driver, VkInstance instance, const DeviceInfo& device_info,
    VkPhysicalDevice physical_device, VkDevice logical_device,
    iree_hal_vulkan_device_options_t options,
    const iree_hal_vulkan_queue_set_t& compute_queue_set,
    const iree_hal_vulkan_queue_set_t& transfer_queue_set,
    const ref_ptr<DynamicSymbols>& syms) {
  IREE_TRACE_SCOPE0("VulkanDevice::Wrap");

  uint64_t compute_queue_count =
      iree_math_count_ones_u64(compute_queue_set.queue_indices);
  uint64_t transfer_queue_count =
      iree_math_count_ones_u64(transfer_queue_set.queue_indices);

  if (compute_queue_count == 0) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "At least one compute queue is required";
  }

  // Find the extensions we need (or want) that are also available on the
  // device. This will fail when required ones are not present.
  //
  // Since the device is already created, we can't actually enable any
  // extensions or query if they are really enabled - we just have to trust
  // that the caller already enabled them for us (or we may fail later).
  IREE_ASSIGN_OR_RETURN(
      auto enabled_extension_names,
      MatchAvailableDeviceExtensions(physical_device,
                                     options.extensibility_spec, *syms));
  iree_hal_vulkan_device_extensions_t enabled_device_extensions =
      iree_hal_vulkan_populate_enabled_device_extensions(
          enabled_extension_names);

  // Wrap the provided VkDevice with a VkDeviceHandle for use within the HAL.
  auto device_handle = make_ref<VkDeviceHandle>(
      syms, enabled_device_extensions,
      /*owns_device=*/false, iree_allocator_system(), /*allocator=*/nullptr);
  *device_handle->mutable_value() = logical_device;

  return CreateVulkanDevice(std::move(driver), device_info, instance,
                            physical_device, std::move(device_handle), options,
                            compute_queue_set, transfer_queue_set);
}

VulkanDevice::VulkanDevice(
    ref_ptr<Driver> driver, const DeviceInfo& device_info,
    VkPhysicalDevice physical_device, ref_ptr<VkDeviceHandle> logical_device,
    iree_hal_allocator_t* device_allocator,
    absl::InlinedVector<std::unique_ptr<CommandQueue>, 4> command_queues,
    ref_ptr<VkCommandPoolHandle> dispatch_command_pool,
    ref_ptr<VkCommandPoolHandle> transfer_command_pool,
    ref_ptr<TimePointSemaphorePool> semaphore_pool,
    ref_ptr<TimePointFencePool> fence_pool)
    : DeviceBase(host_allocator, device_allocator),
      driver_(std::move(driver)),
      physical_device_(physical_device),
      logical_device_(std::move(logical_device)),
      command_queues_(std::move(command_queues)),
      descriptor_pool_cache_(
          make_ref<DescriptorPoolCache>(add_ref(logical_device_))),
      dispatch_command_pool_(std::move(dispatch_command_pool)),
      transfer_command_pool_(std::move(transfer_command_pool)),
      semaphore_pool_(std::move(semaphore_pool)),
      fence_pool_(std::move(fence_pool)) {
  // Populate the queue lists based on queue capabilities.
  for (auto& command_queue : command_queues_) {
    if (command_queue->can_dispatch()) {
      dispatch_queues_.push_back(command_queue.get());
      if (transfer_command_pool_ == VK_NULL_HANDLE) {
        transfer_queues_.push_back(command_queue.get());
      }
    } else {
      transfer_queues_.push_back(command_queue.get());
    }
  }
}

VulkanDevice::~VulkanDevice() {
  IREE_TRACE_SCOPE0("VulkanDevice::dtor");

  // Drop all command queues. These may wait until idle.
  command_queues_.clear();
  dispatch_queues_.clear();
  transfer_queues_.clear();

  // Drop command pools now that we know there are no more outstanding command
  // buffers.
  dispatch_command_pool_.reset();
  transfer_command_pool_.reset();

  // Now that no commands are outstanding we can release all descriptor sets.
  descriptor_pool_cache_.reset();

  // Finally, destroy the device.
  logical_device_.reset();
}

Status VulkanDevice::CreateExecutableCache(
    iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  return iree_hal_vulkan_nop_executable_cache_create(
      logical_device_.get(), identifier, out_executable_cache);
}

Status VulkanDevice::CreateDescriptorSetLayout(
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    absl::Span<const iree_hal_descriptor_set_layout_binding_t> bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  return iree_hal_vulkan_native_descriptor_set_layout_create(
      logical_device_.get(), usage_type, bindings.size(), bindings.data(),
      out_descriptor_set_layout);
}

Status VulkanDevice::CreateExecutableLayout(
    absl::Span<iree_hal_descriptor_set_layout_t*> set_layouts,
    size_t push_constants,
    iree_hal_executable_layout_t** out_executable_layout) {
  return iree_hal_vulkan_native_executable_layout_create(
      logical_device_.get(), set_layouts.size(), set_layouts.data(),
      push_constants, out_executable_layout);
}

Status VulkanDevice::CreateDescriptorSet(
    iree_hal_descriptor_set_layout_t* set_layout,
    absl::Span<const iree_hal_descriptor_set_binding_t> bindings,
    iree_hal_descriptor_set_t** out_descriptor_set) {
  return UnimplementedErrorBuilder(IREE_LOC)
         << "CreateDescriptorSet not yet implemented (needs timeline)";
}

Status VulkanDevice::CreateCommandBuffer(
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_command_buffer_t** out_command_buffer) {
  // Select the command pool to used based on the types of commands used.
  // Note that we may not have a dedicated transfer command pool if there are
  // no dedicated transfer queues.
  VkCommandPoolHandle* command_pool = NULL;
  if (transfer_command_pool_ &&
      !iree_all_bits_set(command_categories,
                         IREE_HAL_COMMAND_CATEGORY_DISPATCH)) {
    command_pool = transfer_command_pool_.get();
  } else {
    command_pool = dispatch_command_pool_.get();
  }
  return iree_hal_vulkan_direct_command_buffer_allocate(
      logical_device_.get(), command_pool, mode, command_categories,
      descriptor_pool_cache_.get(), out_command_buffer);
}

Status VulkanDevice::CreateEvent(iree_hal_event_t** out_event) {
  return iree_hal_vulkan_native_event_create(logical_device_.get(), out_event);
}

Status VulkanDevice::CreateSemaphore(uint64_t initial_value,
                                     iree_hal_semaphore_t** out_semaphore) {
  if (emulating_timeline_semaphores()) {
    return iree_hal_vulkan_emulated_semaphore_create(
        logical_device_.get(),
        ReinterpretSpan<CommandQueue*>(absl::MakeSpan(command_queues_)),
        semaphore_pool_.get(), initial_value, out_semaphore);
  }
  return iree_hal_vulkan_native_semaphore_create(logical_device_.get(),
                                                 initial_value, out_semaphore);
}

Status VulkanDevice::WaitAllSemaphores(
    const iree_hal_semaphore_list_t* semaphore_list, iree_time_t deadline_ns) {
  return WaitSemaphores(semaphore_list, deadline_ns, /*wait_flags=*/0);
}

Status VulkanDevice::WaitAnySemaphore(
    const iree_hal_semaphore_list_t* semaphore_list, iree_time_t deadline_ns) {
  return WaitSemaphores(semaphore_list, deadline_ns,
                        /*wait_flags=*/VK_SEMAPHORE_WAIT_ANY_BIT);
}

Status VulkanDevice::WaitSemaphores(
    const iree_hal_semaphore_list_t* semaphore_list, iree_time_t deadline_ns,
    VkSemaphoreWaitFlags wait_flags) {
  if (emulating_timeline_semaphores()) {
    return iree_hal_vulkan_emulated_semaphore_multi_wait(
        logical_device_.get(), semaphore_list, deadline_ns, wait_flags);
  }
  return iree_hal_vulkan_native_semaphore_multi_wait(
      logical_device_.get(), semaphore_list, deadline_ns, wait_flags);
}

Status VulkanDevice::WaitIdle(iree_time_t deadline_ns) {
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    // Fast path for using vkDeviceWaitIdle, which is usually cheaper (as it
    // requires fewer calls into the driver).
    VK_RETURN_IF_ERROR(syms()->vkDeviceWaitIdle(*logical_device_));
    return OkStatus();
  }
  for (auto& command_queue : command_queues_) {
    IREE_RETURN_IF_ERROR(command_queue->WaitIdle(deadline_ns));
  }
  return OkStatus();
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
