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

#include "iree/hal/vulkan/emulated_semaphore.h"

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "iree/base/intrusive_list.h"
#include "iree/base/ref_ptr.h"
#include "iree/base/status.h"
#include "iree/base/time.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/status_util.h"

namespace iree {
namespace hal {
namespace vulkan {

// A timeline semaphore emulated via `VkFence`s and binary `VkSemaphore`s.
//
// Vulkan provides several explicit synchronization primitives: fences,
// (binary/timeline) semaphores, events, pipeline barriers, and render passes.
// See "6. Synchronization and Cache Control" of the Vulkan specification
// for the details.
//
// Render passes are for graphics pipelines so IREE does not care about them.
// Pipeline barriers synchronize control within a command buffer at a single
// point. Fences, (binary/timeline) semaphores, and events are synchronization
// primitives that have separate signal and wait operations. Events are more
// fine-grained compared to fences and semaphores given that they can be
// signaled or waited within a command buffer while fences and semaphores are
// at queue submissions. Each of them have its usage requirements:
//
// * Fences must be signaled on GPU and waited on CPU. Fences must be reset
//   before reuse.
// * Binary semaphores must be signaled on GPU and waited on GPU. They do not
//   support wait-before-signal submission order. More importantly, binary
//   semaphore wait also unsignals the semaphore. So binary semaphore signals
//   and waits should occur in discrete 1:1 pairs.
// * Timeline semaphores can be signaled on CPU or GPU and waited on CPU or GPU.
//   They support wait-before-signal submission order. Timeline semaphores do
//   not need to be reset.
//
// It's clear that timeline semaphore is more flexible than fences and binary
// semaphores: it unifies GPU and CPU synchronization with a single primitive.
// But it's not always available: it requires the VK_KHR_timeline_semaphore
// or Vulkan 1.2. When it's not available, it can be emulated via `VkFence`s
// and binary `VkSemaphore`s. The emulation need to provide the functionality of
// timeline semaphores and also not violate the usage requirements of `VkFence`s
// and binary `VkSemaphore`s.
//
// The basic idea is to create a timeline object with time points to emulate the
// timeline semaphore, which consists of a monotonically increasing 64-bit
// integer value. Each time point represents a specific signaled/waited integer
// value of the timeline semaphore; each time point can associate with binary
// `VkSemaphore`s and/or `VkFence`s for emulating the synchronization.
//
// Concretely, for each of the possible signal -> wait scenarios timeline
// semaphore supports:
//
// ### GPU -> GPU (via `vkQueueSubmit`)
//
// Each `vkQueueSubmit` can attach a `VkTimelineSemaphoreSubmitInfo` to describe
// the timeline semaphore values signaled and waited. Each of the signaled value
// will be a time point and emulated by a binary `VkSemaphore`. We submit the
// binary `VkSemahpore`s to the GPU under the hood. For the waited values, the
// situation is more complicated because of the differences between binary and
// timeline semaphores:
//
// * Binary semaphore signal-wait relationship is strictly 1:1, unlike timeline
//   semaphore where we can have 1:N cases. This means for a specific binary
//   `VkSemaphore` used to emulate a signaled time point, we can have at most
//   one subsequent `vkQueueSubmit` waits on it. We need other mechanisms for
//   additional waits. A simple way is to involve the CPU and don't sumbit
//   the additional work to queue until the desired value is already signaled
//   past. This requires `VkFence`s for letting the CPU know the status of
//   GPU progress, but `VkFence` is needed anyway because of GPU -> CPU
//   synchronization.
// * Binary semaphores does not support wait-before-signal submission order.
//   This means we need to put the submission into a self-managed queue if the
//   binary semaphores used to emulate the time points waited by the submission
//   are not submitted to GPU yet.
//
// ### GPU -> CPU (via `vkWaitSemaphores`)
//
// Without timeline semaphore, we need to use fences to let CPU wait on GPU
// progress. So this direction can be emulated by `vkWaitFences`. It means we
// need to associate a `VkFence` with the given waited timeline semaphores.
// Because we don't know whether a particular `vkQueueSubmit` with timeline
// semaphores will be later waited on by CPU beforehand, we need to bundle each
// of them with a `VkFence` just in case they will be waited on later.
//
// ### CPU -> GPU (via `vkSignalSemaphore`)
//
// This direction can be handled by bumping the signaled timeline value and
// scan the self-managed queue to submit more work to GPU if possible.
//
// ### CPU -> CPU (via `vkWaitSemaphores`)
//
// This is similar to CPU -> GPU direction; we just need to enable other threads
// on CPU side and let them progress.
//
// The implementation is inspired by the Vulkan-ExtensionLayer project:
// https://github.com/KhronosGroup/Vulkan-ExtensionLayer. We don't handle all
// the aspects of the full spec though given that IREE only uses a subset of
// synchronization primitives. So this should not be treated as a full
// emulation of the Vulkan spec and thus does not substitute
// Vulkan-ExtensionLayer.
class EmulatedTimelineSemaphore final : public Semaphore {
 public:
  // Creates a timeline semaphore with the given |initial_value|.
  static StatusOr<ref_ptr<Semaphore>> Create(
      ref_ptr<VkDeviceHandle> logical_device,
      std::function<Status(Semaphore*)> on_semaphore_signal,
      std::function<void(Semaphore*)> on_semaphore_failure,
      std::function<void(absl::Span<VkFence>)> on_fence_signal,
      ref_ptr<TimePointSemaphorePool> semaphore_pool, uint64_t initial_value);

  EmulatedTimelineSemaphore(
      ref_ptr<VkDeviceHandle> logical_device,
      std::function<Status(Semaphore*)> on_semaphore_signal,
      std::function<void(Semaphore*)> on_semaphore_failure,
      std::function<void(absl::Span<VkFence>)> on_fence_signal,
      ref_ptr<TimePointSemaphorePool> semaphore_pool, uint64_t initial_value);

  ~EmulatedTimelineSemaphore() override;

  StatusOr<uint64_t> Query() override;

  Status Signal(uint64_t value) override;

  Status Wait(uint64_t value, Time deadline_ns) override;

  void Fail(Status status) override;

  // Gets a binary semaphore for waiting on the timeline to advance to the given
  // |value|. The semaphore returned won't be waited by anyone else. Returns
  // VK_NULL_HANDLE if no available semaphores for the given |value|.
  // |wait_fence| is the fence associated with the queue submission that waiting
  // on this semaphore.
  VkSemaphore GetWaitSemaphore(uint64_t value,
                               const ref_ptr<TimePointFence>& wait_fence);

  // Cancels the waiting attempt on the given binary |semaphore|. This allows
  // the |semaphore| to be waited by others.
  Status CancelWaitSemaphore(VkSemaphore semaphore);

  // Gets a binary semaphore for signaling the timeline to the given |value|.
  // |value| must be smaller than the current timeline value. |signal_fence| is
  // the fence associated with the queue submission that signals this semaphore.
  StatusOr<VkSemaphore> GetSignalSemaphore(
      uint64_t value, const ref_ptr<TimePointFence>& signal_fence);

 private:
  // Tries to advance the timeline to the given |to_upper_value| without
  // blocking and returns whether the |to_upper_value| is reached.
  StatusOr<bool> TryToAdvanceTimeline(uint64_t to_upper_value)
      ABSL_LOCKS_EXCLUDED(mutex_);
  // Similar to the above, but also returns the fences that are known to have
  // already signaled via |signaled_fences|.
  StatusOr<bool> TryToAdvanceTimeline(
      uint64_t to_upper_value, absl::InlinedVector<VkFence, 4>* signaled_fences)
      ABSL_LOCKS_EXCLUDED(mutex_);

  std::atomic<uint64_t> signaled_value_;

  ref_ptr<VkDeviceHandle> logical_device_;

  // Callback to inform that this timeline semaphore has signaled a new value.
  std::function<Status(Semaphore*)> on_semaphore_signal_;

  // Callback to inform that this timeline semaphore has encountered a failure.
  std::function<void(Semaphore*)> on_semaphore_failure_;

  // Callback to inform that the given fences have signaled.
  std::function<void(absl::Span<VkFence>)> on_fence_signal_;

  ref_ptr<TimePointSemaphorePool> semaphore_pool_;

  mutable absl::Mutex mutex_;

  // A list of outstanding semaphores used to emulate time points.
  //
  // The life time of each semaphore is in one of the following state:
  //
  // * Unused state: value = UINT64_MAX, signal/wait fence = nullptr. This is
  //   the state of the semaphore when it's initially acquired from the pool and
  //   not put in the queue for emulating a time point yet.
  // * Pending state: signaled value < value < UINT64_MAX, signal fence =
  //   <some-fence>, wait fence == nullptr. This is the state of the semaphore
  //   when it's put into the GPU queue for emulating a time point.
  // * Pending and waiting state: signaled value < value < UINT64_MAX, signal
  //   fence = <some-fence>, wait fence == <some-fence>. This is the state of
  //   the semaphore when it's put into the GPU queue for emulating a time
  //   point and there is another queue submission waiting on it in GPU.
  // * Signaled and not ever waited state: value <= signaled value, singal/wait
  //   fence = nullptr. This is the state of the semaphore when we know it's
  //   already signaled on GPU and there is no waiters for it.
  // * Signaled and waiting state: value <= signaled value, signal fence =
  //   nullptr, wait fence = <some-fence>. This is the state of the semaphore
  //   when we know it's already signaled on GPU and there is still one queue
  //   submission on GPU is waiting for it.
  IntrusiveList<TimePointSemaphore> outstanding_semaphores_
      ABSL_GUARDED_BY(mutex_);

  // NOTE: We only need to access this status (and thus take the lock) when we
  // want to either signal failure or query the status in the case of the
  // semaphore being set to UINT64_MAX.
  Status status_ ABSL_GUARDED_BY(mutex_);
};

// static
StatusOr<ref_ptr<Semaphore>> EmulatedTimelineSemaphore::Create(
    ref_ptr<VkDeviceHandle> logical_device,
    std::function<Status(Semaphore*)> on_semaphore_signal,
    std::function<void(Semaphore*)> on_semaphore_failure,
    std::function<void(absl::Span<VkFence>)> on_fence_signal,
    ref_ptr<TimePointSemaphorePool> semaphore_pool, uint64_t initial_value) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Create");
  return make_ref<EmulatedTimelineSemaphore>(
      std::move(logical_device), std::move(on_semaphore_signal),
      std::move(on_semaphore_failure), std::move(on_fence_signal),
      std::move(semaphore_pool), initial_value);
}

EmulatedTimelineSemaphore::EmulatedTimelineSemaphore(
    ref_ptr<VkDeviceHandle> logical_device,
    std::function<Status(Semaphore*)> on_semaphore_signal,
    std::function<void(Semaphore*)> on_semaphore_failure,
    std::function<void(absl::Span<VkFence>)> on_fence_signal,
    ref_ptr<TimePointSemaphorePool> semaphore_pool, uint64_t initial_value)
    : signaled_value_(initial_value),
      logical_device_(std::move(logical_device)),
      on_semaphore_signal_(std::move(on_semaphore_signal)),
      on_semaphore_failure_(std::move(on_semaphore_failure)),
      on_fence_signal_(std::move(on_fence_signal)),
      semaphore_pool_(std::move(semaphore_pool)) {}

EmulatedTimelineSemaphore::~EmulatedTimelineSemaphore() {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::dtor");
  IREE_CHECK_OK(TryToAdvanceTimeline(UINT64_MAX).status());
  absl::MutexLock lock(&mutex_);
  IREE_CHECK(outstanding_semaphores_.empty())
      << "Destroying an emulated timeline semaphore without first waiting on "
         "outstanding signals";
}

StatusOr<uint64_t> EmulatedTimelineSemaphore::Query() {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Query");
  IREE_DVLOG(2) << "EmulatedTimelineSemaphore::Query";
  IREE_ASSIGN_OR_RETURN(bool signaled, TryToAdvanceTimeline(UINT64_MAX));
  (void)signaled;
  uint64_t value = signaled_value_.load();
  IREE_DVLOG(2) << "Current timeline value: " << value;
  if (value == UINT64_MAX) {
    absl::MutexLock lock(&mutex_);
    return status_;
  }
  return value;
}

Status EmulatedTimelineSemaphore::Signal(uint64_t value) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Signal");
  IREE_DVLOG(2) << "EmulatedTimelineSemaphore::Signal";
  auto signaled_value = signaled_value_.exchange(value);
  IREE_DVLOG(2) << "Previous value: " << signaled_value
                << "; new value: " << value;
  // Make sure the previous signaled value is smaller than the new value.
  IREE_CHECK(signaled_value < value)
      << "Attempting to signal a timeline value out of order; trying " << value
      << " but " << signaled_value << " already signaled";

  // Inform the device to make progress given we have a new value signaled now.
  IREE_RETURN_IF_ERROR(on_semaphore_signal_(this));

  return OkStatus();
}

Status EmulatedTimelineSemaphore::Wait(uint64_t value, Time deadline_ns) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Wait");
  IREE_DVLOG(2) << "EmulatedTimelineSemaphore::Wait";

  VkFence fence = VK_NULL_HANDLE;
  do {
    IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Wait#loop");
    // First try to advance the timeline without blocking to see whether we've
    // already reached the desired value.
    IREE_ASSIGN_OR_RETURN(bool reached_desired_value,
                          TryToAdvanceTimeline(value));
    if (reached_desired_value) return OkStatus();

    // We must wait now. Find the first emulated time point that has a value >=
    // the desired value so we can wait on its associated signal fence to make
    // sure the timeline is advanced to the desired value.
    absl::MutexLock lock(&mutex_);
    auto semaphore = outstanding_semaphores_.begin();
    for (; semaphore != outstanding_semaphores_.end(); ++semaphore) {
      if ((*semaphore)->value >= value) break;
    }
    if (semaphore != outstanding_semaphores_.end()) {
      if (!(*semaphore)->signal_fence) {
        return InternalErrorBuilder(IREE_LOC)
               << "Timeline should have a signal fence for the first time "
                  "point beyond the signaled value";
      }
      IREE_DVLOG(2) << "Found timepoint semaphore " << *semaphore
                    << " (value: " << (*semaphore)->value
                    << ") to wait for desired timeline value: " << value;
      fence = (*semaphore)->signal_fence->value();
      // Found; we can break the loop and proceed to waiting now.
      break;
    }
    // TODO(antiagainst): figure out a better way instead of the busy loop here.
  } while (Now() < deadline_ns);

  if (fence == VK_NULL_HANDLE) {
    return DeadlineExceededErrorBuilder(IREE_LOC)
           << "Deadline reached when waiting timeline semaphore";
  }

  uint64_t timeout_ns =
      static_cast<uint64_t>(DeadlineToRelativeTimeoutNanos(deadline_ns));
  VK_RETURN_IF_ERROR(logical_device_->syms()->vkWaitForFences(
      *logical_device_, /*fenceCount=*/1, &fence, /*waitAll=*/true,
      timeout_ns));

  return TryToAdvanceTimeline(value).status();
}

void EmulatedTimelineSemaphore::Fail(Status status) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::Fail");
  absl::MutexLock lock(&mutex_);
  status_ = std::move(status);
  signaled_value_.store(UINT64_MAX);
}

VkSemaphore EmulatedTimelineSemaphore::GetWaitSemaphore(
    uint64_t value, const ref_ptr<TimePointFence>& wait_fence) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::GetWaitSemaphore");
  IREE_DVLOG(2) << "EmulatedTimelineSemaphore::GetWaitSemaphore";

  absl::MutexLock lock(&mutex_);

  VkSemaphore semaphore = VK_NULL_HANDLE;
  for (TimePointSemaphore* point : outstanding_semaphores_) {
    if (point->value > value && point->wait_fence) {
      point->wait_fence = add_ref(wait_fence);
      semaphore = point->semaphore;
      break;
    }
  }

  IREE_DVLOG(2) << "Binary VkSemaphore to wait on for timeline value (" << value
                << ") and wait fence (" << wait_fence.get()
                << "): " << semaphore;

  return semaphore;
}

Status EmulatedTimelineSemaphore::CancelWaitSemaphore(VkSemaphore semaphore) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::CancelWaitSemaphore");
  IREE_DVLOG(2) << "EmulatedTimelineSemaphore::CancelWaitSemaphore";

  absl::MutexLock lock(&mutex_);
  for (TimePointSemaphore* point : outstanding_semaphores_) {
    if (point->semaphore != semaphore) continue;

    if (!point->wait_fence) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Time point wasn't waited before";
    }
    point->wait_fence = nullptr;
    IREE_DVLOG(2) << "Cancelled waiting on binary VkSemaphore: " << semaphore;
    return OkStatus();
  }
  return InvalidArgumentErrorBuilder(IREE_LOC)
         << "No time point for the given semaphore";
}

StatusOr<VkSemaphore> EmulatedTimelineSemaphore::GetSignalSemaphore(
    uint64_t value, const ref_ptr<TimePointFence>& signal_fence) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::GetSignalSemaphore");
  IREE_DVLOG(2) << "EmulatedTimelineSemaphore::GetSignalSemaphore";

  if (signaled_value_.load() >= value) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "Timeline semaphore already signaled past " << value;
  }

  absl::MutexLock lock(&mutex_);

  auto insertion_point = outstanding_semaphores_.begin();
  while (insertion_point != outstanding_semaphores_.end()) {
    if ((*insertion_point)->value > value) break;
  }

  IREE_ASSIGN_OR_RETURN(TimePointSemaphore * semaphore,
                        semaphore_pool_->Acquire());
  semaphore->value = value;
  semaphore->signal_fence = add_ref(signal_fence);
  if (semaphore->wait_fence) {
    return InternalErrorBuilder(IREE_LOC)
           << "Newly acquired time point semaphore should not have waiters";
  }
  outstanding_semaphores_.insert(insertion_point, semaphore);
  IREE_DVLOG(2) << "Timepoint semaphore to signal for timeline value (" << value
                << ") and wait fence (" << signal_fence.get()
                << "): " << semaphore
                << " (binary VkSemaphore: " << semaphore->semaphore << ")";

  return semaphore->semaphore;
}

StatusOr<bool> EmulatedTimelineSemaphore::TryToAdvanceTimeline(
    uint64_t to_upper_value) {
  absl::InlinedVector<VkFence, 4> signaled_fences;
  auto status = TryToAdvanceTimeline(to_upper_value, &signaled_fences);
  // Inform the queue that some fences are known to have signaled. This should
  // happen here instead of inside the other TryToAdvanceTimeline to avoid
  // potential mutex deadlock, given here we are not holding a mutex anymore.
  if (!signaled_fences.empty()) {
    on_fence_signal_(absl::MakeSpan(signaled_fences));
  }
  return status;
}

StatusOr<bool> EmulatedTimelineSemaphore::TryToAdvanceTimeline(
    uint64_t to_upper_value, absl::InlinedVector<VkFence, 4>* signaled_fences) {
  IREE_TRACE_SCOPE0("EmulatedTimelineSemaphore::TryToAdvanceTimeline");
  IREE_DVLOG(3) << "EmulatedTimelineSemaphore::TryToAdvanceTimeline";

  uint64_t past_value = signaled_value_.load();
  IREE_DVLOG(3) << "Current timeline value: " << past_value
                << "; desired timeline value: " << to_upper_value;

  // Fast path for when already signaled past the desired value.
  if (past_value >= to_upper_value) return true;

  // We hold the lock during the entire resolve process so that we can resolve
  // to the furthest possible value.
  absl::MutexLock lock(&mutex_);

  IREE_DVLOG(3) << "# outstanding semaphores: "
                << outstanding_semaphores_.size();

  // The timeline has not signaled past the desired value and there is no
  // binary semaphore pending on GPU yet: certainly the timeline cannot
  // advance to the desired value.
  if (outstanding_semaphores_.empty()) return false;

  IntrusiveList<TimePointSemaphore> resolved_semaphores;

  auto clear_signal_fence = [&signaled_fences](ref_ptr<TimePointFence>& fence) {
    if (fence) {
      if (signaled_fences) signaled_fences->push_back(fence->value());
      fence = nullptr;
    }
  };

  bool keep_resolving = true;
  bool reached_desired_value = false;
  while (keep_resolving && !outstanding_semaphores_.empty()) {
    auto* semaphore = outstanding_semaphores_.front();
    IREE_DVLOG(3) << "Looking at timepoint semaphore " << semaphore << "..";
    IREE_DVLOG(3) << "  value: " << semaphore->value;
    IREE_DVLOG(3) << "  VkSemaphore: " << semaphore->semaphore;
    IREE_DVLOG(3) << "  signal fence: " << semaphore->signal_fence.get();
    IREE_DVLOG(3) << "  wait fence: " << semaphore->wait_fence.get();

    // If the current semaphore is for a value beyond our upper limit, then
    // early exit so that we don't spend time dealing with signals we don't yet
    // care about. This can prevent live lock where one thread is signaling
    // fences as fast/faster than another thread can consume them.
    if (semaphore->value > to_upper_value) {
      keep_resolving = false;
      reached_desired_value = true;
      break;
    }

    // If the current semaphore is for a value not greater than the past
    // signaled value, then we know it was signaled previously. But there might
    // be a waiter on it on GPU.
    if (semaphore->value <= past_value) {
      if (semaphore->signal_fence) {
        return InternalErrorBuilder(IREE_LOC)
               << "Timeline should already signaled past this time point and "
                  "cleared the signal fence";
      }

      // If ther is no waiters, we can recycle this semaphore now. If there
      // exists one waiter, then query its status and recycle on success. We
      // only handle success status here. Others will be handled when the fence
      // is checked for other semaphores' signaling status for the same queue
      // submission.
      if (!semaphore->wait_fence ||
          semaphore->wait_fence->GetStatus() == VK_SUCCESS) {
        clear_signal_fence(semaphore->signal_fence);
        semaphore->wait_fence = nullptr;
        outstanding_semaphores_.erase(semaphore);
        resolved_semaphores.push_back(semaphore);
        IREE_DVLOG(3) << "Resolved and recycling semaphore " << semaphore;
      }

      continue;
    }

    // This semaphore represents a value gerater than the known previously
    // signaled value. We don't know its status so we need to really query now.

    if (!semaphore->signal_fence) {
      return InternalErrorBuilder(IREE_LOC)
             << "The status of this time point in the timeline should still be "
                "pending with a singal fence";
    }
    VkResult signal_status = semaphore->signal_fence->GetStatus();

    switch (signal_status) {
      case VK_SUCCESS:
        IREE_DVLOG(3) << "..semaphore signaled";
        signaled_value_.store(semaphore->value);
        clear_signal_fence(semaphore->signal_fence);
        // If no waiters, we can recycle this semaphore now.
        if (!semaphore->wait_fence) {
          semaphore->wait_fence = nullptr;
          outstanding_semaphores_.erase(semaphore);
          resolved_semaphores.push_back(semaphore);
          IREE_DVLOG(3) << "Resolved and recycling semaphore " << semaphore;
        }
        break;
      case VK_NOT_READY:
        // The fence has not been signaled yet so this is the furthest time
        // point we can go in this timeline.
        keep_resolving = false;
        IREE_DVLOG(3) << "..semaphore not yet signaled";
        break;
      default:
        // Fence indicates an error (device lost, out of memory, etc).
        // Propagate this back to our status (and thus any waiters).
        // Since we only take the first error we find we skip all remaining
        // fences.
        keep_resolving = false;
        clear_signal_fence(semaphore->signal_fence);
        status_ = VkResultToStatus(signal_status, IREE_LOC);
        signaled_value_.store(UINT64_MAX);
        break;
    }
  }

  IREE_DVLOG(3) << "Releasing " << resolved_semaphores.size()
                << " resolved semaphores; " << outstanding_semaphores_.size()
                << " still outstanding";
  semaphore_pool_->ReleaseResolved(&resolved_semaphores);
  if (!status_.ok()) {
    on_semaphore_failure_(this);
    semaphore_pool_->ReleaseUnresolved(&outstanding_semaphores_);
    return status_;
  }

  return reached_desired_value;
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
