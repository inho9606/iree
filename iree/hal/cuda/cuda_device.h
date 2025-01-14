// Copyright 2021 Google LLC
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

#ifndef IREE_HAL_CUDA_CUDA_DEVICE_H_
#define IREE_HAL_CUDA_CUDA_DEVICE_H_

#include "iree/hal/api.h"
#include "iree/hal/cuda/api.h"
#include "iree/hal/cuda/dynamic_symbols.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a device that owns and manages its own CUcontext.
iree_status_t iree_hal_cuda_device_create(iree_hal_driver_t* driver,
                                          iree_string_view_t identifier,
                                          iree_hal_cuda_dynamic_symbols_t* syms,
                                          CUdevice device,
                                          iree_allocator_t host_allocator,
                                          iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_CUDA_CUDA_DEVICE_H_
