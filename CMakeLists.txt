# @lint-ignore-every LICENSELINT
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)

set(FAISS_LANGUAGES CXX)

if(FAISS_ENABLE_GPU)
  if (FAISS_ENABLE_ROCM)
    list(APPEND FAISS_LANGUAGES HIP)
    list(PREPEND CMAKE_MODULE_PATH "/opt/rocm/lib/cmake")
    list(PREPEND CMAKE_PREFIX_PATH "/opt/rocm")
  else()
    list(APPEND FAISS_LANGUAGES CUDA)
  endif()
endif()

if(FAISS_ENABLE_CUVS)
  include(cmake/thirdparty/fetch_rapids.cmake)
  include(rapids-cmake)
  include(rapids-cpm)
  include(rapids-cuda)
  include(rapids-export)
  include(rapids-find)

  rapids_cuda_init_architectures(faiss)
  rapids_cuda_init_architectures(pyfaiss)
  rapids_cuda_init_architectures(faiss_c_library)
endif()

project(faiss
  VERSION 1.13.2
  DESCRIPTION "A library for efficient similarity search and clustering of dense vectors."
  HOMEPAGE_URL "https://github.com/facebookresearch/faiss"
  LANGUAGES ${FAISS_LANGUAGES})
include(GNUInstallDirs)

if(FAISS_ENABLE_CUVS)
  set(CMAKE_CXX_STANDARD 17)
else()
  set(CMAKE_CXX_STANDARD 20)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Valid values are "generic", "avx2", "avx512", "avx512_spr", "sve".
option(FAISS_OPT_LEVEL "" "generic")
option(FAISS_ENABLE_GPU "Enable support for GPU indexes." ON)
option(FAISS_GPU_STATIC "Link GPU libraries statically." OFF)
option(FAISS_ENABLE_CUVS "Enable cuVS for GPU indexes." OFF)
option(FAISS_ENABLE_ROCM "Enable ROCm for GPU indexes." OFF)
option(FAISS_ENABLE_MKL "Enable MKL." ON)
option(FAISS_ENABLE_PYTHON "Build Python extension." ON)
option(FAISS_ENABLE_C_API "Build C API." OFF)
option(FAISS_ENABLE_EXTRAS "Build extras like benchmarks and demos" ON)
option(FAISS_USE_LTO "Enable Link-Time optimization" OFF)
option(FAISS_ENABLE_SVS "Enable SVS (Intel(R) Scalable Vector Search) integration." OFF)
set(FAISS_SVS_RUNTIME_VERSION "v0" CACHE STRING "Version of the SVS runtime API to use")
set_property(CACHE FAISS_SVS_RUNTIME_VERSION PROPERTY STRINGS "v0")

if(FAISS_ENABLE_GPU)
  if(FAISS_ENABLE_ROCM)
    enable_language(HIP)
    add_definitions(-DUSE_AMD_ROCM)
    find_package(HIP REQUIRED)
    find_package(hipBLAS REQUIRED)
    set(GPU_EXT_PREFIX "hip")
    execute_process(COMMAND ${PROJECT_SOURCE_DIR}/faiss/gpu/hipify.sh)
  else ()
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    enable_language(CUDA)
    set(GPU_EXT_PREFIX "cu")
  endif()
endif()

if(FAISS_ENABLE_CUVS AND NOT TARGET cuvs::cuvs)
   find_package(cuvs)
 endif()

add_subdirectory(faiss)

if(FAISS_ENABLE_GPU)
  add_subdirectory(faiss/gpu)
endif()

if(FAISS_ENABLE_PYTHON)
  add_subdirectory(faiss/python)
endif()

if(FAISS_ENABLE_C_API)
  add_subdirectory(c_api)
endif()

if(FAISS_ENABLE_EXTRAS)
  add_subdirectory(demos)
  add_subdirectory(benchs)
  add_subdirectory(tutorial/cpp)
endif()

# CTest must be included in the top level to enable `make test` target.
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
  add_subdirectory(perf_tests)
  if(FAISS_ENABLE_GPU)
    if(FAISS_ENABLE_ROCM)
      add_subdirectory(faiss/gpu/test)
    else()
      add_subdirectory(faiss/gpu/test)
    endif()
  endif()
endif()

##
## HPDIC Experiments
##

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# --- Deep1B Benchmarks ---
add_executable(test_sivf_deep_add hpdic/experiment/deep/test_sivf_deep_add.cpp)
target_link_libraries(test_sivf_deep_add faiss OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)
add_executable(test_sivf_deep_search hpdic/experiment/deep/test_sivf_deep_search.cpp)
target_link_libraries(test_sivf_deep_search faiss OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)
add_executable(test_sivf_deep_delete hpdic/experiment/deep/test_sivf_deep_delete.cpp)
target_link_libraries(test_sivf_deep_delete faiss OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)
add_executable(test_sivf_deep_pareto hpdic/experiment/deep/test_sivf_deep_pareto.cpp)
target_link_libraries(test_sivf_deep_pareto faiss OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)

# --- SIFT1M Benchmark ---
add_executable(test_sivf_sift_add hpdic/experiment/sift/test_sivf_sift_add.cpp)
target_link_libraries(test_sivf_sift_add 
    faiss 
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
)
add_executable(test_sivf_sift_search hpdic/experiment/sift/test_sivf_sift_search.cpp)
target_link_libraries(test_sivf_sift_search 
    faiss 
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
)
add_executable(test_sivf_sift_delete hpdic/experiment/sift/test_sivf_sift_delete.cpp)
target_link_libraries(test_sivf_sift_delete 
    faiss 
    OpenMP::OpenMP_CXX 
    CUDA::cudart 
    CUDA::cublas
)
add_executable(test_sivf_sift_pareto hpdic/experiment/sift/test_sivf_sift_pareto.cpp)
target_link_libraries(test_sivf_sift_pareto 
    faiss 
    OpenMP::OpenMP_CXX 
    CUDA::cudart 
    CUDA::cublas
)

# --- GIST1M Benchmarks ---
add_executable(test_sivf_gist_add hpdic/experiment/gist/test_sivf_gist_add.cpp)
target_link_libraries(test_sivf_gist_add faiss OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)
add_executable(test_sivf_gist_search hpdic/experiment/gist/test_sivf_gist_search.cpp)
target_link_libraries(test_sivf_gist_search faiss OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)
add_executable(test_sivf_gist_delete hpdic/experiment/gist/test_sivf_gist_delete.cpp)
target_link_libraries(test_sivf_gist_delete faiss OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)
add_executable(test_sivf_gist_pareto hpdic/experiment/gist/test_sivf_gist_pareto.cpp)
target_link_libraries(test_sivf_gist_pareto faiss OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)

# --- T2I-1B Benchmarks ---
add_executable(test_sivf_t2i_add hpdic/experiment/t2i/test_sivf_t2i_add.cpp)
target_link_libraries(test_sivf_t2i_add faiss OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)
add_executable(test_sivf_t2i_search hpdic/experiment/t2i/test_sivf_t2i_search.cpp)
target_link_libraries(test_sivf_t2i_search faiss OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)
add_executable(test_sivf_t2i_delete hpdic/experiment/t2i/test_sivf_t2i_delete.cpp)
target_link_libraries(test_sivf_t2i_delete faiss OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)
add_executable(test_sivf_t2i_pareto hpdic/experiment/t2i/test_sivf_t2i_pareto.cpp)
target_link_libraries(test_sivf_t2i_pareto faiss OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)

# --- DINO10B Benchmarks ---
add_executable(test_sivf_dino_add hpdic/experiment/dino/test_sivf_dino_add.cpp)
target_link_libraries(test_sivf_dino_add faiss MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)
add_executable(test_sivf_dino_delete hpdic/experiment/dino/test_sivf_dino_delete.cpp)
target_link_libraries(test_sivf_dino_delete faiss MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas)

# --- Sliding Window ---
add_executable(test_sivf_sliding hpdic/experiment/test_sivf_sliding.cpp)
target_link_libraries(test_sivf_sliding 
    faiss 
    OpenMP::OpenMP_CXX 
    CUDA::cudart 
    CUDA::cublas
)

# --- Memory Usage ---
add_executable(test_sivf_memory hpdic/experiment/test_sivf_memory.cpp)
target_link_libraries(test_sivf_memory 
    faiss 
    OpenMP::OpenMP_CXX 
    CUDA::cudart 
    CUDA::cublas
)

# --- Non-IVF ---
add_executable(test_sivf_nonivf hpdic/experiment/test_sivf_noivf.cpp)
target_link_libraries(test_sivf_nonivf 
    faiss 
    OpenMP::OpenMP_CXX 
    CUDA::cudart 
    CUDA::cublas
)

# --- Profiling ---
add_executable(test_sivf_profiling hpdic/experiment/test_sivf_profiling.cpp)
target_link_libraries(test_sivf_profiling 
    faiss 
    OpenMP::OpenMP_CXX 
    CUDA::cudart 
    CUDA::cublas
)

# --- P99 ---
add_executable(test_sivf_p99 hpdic/experiment/test_sivf_p99.cpp)
target_link_libraries(test_sivf_p99 
    faiss 
    OpenMP::OpenMP_CXX 
    CUDA::cudart 
    CUDA::cublas
)

# --- Mixed Workload ---
add_executable(test_sivf_mixed hpdic/experiment/test_sivf_mixed.cpp)
target_link_libraries(test_sivf_mixed 
    faiss 
    OpenMP::OpenMP_CXX 
    CUDA::cudart 
    CUDA::cublas
)
