# Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

include(FetchContent)

set(MSCCLPP_DIR "" CACHE STRING "Location of local MSCCL++ repo to build against")

if(MSCCLPP_DIR)
  set(FETCHCONTENT_SOURCE_DIR_MSCCLPP ${MSCCLPP_DIR} CACHE STRING "MSCCL++ source directory override")
endif()

set(MSCCLPP_REPOSITORY "https://github.com/microsoft/mscclpp.git" CACHE STRING "MSCCL++ repo to fetch")
FetchContent_Declare(
  mscclpp
  GIT_REPOSITORY ${MSCCLPP_REPOSITORY}
  GIT_TAG        chhwang/fix-cmake
)

FetchContent_GetProperties(mscclpp)

if(NOT mscclpp_POPULATED)
  FetchContent_Populate(mscclpp)
  set(MSCCLPP_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
  add_subdirectory(${mscclpp_SOURCE_DIR} ${mscclpp_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
