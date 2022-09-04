/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/epilogue/threadblock/output_tile_thread_map.h"


////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////

namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load output tile from shared memory in epilogue.
///
/// Satisfies: ReadableTileIterator | PredicatedTileIterator | ForwardTileIterator
///
template <
  typename ThreadMap_,      ///< Thread map (conept: OutputTileThreadMap)
  typename Element_,        ///< Element data type
  int Stride_               ///
>
class PredicatedTileStaticIterator {
public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_;

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  static_assert( ThreadMap::Iterations::kRow > 0,"ThreadMap::Iterations::kRow must be > 0");
  static_assert( ThreadMap::Iterations::kGroup > 0,"ThreadMap::Iterations::kGroup must be > 0");
  static_assert( ThreadMap::Iterations::kCluster > 0,"ThreadMap::Iterations::kCluster must be > 0");
  static_assert( ThreadMap::Iterations::kColumn > 0,"ThreadMap::Iterations::kColumn must be > 0");

  /// Fragment object
  using Fragment = Array<
    Element, 
    ThreadMap::Iterations::kColumn * 
    ThreadMap::Iterations::kRow * 
    ThreadMap::Iterations::kGroup * 
    ThreadMap::Iterations::kCluster * ThreadMap::kElementsPerAccess>;

  /// Memory access size
  using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

  ///
  static LongIndex const stride = Stride_ * int(sizeof(AccessType)) / kElementsPerAccess;
  static LongIndex const increment_row = stride * ThreadMap::Delta::kRow;
  static LongIndex const increment_group = stride * ThreadMap::Delta::kGroup
        - stride * ThreadMap::Delta::kRow * (ThreadMap::Iterations::kRow - 1);

  static LongIndex const increment_cluster = stride * ThreadMap::Delta::kCluster
        - stride * ThreadMap::Delta::kGroup * (ThreadMap::Iterations::kGroup - 1)
        - stride * ThreadMap::Delta::kRow * (ThreadMap::Iterations::kRow - 1);

  static LongIndex const advance_row = stride * ThreadMap::Shape::kRow;

  static LongIndex const advance_group = stride * (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow * ThreadMap::Count::kRow;
      
  static LongIndex const advance_cluster = 
        stride * 
        ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;;
      
  static LongIndex const advance_tile = 
        stride * 
        ThreadMap::Shape::kGroup * 
        ThreadMap::Shape::kRow * 
        ThreadMap::Shape::kCluster * 
        ThreadMap::Shape::kTile;

  /// Mask object
  struct Mask {

    static int const kCount = ThreadMap::Iterations::kColumn;

    /// Predicate state
    bool predicates[kCount];

    //
    // Mask
    //
    CUTLASS_HOST_DEVICE
    Mask() {
      enable();
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_HOST_DEVICE void clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = false;
      }
    }

    ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
    CUTLASS_DEVICE void enable() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = true;
      }
    }
  };

private:

  //
  // Data members
  //

  /// Byte-level pointer
  uint8_t *byte_pointer_;

  /// Array of boolean values to contain steady-state predicates
  Mask mask_;

  /// Extent of the matrix tile in rows
  Index extent_row_;

  /// A thread's starting row position (assuming steady-state predicates have been computed)
  Index thread_start_row_;

  /// Internal state counter
  int state_[3];

private:

  //
  // Methods
  //

public:

  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE
  PredicatedTileStaticIterator(
    Element *pointer,
    TensorCoord extent,
    int thread_idx,
    TensorCoord threadblock_offset = TensorCoord()
  ) {

    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    extent_row_ = extent.row();
    thread_start_row_ = thread_offset.row();

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {

      mask_.predicates[c] = ((thread_offset.column() 
        + ThreadMap::Delta::kColumn * c) < extent.column());
    }

    // Initialize pointer
    byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) + 
      LongIndex(thread_offset.row()) * stride + 
      LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;

    // Initialize internal state counter
    state_[0] = state_[1] = state_[2] = 0;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, int64_t byte_offset) {

    uint8_t *byte_pointer = byte_pointer_;
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx = 
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow 
            + group * ThreadMap::Delta::kGroup 
            + cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            bool guard = row_guard && mask_.predicates[column];

            if (guard) {
              frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column] = 
                memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess];
            }
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            byte_pointer += increment_row;
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          byte_pointer += increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += increment_cluster;
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_byte_offset(frag, 0);
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, int64_t byte_offset) {
    uint8_t *byte_pointer = byte_pointer_;
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx = 
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow 
            + group * ThreadMap::Delta::kGroup 
            + cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            bool guard = row_guard && mask_.predicates[column];

            if (guard) {
              
              memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess] =
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column];
            }
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            byte_pointer += increment_row;
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          byte_pointer += increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += increment_cluster;
      }
    }
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_byte_offset(frag, 0);
  }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  PredicatedTileStaticIterator &operator++() {

    ++state_[0];
    byte_pointer_ += advance_row;
    thread_start_row_ += ThreadMap::Shape::kRow;
    
    if (state_[0] == ThreadMap::Count::kRow) {

      state_[0] = 0;
      ++state_[1];
      byte_pointer_ += advance_group;

      thread_start_row_ += (ThreadMap::Shape::kGroup - 1) * 
        ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {

        state_[1] = 0;
        ++state_[2];
        byte_pointer_ += advance_cluster;

        thread_start_row_ += ThreadMap::Count::kGroup * 
          ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;
          byte_pointer_ += advance_tile;
        }
      }
    }

    return *this;
  }

  ///< Efficiently disables all accesses guarded by mask
  CUTLASS_DEVICE void clear_mask() {
    mask_.clear();
  }

  ///< Efficiently enables all accesses guarded by mask
  CUTLASS_DEVICE void enable_mask() {
    mask_.enable();
  }

  ///< Sets the mask
  CUTLASS_DEVICE void get_mask(Mask &mask) {
    return mask_;
  }

  ///< Sets the mask
  CUTLASS_DEVICE void set_mask(Mask const &mask) {
    mask_ = mask;
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
