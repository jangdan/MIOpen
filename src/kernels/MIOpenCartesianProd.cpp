/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

# ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
# include <hip/hipf16.h>
# include <hip/hip_runtime.h>
# endif

# include "float_types.h"
# include "tensor_view.hpp"

template <typename T> __device__ void PopulateRowImpl(
	size_t const					operand_number,			// aka `dim1_idx`; the row to populate
	T const * const __restrict__	operand,				// `input`
	// tensor_view_t<1>				operand_tensor_view,	// `input_tv`
	size_t const					operand_length,
	T * const __restrict__			workspace,				// `output_ws`
	// tensor_view_t<2>				workspace_tensor_view,	// `output_tv`
	size_t const					workspace_width,
	size_t const					smear_length			// `stride`; how many times each component of the operand should be "smeared"
) {
	// Fill in a row of the workspace by "smear-repeating" the i-th operand
	auto const gid = blockIdx.x * blockDim.x + threadIdx.x; // column number
	// auto const workspace_width = workspace_tensor_view.size[0];
	if (gid >= workspace_width) return;
	// auto const operand_length = operand_tensor_view.size(0);
	workspace[operand_number * workspace_width /* = row offset */ + gid] = operand[(gid / smear_length) % operand_length];
}
extern "C" __global__ void PopulateRow(
	size_t const						operand_number,			// aka `dim1_idx`; the row to populate
	FLOAT const * const __restrict__	operand,				// `input`
	// tensor_view_t<1>					operand_tensor_view,	// `input_tv`
	size_t const						operand_length,
	FLOAT * const __restrict__			workspace,				// `output_ws`
	// tensor_view_t<2>					workspace_tensor_view,	// `output_tv`
	size_t const						workspace_width,
	size_t const						smear_length			// `stride`; how many times each component of the operand should be "smeared"
) {
	PopulateRowImpl<FLOAT>(operand_number, operand, operand_length, workspace, workspace_width, smear_length);
}

# define TILE_SIZE 16
template <typename T> __device__ void TransposeImpl(
	T const * const __restrict__	workspace,			// `output_ws`
	T * const __restrict__			output,				// `output`
	tensor_view_t<2> const			output_tensor_view	// `output_tv`
) {
	T __shared__ block [TILE_SIZE][TILE_SIZE + 1];
	auto const gid {blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
	if (gid.x < output_tensor_view.size[0] && gid.y < output_tensor_view.size[1]) {
		block[threadIdx.y][threadIdx.x] = workspace[gid.y * output_tensor_view.size[0] /* = row offset */ + gid.x];
	}
	__syncthreads();
	auto const output_n {blockIdx.x * TILE_SIZE + threadIdx.y, blockIdx.y * TILE_SIZE + threadIdx.x};
	if (output_n.x < output_tensor_view.size[0] && output_n.y < output_tensor_view.size[1]) {
		output[output_tensor_view.get_tensor_view_idx({output_n[0], output_n[1]})] = block[threadIdx.x][threadIdx.y];
	}
}
extern "C" __global__ void Transpose(
	T const * const __restrict__	workspace,			// `output_ws`
	T * const __restrict__			output,				// `output`
	tensor_view_t<2> const			output_tensor_view	// `output_tv`
) {
	TransposeImpl<FLOAT>(workspace, output, output_tensor_view);
}

template <typename T> __device__ void CartesianProdBackwardImpl() {
	
}
extern "C" __global__ void CartesianProdBackward() {
	CartesianProdBackwardImpl<FLOAT>();
}
