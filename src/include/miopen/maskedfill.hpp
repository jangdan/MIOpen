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

#ifndef MIOPEN_MASKEDFILL_HPP_
#define MIOPEN_MASKEDFILL_HPP_

#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

miopenStatus_t MaskedFillForward(Handle& handle,
                                 TensorDescriptor const& inputDesc,
                                 ConstData_t input,
                                 TensorDescriptor const& outputDesc,
                                 Data_t output,
                                 TensorDescriptor const& maskDesc,
                                 ConstData_t mask,
                                 float value);

miopenStatus_t MaskedFillBackward(Handle& handle,
                                  TensorDescriptor const& outputGradientDesc,
                                  ConstData_t outputGradient,
                                  TensorDescriptor const& inputGradientDesc,
                                  Data_t inputGradient,
                                  TensorDescriptor const& maskDesc,
                                  ConstData_t mask,
                                  float value);

} // namespace miopen

#endif
