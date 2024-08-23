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

# include <miopen/cartesianprod/solvers.hpp>

# include <miopen/cartesianprod/invoke_params.hpp>

# include <miopen/kernel_build_params.hpp>
# include <miopen/tensor_view_utils.hpp>

# define LOCAL_SIZE 256

namespace miopen :: solver :: cartesianprod {

	bool IsImprovementOverROCm(miopen :: cartesianprod :: ProblemDescription const & problem) {
		if (problem.IsBackward()) return true;
		else {
			if (problem.IsAllContiguous()) {
				switch (problem.GetOutputDesc().GetType()) {
					case miopenFloat:
					return problem.GetOutputDesc().GetElementSize() < float32contiguousfwdminimumnonimprovementnumel;
					case miopenHalf:
					return problem.GetOutputDesc().GetElementSize() < float16contiguousfwdminimumnonimprovementnumel;
					case miopenBFloat16:
					return problem.GetOutputDesc().GetElementSize() < bfloat16contiguousfwdminimumnonimprovementnumel;
				}
			} else return problem.GetOutputDesc().GetElementSize() < noncontiguousfwdminimumnonimprovementnumel;
		}
		return false;
	}
	bool CartesianProd :: IsApplicable(ExecutionContext const & context, miopen :: cartesianprod :: ProblemDescription const & problem) const {
		if (!IsImprovementOverROCm(problem)) return false;
		return true;
	}

	ConvSolution CartesianProd :: GetSolution(ExecutionContext const & context, miopen :: cartesianprod :: ProblemDescription const & problem) const {
		auto result = ConvSolution {miopenStatusSuccess};
		for (signed i = problem.GetOperands().size() - 1; i >= 0; --i) {
			auto const workspacewidth = problem.GetOutputDesc().GetLengths()[0];
			size_t xlocalsize = LOCAL_SIZE;
			size_t ylocalsize = 1;
			size_t zlocalsize = 1;
			size_t xgridsize  = AlignUp(workspacewidth, xlocalsize);
			size_t ygridsize  = 1;
			size_t zgridsize  = 1;
			auto kernel = KernelInfo {};
			kernel.kernel_file = "MIOpenCartesianProd.cpp";
			kernel.kernel_name = "PopulateRow";
			auto const dtype = problem.GetOutputDesc().GetType();
			auto const buildparams = KernelBuildParameters {
				{"MIOPEN_USE_FP16",		static_cast<int>(dtype == miopenHalf)},
				{"MIOPEN_USE_FP32",		static_cast<int>(dtype == miopenFloat)},
				{"MIOPEN_USE_FP64",		static_cast<int>(dtype == miopenDouble)},
				{"MIOPEN_USE_BFP16",	static_cast<int>(dtype == miopenBFloat16)},
				{"LOCAL_SIZE",			LOCAL_SIZE},
			};
			kernel.comp_options = buildparams.GenerateFor(kbp :: HIP {});
			kernel.l_wk.push_back(xlocalsize);
			kernel.l_wk.push_back(ylocalsize);
			kernel.l_wk.push_back(zlocalsize);
			kernel.g_wk.push_back(xgridsize);
			kernel.g_wk.push_back(ygridsize);
			kernel.g_wk.push_back(zgridsize);
			result.construction_params.push_back(kernel);
		}
		{
			size_t xlocalsize = TILE_SIZE;
			size_t ylocalsize = TILE_SIZE;
			size_t zlocalsize = 1;
			size_t xgridsize  = AlignUp(problem.GetOutputDesc().GetLengths(0), xlocalsize);
			size_t ygridsize  = AlignUp(problem.GetOutputDesc().GetLengths(1), ylocalsize);
			size_t zgridsize  = 1;
			auto kernel = KernelInfo {};
			kernel.kernel_file = "MIOpenCartesianProd.cpp";
			kernel.kernel_name = "Transpose";
			auto const dtype = problem.GetOutputDesc().GetType();
			auto const buildparams = KernelBuildParameters {
				{"MIOPEN_USE_FP16",		static_cast<int>(dtype == miopenHalf)},
				{"MIOPEN_USE_FP32",		static_cast<int>(dtype == miopenFloat)},
				{"MIOPEN_USE_FP64",		static_cast<int>(dtype == miopenDouble)},
				{"MIOPEN_USE_BFP16",	static_cast<int>(dtype == miopenBFloat16)},
				{"LOCAL_SIZE",			LOCAL_SIZE},
			};
			kernel.comp_options = buildparams.GenerateFor(kbp :: HIP {});
			kernel.l_wk.push_back(xlocalsize);
			kernel.l_wk.push_back(ylocalsize);
			kernel.l_wk.push_back(zlocalsize);
			kernel.g_wk.push_back(xgridsize);
			kernel.g_wk.push_back(ygridsize);
			kernel.g_wk.push_back(zgridsize);
			result.construction_params.push_back(kernel);
		}
		result.invoker_factory = [] (std :: vector<Kernel> const & kernels) {
			return [=] (Handle const & handle, AnyInvokeParams const & rawparams) {
				decltype(auto) PopulateRow = handle.Run(kernels[params.GetOperands().size()]);
				decltype(auto) params = rawparams.CastTo<miopen :: cartesianprod :: InvokeParams>();
				auto const numel = params.outputDesc -> GetElementSize();
				auto smearlength = 1;
				for (signed i = params.GetOperands().size() - 1; i >= 0; --i) {
					decltype(auto) Transpose = handle.Run(kernels[1]);
					auto const operand = params.GetOperands()[i];
					auto const operandlength = get_inner_expanded_tv<1>(params.GetOperandDescriptors()[i]).size(0);
					PopulateRow(
						i,
						operand,
						operandlength
						params.GetWorkspace(),
						params.GetWorkspaceSize(),
						smearlength
					);
					smearlength *= operandlength;
				}
				Transpose(
					
				);
			};
		};
		return result;
	}

}
