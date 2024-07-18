/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

# include "maskedfill.hpp"

# include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL	(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_STR	(MIOPEN_TEST_FLOAT_ARG)

namespace maskedfill {

	std :: string GetFloatArg() {
		auto const tmp = miopen :: env :: value(MIOPEN_TEST_FLOAT_ARG);
		return tmp.empty()? "" : tmp;
	}

}

using namespace maskedfill;

struct MaskedFillForwardTestFloat:	MaskedFillForwardTest<float>	{};
struct MaskedFillBackwardTestFloat:	MaskedFillBackwardTest<float>	{};

TEST_P(MaskedFillForwardTestFloat, Ok) {
	if (miopen :: env :: enabled(MIOPEN_TEST_ALL) && GetFloatArg() == "--float") {
		RunTest();
		Verify();
	} else {
		GTEST_SKIP();
	}
}

TEST_P(MaskedFillBackwardTestFloat, Ok) {
	if (miopen :: env :: enabled(MIOPEN_TEST_ALL) && GetFloatArg() == "--float") {
		RunTest();
		Verify();
	} else {
		GTEST_SKIP();
	}
}

INSTANTIATE_TEST_SUITE_P(, MaskedFillForwardTestFloat,	testing :: ValuesIn(MaskedFillTestConfigs(MIOPEN_MASKEDFILL_FORWARD)));
INSTANTIATE_TEST_SUITE_P(, MaskedFillBackwardTestFloat,	testing :: ValuesIn(MaskedFillTestConfigs(MIOPEN_MASKEDFILL_BACKWARD)));
