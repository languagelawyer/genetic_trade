#pragma once

#include <cstddef>


namespace NN
{
template<typename T, std::size_t In, std::size_t Out, bool Bias = true>
struct Linear {
	static_assert(In > 0);
	static_assert(Out > 0);

	static constexpr std::size_t ParamCount = (In + Bias) * Out;

	void operator()(const T* __restrict params, const T* __restrict in, T* __restrict out) const
	{
		// y = xW^T [+ b]
		for (std::size_t o = 0; o < Out; ++o)
		{
			T sum = 0;
			for (std::size_t i = 0; i < In; ++i)
				sum += in[i] * params[In * o + i];
			if constexpr (Bias) sum += params[In * Out + o];
			out[o] = sum;
		}
	}
};
}
