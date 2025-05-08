#pragma once

#include "linear.hpp"

#include <cmath>
#include <cstddef>

#include <array>


namespace NN
{
template<typename T, std::size_t In, std::size_t Hidden, bool Bias = true>
struct SimpleRNNCell {
	using X2H = Linear<T, In, Hidden, false>; // only one bias vector, unlike in PyTorch
	using H2H = Linear<T, Hidden, Hidden, Bias>;

	static constexpr std::size_t ParamCount = X2H::ParamCount + H2H::ParamCount;
	static_assert(ParamCount == (In + Hidden + Bias) * Hidden);

	std::array<T, Hidden> h = {};

	void operator()(const T* params, const T* in)
	{
		decltype(h) ox, oh;

		X2H{}(params, in, ox.data());
		H2H{}(params + X2H::ParamCount, h.data(), oh.data());

		for (std::size_t i = 0; i < Hidden; ++i)
			h[i] = std::tanh(ox[i] + oh[i]);
	}
};
}
