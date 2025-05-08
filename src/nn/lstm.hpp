#pragma once

#include "linear.hpp"

#include <cmath>
#include <cstddef>

#include <array>


namespace NN
{
template<typename T, std::size_t In, std::size_t Hidden, bool Bias = true>
struct LSTMCell {
	using X2IFGO = Linear<T, In, Hidden, false>; // only one bias vector per gate, unlike in PyTorch
	using H2IFGO = Linear<T, Hidden, Hidden, Bias>;

	static constexpr std::size_t ParamCount = 4 * (X2IFGO::ParamCount + H2IFGO::ParamCount);

	std::array<T, Hidden> h = {};
	std::array<T, Hidden> c = {};

	void operator()(const T* params, const T* in)
	{
		decltype(h) ifgox[4], ifgoh[4];

		for (std::size_t i = 0; i < 4; ++i) X2IFGO{}(params, in, ifgox[i].data()), params += X2IFGO::ParamCount;
		for (std::size_t i = 0; i < 4; ++i) H2IFGO{}(params, h.data(), ifgoh[i].data()), params += H2IFGO::ParamCount;

		auto sigmoid = [](T x) { return T(1) / (T(1) + std::exp(-x)); };

		// tanh for g, sigmoid for the rest
		for (std::size_t i = 0; i < Hidden; ++i) {
			// c' = f . c + i . g
			c[i] = sigmoid(ifgox[1][i] + ifgoh[1][i]) * c[i]
			     + sigmoid(ifgox[0][i] + ifgoh[0][i]) * std::tanh(ifgox[2][i] + ifgoh[2][i]);

			// h' = o . tanh(c')
			h[i] = sigmoid(ifgox[3][i] + ifgoh[3][i]) * std::tanh(c[i]);
		}
	}
};
}
