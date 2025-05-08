#pragma once

#include "linear.hpp"

#include <cmath>
#include <cstddef>

#include <array>


namespace NN
{
template<template<typename T, std::size_t In, std::size_t Out, bool Bias> typename Cell,
	typename T, std::size_t In, std::size_t Hidden, std::size_t Layers, bool Bias = true>
struct RNN {
	static_assert(Layers > 0);

	using L0 = Cell<T, In, Hidden, Bias>;
	using LS = Cell<T, Hidden, Hidden, Bias>;

	static constexpr std::size_t ParamCount = L0::ParamCount + LS::ParamCount * (Layers - 1);

	static_assert(sizeof(L0) == sizeof(LS));
	// They are all of the same size/layout,
	// so we will use a single array of Ll
	// with some dirty reinterpret_cast hacks on the first layer
	// This allows accessing ls.back() even for Layers == 1
	std::array<LS, Layers> ls;

	void operator()(const T* params, const T* in)
	{
		reinterpret_cast<L0&>(ls[0])(params, in), params += L0::ParamCount;
		for (std::size_t l = 1; l < Layers; ++l)
			ls[l](params, ls[l - 1].h.data()), params += LS::ParamCount;
	}
};
}
