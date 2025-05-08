#pragma once

#include "candle.hpp"
#include "signal.hpp"

#include <span>


struct trader
{
	virtual ~trader() = default;

	virtual Signal operator()(std::span<const candle> past_data) = 0;
};
