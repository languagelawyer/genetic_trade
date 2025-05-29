#pragma once

#include <cstdint>


struct candle {
	std::uint32_t open_time;
	float open;
	float high;
	float low;
	float close;
	float volume;
};
