#pragma once

#include <cstdint>


struct candle {
	std::int64_t open_time;
	double open;
	double high;
	double low;
	double close;
	double volume;
};
