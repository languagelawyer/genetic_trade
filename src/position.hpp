#pragma once

#include <cstddef>
#include <cstdint>

#include <vector>


struct position
{
	double size = 0;
	double total_value = 0;

	double current_value = 0;

	std::vector<std::size_t> orders;

	std::uint64_t close_time = 0;
};
