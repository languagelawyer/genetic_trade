#pragma once

#include "candle.hpp"
#include "order.hpp"
#include "position.hpp"
#include "signal.hpp"
#include "trader.hpp"

#include <span>
#include <vector>


struct spot
{
	double min_order_value;
	double commission;

	double balance;
	double max_equity = balance;
	double mdd = 0;

	std::vector<order> orders;
	std::vector<position> positions;

	position* pos = nullptr;

	void trade(trader& trader, std::span<const candle> past_data);
};
