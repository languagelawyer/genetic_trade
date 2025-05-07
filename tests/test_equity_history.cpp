#include "trader.hpp"
#include "spot.hpp"

#include <cassert>

#include <iostream>
#include <vector>


class test_trader : public trader
{
	Signal operator()(std::span<candle> past_data) override
	{
		if (past_data.size() == 1) return Signal::BUY;
		if (past_data.size() == 3) return Signal::SELL;
		return Signal::HOLD;
	}
};

int main()
{
	std::vector<candle> candles = {
		{ 1, 5, 5, 5, 5, },
		{ 2, 5, 5, 5, 5, },
		{ 3, 4, 4, 4, 4, },
		{ 4, 5, 5, 5, 5, },
	};

	spot spot {
		.min_order_value = 5,
		.commission = 0,
		.balance = 100,
	};
	test_trader tt;
	spot.trade(tt, candles);

	assert((spot.balance_open == std::vector<double>{ 100, 100, 95, 95, }));
	assert((spot.balance_close == std::vector<double>{ 100, 95, 95, 100, }));
	assert((spot.equity_open == std::vector<double>{ 100, 100, 99, 100, }));
	assert((spot.equity_close == std::vector<double>{ 100, 100, 99, 100, }));
}
