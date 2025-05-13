#include "spot.hpp"
#include "order.hpp"

#include <functional>


void spot::trade(trader& trader, std::span<const candle> candles)
{
	// Loop till one before the last, so we can haz the "current" bar
	for (auto i = 0u; i + 1 < candles.size(); ++i)
	{
		const auto& curr = candles[i + 1];

		// Since we assume trading at 1s intervals,
		// update the equity and MDD at the "current" bar,
		// before executing the order
		auto equity = balance;
		if (pos) equity += pos->size * curr.low;

		max_equity = std::max(max_equity, equity);
		auto drawdown = (max_equity - equity) / max_equity;
		mdd = std::max(mdd, drawdown);

		// Since we assume trading at 1s intervals, all orders are marked orders
		// and are executed conservatively/pessimistically, at worst prices:
		// High for BUY, Low for SELL
		auto signal = trader(candles.first(i + 1));

		// Buy for the min_order_size, sell the whole position
		if (signal == Signal::BUY)
		{
			if (!pos) pos = &positions.emplace_back();

			balance -= commission;
			balance -= min_order_value;

			auto o = order {
				.create_time = curr.open_time,
				.price = curr.high,
				.size = min_order_value / curr.high,
			};
			pos->size += o.size;
			pos->total_value += min_order_value;
			pos->orders.push_back(orders.size());
			orders.push_back(std::move(o));
		}

		if (signal == Signal::SELL and pos)
		{
			auto close_value = pos->size * curr.low;
			// Can't sell for less than min_order_size
			if (close_value < min_order_value) continue;

			balance -= commission * pos->orders.size();
			balance += close_value;

			pos->close_time = curr.open_time;
			pos->close_value = close_value;
			pos = nullptr;
		}
	}
}
