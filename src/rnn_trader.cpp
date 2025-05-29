#include "candle.hpp"
#include "signal.hpp"
#include "spot.hpp"
#include "trader.hpp"

#include "nn/linear.hpp"
#include "nn/rnn.hpp"
#include "nn/lstm.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <algorithm>


namespace
{
	using param_t = decltype(candle::open);

	constexpr double initial_balance = 200;
	constexpr double min_order_value = 5;
	constexpr double commission = 0.01;
	constexpr size_t warmup_period = 600;
	constexpr double max_position_value = 100;

	struct Network
	{
		static constexpr size_t In = 2;
		static constexpr size_t Hidden = 4;
		static constexpr size_t Layers = 4;
		static constexpr size_t Out = 1;
		static constexpr bool Bias = false;

		using RNN = NN::RNN<NN::LSTMCell, param_t, In, Hidden, Layers, Bias>;
		using Linear = NN::Linear<param_t, Hidden, Out, Bias>;

		RNN rnn;

		static constexpr size_t ParamCount = RNN::ParamCount + Linear::ParamCount;

		void operator()(const param_t* params, const param_t* in, param_t* out)
		{
			rnn(params, in), params += decltype(rnn)::ParamCount;
			Linear{}(params, rnn.ls.back().h.data(), out);

			std::for_each(out, out + Out, [](auto& x) { x = std::tanh(x); });
		}
	};

	struct NNTrader : trader
	{
		spot& engine;
		Network& nn;
		const param_t* params;

		NNTrader(spot& engine, Network& nn, const param_t* params)
		: engine(engine)
		, nn(nn)
		, params(params)
		{}

		Signal operator()(std::span<const candle> past_data) override
		{
			if (past_data.size() < 2) return Signal::HOLD;

			const auto& last = past_data[past_data.size() - 1];
			const auto& prev = past_data[past_data.size() - 2];

			param_t in[Network::In] = {
				std::log(last.open / prev.open),
				std::log(last.volume + 1),
			};
			param_t out[Network::Out]; // tanh output, (-1, 1) range
			nn(params, in, out);

			if (past_data.size() < warmup_period) return Signal::HOLD;

			if (out[0] < -0.5) return Signal::SELL;

			if (engine.pos and engine.pos->total_value >= max_position_value) return Signal::HOLD;

			if (out[0] > +0.5) return Signal::BUY;

			return Signal::HOLD;
		}
	};

	struct Out {
		double ret;
		double mdd;
		std::uint64_t positions;
	};
}


extern "C"
{
	extern const size_t rnn_trader_var = Network::ParamCount;

	void rnn_trader_run(
		const candle* candles,
		std::size_t candle_count,
		const param_t params[rnn_trader_var],
		Out* out
	)
	{
		spot engine{ min_order_value, commission, initial_balance, };
		Network nn;
		NNTrader trader(engine, nn, params);
		engine.trade(trader, { candles, candle_count });

		// Roll-back the position, the trader couldn't know the time is finite
		if (engine.pos) engine.balance += engine.pos->orders.size() * (min_order_value + commission);

		out->ret = engine.balance - initial_balance;
		out->mdd = engine.mdd;
		out->positions = engine.positions.size();
	}
}
