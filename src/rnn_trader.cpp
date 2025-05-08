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
	constexpr double initial_balance = 200;
	constexpr double min_order_value = 5;
	constexpr double commission = 0.002;
	constexpr size_t warmup_period = 600;
	constexpr double max_position_value = 100;

	struct Network
	{
		static constexpr size_t In = 2;
		static constexpr size_t Hidden = 10;
		static constexpr size_t Layers = 2;
		static constexpr size_t Out = 1;
		static constexpr bool Bias = true;

		using RNN = NN::RNN<NN::LSTMCell, double, In, Hidden, Layers, Bias>;
		using Linear = NN::Linear<double, Hidden, Out, Bias>;

		RNN rnn;

		static constexpr size_t ParamCount = RNN::ParamCount + Linear::ParamCount;

		void operator()(const double* params, const double* in, double* out)
		{
			rnn(params, in), params += decltype(rnn)::ParamCount;
			Linear{}(params, rnn.ls.back().h.data(), out);

			std::for_each(out, out + Out, [](double& x) { x = std::tanh(x); });
		}
	};

	struct NNTrader : trader
	{
		spot& engine;
		Network& nn;
		const double *params;

		NNTrader(spot& engine, Network& nn, const double* params)
		: engine(engine)
		, nn(nn)
		, params(params)
		{}

		Signal operator()(std::span<const candle> past_data) override
		{
			if (past_data.size() < 2) return Signal::HOLD;

			const auto& last = past_data[past_data.size() - 1];
			const auto& prev = past_data[past_data.size() - 2];

			double in[Network::In] = {
				std::log(last.open / prev.open),
				std::log(last.volume + 1),
			};
			double out[Network::Out]; // tanh output, (-1, 1) range
			nn(params, in, out);

			if (past_data.size() < warmup_period) return Signal::HOLD;

			if (out[0] < -0.5) return Signal::SELL;

			if (engine.pos and engine.pos->total_value >= max_position_value) return Signal::HOLD;

			if (out[0] > +0.5) return Signal::BUY;

			return Signal::HOLD;
		}
	};
}


extern "C"
{
	extern const size_t rnn_trader_var = Network::ParamCount;
	extern const size_t rnn_trader_obj = 3; // (return, MDD, number of positions)

	const candle* rnn_trader_candles;
	std::size_t rnn_trader_candlle_count;

	void rnn_trader_run(const double params[rnn_trader_var], double out[rnn_trader_obj])
	{
		// lets get worst over several periods
		// this should help against overfitting
		constexpr size_t periods = 4;
		const size_t candles_per_period = rnn_trader_candlle_count / periods;

		double worst_balance = 1e6;
		double worst_mdd = 0;
		std::size_t total_positions = 0;
		for (int i = 0; i < periods; i++)
		{
			auto candles = rnn_trader_candles + i * candles_per_period;

			spot engine{ min_order_value, commission, initial_balance, };
			Network nn;
			NNTrader trader(engine, nn, params);
			engine.trade(trader, { candles, candles_per_period });

			// Do not punish for no trades:
			// if the model understands the market is not great
			// why should it be punished?
			if (engine.positions.empty()) continue;

			worst_balance = std::min(worst_balance, engine.balance);
			worst_mdd = std::max(worst_mdd, engine.mdd);
			total_positions += engine.positions.size();
		}

		if (total_positions == 0) worst_balance = initial_balance;

		// we have a minimization task
		// so negate objectives that should be maximized
		out[0] = -(worst_balance - initial_balance);
		out[1] = worst_mdd;
		out[2] = -std::log(total_positions + 1);
	}
}
