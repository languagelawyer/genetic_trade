#include "nn/linear.hpp"

#include <cassert>

#include <vector>


int main()
{
	using namespace NN;

	{
		Linear<double, 2, 2, false> linear;
		// identity matrix
		double params[decltype(linear)::ParamCount] = { 1, 0, 0, 1, };
		double in[2] = { 1, 2 };
		double out[2];

		linear(params, in, out);

		assert(out[0] == 1);
		assert(out[1] == 2);
	}

	{
		Linear<double, 2, 2, true> linear;
		// identity matrix with bias
		double params[decltype(linear)::ParamCount] = { 1, 0, 0, 1, 1, 2, };
		double in[2] = { 1, 2 };
		double out[2];

		linear(params, in, out);

		assert(out[0] == 2);
		assert(out[1] == 4);
	}

	{
		Linear<double, 2, 2, false> linear;
		double params[decltype(linear)::ParamCount] = { 2, 1, 0, 1, };
		double in[2] = { 1, 2 };
		double out[2];

		linear(params, in, out);

		assert(out[0] == 4);
		assert(out[1] == 2);
	}
}
