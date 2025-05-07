Trying to evolve a neural network population maximizing return and minimizing MDD using genetic algorithms.
Using direct encoding of NN parameters (weights and biases).

Simple architecture with a recursive (vanilla RNN, GRU or LSTM) module
followed by a linear layer with the final `tanh` activation.
Output $> 0.5$ results in a Buy signal, output $< -0.5$ results in a Sell signal.
Buying is for the minimal possible order size, Sell — closes the whole position.
Testing on 1s spot data for simplicity (all orders are Market orders).

After initial experiments with a pure Python implementation,
it was decided that the performance is terrible.
At first, I thought to rewrite everything in Rust using some NSGA-II Rust implementation,
but could not find any having `pymoo`-like low-lever interface.
So I decided to continue using `pymoo`, but rewrite the computationally-intensive part in C++,
providing simple `ctypes`-compatible interface.
