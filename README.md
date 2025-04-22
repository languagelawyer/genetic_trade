Trying to evolve a neural network population maximizing return and minimizing MDD using genetic algorithms.
Using direct encoding of NN parameters (weights and biases).

Simple architecture with an LSTM module followed by a linear layer with the final `tanh` activation.
Output $> 0.5$ results in a Buy signal, output $< -0.5$ results in a Sell signal.
Buying is for the minimal possible order size, Sell — closes the whole position.
Testing on 1s spot data for simplicity (all orders are Market orders).

The input to the model is very poor: only the ratio of the last two bars.
Richer input (price, volume, even the current position size etc.) should theoretically help the model.

After ~10 generations in pretty pleasant conditions (the price is rising), one of the models managed to earn $0.001753$ USDT.
After a few more generations, another model evolved to earn $0.002786$ USDT (after 2h of trading).
Impressive, innit?
The later result (weights) is saved in the `results.txt` file.

With 100 individuals in the population and with more generations, the results should be more impressive.
But that would be very time-consuming, especially when backtesting over a month — let alone an entire year.