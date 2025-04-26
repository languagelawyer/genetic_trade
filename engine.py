from __future__ import annotations

from enum import Enum
from typing import Optional

from pandas import DataFrame

from order import Order
from position import Position
from usdt import usdt


class Signal(Enum):
	BUY = 1
	HOLD = 0
	SELL = -1

class Trader:
	def __call__(self, *, data_so_far: DataFrame, engine: Spot) -> Signal:
		return Signal.HOLD

class Spot:
	def trade(self, *, trader: Trader, data: DataFrame, min_order_size: usdt, initial_quote: usdt = 0, commission: usdt = 0):
		self.min_order_size = min_order_size
		self.commission = commission

		self.quote: usdt = initial_quote
		self.max_quote: usdt = initial_quote
		self.mdd: usdt = 0

		self.quotes: list[usdt] = [initial_quote, initial_quote]
		self.drawdowns: list[usdt] = [0, 0]

		self.orders: list[Order] = []
		self.positions: list[Position] = []

		self.position: Optional[Position] = None
		# Start from 2 so that the trader could have a ratio of 2 bars
		# End at len(data)-1 so that we could execute the order at the "next" bar
		for i in range(2, len(data) - 1):
			next = data.iloc[i+1]

			# Since we assume trading at 1s intervals,
			# update the quote and MDD at the "next" bar,
			# before executing the order
			quote = self.quote
			if self.position:
				quote += self.position.size * next['Low']

			self.max_quote = max(self.max_quote, quote)
			drawdown = (self.max_quote - quote) / self.max_quote
			self.mdd = max(self.mdd, drawdown)

			self.quotes.append(quote)
			self.drawdowns.append(drawdown)

			# Since we assume trading at 1s intervals, all orders are marked orders
			# and are executed conservatively/pessimistically, at worst prices:
			# High for BUY, Low for SELL
			signal = trader(data_so_far = data[:i], engine = self)
			if signal == Signal.HOLD:
				continue

			# Buy for the min_order_size, sell all
			if signal == Signal.BUY:
				if not self.position:
					self.position = Position(id = len(self.positions))
					self.positions.append(self.position)

				self.quote -= self.commission
				self.quote -= self.min_order_size
				order = Order(
					id = len(self.orders),
					create_time = next.index,
					price = next['High'],
					size = self.min_order_size / next['High'],
				)
				self.orders.append(order)
				self.position.add_order(order)

			if signal == Signal.SELL and self.position:
				# Can't sell for less than min_order_size
				if self.position.size * next['Low'] < self.min_order_size:
					continue

				self.quote -= self.commission * len(self.position.orders)

				self.position.close_time = next.index
				self.position.close_value = self.position.size * next['Low']
				self.quote += self.position.close_value
				self.position = None
