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
	def trade(self, *, trader: Trader, data: DataFrame, min_order_value: usdt, initial_balance: usdt = 0, commission: usdt = 0):
		self.min_order_value = min_order_value
		self.commission = commission

		self.balance: usdt = initial_balance # available quote
		self.max_equity: usdt = initial_balance
		self.mdd: usdt = 0

		self.equities: list[usdt] = [initial_balance, initial_balance]

		self.orders: list[Order] = []
		self.positions: list[Position] = []

		self.position: Optional[Position] = None
		# Start from 2 so that the trader could have a ratio of 2 bars
		# End at len(data)-1 so that we could execute the order at the "current" bar
		for i in range(2, len(data) - 1):
			curr = data.iloc[i+1]

			# Since we assume trading at 1s intervals,
			# update the equity and MDD at the "current" bar,
			# before executing the order
			equity = self.balance # free quote + asset value
			if self.position:
				equity += self.position.size * curr['Low']

			self.max_equity = max(self.max_equity, equity)
			drawdown = (self.max_equity - equity) / self.max_equity
			self.mdd = max(self.mdd, drawdown)

			self.equities.append(equity)

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

				self.balance -= self.commission
				self.balance -= self.min_order_value
				order = Order(
					id = len(self.orders),
					create_time = curr.index,
					price = curr['High'],
					size = self.min_order_value / curr['High'],
				)
				self.orders.append(order)
				self.position.add_order(order)

			if signal == Signal.SELL and self.position:
				close_value = self.position.size * curr['Low']
				# Can't sell for less than min_order_size
				if close_value < self.min_order_value:
					continue

				self.balance -= self.commission * len(self.position.orders)

				self.position.close_time = curr.index
				self.position.close_value = close_value
				self.balance += close_value
				self.position = None
