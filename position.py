from dataclasses import dataclass, field
from typing import Any

from order import Order
from usdt import usdt


@dataclass
class Position:
	id: int
	size: float = 0
	tot_price: usdt = 0

	# avg_price: usdt = 0

	close_time: Any = None
	close_value: usdt = 0

	orders: list[Order] = field(default_factory=list)

	def add_order(self, order: Order):
		self.tot_price += order.price
		# self.avg_price = (self.avg_price * self.amount + order.price * order.amount) / (self.amount + order.amount)
		self.size += order.size
		self.orders.append(order)
