from dataclasses import dataclass
from typing import Any

from usdt import usdt


# Currently, we assume Orders are fully fulfilled, so that there is no Trade class
@dataclass
class Order:
	id: int
	create_time: Any

	size: float
	price: usdt

