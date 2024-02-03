from enum import Enum


class Position(Enum):
    LONG = 0
    """Long Buy:

    Act of buying and selling later usually to profit on the rise of the stock prices.

    The profits came buying at low prices and selling at high prices.
    """
    SHORT = 1
    """Short Sell:

    Act of borrowing/selling and buying later usually to profit on the falling of the stock prices.

    The profits came borrowing and selling at high prices and paying at low prices at the end of the position.
    """
