from enum import Enum


class Contract(Enum):
    LINEAR = 0
    """In linear futures contracts all contracts use a common coin as the margin.

    For example, in BTCUSDT and ETHUSDT holding USDT grant access to both markets.
    """
    INVERSE = 1
    """In inverse futures contracts use its trading coins as the margin.

    For example, in BTCUSDT the BTC is the margin coin and in ETHUSDT the ETH is the margin coin.
    """
