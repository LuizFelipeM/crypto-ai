from typing_extensions import TypedDict


class PnLConfig(TypedDict, total=False):
    position_qty: float
    """Required when is a Linear Contract calculation"""
    contract_qty: int
    """Required when is a Inverse Contract calculation"""
    contract_value: float
    """Required when is a Inverse Contract calculation"""
