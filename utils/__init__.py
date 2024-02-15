import numpy as np
import pandas as pd
import torch
from typing_extensions import Unpack
from sqlalchemy import ScalarResult
from utils._pnlConfig import PnLConfig
from utils._contract import Contract
from utils._position import Position


__all__ = ["Position", "Contract"]


def create_dataframe(values: ScalarResult, selected_fields: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([{f: getattr(r, f) for f in selected_fields} for r in values])
    df.shape
    return df


def split_X_y(
    dataset_shape: tuple[int, int], training_set: pd.DataFrame, padding: int
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    padded_range = range(padding, dataset_shape[0])
    X = [training_set.iloc[i - padding : i] for i in padded_range]
    y = [training_set.iloc[[i]] for i in padded_range]

    # X: list[pd.DataFrame] = []
    # y: list[pd.DataFrame] = []
    # Convert to List Comprehension in order to use C++ performance improvements
    # for i in range(padding, dataset_shape[0]):
    #     X.append(training_set.iloc[i - padding : i])
    #     y.append(training_set.iloc[[i]])

    return (X, pd.concat(y))


def diminishing_return(value: float, factor: float) -> float:
    return factor / (factor + value)


def pnl(
    position: Position,
    contract: Contract,
    open_price: float,
    close_price: float,
    **kwargs: Unpack[PnLConfig],
) -> float:
    """Position and Contract types affect the PnL calculation as following:

    Linear:
        Long: PnL = position_qty * (close_price - open_price)
        Short: PnL = position_qty * (open_price - close_price)

    Inverse:
        Long: PnL = contract_qty * contract_value * (1/open_price - 1/close_price)
        Short: PnL = contract_qty * contract_value * (1/close_price - 1/open_price)
    """
    (price1, price2) = (
        (open_price, close_price)
        if position == Position.SHORT
        else (close_price, open_price)
    )

    if contract == Contract.LINEAR:
        if kwargs.get("position_qty") == None:
            raise ValueError(
                "position_qty is required for Linear contracts PnL calculation"
            )
        return kwargs.get("position_qty", 0.0) * (price1 - price2)

    if kwargs.get("position_qty") == None or kwargs.get("contract_size") == None:
        raise ValueError(
            "position_qty and contract_size are required for Inverse contracts PnL calculation"
        )
    return (
        kwargs.get("contract_qty", 0.0)
        * kwargs.get("contract_size", 0.0)
        * ((1 / price2) - (1 / price1))
    )


def get_moving_average(period, values) -> np.ndarray:
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = (
            values.unfold(dimension=0, size=period, step=1)
            .mean(dim=1)
            .flatten(start_dim=0)
        )
        moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))
        return moving_avg.numpy()

    moving_avg = torch.zeros(len(values))
    return moving_avg.numpy()
