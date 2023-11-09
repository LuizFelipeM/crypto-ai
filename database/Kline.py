from database import Base
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column


class Kline(Base):
    __tablename__ = "Kline"

    id: Mapped[int] = mapped_column(name="Id", primary_key=True)
    open_time: Mapped[datetime] = mapped_column(name="OpenTime")
    close_time: Mapped[datetime] = mapped_column(name="CloseTime")
    symbol: Mapped[str] = mapped_column(name="Symbol")
    interval: Mapped[str] = mapped_column(name="Interval")
    open_price: Mapped[float] = mapped_column(name="OpenPrice")
    close_price: Mapped[float] = mapped_column(name="ClosePrice")
    high_price: Mapped[float] = mapped_column(name="HighPrice")
    low_price: Mapped[float] = mapped_column(name="LowPrice")
    base_asset_volume: Mapped[float] = mapped_column(name="BaseAssetVolume")
    number_of_trades: Mapped[int] = mapped_column(name="NumberOfTrades")
    is_kline_closed: Mapped[bool] = mapped_column(name="IsKlineClosed")
    quote_asset_volume: Mapped[float] = mapped_column(name="QuoteAssetVolume")
    taker_buy_base_asset_volume: Mapped[float] = mapped_column(
        name="TakerBuyBaseAssetVolume"
    )
    taker_buy_quote_asset_volume: Mapped[float] = mapped_column(
        name="TakerBuyQuoteAssetVolume"
    )
