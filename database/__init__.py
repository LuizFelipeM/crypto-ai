from sqlalchemy import Column, Integer, create_engine
from sqlalchemy.ext.declarative import declarative_base


def p():
    print("Database")


connection_string = "mysql+<drivername>://<username>:<password>@<server>:<port>/dbname"
engine = create_engine(connection_string, echo=True)

Base = declarative_base()


class Kline(Base):
    __tablename__ = "Kline"
    id = Column(Integer, primary_key=True)
    open_time = DateTime
    close_tim = DateTime
    symbol = string
    interval = string
    open_price = double
    close_price = double
    high_price = double
    low_price = double
    base_asset_volume = double
    number_of_trades = long
    is_kline_closed = bool
    quote_asset_volume = double
    taker_buy_base_asset_volume = double
    taker_buy_quote_asset_volume = double
