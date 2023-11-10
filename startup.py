import os
import pandas as pd
from dotenv import load_dotenv
from aimodels import LSTMModel
from database import DbContext, MySqlConfig
from database.repositories import KlineRepository


if __name__ == "__main__":
    load_dotenv()

    mysql_config = MySqlConfig(
        os.environ.get("MySql:User"),
        os.environ.get("MySql:Password"),
        "mysql+mysqlconnector://<User>:<Password>@aws.connect.psdb.cloud:3306/crypto-bot",
    )

    dbc = DbContext(mysql_config)
    kr = KlineRepository(dbc)
    batch = kr.batch(10)

    selected_fields = [
        "id",
        "open_time",
        "close_time",
        "symbol",
        "interval",
        "open_price",
        "close_price",
        "high_price",
        "low_price",
        "base_asset_volume",
        "number_of_trades",
        "is_kline_closed",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    df = pd.DataFrame(
        [{f: getattr(r, f) for f in selected_fields} for r in batch.next()]
    )

    lstm = LSTMModel()
    lstm.fit()
    print(df.head())
