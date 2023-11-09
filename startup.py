import os
import aimodels
from database import DbContext, MySqlConfig
from database.repositories import KlineRepository


if __name__ == "__main__":
    dbc = DbContext(
        MySqlConfig(
            os.environ.get("MySql:User"),
            os.environ.get("MySql:Password"),
            "mysql+mysqlconnector://<User>:<Password>@aws.connect.psdb.cloud:3306/crypto-bot",
        )
    )

    kr = KlineRepository(dbc)
    batch = kr.batch(10)

    print(batch.next().all())
    print("\n----------------------------\n")
    print(batch.next().all())
