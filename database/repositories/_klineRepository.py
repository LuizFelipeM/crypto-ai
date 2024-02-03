from sqlalchemy import select
from database._dbContext import DbContext
from database.models import Kline
from database.repositories._baseRepository import BaseRepository


class KlineRepository(BaseRepository[Kline]):
    def __init__(self, db_context: DbContext) -> None:
        select(Kline).order_by(Kline.open_time)
        super().__init__(db_context, Kline)
