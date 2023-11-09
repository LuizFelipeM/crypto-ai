from typing import TypeVar
from database.Base import Base
from database.MySqlConfig import MySqlConfig
from sqlalchemy import Engine, ScalarResult, create_engine, select
from sqlalchemy.orm import Session


TEntity = TypeVar("TEntity", Base)


class DbContext:
    _engine: Engine
    _session: Session

    def __init__(self, config: MySqlConfig) -> None:
        self._engine = create_engine(config.connection_string(), echo=True)
        self._session = Session(self._engine)

    def get_all(self, entity: TEntity) -> ScalarResult[TEntity]:
        result = select(entity)
        return self._session.scalars(result)
