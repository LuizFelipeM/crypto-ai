from database._mySqlConfig import MySqlConfig
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session


MySqlConfig


class DbContext:
    _engine: Engine
    session: Session

    def __init__(self, config: MySqlConfig) -> None:
        self._engine = create_engine(config.connection_string(), echo=True)
        self.session = Session(self._engine)
