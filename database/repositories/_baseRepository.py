from abc import ABCMeta, abstractmethod
from typing import Callable, Generic, Type, TypeVar
from sqlalchemy import ColumnExpressionArgument, ScalarResult, select
from database import DbContext
from database.models._base import Base

TEntity = TypeVar("TEntity", bound=Base)


class Batch(Generic[TEntity]):
    _db_context: DbContext
    _entity: Type[TEntity]
    _current_offset: int = 0
    _whereclause: ColumnExpressionArgument
    batch_size: int

    def __init__(
        self, db_context: DbContext, entity: Type[TEntity], batch_size: int
    ) -> None:
        self._db_context = db_context
        self._entity = entity
        self.batch_size = batch_size

    def next(self) -> ScalarResult[Type[TEntity]]:
        """Get next batch using the batch size previous defined"""
        stmt = (
            select(self._entity)
            .where(self._whereclause)
            .limit(self.batch_size)
            .offset(self._current_offset * self.batch_size)
        )
        self._current_offset += 1
        return self._db_context.session.scalars(stmt)

    def where(self, whereclause: ColumnExpressionArgument) -> None:
        """Set where clause to be used in batch selects"""
        self._whereclause = whereclause


class BaseRepository(Generic[TEntity], metaclass=ABCMeta):
    _db_context: DbContext
    _entity: Type[TEntity]

    def __init__(self, db_context: DbContext, entity: Type[TEntity]) -> None:
        self._db_context = db_context
        self._entity = entity

    def get_all(self) -> ScalarResult[Type[TEntity]]:
        stmt = select(self._entity)
        return self._db_context.session.scalars(stmt)

    def batch(self, batch_size: int) -> Batch[TEntity]:
        """Create a batch object for batch processing iterations"""
        return Batch[TEntity](self._db_context, self._entity, batch_size)
