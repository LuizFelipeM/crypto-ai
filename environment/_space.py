from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class Space(Generic[K, V]):
    actions: dict[K, V]

    def __init__(self, actions: dict[K, V]) -> None:
        self.actions = actions

    @property
    def n(self) -> int:
        return len(self.actions)

    def __getitem__(self, key) -> V:
        return self.actions[key]
