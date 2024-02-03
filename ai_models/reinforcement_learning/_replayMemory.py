from collections import namedtuple
import random


Experience = namedtuple("Experience", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory: list[Experience] = []
        self.push_count = 0

    def push(self, experience: Experience) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size: int) -> list[Experience]:
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size: int) -> bool:
        return len(self.memory) >= batch_size
