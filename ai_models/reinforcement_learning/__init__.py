from ai_models.reinforcement_learning.Environment.Environment import Environment
from ai_models.reinforcement_learning.Environment.Space import Space


def diminishing_return(value: float, factor: float) -> float:
    return factor / (factor + value)


env = Environment(
    Space(
        {
            # Comprar
            0: (
                lambda x: diminishing_return(
                    x.count((0, False)), 0.5  # Improve with better formula
                ),
                False,
            ),
            # Manter
            1: (lambda x: 1, False),
            # Vender
            2: (lambda x: 2, True),
        }
    )
)
