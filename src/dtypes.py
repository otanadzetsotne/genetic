from typing import TypedDict, Literal, Iterable

activations = [
    'relu',
    # 'sigmoid',
    # 'softmax',
    # 'softplus',
    # 'softsign',
    # 'tanh',
    # 'selu',
    # 'elu',
    # 'exponential',
    # 'linear',
]

DNA = list[int, str]
Individual = list[DNA]
Population = list[Individual]
Parents = list[Individual, Individual]
Scores = list[float]
