from utils import *
import numpy as np
from random import random, randint
from functools import total_ordering

@total_ordering
class Individual:
    X: np.matrix
    r: int
    LW: int
    UW: int
    LH: int
    UH: int

    MUTATION_CHANCE: float

    @classmethod
    def init_class(
        cls,
        X: np.matrix,
        r: int,
        LW: int,
        UW: int,
        LH: int,
        UH: int,
        MUTATION_CHANCE: float
    ) -> None:
        cls.X = X 
        cls.r = r 
        cls.LW = LW 
        cls.UW = UW 
        cls.LH = LH 
        cls.UH = UH 
        cls.MUTATION_CHANCE = MUTATION_CHANCE


    def __init__(self, W: np.matrix, H: np.matrix) -> None:
        self.cls = self.__class__
        self.W = W 
        self.H = H 
        self.possible_mutate()
        self.cost = self.compute_cost()

    def __lt__(self, other: "Individual") -> bool: # sort ability
        return self.cost < other.cost

    def compute_cost(self) -> int:
        return fobj(self.cls.X, self.W, self.H)

    def mutate(self) -> None: 
        # Neighborhood -> complete enum (can change)
        in_W = randint(0, 1)
        M = self.W if in_W else self.H
        LB, HB = (self.cls.LW, self.cls.UW) if in_W else (self.cls.LH, self.cls.UH)
        i = randint(0, M.shape[0] - 1)
        j = randint(0, M.shape[1] - 1)
        M[i, j] = randint(LB, HB)

    def possible_mutate(self):
        if random() < self.cls.MUTATION_CHANCE : self.mutate()
    
    def reproduce(self, other: "Individual") -> "Individual":
        # "Matrix k swap crossover" (can change)
        MAX_K = 2*self.cls.r
        MIN_K = self.cls.r // 2
        new_W, new_H = self.W.copy(), self.H.copy()
        k = randint(MIN_K, MAX_K)
        for _ in range(k):
            in_W = randint(0, 1)
            M = new_W if in_W else new_H
            other_M = other.W if in_W else other.H
            i = randint(0, M.shape[0] - 1)
            j = randint(0, M.shape[1] - 1)
            M[i, j] = other_M[i, j]
        return Individual(new_W, new_H)

