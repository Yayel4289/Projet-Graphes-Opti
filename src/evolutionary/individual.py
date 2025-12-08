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

    BETA: float

    MIN_K: int 
    MAX_K: int

    @classmethod
    def init_class(
        cls,
        X: np.matrix,
        r: int,
        LW: int,
        UW: int,
        LH: int,
        UH: int,
        BETA: float,
        K_OBJ: int
    ) -> None:
        cls.X = X 
        cls.r = r 
        cls.LW = LW 
        cls.UW = UW 
        cls.LH = LH 
        cls.UH = UH 
        cls.BETA = BETA
        cls.K_OBJ = K_OBJ
        cls.MAX_K = cls.r * 2
        cls.MIN_K = cls.r // 2


    def __init__(self, W: np.matrix, H: np.matrix, cost: int=-1) -> None:
        self.cls = self.__class__
        self.W = W 
        self.H = H 

        # cost != -1 => copy
        if cost == -1:
            self.possible_mutate()
            self.cost = self.compute_cost()
        else:
            self.cost = cost

    def __eq__(self, other: object) -> bool:
        return False if not isinstance(other, Individual) else self.cost == other.cost 

    def __lt__(self, other: "Individual") -> bool: # sort ability
        return self.cost < other.cost

    def copy(self):
        # only for min update in population !
        return Individual(self.W, self.H, self.compute_real_cost())

    def compute_cost(self) -> int:
        return \
            partial_fobj(self.cls.X, self.W, self.H, self.cls.K_OBJ) if self.cls.K_OBJ != -1 \
            else fobj(self.cls.X, self.W, self.H)

    def compute_real_cost(self) -> int:
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
        if random() < self.cls.BETA : self.mutate()
    
    def reproduce(self, other: "Individual") -> "Individual":
        # "Matrix k swap crossover" (can change)
        new_W, new_H = self.W.copy(), self.H.copy()
        k = randint(self.cls.MIN_K, self.cls.MAX_K)
        for _ in range(k):
            in_W = randint(0, 1)
            M = new_W if in_W else new_H
            other_M = other.W if in_W else other.H
            i = randint(0, M.shape[0] - 1)
            j = randint(0, M.shape[1] - 1)
            M[i, j] = other_M[i, j]
        return Individual(new_W, new_H)

