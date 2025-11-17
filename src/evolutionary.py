import numpy as np 
import utils
from random import randint, random

MAX_GENERATION = 1000
N_POP = 1000 
MUTATION_CHANCE = 0.2

X: np.matrix
r: int
LW: int 
UW: int 
LH: int
UH: int

class Individual:
    def __init__(self, W: np.matrix, H: np.matrix) -> None:
        self.W = W 
        self.H = H 
        self.possible_mutate()
        self.fitness = self.compute_fitness()

    def compute_fitness(self) -> int:
        return utils.fobj(X, self.W, self.H)

    def mutate(self) -> None: 
        # Neighborhood -> complete enum (can change)
        in_W = randint(0, 1)
        M = self.W if in_W else self.H
        LB, HB = (LW, UW) if in_W else (LH, UH)
        i = randint(0, M.shape[0] - 1)
        j = randint(0, M.shape[1] - 1)
        M[i, j] = randint(LB, HB)

    def possible_mutate(self):
        if random() < MUTATION_CHANCE : self.mutate()
    
    def reproduce(self, other: "Individual") -> "Individual":
        # "Matrix k swap crossover" (can change)
        MAX_K = 2*r
        MIN_K = r // 2
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

