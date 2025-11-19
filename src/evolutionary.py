import numpy as np 
from utils import *
from random import randint, random
from functools import total_ordering
import os

MAX_GENERATION = 1000
N_POP = 1000 
MUTATION_CHANCE = 0.2

X: np.matrix
r: int
LW: int 
UW: int 
LH: int
UH: int

@total_ordering
class Individual:
    def __init__(self, W: np.matrix, H: np.matrix) -> None:
        self.W = W 
        self.H = H 
        self.possible_mutate()
        self.cost = self.compute_cost()

    def __lt__(self, other: "Individual") -> bool: # sort ability
        return self.cost < other.cost

    def compute_cost(self) -> int:
        return fobj(X, self.W, self.H)

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


class Population:
    def __init__(self) -> None:
        self.population: list["Individual"] = [
            Individual(
                np.matrix(np.random.randint(LW, UW, size=(X.shape[0], r))), # W
                np.matrix(np.random.randint(LH, UH, size=(r, X.shape[1])))  # H
            ) for _ in range(N_POP)
        ]
        self.sort()
        self.min = self[0]

    def __getitem__(self, i) -> "Individual":
        return self.population.__getitem__(i)
        
    def __len__(self) -> int:
        return self.population.__len__()

    def add(self, ind: "Individual") -> None:
        self.population.append(ind)

    def sort(self) -> None:
        # low cost first
        self.population.sort() 

    def make_reproduction(self) -> None:
        # can change
        for i in range(0, len(self), 2):
            child = self[i].reproduce(self[i + 1])
            self.add(child) 
        self.sort()
        print(self.min.cost, self[0].cost)
        self.min = min(self.min, self[0])

    def select(self) -> None: 
        # naive selection
        self.population = self.population[:N_POP]

    def generate(self):
        self.make_reproduction()
        self.select()

    def evolve(self):
        for _ in range(MAX_GENERATION):
            self.generate()

if __name__ == "__main__":
    TEST_PATH = "../tests/testMatrices/"
    TRAINING_PATH = "../tests/trainingMatrices/"
    RESULT_PATH = "../tests/evolutionaryResults/"

    test = os.listdir(TEST_PATH)
    training = os.listdir(TRAINING_PATH)

    for i, file in enumerate(test + training):
        if "houdain" in file:
            continue
        print(file)
        path = TEST_PATH if i < len(test) else TRAINING_PATH
        X, r, LW, UW, LH, UH = parse_input(path + file)
        W = np.matrix(np.random.randint(LW, UW, size=(X.shape[0], r)))
        H = np.matrix(np.random.randint(LH, UH, size=(r, X.shape[1])))
        pop = Population()
        pop.evolve()
        best = pop.min
        generate_output(best.cost, best.W, best.H, RESULT_PATH + file)

