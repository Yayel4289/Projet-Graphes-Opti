from typing import Iterator
import numpy as np
import time
from random import sample, random, choice
from .individual import Individual

class Population:
    def __init__(
        self,
        X: np.matrix,
        r: int,
        LW: int,
        UW: int,
        LH: int,
        UH: int,
        N_POPULATION: int,
        TIME_SEC: int,
        ALPHA: float,
        BETA: float
    ) -> None:

        self.N_POPULATION = N_POPULATION
        self.TIME_SEC = TIME_SEC
        self.ALPHA = ALPHA
        self.K_TOURNAMENT_SELECT = 3
        Individual.init_class(X, r, LW, UW, LH, UH, BETA)
        self.population: list["Individual"] = [
            Individual(
                np.matrix(np.random.randint(LW, UW, size=(X.shape[0], r))), # W
                np.matrix(np.random.randint(LH, UH, size=(r, X.shape[1])))  # H
            ) for _ in range(self.N_POPULATION)
        ]
        self.min = choice(self.population) # random min start

    def __getitem__(self, key):
        return self.population.__getitem__(key)

    def __iter__(self) -> Iterator["Individual"]:
        return self.population.__iter__()
        
    def __len__(self) -> int:
        return self.population.__len__()

    def add(self, ind: "Individual") -> None:
        self.population.append(ind)

    def pop(self, i: int) -> None:
        self.population.pop(i)

    def sort(self) -> None: # n log n -> might be too slow
        # low cost first
        self.population.sort() 

    def make_reproduction(self) -> None:
        n = len(self)
        for individual in self[:n]:
            if random() < self.ALPHA:
                self.add(individual.reproduce(choice(self.population)))

    def make_mutation(self) -> None:
        for individual in self:
            if individual is not self.min: # avoid mutating our min
                individual.possible_mutate()

    def select(self) -> None: 
        # naive selection
        # needs sorted list
        self.population = self.population[:self.N_POPULATION]

    def tournament_select(self) -> None:
        # tournament
        while len(self) > self.N_POPULATION:
            i_group = sample(range(len(self)), self.K_TOURNAMENT_SELECT)
            i_group.pop(min(range(len(i_group)), key=lambda i: self[i_group[i]]))

            for i in sorted(i_group, reverse=True): 
                self.pop(i)

    def update_min(self) -> None:
        self.min = min(self.population)
        print(self.min.cost)
            
    def generate(self):
        self.make_reproduction()
        self.make_mutation()
        self.tournament_select()
        self.update_min()

    def evolve(self):
        start = time.time()
        while time.time() - start < self.TIME_SEC:
            self.generate()


