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
        BETA: float,
        K_OBJ: int
    ) -> None:

        self.X = X
        self.r = r 
        self.LW = LW 
        self.UW = UW 
        self.LH = LH 
        self.UH = UH
        self.N_POPULATION = N_POPULATION
        self.TIME_SEC = TIME_SEC
        self.ALPHA = ALPHA
        self.K_TOURNAMENT_SELECT = 3
        Individual.init_class(X, r, LW, UW, LH, UH, BETA, K_OBJ)
        self.population: list["Individual"] = [
            self.randIndividual() 
            for _ in range(self.N_POPULATION)
        ]
        self.still_count = 0
        self.STILL_COUNT_MAX = 100 # to trigger diversification
        self.WAIT_DIVERSIFICATION = 10
        self.min = choice(self.population).copy() # random min start

    def __getitem__(self, key):
        return self.population.__getitem__(key)

    def __iter__(self) -> Iterator["Individual"]:
        return self.population.__iter__()
        
    def __len__(self) -> int:
        return self.population.__len__()

    def randIndividual(self) -> "Individual":
        return Individual(
            np.matrix(np.random.randint(self.LW, self.UW, size=(self.X.shape[0], self.r))), # W
            np.matrix(np.random.randint(self.LH, self.UH, size=(self.r, self.X.shape[1])))  # H
        )

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
        for individual in self: individual.possible_mutate()

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

    def diversify(self) -> None:
        self.sort()
        self.population = self.population[self.N_POPULATION // 10:] # 10% elitism
        while len(self) < self.N_POPULATION:
            self.add(self.randIndividual())

        self.ALPHA -= 0.06

    def intensify(self) -> None:
        self.ALPHA += 0.1

    def update_min(self) -> None:
        old = self.min
        self.min = min(old, min(self.population).copy())

        if self.min < old:
            self.still_count = 0
            print(self.min.cost)
        else:
            self.still_count += 1

        if self.still_count >= self.STILL_COUNT_MAX:
            print("Diversification")
            self.still_count = -self.WAIT_DIVERSIFICATION * self.STILL_COUNT_MAX
            self.diversify()
            self.intensify()
            
    def generate(self):
        self.make_reproduction()
        self.make_mutation()
        self.tournament_select()
        self.update_min()

    def evolve(self):
        start = time.time()
        while time.time() - start < self.TIME_SEC:
            self.generate()


