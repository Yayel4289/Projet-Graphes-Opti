import numpy as np
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
        N_POP: int,
        MAX_GENERATION: int,
        MUTATION_CHANCE: float
    ) -> None:

        self.N_POP = N_POP
        self.MAX_GENERATION = MAX_GENERATION
        Individual.init_class(X, r, LW, UW, LH, UH, MUTATION_CHANCE)
        self.population: list["Individual"] = [
            Individual(
                np.matrix(np.random.randint(LW, UW, size=(X.shape[0], r))), # W
                np.matrix(np.random.randint(LH, UH, size=(r, X.shape[1])))  # H
            ) for _ in range(self.N_POP)
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
        self.population = self.population[:self.N_POP]

    def generate(self):
        self.make_reproduction()
        self.select()

    def evolve(self):
        for _ in range(self.MAX_GENERATION):
            self.generate()


