import numpy as np 
from evolutionary.population import Population
from utils import *
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
        pop = Population(
           X, r, LW, UW, LH, UH,
           N_POP, MAX_GENERATION, MUTATION_CHANCE
        )
        pop.evolve()
        best = pop.min
        generate_output(best.cost, best.W, best.H, RESULT_PATH + file)

