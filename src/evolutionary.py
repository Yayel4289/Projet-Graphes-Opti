import numpy as np 
from evolutionary.population import Population
from utils import *
import os


if __name__ == "__main__":
    # hyperparams
    TIME_SEC = 60 # time in sec for each file
    N_POPULATION = 500
    ALPHA = 0.7 # reproduction chance
    BETA = 0.2 # mutation chance

    # paths
    TEST_PATH = "../tests/testMatrices/"
    TRAINING_PATH = "../tests/trainingMatrices/"
    RESULT_PATH = "../tests/evolutionaryResults/"

    test = os.listdir(TEST_PATH)
    training = os.listdir(TRAINING_PATH)

    for i, file in enumerate(test + training):
        if "houdain" not in file:
            continue
        print(file)
        path = TEST_PATH if i < len(test) else TRAINING_PATH

        K_OBJ = -1
        X, r, LW, UW, LH, UH = parse_input(path + file)
        print(X.shape, r)
        W = np.matrix(np.random.randint(LW, UW, size=(X.shape[0], r)))
        H = np.matrix(np.random.randint(LH, UH, size=(r, X.shape[1])))
        if "houdain" in file:
            K_OBJ = 100
        population = Population(
           X, r, LW, UW, LH, UH,
           N_POPULATION, TIME_SEC, ALPHA, BETA, K_OBJ
        )
        population.evolve()

        best = population.min
        generate_output(best.compute_real_cost(), best.W, best.H, RESULT_PATH + file)

