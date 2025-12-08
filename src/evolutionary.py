import numpy as np 
from evolutionary.population import Population
from utils import *
import os


if __name__ == "__main__":
    # hyperparams
    TIME_SEC = 280 # time in sec for each file
    N_POPULATION = 100
    ALPHA = 0.75 # reproduction chance
    BETA = 0.25 # mutation chance

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
        
        W_START, H_START = least_square_optimization(X, r, LW, UW, LH, UH)
        print(fobj(X, W_START, H_START))

        if "houdain" in file:
            K_OBJ = -1
        population = Population(
            X, r, LW, UW, LH, UH,
            W_START, H_START,
            N_POPULATION, TIME_SEC, ALPHA, BETA, K_OBJ
        )

        population.evolve()

        best = population.min
        generate_output(best.cost, best.W, best.H, RESULT_PATH + file)
