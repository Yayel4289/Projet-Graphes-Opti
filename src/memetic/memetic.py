from .population import Population
from .utils import *
import os

# hyperparams
TIME_SEC = 270 # time in sec for each file
N_POPULATION = 10
ALPHA = 0.75 # reproduction chance
BETA = 0.1 # mutation chance

RESULT_PATH = "../tests/memeticResults/"

def memetic(path: str):
    X, r, LW, UW, LH, UH = parse_input(path)
    
    W_START, H_START = least_square_optimization(X, r, LW, UW, LH, UH)
    print(fobj(X, W_START, H_START))

    population = Population(
        X, r, LW, UW, LH, UH,
        W_START, H_START,
        N_POPULATION, TIME_SEC, ALPHA, BETA
    )

    population.evolve()

    best = population.min
    return best.cost, best.W, best.H


if __name__ == "__main__":
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
        
        cost, W, H = memetic(path + file)

        generate_output(cost, W, H, RESULT_PATH + file)
