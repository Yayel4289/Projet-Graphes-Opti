from .population import Population
from .utils import *
import os

# hyperparams
TIME_SEC = 270 # time in sec for each file
N_POPULATION = 10
ALPHA = 0.75 # reproduction chance
BETA = 0.1 # mutation chance

RESULT_PATH = os.path.join("..", "tests", "memeticResults")

def memetic(path: str, generate=False):
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

    if generate:
        # On récupère seulement le nom du fichier, pas tout le chemin
        filename = os.path.basename(path)

        # On construit un chemin OS-indépendant
        output_path = os.path.join(RESULT_PATH, filename)

        # Génère le fichier de sortie proprement
        generate_output(best.cost, best.W, best.H, output_path)

    return best.cost, best.W, best.H


if __name__ == "__main__":
    # paths
    TEST_PATH = "../tests/testMatrices/"
    TRAINING_PATH = "../tests/trainingMatrices/"

    test = os.listdir(TEST_PATH)
    training = os.listdir(TRAINING_PATH)

    for i, file in enumerate(test + training):
        if "houdain" not in file:
            continue
        print(file)
        path = TEST_PATH if i < len(test) else TRAINING_PATH
        
        cost, W, H = memetic(path + file, True)
        print(cost)
