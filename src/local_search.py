import numpy as np
from utils import *
import os


def local_search(
    X: np.matrix, 
    W: np.matrix, 
    H: np.matrix, 
    r: int, 
    LW: int, 
    UW: int, 
    LH: int, 
    UH: int
    ) -> tuple[int, np.matrix, np.matrix]:
    """
    Local Search

    Parameters
    ----------
    X : np.matrix
        The matrix to estimate
    W : np.matrix
        The multiplicand
    H : np.matrix
        The multiplier
    r : int
        X's rank
    LW : int 
        The lowest possible W element
    UW : int 
        The highest possible W element
    LH : int 
        The lowest possible H element
    UH : int 
        The highest possible H element

    Returns
    -------
    obj : int
        The result of a local search using 
        the given parameters
    W_min : 
        The result of a local search for W
    H_min : 
        The result of a local search for H
    """

    min_fobj = fobj(X, W, H)

    min_W, min_H = W.copy(), H.copy()
    last_min = min_fobj

    for i in range(W.shape[0]):
        for j in range(r):
            for w in range(LW, UW + 1):
                base_w = W[i, j]
                W[i, j] = w
                neighbor_fobj = fobj(X, W, H)
                if neighbor_fobj < min_fobj:
                    min_fobj = neighbor_fobj
                    min_W = W.copy()
                W[i, j] = base_w 

    for i in range(r):
        for j in range(H.shape[1]):
            for h in range(LH, UH + 1):
                base_h = H[i, j]
                H[i, j] = h
                neighbor_fobj = fobj(X, W, H)
                if neighbor_fobj < min_fobj:
                    min_fobj = neighbor_fobj
                    min_H = H.copy()
                H[i, j] = base_h
    return \
        local_search(X, min_W, min_H, r, LW, UW, LH, UH) if last_min != min_fobj else \
        (min_fobj, min_W, min_H)


if __name__ == "__main__":

    TEST_PATH = "../tests/testMatrices/"
    TRAINING_PATH = "../tests/trainingMatrices/"
    RESULT_PATH = "../tests/localSearchResults/"

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
        obj, W, H = local_search(X, W, H, r, LW, UW, LH, UH)
        generate_output(obj, W, H, RESULT_PATH + file)
