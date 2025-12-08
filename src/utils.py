import numpy as np
from random import randrange
import re 
from math import ceil


def fobj(X: np.matrix, W: np.matrix, H: np.matrix) -> int:
    """
    Parameters
    ----------
    X : np.matrix
        The matrix to estimate
    W : np.matrix
        The multiplicand
    H : np.matrix
        The multiplier

    Returns
    -------
    int 
        The result of the objective function
    """

    f = np.linalg.norm(X-W@H,'fro')**2 # ou alors, sans NumPy : f = sum(sum((X-W@H)**2))
    return ceil(f)

def point_fobj(X: np.matrix, W: np.matrix, H: np.matrix, i: int, j: int) -> int:
    true_point = X[i, j]
    r = W.shape[1]
    approx_point = 0
    for ind in range(r):
        approx_point += W[i, ind] * H[ind, j]
    return (true_point - approx_point)**2

def partial_fobj(X: np.matrix, W: np.matrix, H: np.matrix, k: int) -> int:
    # Compute AT MOST k points
    res = 0
    rows, cols = X.shape 
    # Choose k <<< rows * cols
    computed_set = set()
    for _ in range(k):
        i = randrange(rows)
        j = randrange(cols)
        if (i, j) in computed_set:
            continue
        
        res += point_fobj(X, W, H, i, j)
        computed_set.add((i, j))

    return res


def is_solution_feasible(W: np.matrix, H: np.matrix, r: int, LW: int, UW: int, LH: int, UH: int) -> bool:
    """
    Test whether the given solution is feasible or not

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
    bool 
        True if feasible, False if not
    """
    if W.shape[1] != r or H.shape[0] != r:
        return False
    Wi = np.issubdtype(W.dtype, np.integer) or (np.issubdtype(W.dtype, np.floating) and np.all(np.isfinite(W)) and np.all(W == np.floor(W)))
    Hi = np.issubdtype(H.dtype, np.integer) or (np.issubdtype(H.dtype, np.floating) and np.all(np.isfinite(H)) and np.all(H == np.floor(H)))
    if not (Wi and Hi):
        return False
    return bool(np.all((W >= LW) & (W <= UW)) and np.all((H >= LH) & (H <= UH)))


def parse_input(path: str) -> tuple[np.matrix, int, int, int, int, int]:
    """
    Parse the given input

    Parameters
    ----------
    path : str 
        The input file path

    Returns
    -------
    X : np.matrix
        The matrix to estimate
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
    """
    with open(path, 'r') as file:
        lines = file.readlines()
        _, _, r, LW, UW, LH, UH = map(int, lines[0].split())
        X = np.matrix(
            [ list(map(int, line.split())) for line in lines[1:] ], 
            dtype=int
        )

    return X, r, LW, UW, LH, UH


def get_matrix_str_output(M: np.matrix) -> str:
    """
    Returns a matrix M to the asked output format : 
        [[1 2]      1 2
         [3 4]  ->  3 4
         [5 6]]     5 6

    Parameters
    ----------
    M : np.matrix
        The matrix to output

    Returns 
    -------
    str 
        The matrix M in the asked output format
    """

    return re.sub(r" \[|\[|\]", '', str(M))


def generate_output(obj: int, W: np.matrix, H: np.matrix, path: str):
    """
    Generate the ouptut file

    Parameters
    ----------
    obj: int
        The found solution's value
    W : np.matrix 
        The W matrix solution
    H : np.matrix 
        The H matrix solution
    path: str
        The location to generate the output file
    """

    s = f"{obj}\n{get_matrix_str_output(W)}\n{get_matrix_str_output(H)}"
    with open(path, "w") as file:
        file.write(s)
    

"""
def metaheuristic(X,r,LW,UW,LH,UH):
    bestW = np.zeros((X.shape[0],r))
    bestH = np.zeros((r,X.shape[1]))

    # ...

    return bestW, bestH
"""

