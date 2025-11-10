import numpy as np

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
    return int(f)


def solutionIsFeasible(W: np.matrix, H: np.matrix, r: int, LW: int, UW: int, LH: int, UH: int) -> bool:
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

def parseInput(path: str):
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
        X = np.array(map(int, lines[1:]), dtype=int)
    return X, r, LW, UW, LH, UH

def generateOutput(obj: int, W: np.matrix, H: np.matrix, path: str):
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

    with open(path, "w") as file:
        pass # TODO
    

def localSearch(X: np.matrix, W: np.matrix, H: np.matrix, r: int, LW: int, UW: int, LH: int, UH: int) -> int:
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
    int 
        The result of a local search using 
        the given parameters
    """

    min_fobj = fobj(X, W, H)
    W_min, H_min = W.copy(), H.copy()
    last_min = min_fobj
    for i in range(W.shape[0]):
        for j in range(r):
            for w in range(LW, UW + 1):
                W_copy = W.copy()
                W_copy[i][j] = w
                neighbor_fobj = fobj(X, W_copy, H)
                if neighbor_fobj < min_fobj:
                    min_fobj = neighbor_fobj
                    W_min = W_copy

    for i in range(r):
        for j in range(H.shape[1]):
            for h in range(LH, UH + 1):
                H_copy = H.copy()
                H_copy[i][j] = h
                neighbor_fobj = fobj(X, W, H_copy)
                if neighbor_fobj < min_fobj:
                    min_fobj = neighbor_fobj
                    H_min = H_copy
    return \
        localSearch(X, W_min, H_min, r, LW, UW, LH, UH) if last_min != min_fobj else \
        min_fobj


"""
def metaheuristic(X,r,LW,UW,LH,UH):
  bestW = np.zeros((X.shape[0],r))
  bestH = np.zeros((r,X.shape[1]))

  ... #votre méthode

  return bestW, bestH
"""
