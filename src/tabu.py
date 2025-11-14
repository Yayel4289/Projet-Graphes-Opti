import numpy as np
from utils import *
import os


def taboo_search(
    X: np.matrix,
    W: np.matrix,
    H: np.matrix,
    r: int,
    LW: int,
    UW: int,
    LH: int,
    UH: int,
    max_no_improv: int,
    tabu_tenure: int
) -> tuple[int, np.matrix, np.matrix]:
    """
    Tabu Search pour la factorisation entière X ≈ W H
    (voisin = flip sur W, calcul naïf de fobj)
    """
    # Solution courante
    current_W = W.copy()
    current_H = H.copy()
    current_obj = fobj(X, current_W, current_H)

    # Meilleur global
    best_W = current_W.copy()
    best_H = current_H.copy()
    best_obj = current_obj

    # Liste tabou : on force un ndarray pour éviter les soucis de np.matrix
    tabu_W = np.zeros(current_W.shape, dtype=int)

    no_improv = 0
    n_rows, _ = current_W.shape

    while no_improv < max_no_improv:
        best_neighbor_obj = None
        best_move = None  # (i, k, delta)

        # Parcours de tous les voisins (flip ±1 sur W[i,k])
        for i in range(n_rows):
            for k in range(r):
                for delta in (-1, 1):
                    new_val = current_W[i, k] + delta

                    # Respect des bornes
                    if new_val < LW or new_val > UW:
                        continue

                    # Copie temporaire de W
                    neighbor_W = current_W.copy()
                    neighbor_W[i, k] = new_val

                    # Calcul naïf de la valeur de l'objectif
                    neighbor_obj = fobj(X, neighbor_W, current_H)

                    # Mouvement tabou ?
                    is_tabu = tabu_W[i, k] > 0

                    # Aspiration : on autorise un mouvement tabou
                    # s'il améliore le meilleur global
                    if is_tabu and neighbor_obj >= best_obj:
                        continue  # mouvement interdit

                    # Sélection du meilleur voisin admissible
                    if (best_neighbor_obj is None) or (neighbor_obj < best_neighbor_obj):
                        best_neighbor_obj = neighbor_obj
                        best_move = (i, k, delta)

        # Si aucun mouvement admissible trouvé, on arrête
        if best_move is None:
            break

        # Mise à jour de la liste tabou : décrémenter tous les > 0
        # (version qui marche bien quel que soit le type)
        tabu_W = np.where(tabu_W > 0, tabu_W - 1, 0)

        # Application du meilleur mouvement
        i_move, k_move, delta_move = best_move
        current_W[i_move, k_move] += delta_move
        current_obj = best_neighbor_obj

        # Position (i_move, k_move) devient tabou pour 'tabu_tenure' itérations
        tabu_W[i_move, k_move] = tabu_tenure

        # Mise à jour du meilleur global
        if current_obj < best_obj:
            best_obj = current_obj
            best_W = current_W.copy()
            best_H = current_H.copy()
            no_improv = 0
        else:
            no_improv += 1

    return best_obj, best_W, best_H


if __name__ == "__main__":
    TEST_PATH = "../tests/testMatrices/"
    TRAINING_PATH = "../tests/trainingMatrices/"
    RESULT_PATH = "../tests/tabuResults/"

    test = os.listdir(TEST_PATH)
    training = os.listdir(TRAINING_PATH)

    for i, file in enumerate(test + training):
        if "houdain" in file:
            continue
        print(file)
        path = TEST_PATH if i < len(test) else TRAINING_PATH
        X, r, LW, UW, LH, UH = parse_input(path + file)

        # solution initiale (comme dans local_search, avec bornes inclusives)
        W0 = np.matrix(np.random.randint(LW, UW + 1, size=(X.shape[0], r)))
        H0 = np.matrix(np.random.randint(LH, UH + 1, size=(r, X.shape[1])))

        obj, W_best, H_best = taboo_search(
            X, W0, H0, r, LW, UW, LH, UH,
            max_no_improv=1000,
            tabu_tenure=5
        )

        # 🔍 Check feasibility
        is_feasible = is_solution_feasible(W_best, H_best, r, LW, UW, LH, UH)
        print(f"Solution feasible ? {is_feasible}")

        if not is_feasible:
            print("⚠️  ERREUR : solution tabou NON FEASIBLE !!")

        generate_output(obj, W_best, H_best, RESULT_PATH + file)

