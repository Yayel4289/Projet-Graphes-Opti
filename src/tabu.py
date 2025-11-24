import numpy as np
import time
from utils import *
import os


def tabu_search(
    X: np.matrix,
    W: np.matrix,
    H: np.matrix,
    r: int,
    LW: int,
    UW: int,
    LH: int,
    UH: int,
    max_no_improv: int,
    tabu_tenure: int,
    max_iters: int = 200,
) -> tuple[int, np.matrix, np.matrix]:
    """
    Tabu search avec:
    - calcul incrémental via le résidu R
    - float pour l'objectif (évite les overflow)
    - sampling du voisinage W/H
    - logs de temps par itération
    """

    # ===== Paramètres de sampling du voisinage =====
    # Fraction de couples (i,k) et (k,j) testés.
    # 1.0 = tous les voisins (complet), 0.3 = environ 30%.
    FRAC_W = 0.3
    FRAC_H = 0.3

    # ----- Initialisation -----
    current_W = W.copy()
    current_H = H.copy()

    P = current_W @ current_H
    R = np.asarray(X - P, dtype=float)   # résidu en float
    current_obj = float(np.sum(np.square(R)))

    best_W = current_W.copy()
    best_H = current_H.copy()
    best_obj = current_obj

    tabu_W = np.zeros(current_W.shape, dtype=int)
    tabu_H = np.zeros(current_H.shape, dtype=int)

    no_improv = 0
    n_rows_W, _ = current_W.shape
    n_rows_H, n_cols_H = current_H.shape

    iter_idx = 0

    # ============================
    #     BOUCLE PRINCIPALE
    # ============================
    while no_improv < max_no_improv and iter_idx < max_iters:

        iter_start = time.perf_counter()

        best_neighbor_obj = None
        best_move = None

        # ==========================================
        # 1) Voisins via W (sampling sur (i,k))
        # ==========================================
        tW_start = time.perf_counter()

        for i in range(n_rows_W):
            row_R = R[i, :].ravel()
            old_cost_row = float(np.dot(row_R, row_R))   # même pour tous les moves (i, *)

            for k in range(r):

                # sampling du couple (i,k)
                if np.random.random() > FRAC_W:
                    continue

                h_k = np.asarray(current_H[k, :]).ravel()

                for delta in (-1, 1):
                    new_val = current_W[i, k] + delta
                    if new_val < LW or new_val > UW:
                        continue

                    delta_w = delta

                    new_row_R = row_R - delta_w * h_k
                    new_cost_row = float(np.dot(new_row_R, new_row_R))

                    neighbor_obj = current_obj + (new_cost_row - old_cost_row)

                    is_tabu = tabu_W[i, k] > 0
                    if is_tabu and neighbor_obj >= best_obj:
                        continue

                    if (best_neighbor_obj is None) or (neighbor_obj < best_neighbor_obj):
                        best_neighbor_obj = neighbor_obj
                        best_move = ("W", i, k, delta)

        tW = time.perf_counter() - tW_start

        # ==========================================
        # 2) Voisins via H (sampling sur (k,j))
        # ==========================================
        tH_start = time.perf_counter()

        for j in range(n_cols_H):
            col_R = R[:, j].ravel()
            old_cost_col = float(np.dot(col_R, col_R))   # idem pour tous les moves (*, j)

            for k in range(n_rows_H):

                # sampling du couple (k,j)
                if np.random.random() > FRAC_H:
                    continue

                w_k = np.asarray(current_W[:, k]).ravel()

                for delta in (-1, 1):
                    new_val = current_H[k, j] + delta
                    if new_val < LH or new_val > UH:
                        continue

                    delta_h = delta

                    new_col_R = col_R - delta_h * w_k
                    new_cost_col = float(np.dot(new_col_R, new_col_R))

                    neighbor_obj = current_obj + (new_cost_col - old_cost_col)

                    is_tabu = tabu_H[k, j] > 0
                    if is_tabu and neighbor_obj >= best_obj:
                        continue

                    if (best_neighbor_obj is None) or (neighbor_obj < best_neighbor_obj):
                        best_neighbor_obj = neighbor_obj
                        best_move = ("H", k, j, delta)

        tH = time.perf_counter() - tH_start

        # Aucun mouvement admissible
        if best_move is None:
            print("[Tabu] Aucun voisin admissible → arrêt.")
            break

        # Décrément tabou
        tabu_W = np.where(tabu_W > 0, tabu_W - 1, 0)
        tabu_H = np.where(tabu_H > 0, tabu_H - 1, 0)

        # Appliquer le meilleur mouvement
        move_type, a, b, delta_move = best_move

        if move_type == "W":
            current_W[a, b] += delta_move
            h_b = np.asarray(current_H[b, :]).ravel()
            R[a, :] = R[a, :].ravel() - delta_move * h_b
            tabu_W[a, b] = tabu_tenure
        else:
            current_H[a, b] += delta_move
            w_a = np.asarray(current_W[:, a]).ravel()
            R[:, b] = R[:, b].ravel() - delta_move * w_a
            tabu_H[a, b] = tabu_tenure

        current_obj = best_neighbor_obj

        # Mise à jour meilleure solution
        if current_obj < best_obj:
            best_obj = current_obj
            best_W = current_W.copy()
            best_H = current_H.copy()
            no_improv = 0
        else:
            no_improv += 1

        iter_time = time.perf_counter() - iter_start

        print(
            f"[Tabu] iter={iter_idx}  time={iter_time:.3f}s  "
            f"W={tW:.3f}s  H={tH:.3f}s  "
            f"obj={current_obj:.2f}  best={best_obj:.2f}  no_improv={no_improv}"
        )

        iter_idx += 1

    return best_obj, best_W, best_H


# =====================================================
#                     MAIN
# =====================================================

if __name__ == "__main__":

    TEST_PATH = "../tests/testMatrices/"
    TRAINING_PATH = "../tests/trainingMatrices/"
    RESULT_PATH = "../tests/tabuResults/"

    os.makedirs(RESULT_PATH, exist_ok=True)

    test = os.listdir(TEST_PATH)
    training = os.listdir(TRAINING_PATH)

    for i, file in enumerate(test + training):

        print("\n===== FICHIER :", file, "=====")
        path = TEST_PATH if i < len(test) else TRAINING_PATH
        X, r, LW, UW, LH, UH = parse_input(path + file)

        W0 = np.matrix(np.random.randint(LW, UW + 1, size=(X.shape[0], r)))
        H0 = np.matrix(np.random.randint(LH, UH + 1, size=(r, X.shape[1])))

        start = time.perf_counter()
        obj, W_best, H_best = tabu_search(
            X, W0, H0, r, LW, UW, LH, UH,
            max_no_improv=10,
            tabu_tenure=10,
            max_iters=500,     # toujours la sécurité globale
        )
        end = time.perf_counter()

        print(f"⏱ Temps total tabu pour {file} : {end - start:.3f} s")

        feasible = is_solution_feasible(W_best, H_best, r, LW, UW, LH, UH)
        print("Solution feasible ?", feasible)

        generate_output(obj, W_best, H_best, RESULT_PATH + file)
