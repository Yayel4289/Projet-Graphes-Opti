import numpy as np
from utils import *
import os
import cProfile, pstats, io
import time



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
        debug=False
) -> tuple[int, np.matrix, np.matrix]:
    """
    Tabu Search

    Parameters
    ----------
    X : np.matrix
        The matrix to estimate
    W : np.matrix
        The multiplicand (initial solution)
    H : np.matrix
        The multiplier (initial solution)
    r : int
        Rank of the decomposition
    LW : int
        Minimum possible integer for W
    UW : int
        Maximum possible integer for W
    LH : int
        Minimum possible integer for H
    UH : int
        Maximum possible integer for H
    max_no_improv : int
        Maximum iterations without improving the global best
    tabu_tenure : int
        Number of iterations during which a move is forbidden

    Returns
    -------
    best_obj : int
        Best objective value found by the tabu search
    best_W : np.matrix
        Best W matrix found
    best_H : np.matrix
        Best H matrix found
    """

    # ----- Initialisation de la solution courante -----
    current_W = W.copy()
    current_H = H.copy()

    # Résidu R = X - W H et valeur objective courante ||R||_F^2
    R = X - current_W @ current_H
    current_obj = int(np.linalg.norm(R, 'fro')**2)

    # ----- Meilleure solution globale -----
    best_W = current_W.copy()
    best_H = current_H.copy()
    best_obj = current_obj

    # ----- Listes tabou (entiers : temps d'interdiction restants) -----
    tabu_W = np.zeros(current_W.shape, dtype=int)
    tabu_H = np.zeros(current_H.shape, dtype=int)

    no_improv = 0           # compteur d'itérations sans amélioration
    n_rows_W, _ = current_W.shape
    n_rows_H, n_cols_H = current_H.shape   # H est r x n

    # ============================
    #     BOUCLE PRINCIPALE
    # ============================
    while no_improv < max_no_improv:
        if debug:
            t0 = time.perf_counter()

        best_neighbor_obj = None
        best_move = None  # ("W" ou "H", i/k, j, delta)

        # ==========================================
        # 1) Génération des voisins via W
        # ==========================================
        for i in range(n_rows_W):
            for k in range(r):
                for delta in (-1, 1):

                    new_val = current_W[i, k] + delta

                    # Bornes respectées
                    if new_val < LW or new_val > UW:
                        continue

                    # Mouvement tabou ?
                    is_tabu = tabu_W[i, k] > 0

                    # Évaluation locale du voisin :
                    # On change W[i,k] -> W[i,k] + delta
                    # Cela modifie uniquement la ligne i de R.
                    # Formule : f' = f - 2*delta*<R_i., H_k.> + delta^2*||H_k.||^2
                    r_row = np.asarray(R[i, :]).ravel()
                    h_row = np.asarray(current_H[k, :]).ravel()
                    dot_rh = np.dot(r_row, h_row)
                    h_norm2 = np.dot(h_row, h_row)

                    neighbor_obj = current_obj - 2 * delta * dot_rh + (delta ** 2) * h_norm2
                    neighbor_obj = int(neighbor_obj)

                    # Aspiration (on autorise tabou si améliore le global)
                    if is_tabu and neighbor_obj >= best_obj:
                        continue

                    # Sélection du meilleur voisin admissible
                    if (best_neighbor_obj is None) or (neighbor_obj < best_neighbor_obj):
                        best_neighbor_obj = neighbor_obj
                        best_move = ("W", i, k, delta)

        # ==========================================
        # 2) Génération des voisins via H
        # ==========================================
        for k in range(n_rows_H):          # ligne 0..r-1
            for j in range(n_cols_H):      # colonne 0..n-1
                for delta in (-1, 1):

                    new_val = current_H[k, j] + delta

                    # Bornes respectées
                    if new_val < LH or new_val > UH:
                        continue

                    # Mouvement tabou ?
                    is_tabu = tabu_H[k, j] > 0

                    # Évaluation locale du voisin :
                    # On change H[k,j] -> H[k,j] + delta
                    # Cela modifie uniquement la colonne j de R.
                    # Formule : f' = f - 2*delta*<R_.j, W_.k> + delta^2*||W_.k||^2
                    r_col = np.asarray(R[:, j]).ravel()
                    w_col = np.asarray(current_W[:, k]).ravel()
                    dot_rw = np.dot(r_col, w_col)
                    w_norm2 = np.dot(w_col, w_col)

                    neighbor_obj = current_obj - 2 * delta * dot_rw + (delta ** 2) * w_norm2
                    neighbor_obj = int(neighbor_obj)

                    # Aspiration
                    if is_tabu and neighbor_obj >= best_obj:
                        continue

                    # Sélection du meilleur voisin admissible
                    if (best_neighbor_obj is None) or (neighbor_obj < best_neighbor_obj):
                        best_neighbor_obj = neighbor_obj
                        best_move = ("H", k, j, delta)

        # Si aucun mouvement admissible, fin
        if best_move is None:
            break

        # ==========================================
        # Mise à jour des listes tabou (décrément)
        # ==========================================
        tabu_W = np.where(tabu_W > 0, tabu_W - 1, 0)
        tabu_H = np.where(tabu_H > 0, tabu_H - 1, 0)

        # ==========================================
        # Application du meilleur mouvement trouvé
        # ==========================================
        move_type, a, b, delta_move = best_move

        if move_type == "W":
            # Mise à jour de W
            current_W[a, b] += delta_move
            tabu_W[a, b] = tabu_tenure

            # Mise à jour du résidu R : ligne a modifiée
            # R[a, :] <- R[a, :] - delta * H[b, :]
            R[a, :] = R[a, :] - delta_move * current_H[b, :]

        else:  # move on H
            # Mise à jour de H
            current_H[a, b] += delta_move
            tabu_H[a, b] = tabu_tenure

            # Mise à jour du résidu R : colonne b modifiée
            # R[:, b] <- R[:, b] - delta * W[:, a]
            R[:, b] = R[:, b] - delta_move * current_W[:, a]

        # La valeur objective correspond au voisin choisi
        current_obj = best_neighbor_obj

        # ==========================================
        # Mise à jour du meilleur global
        # ==========================================
        if current_obj < best_obj:
            best_obj = current_obj
            best_W = current_W.copy()
            best_H = current_H.copy()
            no_improv = 0
        else:
            no_improv += 1

        if debug:
            dt = time.perf_counter() - t0
            print(f"[HOUDAIN] iter={no_improv}   time={dt:.4f}s   obj={current_obj}")

    return best_obj, best_W, best_H


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    TEST_PATH = "../tests/testMatrices/"
    TRAINING_PATH = "../tests/trainingMatrices/"
    RESULT_PATH = "../tests/tabuResults/"

    os.makedirs(RESULT_PATH, exist_ok=True)

    test = os.listdir(TEST_PATH)
    training = os.listdir(TRAINING_PATH)

    for i, file in enumerate(test + training):
        #if "houdain" in file:
        #    continue

        print(file)
        path = TEST_PATH if i < len(test) else TRAINING_PATH
        X, r, LW, UW, LH, UH = parse_input(path + file)

        # Initialisation aléatoire
        W0 = np.matrix(np.random.randint(LW, UW + 1, size=(X.shape[0], r)))
        H0 = np.matrix(np.random.randint(LH, UH + 1, size=(r, X.shape[1])))

        # Lancement Tabu
        is_houdain = ("houdain" in file.lower())

        obj, W_best, H_best = tabu_search(
            X, W0, H0, r, LW, UW, LH, UH,
            max_no_improv=100,
            tabu_tenure=5,
            debug=is_houdain
        )

        # Vérification faisabilité
        feasible = is_solution_feasible(W_best, H_best, r, LW, UW, LH, UH)
        print("Solution feasible ?", feasible)
        if not feasible:
            print("⚠️ ERREUR : solution NON faisable")

        generate_output(obj, W_best, H_best, RESULT_PATH + file)

        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
        ps.print_stats(30)

        print(s.getvalue())
