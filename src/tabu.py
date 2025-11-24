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
    debug: bool = False
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
    current_obj = int(np.linalg.norm(R, "fro") ** 2)

    # ----- Meilleure solution globale -----
    best_W = current_W.copy()
    best_H = current_H.copy()
    best_obj = current_obj

    # ----- Listes tabou (entiers : temps d'interdiction restants) -----
    tabu_W = np.zeros(current_W.shape, dtype=int)
    tabu_H = np.zeros(current_H.shape, dtype=int)

    no_improv = 0  # compteur d'itérations sans amélioration
    n_rows_W, _ = current_W.shape
    n_rows_H, n_cols_H = current_H.shape  # H est r x n

    # ============================
    #     BOUCLE PRINCIPALE
    # ============================
    while no_improv < max_no_improv:
        if debug:
            t0 = time.perf_counter()

        # --- Pré-calcul des normes L2 des lignes de H et colonnes de W ---
        # H_norm2[k] = || H[k, :] ||^2
        # W_norm2[k] = || W[:, k] ||^2
        H_norm2 = np.sum(np.asarray(current_H) ** 2, axis=1)  # shape (r,)
        W_norm2 = np.sum(np.asarray(current_W) ** 2, axis=0)  # shape (r,)

        # --- Voisinage restreint : on ne parcourt qu'un sous-ensemble de lignes/colonnes ---
        MAX_ROWS_W = 100   # à ajuster si tu veux
        MAX_COLS_H = 100

        rows_W_to_scan = np.random.choice(
            n_rows_W,
            size=min(MAX_ROWS_W, n_rows_W),
            replace=False
        )
        cols_H_to_scan = np.random.choice(
            n_cols_H,
            size=min(MAX_COLS_H, n_cols_H),
            replace=False
        )

        # === First improving move + best worsening backup ===
        improving_move_found = False
        improving_move = None
        improving_obj = None

        best_worse_obj = None
        best_worse_move = None  # ("W" ou "H", i/k, j, delta)

        # ==========================================
        # 1) Génération des voisins via W
        #    (on arrête dès qu'on trouve un move améliorant)
        # ==========================================
        for i in rows_W_to_scan:
            r_row = np.asarray(R[i, :]).ravel()  # calculé 1 fois pour i
            for k in range(r):
                h_row = np.asarray(current_H[k, :]).ravel()  # 1 fois pour (i, k)
                dot_rh = np.dot(r_row, h_row)
                h_norm2 = H_norm2[k]

                # Test du meilleur delta local (+1 ou -1)
                best_local_obj = None
                best_local_delta = None

                # delta = +1
                if current_W[i, k] < UW:
                    obj_plus = current_obj - 2 * (+1) * dot_rh + h_norm2
                    best_local_obj = obj_plus
                    best_local_delta = +1

                # delta = -1
                if current_W[i, k] > LW:
                    obj_minus = current_obj - 2 * (-1) * dot_rh + h_norm2
                    if (best_local_obj is None) or (obj_minus < best_local_obj):
                        best_local_obj = obj_minus
                        best_local_delta = -1

                # Si aucun delta possible, passer
                if best_local_delta is None:
                    continue

                # Vérif tabou + aspiration
                is_tabu = tabu_W[i, k] > 0
                if is_tabu and best_local_obj >= best_obj:
                    # Mouvement tabou sans aspiration -> interdit
                    continue

                # Move admissible
                if best_local_obj < current_obj:
                    # === FIRST IMPROVING MOVE ===
                    improving_move_found = True
                    improving_move = ("W", i, k, best_local_delta)
                    improving_obj = best_local_obj
                    break  # stop boucle k
                else:
                    # Move dégradant mais admissible : on garde le meilleur au cas où
                    if (best_worse_obj is None) or (best_local_obj < best_worse_obj):
                        best_worse_obj = best_local_obj
                        best_worse_move = ("W", i, k, best_local_delta)

            if improving_move_found:
                break  # stop boucle i

        # ==========================================
        # 2) Génération des voisins via H
        #    (seulement si aucun improving trouvé via W)
        # ==========================================
        if not improving_move_found:
            for j in cols_H_to_scan:
                r_col = np.asarray(R[:, j]).ravel()  # calculé 1 fois pour j
                for k in range(n_rows_H):
                    w_col = np.asarray(current_W[:, k]).ravel()  # 1 fois pour (k, j)
                    dot_rw = np.dot(r_col, w_col)
                    w_norm2 = W_norm2[k]

                    best_local_obj = None
                    best_local_delta = None

                    # delta = +1
                    if current_H[k, j] < UH:
                        obj_plus = current_obj - 2 * (+1) * dot_rw + w_norm2
                        best_local_obj = obj_plus
                        best_local_delta = +1

                    # delta = -1
                    if current_H[k, j] > LH:
                        obj_minus = current_obj - 2 * (-1) * dot_rw + w_norm2
                        if (best_local_obj is None) or (obj_minus < best_local_obj):
                            best_local_obj = obj_minus
                            best_local_delta = -1

                    if best_local_delta is None:
                        continue

                    is_tabu = tabu_H[k, j] > 0
                    if is_tabu and best_local_obj >= best_obj:
                        continue

                    if best_local_obj < current_obj:
                        # === FIRST IMPROVING MOVE ===
                        improving_move_found = True
                        improving_move = ("H", k, j, best_local_delta)
                        improving_obj = best_local_obj
                        break  # stop boucle k
                    else:
                        if (best_worse_obj is None) or (best_local_obj < best_worse_obj):
                            best_worse_obj = best_local_obj
                            best_worse_move = ("H", k, j, best_local_delta)

                if improving_move_found:
                    break  # stop boucle j

        # ==========================================
        # Sélection du mouvement à appliquer
        # ==========================================
        if improving_move_found:
            move_type, a, b, delta_move = improving_move
            new_obj = improving_obj
        else:
            # Pas de move améliorant : on prend le "moins pire" si existe
            if best_worse_move is None:
                # Aucun mouvement admissible -> on s'arrête
                break
            move_type, a, b, delta_move = best_worse_move
            new_obj = best_worse_obj

        # ==========================================
        # Mise à jour des listes tabou (décrément)
        # ==========================================
        tabu_W = np.where(tabu_W > 0, tabu_W - 1, 0)
        tabu_H = np.where(tabu_H > 0, tabu_H - 1, 0)

        # ==========================================
        # Application du mouvement choisi
        # ==========================================
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
        current_obj = new_obj

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
        # if "houdain" in file:
        #     continue

        print(file)
        path = TEST_PATH if i < len(test) else TRAINING_PATH
        X, r, LW, UW, LH, UH = parse_input(path + file)

        # Initialisation aléatoire
        W0 = np.matrix(np.random.randint(LW, UW + 1, size=(X.shape[0], r)))
        H0 = np.matrix(np.random.randint(LH, UH + 1, size=(r, X.shape[1])))

        # Lancement Tabu
        is_houdain = ("houdain" in file.lower())

        obj, W_best, H_best = tabu_search(
            X,
            W0,
            H0,
            r,
            LW,
            UW,
            LH,
            UH,
            max_no_improv=100,
            tabu_tenure=5,
            debug=is_houdain,
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
