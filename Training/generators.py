import numpy as np
import random

def generate_cross_states(cube_class, n_samples, max_scramble, allowed_moves):
    all_moves = allowed_moves
    samples = []

    while len(samples) < n_samples:
        cube = cube_class(ndim=3)
        k = random.choices(
            range(1, max_scramble + 1),
            weights=[1/i for i in range(1, max_scramble + 1)]
        )[0]
        for _ in range(k):
            getattr(cube, random.choice(all_moves))()
        if not cube.is_cross_solved():
            samples.append((cube.to_one_hot(), float(k) / max_scramble))

    return samples

def f2l_distance(cube):
    n = cube.ndim
    c = n // 2
    wrong = 0
    total = 0

    for idx in [1, 2, 3, 5]:
        face = cube.faces[idx]
        center_color = int(face[c, c])
        lower_half   = face[c:, :]
        wrong += np.sum(lower_half != center_color)
        total += lower_half.size

    down       = cube.faces[4]
    down_color = int(down[c, c])
    wrong += np.sum(down != down_color)
    total += down.size

    # penalizar más las celdas de las esquinas
    # porque son las más difíciles de colocar
    corner_penalty = 0
    for idx in [1, 2, 3, 5]:
        face         = cube.faces[idx]
        center_color = int(face[c, c])
        corners      = [face[n-1, 0], face[n-1, n-1]]
        corner_penalty += sum(1 for cell in corners if int(cell) != center_color)

    return float(wrong) / total + 0.1 * corner_penalty / 8

def generate_f2l_states(cube_class, n_samples, max_scramble, allowed_moves):
    # secuencias F2L comunes — aplicar su inverso da estados F2L válidos
    f2l_sequences = [
        # inserción básica derecha
        ["move_u", "move_r", "move_u_inv", "move_r_inv"],
        # inserción básica izquierda  
        ["move_u_inv", "move_l_inv", "move_u", "move_l"],
        # inserción frontal
        ["move_u", "move_f", "move_u_inv", "move_f_inv"],
        # inserción back
        ["move_u_inv", "move_b", "move_u", "move_b_inv"],
        # caso esquina arriba borde arriba derecha
        ["move_r", "move_u", "move_r_inv", "move_u_inv"],
        # caso esquina arriba borde arriba izquierda
        ["move_l_inv", "move_u_inv", "move_l", "move_u"],
        # caso esquina arriba borde arriba frente
        ["move_f", "move_u", "move_f_inv", "move_u_inv"],
        # variantes con doble U
        ["move_u", "move_u", "move_r", "move_u_inv", "move_r_inv"],
        ["move_u", "move_u", "move_l_inv", "move_u", "move_l"],
        # casos con recolocación
        ["move_r", "move_u", "move_u", "move_r_inv", "move_u_inv", "move_r", "move_u_inv", "move_r_inv"],
        ["move_l_inv", "move_u", "move_u", "move_l", "move_u", "move_l_inv", "move_u", "move_l"],
        ["move_u", "move_r", "move_u", "move_u", "move_r_inv", "move_u", "move_r", "move_u_inv", "move_r_inv"],
    ]

    samples  = []
    attempts = 0

    while len(samples) < n_samples and attempts < n_samples * 5:
        attempts += 1
        cube = cube_class(ndim=3)  # resuelto → cruz intacta garantizada

        # aplicar k secuencias inversas aleatorias
        k = random.randint(1, 4)  # 1-4 pares desordenados
        moves_applied = 0

        for _ in range(k):
            seq = random.choice(f2l_sequences)
            # aplicar inverso de la secuencia
            for move in reversed(seq):
                inv = move + "_inv" if not move.endswith("_inv") else move[:-4]
                getattr(cube, inv)()
                moves_applied += 1

            # rotación U aleatoria entre secuencias
            u_count = random.randint(0, 3)
            for _ in range(u_count):
                cube.move_u()
                moves_applied += 1

        if cube.is_cross_solved() and not cube.is_f2l_solved():
            dist = f2l_distance(cube)
            if dist > 0.001:
                samples.append((cube.to_one_hot(), dist))

    if len(samples) < n_samples:
        print(f"Advertencia: solo se generaron {len(samples)} muestras válidas")

    return samples

def generate_oll_training_data(cube_class, algorithms: dict, n_samples=5000):
    algo_names = list(algorithms.keys())
    samples    = []

    for idx, (name, algo) in enumerate(algorithms.items()):
        per_algo = n_samples // len(algorithms)

        for _ in range(per_algo):
            cube = cube_class(ndim=3)  # F2L resuelto

            # orientación U aleatoria
            u_count = random.randint(0, 3)
            for _ in range(u_count):
                cube.move_u()

            # aplicar inverso del algoritmo para obtener el estado que lo necesita
            for move in reversed(algo):
                inv = move + "_inv" if not move.endswith("_inv") else move[:-4]
                getattr(cube, inv)()

            # verificar que F2L sigue resuelto
            if cube.is_f2l_solved() and not cube.is_oll_solved():
                samples.append((cube.to_one_hot(), idx))

    random.shuffle(samples)
    return samples


def generate_pll_training_data(cube_class, algorithms: dict, n_samples=5000):
    algo_names = list(algorithms.keys())
    samples    = []

    for idx, (name, algo) in enumerate(algorithms.items()):
        per_algo = n_samples // len(algorithms)

        for _ in range(per_algo):
            cube = cube_class(ndim=3)

            u_count = random.randint(0, 3)
            for _ in range(u_count):
                cube.move_u()

            for move in reversed(algo):
                inv = move + "_inv" if not move.endswith("_inv") else move[:-4]
                getattr(cube, inv)()

            if cube.is_oll_solved() and not cube.is_solved():
                samples.append((cube.to_one_hot(), idx))

    random.shuffle(samples)
    return samples