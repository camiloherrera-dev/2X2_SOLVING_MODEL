import copy
import random
from Models.astar import astar

def evaluate_stage(stage, cube_class, n_tests=10, max_scramble=10, max_nodes=10000, verbose=True):
    solved     = 0
    total_moves = 0

    for test in range(n_tests):
        cube = cube_class(ndim=3)
        
        # scramble sin romper prerequisito
        scramble = []
        for _ in range(max_scramble):
            move = random.choice(stage.allowed_moves)
            getattr(cube, move)()
            scramble.append(move)
            if not stage.prerequisites_ok(cube):
                break
        
        if stage.is_stage_solved(cube):
            continue

        if verbose:
            print(f"\n{'='*40}")
            print(f"Test {test+1}")
            print(f"Scramble: {' → '.join(scramble)}")
            print(f"\nEstado inicial:")
            print(f"V(s) inicial: {stage.value(cube):.4f}")

        solution = astar(cube, stage, max_nodes=max_nodes)

        if solution is None:
            if verbose:
                print("No encontró solución")
            continue

        # reproducir solución paso a paso
        cube_copy = copy.deepcopy(cube)
        if verbose:
            print(f"\nSolución ({len(solution)} movimientos): {' → '.join(solution)}")
            print(f"\n--- Pasos ---")

        for i, move_name in enumerate(solution):
            getattr(cube_copy, move_name)()
            if verbose:
                print(f"\nPaso {i+1}: {move_name} | V(s): {stage.value(cube_copy):.4f}")

        if stage.is_stage_solved(cube_copy):
            solved += 1
            total_moves += len(solution)
            if verbose:
                print(f"\nResuelto en {len(solution)} movimientos")
                cube.plt_faces()
        else:
            if verbose:
                print(f"\nTerminó sin resolver")

    print(f"\n{'='*40}")
    print(f"Resueltos: {solved}/{n_tests}")
    if solved > 0:
        print(f"Promedio de movimientos: {total_moves/solved:.1f}")