
import copy
from cube_class import Cube
from Models.classifier import ClassifierModel
from Stages.oll import OLLStage
from Algorithms.loader import load_algorithms

INPUT_SIZE  = 6 * 9 * 6
oll_algos   = load_algorithms("Algorithms/oll.csv")
NUM_CLASSES = len(oll_algos)

model = ClassifierModel(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
stage = OLLStage(model, "Algorithms/oll.csv")
stage.load_weights("Models/checkpoints/oll.pt")

def test_oll(n_tests=20):
    solved    = 0
    algo_names = list(oll_algos.keys())

    for test in range(n_tests):
        cube = Cube(ndim=3)

        # orientación U aleatoria
        import random
        u_count = random.randint(0, 3)
        for _ in range(u_count):
            cube.move_u()

        # aplicar inverso de un algoritmo aleatorio
        algo_name = random.choice(algo_names)
        algo      = oll_algos[algo_name]
        for move in reversed(algo):
            inv = move + "_inv" if not move.endswith("_inv") else move[:-4]
            getattr(cube, inv)()

        cube.plt_faces()
        
        if not cube.is_f2l_solved() or cube.is_oll_solved():
            print(f"Test {test+1}: estado inválido, saltando")
            continue

        print(f"\nTest {test+1} | Caso real: {algo_name} | U previos: {u_count}")

        result = stage.solve(cube)

        if result is None:
            print("No encontró solución")
            continue

        solved_cube, moves = result
        print(f"Movimientos: {' → '.join(moves)}")

        if stage.is_solved(solved_cube) and stage.prerequisite_ok(solved_cube):
            solved += 1
            print("✅ OLL resuelto")
            solved_cube.plt_faces()
        else:
            print("OLL no resuelto correctamente")
            solved_cube.plt_faces()

    print(f"\n{'='*40}")
    print(f"Resueltos: {solved}/{n_tests}")

test_oll(n_tests=20)