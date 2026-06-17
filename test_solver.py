import random
from cube_class import Cube
from solver import solve

ALL_MOVES = [
    "move_u", "move_u_inv", "move_d", "move_d_inv",
    "move_r", "move_r_inv", "move_l", "move_l_inv",
    "move_f", "move_f_inv", "move_b", "move_b_inv"
]

def test_solver(n_tests=5, scramble_depth=5):
    solved = 0

    for i in range(n_tests):
        cube = Cube(ndim=3)

        scramble = [random.choice(ALL_MOVES) for _ in range(scramble_depth)]
        for m in scramble:
            getattr(cube, m)()

        print(f"\n{'='*40}")
        print(f"Test {i+1} | Scramble ({scramble_depth}): {' '.join(scramble)}")
        # cube.print_cube()

        result = solve(copy.deepcopy(cube), verbose=True)

        if result is not None:
            _, solved_cube = result
            if solved_cube.is_solved():
                solved += 1

    print(f"\n{'='*40}")
    print(f"Resueltos: {solved}/{n_tests}")

if __name__ == "__main__":
    import copy
    test_solver(n_tests=5, scramble_depth=15)