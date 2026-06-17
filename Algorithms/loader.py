import csv

MOVE_MAP = {
    "R":     "move_r",
    "R_inv": "move_r_inv",
    "L":     "move_l",
    "L_inv": "move_l_inv",
    "U":     "move_u",
    "U_inv": "move_u_inv",
    "D":     "move_d",
    "D_inv": "move_d_inv",
    "F":     "move_f",
    "F_inv": "move_f_inv",
    "B":     "move_b",
    "B_inv": "move_b_inv",
    "M":     "move_m",
    "M_inv": "move_m_inv",
    "x": "move_x",
    "x_inv": "move_x_inv",
    "y": "move_y"
}


def parse_algorithm(algo_str: str) -> list[str]:
    return [MOVE_MAP[move] for move in algo_str.strip().split()]

def load_algorithms(csv_path: str) -> dict:
    algorithms = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            algorithms[name] = parse_algorithm(row["algorithm"])
    return algorithms