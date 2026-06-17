import copy
import torch
import numpy as np
from Algorithms.loader import load_algorithms

class OLLStage:
    def __init__(self, model, algorithms_path):
        self.algorithms  = load_algorithms(algorithms_path)
        self.algo_names  = list(self.algorithms.keys())
        self.model       = model
        self.u_moves     = ["move_u", "move_u", "move_u", "move_u_inv"]
        self.model.eval()

    def is_stage_solved(self, cube) -> bool:
        return cube.is_oll_solved()

    def prerequisite_ok(self, cube) -> bool:
        return cube.is_f2l_solved()

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def predict(self, cube) -> int:
        x = torch.tensor(cube.to_one_hot(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            return torch.argmax(logits, dim=1).item()

    def solve(self, cube):
        best_result = None

        # probar las 4 orientaciones U
        for u_count in range(4):
            test_cube = copy.deepcopy(cube)

            # aplicar u_count movimientos U
            for _ in range(u_count):
                test_cube.move_u()

            # la red elige el algoritmo
            algo_idx = self.predict(test_cube)
            algo     = self.algorithms[self.algo_names[algo_idx]]

            # aplicar algoritmo
            result_cube = copy.deepcopy(test_cube)
            for move in algo:
                getattr(result_cube, move)()

            # ajuste U final
            for u_adj in range(4):
                adj_cube = copy.deepcopy(result_cube)
                for _ in range(u_adj):
                    adj_cube.move_u()

                if self.is_solved(adj_cube) and self.prerequisite_ok(adj_cube):
                    moves = (["move_u"] * u_count + algo +
                             ["move_u"] * u_adj)
                    best_result = (adj_cube, moves)
                    break

            if best_result:
                break

        return best_result  # (cubo_resuelto, lista_de_movimientos) o None