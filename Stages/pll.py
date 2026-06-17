import copy
import torch
from Algorithms.loader import load_algorithms

class PLLStage:
    def __init__(self, model, algorithms_path):
        self.algorithms = load_algorithms(algorithms_path)
        self.algo_names = list(self.algorithms.keys())
        self.model      = model
        self.model.eval()

    def is_stage_solved(self, cube) -> bool:
        return cube.is_solved()

    def prerequisite_ok(self, cube) -> bool:
        return cube.is_oll_solved()

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

        for u_pre in range(4):
            test_cube = copy.deepcopy(cube)
            for _ in range(u_pre):
                test_cube.move_u()

            algo_idx = self.predict(test_cube)
            algo     = self.algorithms[self.algo_names[algo_idx]]

            result_cube = copy.deepcopy(test_cube)
            for move in algo:
                getattr(result_cube, move)()

            for u_post in range(4):
                adj_cube = copy.deepcopy(result_cube)
                for _ in range(u_post):
                    adj_cube.move_u()

                if self.is_stage_solved(adj_cube):
                    moves = (["move_u"] * u_pre + algo +
                             ["move_u"] * u_post)
                    best_result = (adj_cube, moves)
                    break

            if best_result:
                break

        return best_result