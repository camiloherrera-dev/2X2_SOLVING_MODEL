from .base_stages import BaseStage

CROSS_MOVES = [
    "move_u", "move_u_inv",
    "move_d", "move_d_inv",
    "move_r", "move_r_inv",
    "move_l", "move_l_inv",
    "move_f", "move_f_inv",
    "move_b", "move_b_inv"
]

class CrossStage(BaseStage):
    def __init__(self, model):
        super().__init__(model, CROSS_MOVES)
        
    def is_stage_solved(self, cube):
        return cube.is_cross_solved()
    
    def prerequisites_ok(self, cube):
        return True