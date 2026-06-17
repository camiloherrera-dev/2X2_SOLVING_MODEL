from .base_stages import BaseStage

F2L_MOVES = [
    "move_u", "move_u_inv",
    "move_r", "move_r_inv",
    "move_l", "move_l_inv",
    "move_f", "move_f_inv",
    "move_b", "move_b_inv"
]

class F2lStage(BaseStage):
    def __init__(self, model):
        super().__init__(model, F2L_MOVES)
        
    def is_stage_solved(self, cube):
        return cube.is_f2l_solved()
    
    def prerequisites_ok(self, cube):
        return cube.is_cross_solved()