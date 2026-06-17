import random
from cube_class import Cube
from Models.base_model import StageModel
from Stages.f2l import F2lStage

INPUT_SIZE = 6 * 9 * 6
model      = StageModel(input_size=INPUT_SIZE, hidden_size=1024)
stage      = F2lStage(model)
stage.load_weights("Models/checkpoints/f2l.pt")

f2l_moves = [
    "move_u", "move_u_inv",
    "move_r", "move_r_inv", "move_l", "move_l_inv",
    "move_f", "move_f_inv", "move_b", "move_b_inv"
]

cube = Cube(ndim=3)
print(f"Resuelto:      V(s) = {stage.value(cube):.4f} | F2L: {stage.is_stage_solved(cube)}")

for k in [1, 3, 5, 10, 15]:
    cube = Cube(ndim=3)
    for _ in range(k):
        getattr(cube, random.choice(f2l_moves))()
    print(f"{k:2d} movimientos: V(s) = {stage.value(cube):.4f} | F2L: {stage.is_stage_solved(cube)}")