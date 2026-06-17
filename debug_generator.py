from cube_class import Cube
import random

f2l_moves = [
    "move_u", "move_u_inv",
    "move_r", "move_r_inv", "move_l", "move_l_inv",
    "move_f", "move_f_inv", "move_b", "move_b_inv"
]

cross_ok  = 0
f2l_ok    = 0
both_ok   = 0
total     = 1000

for _ in range(total):
    cube = Cube(ndim=3)
    for _ in range(7):
        getattr(cube, random.choice(f2l_moves))()
    
    if cube.is_cross_solved():
        cross_ok += 1
    if not cube.is_f2l_solved():
        f2l_ok += 1
    if cube.is_cross_solved() and not cube.is_f2l_solved():
        both_ok += 1

print(f"Cruz intacta:        {cross_ok}/{total}")
print(f"F2L sin resolver:    {f2l_ok}/{total}")
print(f"Ambas condiciones:   {both_ok}/{total}")