
import numpy as np
from cube_class import cube

new_cube = cube(ndim=3)

new_cube.move_r()
new_cube.move_u()
new_cube.move_r_inv()
new_cube.move_u_inv()
# new_cube.move_d()

print(new_cube.is_cross_solved())
# new_cube.plt_faces()
