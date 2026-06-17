from cube_class import Cube

cube = Cube(ndim=3)
n = cube.ndim
c = n // 2

print("=== Cubo resuelto ===")
print(f"Cruz resuelta: {cube.is_cross_solved()}")

# verificar manualmente las celdas que debería revisar
down       = cube.faces[4]
down_color = int(down[c, c])
print(f"Down center: {down_color}")
print(f"down[0,c]={int(down[0,c])} down[c,0]={int(down[c,0])} down[c,n-1]={int(down[c,n-1])} down[n-1,c]={int(down[n-1,c])}")

for idx in [1, 2, 3, 5]:
    face         = cube.faces[idx]
    center_color = int(face[c, c])
    bottom_edge  = int(face[n-1, c])
    print(f"Cara {idx}: center={center_color} bottom_edge={bottom_edge} ok={bottom_edge==center_color}")

print("\n=== Después de move_r ===")
cube.move_r()
print(f"Cruz resuelta: {cube.is_cross_solved()}")
down = cube.faces[4]
print(f"down[0,c]={int(down[0,c])} down[c,0]={int(down[c,0])} down[c,n-1]={int(down[c,n-1])} down[n-1,c]={int(down[n-1,c])}")
for idx in [1, 2, 3, 5]:
    face         = cube.faces[idx]
    center_color = int(face[c, c])
    bottom_edge  = int(face[n-1, c])
    print(f"Cara {idx}: center={center_color} bottom_edge={bottom_edge} ok={bottom_edge==center_color}")