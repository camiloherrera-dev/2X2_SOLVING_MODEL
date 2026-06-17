import heapq
import copy
import numpy as np

def astar(cube, stage, max_nodes=10000):
    start = copy.deepcopy(cube)

    h_start = stage.value(start)
    counter = 0
    heap    = [(h_start, counter, 0, start, [])]
    visited = set()
    nodes_explored = 0

    while heap and nodes_explored < max_nodes:
        f, _, g, current, path = heapq.heappop(heap)

        state_key = tuple(int(cell) for face in current.faces for row in face for cell in row)
        if state_key in visited:
            continue
        visited.add(state_key)
        nodes_explored += 1

        if stage.is_stage_solved(current):
            return path

        for move_name in stage.allowed_moves:
            # copia ligera en lugar de deepcopy
            next_cube       = copy.copy(current)
            next_cube.faces = [f.copy() for f in current.faces]
            getattr(next_cube, move_name)()

            if not stage.prerequisites_ok(next_cube):
                continue

            next_key = next_cube.to_one_hot().tobytes()
            if next_key in visited:
                continue

            g_next = g + 1
            h_next = max(0.0, stage.value(next_cube))
            f_next = g_next + h_next
            counter += 1

            heapq.heappush(heap, (f_next, counter, g_next, next_cube, path + [move_name]))

    return None