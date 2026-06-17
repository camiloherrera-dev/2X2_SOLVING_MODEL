import copy
import torch
from cube_class import Cube
from Models.base_model import StageModel
from Models.classifier import ClassifierModel
from Stages.cross import CrossStage
from Stages.f2l import F2lStage
from Stages.oll import OLLStage
from Stages.pll import PLLStage
from Algorithms.loader import load_algorithms
from Models.astar import astar

INPUT_SIZE = 6 * 9 * 6

def load_solver():
    # Cross
    cross_model = StageModel(input_size=INPUT_SIZE)
    cross_stage = CrossStage(cross_model)
    cross_stage.load_weights("Models/checkpoints/cross.pt")

    # F2L
    f2l_model = StageModel(input_size=INPUT_SIZE, hidden_size=1024)
    f2l_stage = F2lStage(f2l_model)
    f2l_stage.load_weights("Models/checkpoints/f2l.pt")

    # OLL
    oll_algos  = load_algorithms("Algorithms/oll.csv")
    oll_model  = ClassifierModel(input_size=INPUT_SIZE, num_classes=len(oll_algos))
    oll_stage  = OLLStage(oll_model, "Algorithms/oll.csv")
    oll_stage.load_weights("Models/checkpoints/oll.pt")

    # PLL
    pll_algos  = load_algorithms("Algorithms/pll.csv")
    pll_model  = ClassifierModel(input_size=INPUT_SIZE, num_classes=len(pll_algos))
    pll_stage  = PLLStage(pll_model, "Algorithms/pll.csv")
    pll_stage.load_weights("Models/checkpoints/pll.pt")

    return cross_stage, f2l_stage, oll_stage, pll_stage


def solve(cube, max_nodes=50000, verbose=True):
    cross_stage, f2l_stage, oll_stage, pll_stage = load_solver()
    solution = []

    def log(msg):
        if verbose:
            print(msg)

    # --- CROSS ---
    log("\n=== CROSS ===")
    if not cross_stage.is_stage_solved(cube):
        print("before astar")
        moves = astar(cube, cross_stage, max_nodes=5000)
        print("past astar")
        if moves is None:
            log("Cross falló")
            return None
        for m in moves:
            getattr(cube, m)()
        solution.extend(moves)
        log(f"Cross resuelto en {len(moves)} movimientos")
    else:
        log("Cross ya estaba resuelto")

    log(f"Cross ok: {cross_stage.is_stage_solved(cube)}")

    # --- F2L ---
    log("\n=== F2L ===")
    if not f2l_stage.is_stage_solved(cube):
        moves = astar(cube, f2l_stage, max_nodes=max_nodes)
        if moves is None:
            log("F2L falló")
            return None
        for m in moves:
            getattr(cube, m)()
        solution.extend(moves)
        log(f"F2L resuelto en {len(moves)} movimientos")
    else:
        log("F2L ya estaba resuelto")

    log(f"F2L ok: {f2l_stage.is_stage_solved(cube)}")

    # --- OLL ---
    log("\n=== OLL ===")
    if not oll_stage.is_stage_solved(cube):
        result = oll_stage.solve(cube)
        if result is None:
            log("OLL falló")
            return None
        cube, moves = result
        solution.extend(moves)
        log(f"OLL resuelto en {len(moves)} movimientos")
    else:
        log("OLL ya estaba resuelto")

    log(f"OLL ok: {oll_stage.is_stage_solved(cube)}")

    # --- PLL ---
    log("\n=== PLL ===")
    if not pll_stage.is_stage_solved(cube):
        result = pll_stage.solve(cube)
        if result is None:
            log("PLL falló")
            return None
        cube, moves = result
        solution.extend(moves)
        log(f"PLL resuelto en {len(moves)} movimientos")
    else:
        log("PLL ya estaba resuelto")

    log(f"\n{'='*40}")
    log(f"Cubo resuelto: {cube.is_solved()}")
    log(f"Total movimientos: {len(solution)}")
    log(f"Solución: {' → '.join(solution)}")

    return solution, cube