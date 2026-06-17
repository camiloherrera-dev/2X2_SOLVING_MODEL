from cube_class import Cube
from Models.base_model import StageModel
from Stages.cross import CrossStage
from Training.evaluate import evaluate_stage

INPUT_SIZE = 6 * 9 * 6

model = StageModel(input_size=INPUT_SIZE, hidden_size=512)
stage = CrossStage(model)
stage.load_weights("models/checkpoints/cross.pt")

# un solo test con verbose completo
evaluate_stage(
    stage       = stage,
    cube_class  = Cube,
    n_tests     = 5,
    max_scramble= 10,
    max_nodes   = 10000,
    verbose     = True
)