from cube_class import Cube
import Models, Stages, Training
from Models.base_model import StageModel
from Stages.cross import CrossStage
from Stages.f2l import F2lStage
from Training.train_adi import train_stage

INPUT_SIZE = 6*9*6

model = StageModel(input_size=INPUT_SIZE, hidden_size=512)
cross_stage = CrossStage(model)
f2l_stage = F2lStage(model)

train_stage(
    stage_name  = "cross",
    stage       = cross_stage,
    cube_class  = Cube,
    n_epochs    = 50,
    n_samples   = 10000,
    max_scramble= 20,
    batch_size  = 256,
    lr          = 1e-4,
    save_path   = "Models/Checkpoints/cross.pt"
)