from cube_class import Cube
from Models.base_model import StageModel
from Stages.f2l import F2lStage
from Training.train_adi import train_stage

INPUT_SIZE = 6 * 9 * 6

model = StageModel(input_size=INPUT_SIZE, hidden_size=1024)
stage = F2lStage(model)

train_stage(
    stage        = stage,
    stage_name   = "f2l",
    cube_class   = Cube,
    n_epochs     = 30,
    n_samples    = 10000,
    max_scramble = 7,      # scrambles cortos primero
    batch_size   = 256,
    lr           = 1e-4,
    save_path    = "Models/checkpoints/f2l.pt"
)

# fase 2 — casos difíciles
train_stage(
    stage        = stage,
    stage_name   = "f2l",
    cube_class   = Cube,
    n_epochs     = 30,
    n_samples    = 10000,
    max_scramble = 15,     # scrambles más profundos
    batch_size   = 256,
    lr           = 5e-5,   # lr más bajo para refinamiento
    save_path    = "Models/checkpoints/f2l.pt"
)