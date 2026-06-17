from cube_class import Cube
from Models.classifier import ClassifierModel
from Stages.pll import PLLStage
from Algorithms.loader import load_algorithms
from Training.generators import generate_pll_training_data
from Training.train_classifier import train_classifier

INPUT_SIZE  = 6 * 9 * 6
pll_algos   = load_algorithms("algorithms/pll.csv")
NUM_CLASSES = len(pll_algos)
print(NUM_CLASSES)

model = ClassifierModel(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
stage = PLLStage(model, "algorithms/pll.csv")

samples = generate_pll_training_data(Cube, pll_algos, n_samples=5000)

train_classifier(
    stage     = stage,
    samples   = samples,
    n_epochs  = 50,
    save_path = "models/checkpoints/pll.pt"
)