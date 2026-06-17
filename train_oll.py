from cube_class import Cube
from Models.classifier import ClassifierModel
from Stages.oll import OLLStage
from Algorithms.loader import load_algorithms
from Training.generators import generate_oll_training_data
from Training.train_classifier import train_classifier

INPUT_SIZE  = 6 * 9 * 6
oll_algos   = load_algorithms("algorithms/oll.csv")
NUM_CLASSES = len(oll_algos)
print(NUM_CLASSES)

model = ClassifierModel(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
stage = OLLStage(model, "algorithms/oll.csv")

samples = generate_oll_training_data(Cube, oll_algos, n_samples=5000)

train_classifier(
    stage     = stage,
    samples   = samples,
    n_epochs  = 50,
    save_path = "models/checkpoints/oll.pt"
)