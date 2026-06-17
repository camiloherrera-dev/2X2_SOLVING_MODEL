import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .classifier_dataset import ClassifierDataset

def train_classifier(
    stage,
    samples,
    n_epochs   = 50,
    batch_size = 256,
    lr         = 1e-4,
    save_path  = None,
    device     = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage.model.to(device)
    stage.model.train()

    dataset    = ClassifierDataset(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(stage.model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    loss_fn = nn.CrossEntropyLoss()  # clasificación, no MSE

    print(f"Entrenando en: {device}")
    print(f"Muestras: {len(dataset)}")

    for epoch in range(n_epochs):
        epoch_loss    = 0.0
        epoch_correct = 0

        for states, labels in dataloader:
            states = states.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits  = stage.model(states)
            loss    = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss    += loss.item()
            epoch_correct += (logits.argmax(dim=1) == labels).sum().item()

        avg_loss = epoch_loss / len(dataloader)
        accuracy = epoch_correct / len(dataset) * 100
        scheduler.step(avg_loss)

        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.1f}%")

        if save_path:
            stage.save_weights(save_path)

    stage.model.eval()
    print("Entrenamiento completado.")