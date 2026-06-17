import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .adi import ADIDataset

def train_stage(
    stage,
    cube_class,
    stage_name,
    n_epochs      = 50,
    n_samples     = 10000,
    max_scramble  = 20,
    batch_size    = 256,
    lr            = 1e-4,
    save_path     = None,
    device        = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    stage.model.to(device)
    stage.model.train()
    
    optimizer = torch.optim.Adam(stage.model.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode     = 'min',
        factor   = 0.5,      # reduce lr a la mitad
        patience = 5        # si no mejora en 5 épocas
    )

    loss_fn   = nn.MSELoss()
    
    print(f"Entrenando en: {device}")
    
    for epoch in range(n_epochs):
        # regenerar datos cada epoch (clave en ADI)
        dataset    = ADIDataset(stage_name, stage, cube_class, n_samples, max_scramble)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        epoch_loss = 0.0
        
        for states, targets in dataloader:
            states  = states.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            preds = stage.model(states)
            loss  = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}")
        
        if save_path:
            stage.save_weights(save_path)
    
    stage.model.eval()
    print("Entrenamiento completado.")