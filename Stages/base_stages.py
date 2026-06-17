from abc import ABC, abstractmethod
import torch

class BaseStage(ABC):
    def __init__(self, model, allowed_moves):
        self.model = model
        self.allowed_moves = allowed_moves
        self.model.eval()
        
    @abstractmethod
    def is_stage_solved(self, cube):
        pass
    
    @abstractmethod
    def prerequisites_ok(self, cube):
        pass
    
    def value(self, cube):
        x = torch.tensor(cube.to_one_hot(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.model(x).item()

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)