import torch
from pipeline.model import SimpleCNNConceptNet
import torchvision.transforms as T
from PIL import Image
import io
import numpy as np

transform = T.Compose([
    T.Grayscale(),
    T.Resize((224,224)),
    T.ToTensor(),
])

class ModelServer:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = SimpleCNNConceptNet(n_concepts=5, n_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.concept_names = ['opacity','infiltrate','diffuse_pattern','airway_changes','enlargement']

    def predict_from_bytes(self, img_bytes):
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        x = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, concepts = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            concepts = concepts.cpu().numpy()[0]
        return {'probs': probs.tolist(), 'concepts': concepts.tolist(), 'concept_names': self.concept_names}
