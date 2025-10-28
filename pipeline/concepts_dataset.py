from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import torchvision.transforms as T

transform = T.Compose([
    T.Grayscale(),
    T.Resize((224,224)),
    T.ToTensor(),
])

class BasicMedicalDataset(Dataset):
    """Expects folder structure data/<label>/*.png and a CSV for concept labels.
    For prototype, random concept vectors can be generated if annotations are missing.
    """
    def __init__(self, root):
        self.samples = []
        for label in os.listdir(root):
            labdir = os.path.join(root, label)
            if not os.path.isdir(labdir):
                continue
            for f in os.listdir(labdir):
                if f.lower().endswith('.png') or f.lower().endswith('.jpg'):
                    self.samples.append((os.path.join(labdir, f), int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('L')
        img = transform(img)
        # prototype: random concept targets (replace with radiologist labels)
        concept_targets = torch.rand(5)
        return img, label, concept_targets
