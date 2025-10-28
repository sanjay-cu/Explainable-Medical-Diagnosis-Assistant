import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNNConceptNet(nn.Module):
    """A small CNN that outputs both task logits and concept logits.
    Concept logits are intermediate interpretable features (e.g., opacity, fluid).
    """
    def __init__(self, n_concepts=5, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*56*56, 256)
        self.concept_head = nn.Linear(256, n_concepts)  # concept logits
        self.class_head = nn.Linear(256 + n_concepts, n_classes)  # uses concepts + features

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        feat = F.relu(self.fc1(x))
        concepts = torch.sigmoid(self.concept_head(feat))
        combined = torch.cat([feat, concepts], dim=1)
        logits = self.class_head(combined)
        return logits, concepts
