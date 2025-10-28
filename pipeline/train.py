import argparse
import torch
from torch.utils.data import DataLoader
from model import SimpleCNNConceptNet
from concepts_dataset import BasicMedicalDataset
import torch.optim as optim
import torch.nn as nn


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNNConceptNet(n_concepts=5, n_classes=2).to(device)
    dataset = BasicMedicalDataset(args.data)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCELoss()

    for epoch in range(10):
        model.train()
        total_loss = 0
        for imgs, labels, concept_targets in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            concept_targets = concept_targets.to(device).float()
            logits, concepts = model(imgs)
            loss_cls = ce(logits, labels)
            loss_concepts = bce(concepts, concept_targets)
            loss = loss_cls + 0.5 * loss_concepts
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} loss {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='model.pth')
    args = parser.parse_args()
    train(args)
