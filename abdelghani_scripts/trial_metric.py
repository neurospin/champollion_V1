# Fake logits and targets
import torch
from sklearn.metrics import accuracy_score

logits = torch.tensor([[1.2, 2.3], [0.5, 1.5], [3.0, 1.0]])
targets = torch.tensor([1, 1, 0])

def accuracy_fn(logits, targets):
    preds = logits.argmax(dim=1).cpu().numpy()
    return accuracy_score(targets.cpu().numpy(), preds)

print("Accuracy:", accuracy_fn(logits, targets))
