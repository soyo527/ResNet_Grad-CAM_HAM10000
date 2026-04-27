import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, labels in tqdm(loader, desc="训练", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_classifier(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []

    for inputs, labels in tqdm(loader, desc="评估", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
