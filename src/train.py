import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from src.dataset import FERDataset, get_transforms
from src.model import EmotionNet

CONFIG = {
    'data_dir'   : 'data/raw',
    'epochs'     : 20,
    'batch_size' : 32,
    'lr'         : 1e-3,
    'num_classes': 7,
    'dropout'    : 0.3,
    'run_name'   : 'baseline_no_clahe',
    'device'     : 'cuda',   # change to 'cpu' to train on CPU
}


def get_dataloaders(config):
    train_dataset = FERDataset(
        config['data_dir'],
        split='train',
        transform=get_transforms('train')
    )
    test_dataset = FERDataset(
        config['data_dir'],
        split='test',
        transform=get_transforms('test')
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def train(config):
    # device setup from config
    if config['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        if config['device'] == 'cuda':
            print('WARNING: cuda not available, falling back to cpu')

    print(f'Training on  : {device}')
    print(f'GPU          : {torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A"}')

    train_loader, test_loader = get_dataloaders(config)

    model     = EmotionNet(config['num_classes'], config['dropout']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )

    best_acc = 0.0

    with mlflow.start_run(run_name=config['run_name']):
        mlflow.log_params(config)

        for epoch in range(config['epochs']):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            test_loss, test_acc = evaluate(
                model, test_loader, criterion, device
            )
            scheduler.step()

            print(
                f"Epoch {epoch+1:02d}/{config['epochs']} | "
                f"train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
                f"test  loss: {test_loss:.4f}  acc: {test_acc:.4f}"
            )

            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc' : train_acc,
                'test_loss' : test_loss,
                'test_acc'  : test_acc,
            }, step=epoch)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), 'models/best.pth')
                print(f'  >> saved best model — acc: {best_acc:.4f}')

        mlflow.log_metric('best_test_acc', best_acc)
        print(f'\nTraining done. Best accuracy: {best_acc:.4f}')


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    train(CONFIG)