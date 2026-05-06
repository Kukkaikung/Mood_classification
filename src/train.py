import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # force RTX 3050 Ti

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import argparse
from src.dataset import FERDataset, get_transforms
from src.model import EmotionNet


CONFIG = {
    'data_dir'   : 'data/raw',
    'epochs'     : 50,
    'batch_size' : 32,
    'lr'         : 1e-3,
    'num_classes': 7,
    'dropout'    : 0.3,
    'run_name'   : 'clahe_epoch50',
    'device'     : 'cuda',   # 'cuda' or 'cpu'
    'gpu_id'     : 0,        # 0, 1, 2... or 'all' for all GPUs
    'num_workers': 0,        # 0 for Windows
    'resume'     : False,    # set True to continue from checkpoint
    'resume_path': 'models/best.pth',
    'early_stop_patience' : 10,
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
        num_workers=config['num_workers'],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    return train_loader, test_loader


def get_class_weights(config, device):
    dataset = FERDataset(
        config['data_dir'],
        split='train',
        transform=None
    )
    counts = [0] * config['num_classes']
    for _, label in dataset.samples:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (config['num_classes'] * c) for c in counts]

    print('Class weights:')
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    for i, (c, w) in enumerate(zip(counts, weights)):
        print(f'  {classes[i]:10s}: {c:5d} images → weight {w:.3f}')

    return torch.tensor(weights, dtype=torch.float).to(device)


def get_device(config):
    if config['device'] == 'cuda' and torch.cuda.is_available():
        if config['gpu_id'] == 'all':
            device = torch.device('cuda')
        else:
            device = torch.device(f"cuda:{config['gpu_id']}")
    else:
        device = torch.device('cpu')
        if config['device'] == 'cuda':
            print('WARNING: cuda not available, falling back to cpu')
    return device


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
    device = get_device(config)

    print(f'Training on   : {device}')
    if device.type == 'cuda':
        print(f'GPU           : {torch.cuda.get_device_name(device)}')
        print(f'GPUs available: {torch.cuda.device_count()}')

    train_loader, test_loader = get_dataloaders(config)

    model = EmotionNet(config['num_classes'], config['dropout'])

    if device.type == 'cuda' and config['gpu_id'] == 'all' \
            and torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs with DataParallel')
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    # UPDATED — save path uses run_name so each run has its own model file
    save_path = f"models/best_{config['run_name']}.pth"

    if config.get('resume') and os.path.exists(config['resume_path']):
        model.load_state_dict(
            torch.load(config['resume_path'], map_location=device)
        )
        print(f'Resumed from  : {config["resume_path"]}')
    elif config.get('resume'):
        print(f'WARNING: resume=True but {config["resume_path"]} not found — starting fresh')

    weights   = get_class_weights(config, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )

    best_acc   = 0.0
    no_improve = 0

    with mlflow.start_run(run_name=config['run_name']):
        mlflow.log_params(config)

        for epoch in range(config['epochs']):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            test_loss, test_acc = evaluate(
                model, test_loader, criterion, device
            )

            scheduler.step(test_acc)
            current_lr = optimizer.param_groups[0]['lr']

            print(
                f"Epoch {epoch+1:02d}/{config['epochs']} | "
                f"train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
                f"test  loss: {test_loss:.4f}  acc: {test_acc:.4f} | "
                f"lr: {current_lr:.6f} | "
                f"no_improve: {no_improve}"
            )

            mlflow.log_metrics({
                'train_loss' : train_loss,
                'train_acc'  : train_acc,
                'test_loss'  : test_loss,
                'test_acc'   : test_acc,
                'lr'         : current_lr,
                'no_improve' : no_improve,
            }, step=epoch)

            if test_acc > best_acc:
                best_acc   = test_acc
                no_improve = 0
                # UPDATED — save to run-specific path instead of overwriting best.pth
                torch.save(model.state_dict(), save_path)
                print(f'  >> saved best model — acc: {best_acc:.4f} → {save_path}')
            else:
                no_improve += 1
                if no_improve >= config['early_stop_patience']:
                    print(
                        f'\n  >> early stopping at epoch {epoch+1}'
                        f' — no improvement for {no_improve} epochs'
                    )
                    break

        mlflow.log_metric('best_test_acc', best_acc)
        print(f'\nTraining done. Best accuracy: {best_acc:.4f}')
        print(f'Model saved to: {save_path}')


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)

    parser = argparse.ArgumentParser(description='Train EmotionNet')
    parser.add_argument('--data_dir',             type=str,   default=CONFIG['data_dir'])
    parser.add_argument('--epochs',               type=int,   default=CONFIG['epochs'])
    parser.add_argument('--batch_size',           type=int,   default=CONFIG['batch_size'])
    parser.add_argument('--lr',                   type=float, default=CONFIG['lr'])
    parser.add_argument('--num_classes',          type=int,   default=CONFIG['num_classes'])
    parser.add_argument('--dropout',              type=float, default=CONFIG['dropout'])
    parser.add_argument('--run_name',             type=str,   default=CONFIG['run_name'])
    parser.add_argument('--device',               type=str,   default=CONFIG['device'])
    parser.add_argument('--gpu_id',               type=str,   default=str(CONFIG['gpu_id']))
    parser.add_argument('--num_workers',          type=int,   default=CONFIG['num_workers'])
    parser.add_argument('--resume',               action='store_true', default=CONFIG['resume'])
    parser.add_argument('--resume_path',          type=str,   default=CONFIG['resume_path'])
    parser.add_argument('--early_stop_patience',  type=int,   default=CONFIG['early_stop_patience'])

    args = parser.parse_args()

    CONFIG['data_dir']            = args.data_dir
    CONFIG['epochs']              = args.epochs
    CONFIG['batch_size']          = args.batch_size
    CONFIG['lr']                  = args.lr
    CONFIG['num_classes']         = args.num_classes
    CONFIG['dropout']             = args.dropout
    CONFIG['run_name']            = args.run_name
    CONFIG['device']              = args.device
    CONFIG['gpu_id']              = int(args.gpu_id) if args.gpu_id.isdigit() else args.gpu_id
    CONFIG['num_workers']         = args.num_workers
    CONFIG['resume']              = args.resume
    CONFIG['resume_path']         = args.resume_path
    CONFIG['early_stop_patience'] = args.early_stop_patience

    train(CONFIG)