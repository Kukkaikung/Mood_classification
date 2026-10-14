import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # force RTX 3050 Ti

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')  # save figures to file, no window
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from src.dataset import FERDataset, get_transforms
from src.model import EmotionNet


CONFIG = {
    'model_path'     : 'models/best_first_train_with_val.pth',
    'data_dir'       : 'data/raw',
    'out_dir'        : 'reports',     # results go to reports/<run_name>/
    'run_name'       : None,          # MLflow run for training curves; None = taken from model filename
    'tracking_uri'   : 'sqlite:///mlflow.db',
    'batch_size'     : 64,
    'num_classes'    : 7,
    'device'         : 'cuda',        # 'cuda' or 'cpu'
    'num_workers'    : 0,
    'n_misclassified': 24,            # images in the misclassified grid
    'log_mlflow'     : False,         # True = also log test metrics + figures into the training run
    'limit'          : None,          # evaluate only the first N test images (quick check)
}


def run_name_from_model_path(model_path):
    # models/best_<run_name>.pth → <run_name>
    name = os.path.splitext(os.path.basename(model_path))[0]
    return name[len('best_'):] if name.startswith('best_') else name


def get_device(config):
    if config['device'] == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if config['device'] == 'cuda':
        print('WARNING: cuda not available, falling back to cpu')
    return torch.device('cpu')


def predict(model, loader, device):
    # returns true labels, predicted labels and softmax probabilities for the whole test set
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            probs = F.softmax(model(images.to(device)), dim=1).cpu()
            all_probs.append(probs)
            all_preds.append(probs.argmax(dim=1))
            all_labels.append(labels)

    return (torch.cat(all_labels).numpy(),
            torch.cat(all_preds).numpy(),
            torch.cat(all_probs).numpy())


def plot_confusion_matrix(cm, classes, path, normalize=False):
    if normalize:
        # row-normalised: each row = 100% of the images of that true class
        cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion matrix (normalized by true class)' if normalize
                 else 'Confusion matrix (image counts)')

    # write the value in every cell, white text on dark cells
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f'{cm[i, j]:.2f}' if normalize else f'{cm[i, j]:d}'
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                    color='white' if cm[i, j] > threshold else 'black')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def find_mlflow_run(client, run_name):
    # latest finished run with this name, across all experiments
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"attributes.run_name = '{run_name}' and attributes.status = 'FINISHED'",
        order_by=['attributes.start_time DESC'],
        output_format='list'
    )
    return runs[0] if runs else None


def plot_training_curves(client, run, path):
    # new runs log val_*, older runs logged test_* (test set was used for selection back then)
    keys = run.data.metrics.keys()
    eval_prefix = 'val' if 'val_acc' in keys else 'test'

    def history(key):
        h = sorted(client.get_metric_history(run.info.run_id, key), key=lambda m: m.step)
        return [m.step + 1 for m in h], [m.value for m in h]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ['loss', 'acc']):
        for split, style in [('train', '-'), (eval_prefix, '--')]:
            epochs, values = history(f'{split}_{metric}')
            ax.plot(epochs, values, style, label=split)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy' if metric == 'acc' else 'Loss')
        ax.set_title(f'{metric.capitalize()} per epoch')
        ax.grid(alpha=0.3)

    # mark the epoch whose checkpoint was saved (best eval accuracy)
    epochs, values = history(f'{eval_prefix}_acc')
    best_epoch = epochs[int(np.argmax(values))]
    for ax in axes:
        ax.axvline(best_epoch, color='gray', linestyle=':', label=f'best epoch ({best_epoch})')
        ax.legend()

    fig.suptitle(f'Training history — {run.info.run_name}')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return best_epoch, max(values)


def save_runs_comparison(path):
    # one row per finished run, for the "experiments" table in the report/slides
    df = mlflow.search_runs(search_all_experiments=True,
                            filter_string="attributes.status = 'FINISHED'",
                            order_by=['attributes.start_time ASC'])
    wanted = {
        'tags.mlflow.runName'     : 'run_name',
        'start_time'              : 'start_time',
        'params.epochs'           : 'max_epochs',
        'params.batch_size'       : 'batch_size',
        'params.lr'               : 'lr',
        'params.dropout'          : 'dropout',
        'params.val_split'        : 'val_split',
        'metrics.best_test_acc'   : 'best_test_acc (selected on test)',
        'metrics.best_val_acc'    : 'best_val_acc',
        'metrics.final_test_acc'  : 'final_test_acc (evaluate.py)',
    }
    df = df[[c for c in wanted if c in df.columns]].rename(columns=wanted)
    df.to_csv(path, index=False)
    return df


def plot_misclassified(samples, labels, preds, probs, classes, path, n):
    # most confident mistakes first — the most informative errors to discuss
    wrong = np.where(labels != preds)[0]
    wrong = wrong[np.argsort(-probs[wrong, preds[wrong]])][:n]

    cols = 6
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis('off')

    for ax, idx in zip(np.atleast_1d(axes).ravel(), wrong):
        img_path, _ = samples[idx]
        ax.imshow(Image.open(img_path).convert('RGB'))
        ax.set_title(f'true: {classes[labels[idx]]}\npred: {classes[preds[idx]]} '
                     f'({probs[idx, preds[idx]]:.0%})', fontsize=8)

    fig.suptitle('Most confident misclassifications (test set)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(config):
    device   = get_device(config)
    run_name = config['run_name'] or run_name_from_model_path(config['model_path'])
    out_dir  = os.path.join(config['out_dir'], run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f'Evaluating    : {config["model_path"]}')
    print(f'Device        : {device}')
    print(f'Output folder : {out_dir}')

    # same preprocessing as validation during training (resize 224 + CLAHE, no augmentation)
    test_dataset = FERDataset(config['data_dir'], split='test', transform=get_transforms('test'))
    classes      = test_dataset.classes
    samples      = test_dataset.samples
    if config['limit']:
        # evenly spaced indices so every class is represented (samples are sorted by class folder)
        idx          = np.linspace(0, len(samples) - 1, config['limit']).astype(int)
        test_dataset = Subset(test_dataset, idx)
        samples      = [samples[i] for i in idx]
    loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'], pin_memory=True)
    print(f'Test images   : {len(test_dataset)}')

    # dropout value does not matter in eval mode, the architecture is the same
    model = EmotionNet(config['num_classes'])
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model = model.to(device)

    labels, preds, probs = predict(model, loader, device)

    # 1) metrics
    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    print(f'\nTest accuracy : {acc:.4f}')
    print(f'Macro F1      : {macro_f1:.4f}\n')

    report_txt = classification_report(labels, preds, labels=range(len(classes)),
                                       target_names=classes, digits=4, zero_division=0)
    print(report_txt)
    with open(os.path.join(out_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Model: {config["model_path"]}\n')
        f.write(f'Test images: {len(labels)}\n\n')
        f.write(report_txt)
    report_dict = classification_report(labels, preds, labels=range(len(classes)),
                                        target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).T.to_csv(os.path.join(out_dir, 'classification_report.csv'))

    # 2) confusion matrices
    cm = confusion_matrix(labels, preds, labels=range(len(classes)))
    plot_confusion_matrix(cm, classes, os.path.join(out_dir, 'confusion_matrix.png'))
    plot_confusion_matrix(cm, classes, os.path.join(out_dir, 'confusion_matrix_normalized.png'),
                          normalize=True)

    # 3) misclassified examples
    plot_misclassified(samples, labels, preds, probs, classes,
                       os.path.join(out_dir, 'misclassified.png'), config['n_misclassified'])

    # 4) training curves + 5) all-runs table, both from MLflow
    mlflow.set_tracking_uri(config['tracking_uri'])
    client = MlflowClient()
    run    = find_mlflow_run(client, run_name)
    if run is None:
        print(f'WARNING: no finished MLflow run named "{run_name}" — skipping training curves')
    else:
        best_epoch, best_eval_acc = plot_training_curves(
            client, run, os.path.join(out_dir, 'training_curves.png'))
        print(f'Best epoch    : {best_epoch} (eval acc {best_eval_acc:.4f})')

        if config['log_mlflow']:
            with mlflow.start_run(run_id=run.info.run_id):
                mlflow.log_metrics({'final_test_acc': acc, 'final_test_macro_f1': macro_f1})
                mlflow.log_artifacts(out_dir, artifact_path='evaluation')
            print('Logged test metrics and figures to the MLflow run')

    save_runs_comparison(os.path.join(out_dir, 'runs_comparison.csv'))

    print(f'\nSaved results to {out_dir}/')
    for fname in sorted(os.listdir(out_dir)):
        print(f'  {fname}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate EmotionNet on the test set')
    parser.add_argument('--model_path',      type=str, default=CONFIG['model_path'])
    parser.add_argument('--data_dir',        type=str, default=CONFIG['data_dir'])
    parser.add_argument('--out_dir',         type=str, default=CONFIG['out_dir'])
    parser.add_argument('--run_name',        type=str, default=CONFIG['run_name'])
    parser.add_argument('--tracking_uri',    type=str, default=CONFIG['tracking_uri'])
    parser.add_argument('--batch_size',      type=int, default=CONFIG['batch_size'])
    parser.add_argument('--num_classes',     type=int, default=CONFIG['num_classes'])
    parser.add_argument('--device',          type=str, default=CONFIG['device'])
    parser.add_argument('--num_workers',     type=int, default=CONFIG['num_workers'])
    parser.add_argument('--n_misclassified', type=int, default=CONFIG['n_misclassified'])
    parser.add_argument('--log_mlflow',      action='store_true', default=CONFIG['log_mlflow'])
    parser.add_argument('--limit',           type=int, default=CONFIG['limit'])

    args = parser.parse_args()
    CONFIG.update(vars(args))

    evaluate(CONFIG)
