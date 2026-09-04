"""GRD+GLCM experiment entry point replacing the notebook."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interformer import InterFormer
from interformer.data import load_polygon_data, make_loader
from interformer.train import load_model, predict, seed_everything, train_test


def main():
    parser = argparse.ArgumentParser(description='InterFormer real-valued GRD+GLCM experiment')
    parser.add_argument('--grd-root', type=Path, default=Path('/home/huangx3/DATA/GRD_s2_v1_flat'))
    parser.add_argument('--glcm-root', type=Path, default=Path('/home/huangx3/DATA/GRD_s2_v1_flat_glcm'))
    parser.add_argument('--split-file', type=Path, help='Reuse a saved polygon split and data seed')
    parser.add_argument('--output', type=Path, required=True, help='New experiment output directory')
    parser.add_argument('--checkpoint', type=Path, help='Evaluate a Lightning checkpoint without training')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--accelerator', choices=['auto', 'cpu', 'gpu', 'mps'], default='auto')
    parser.add_argument('--threshold', type=float, default=0.95)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be in [0,1]')
    if args.checkpoint and args.split_file is None:
        parser.error("--checkpoint requires the original run's --split-file")
    seed_everything(args.seed)
    data = load_polygon_data(args.grd_root, args.glcm_root, seed=args.seed, split_file=args.split_file)
    test_loader = make_loader(data['test']['x'], data['test']['y'], batch_size=args.batch_size)
    if args.checkpoint is None:
        train_loader = make_loader(data['train']['x'], data['train']['y'],
                                   batch_size=args.batch_size, shuffle=True, seed=args.seed)
        val_loader = make_loader(data['val']['x'], data['val']['y'], batch_size=args.batch_size)
    print('Polygon splits:', data['manifest']['polygon_ids'])
    print('Input shape:', data['manifest']['input_shape'])
    x_test, truth = test_loader.dataset.tensors
    if args.checkpoint:
        args.output.mkdir(parents=True, exist_ok=False)
        model = load_model(args.checkpoint)
        probabilities, labels = predict(model, test_loader)
        checkpoint_path = args.checkpoint
    else:
        x_train = train_loader.dataset.tensors[0]
        if x_train.shape[1:] != x_test.shape[1:] or val_loader.dataset.tensors[0].shape[1:] != x_train.shape[1:]:
            raise ValueError('All splits must have identical time and feature dimensions.')
        model = InterFormer(dim=x_train.shape[-1], time_depth=x_train.shape[1],
                            part2=True, dis=False, t_dis=True)
        trainer, logits = train_test(
            model, args.lr, args.epochs, train_loader, val_loader, test_loader,
            args.patience, args.output.name, args.output.parent, accelerator=args.accelerator)
        probabilities = torch.cat(logits).softmax(-1)
        labels = probabilities.argmax(-1)
        checkpoint_path = Path(trainer.checkpoint_callback.best_model_path)
    threshold_labels = torch.where((probabilities[:, 0] > args.threshold) &
                                   (probabilities[:, 0] > probabilities[:, 1]), 0, 1)
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (truth.numpy(), labels.numpy()), 1)
    results = dict(accuracy=(labels == truth).float().mean().item(),
                   confusion_matrix=cm.tolist(), checkpoint=str(checkpoint_path.resolve()),
                   threshold=args.threshold, seed=args.seed, data_seed=data['manifest']['seed'])
    np.savez(args.output / 'predictions.npz', probabilities=probabilities.numpy(),
             labels=labels.numpy(), threshold_labels=threshold_labels.numpy(), truth=truth.numpy(),
             positions=data['test']['positions'], polygon_ids=data['test']['polygon_ids'],
             source_rows=data['test']['source_rows'])
    (args.output / 'split.json').write_text(json.dumps(data['manifest'], indent=2))
    (args.output / 'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
