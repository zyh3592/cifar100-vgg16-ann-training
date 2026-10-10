"""Explicit held-out CIFAR-100 test evaluation; never used for model selection."""
import runtime
import argparse
import json
import torch
from torchvision.datasets import CIFAR100
from dataset import MEAN, STD, transform
from model import build_model
from prepare_data import restore
from train import choose_device, epoch_pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='data/raw')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--max-batches', type=int)
    parser.add_argument('--output', default='output/test_metrics.json')
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or (args.max_batches is not None and args.max_batches < 1):
        parser.error('Invalid batch-size, workers, or max-batches')
    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if tuple(state['normalization']['mean']) != MEAN or tuple(state['normalization']['std']) != STD:
        raise ValueError('Checkpoint normalization differs from dataset.py')
    model = build_model(state['model_config']).to(device)
    model.load_state_dict(state['model'])
    dataset = CIFAR100(str(restore(args.data_root)), train=False, transform=transform(), download=False)
    if dataset.classes != state['class_names']:
        raise ValueError('Checkpoint class order differs from test data')
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         num_workers=args.workers, shuffle=False,
                                         pin_memory=device.type == 'cuda')
    result = epoch_pass(model, loader, device, max_batches=args.max_batches)
    result.update({'checkpoint_epoch': state['epoch'], 'split': 'official_test',
                   'partial_evaluation': args.max_batches is not None,
                   'checkpoint_debug_run': state['debug_run']})
    output = runtime.write_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
