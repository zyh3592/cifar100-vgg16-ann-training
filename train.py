"""Train a conventional ReLU ANN; no SNN or quantized activations."""
import runtime
import argparse
import csv
import json
import hashlib
import math
import random
import time
import sys
from pathlib import Path
import numpy as np
import torch
import torchvision
from torch import nn
from dataset import MEAN, STD, seed_worker, training_datasets
from model import build_model
from prepare_data import restore
from ann_augmentation import mix_batch


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(runtime.ROOT / 'configs/vgg16_cifar100.json'))
    for name in ['epochs', 'batch-size', 'warmup-epochs', 'seed', 'val-per-class',
                 'workers', 'cpu-threads', 'hidden-dim', 'max-train-batches', 'max-val-batches',
                 'prefetch-factor', 'log-interval']:
        parser.add_argument('--' + name, type=int)
    for name in ['lr', 'min-lr', 'momentum', 'weight-decay', 'dropout',
                 'mix-alpha', 'mix-probability', 'label-smoothing']:
        parser.add_argument('--' + name, type=float)
    for name in ['device', 'pool', 'data-root', 'output-dir']:
        parser.add_argument('--' + name)
    parser.add_argument('--lr-schedule', choices=['multistep', 'cosine'])
    parser.add_argument('--augmentation', choices=['none', 'mixup', 'cutmix'])
    parser.add_argument('--image-augmentation', choices=['none', 'autoaugment'])
    for name in ['amp', 'deterministic', 'batch-norm']:
        parser.add_argument('--' + name, action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after-epoch', type=int,
                        help='Stop early without changing the full schedule; useful for verification')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    for key, default in {'image_augmentation': 'none', 'augmentation': 'none', 'mix_alpha': 0.2,
                         'mix_probability': 1.0, 'label_smoothing': 0.0}.items():
        config.setdefault(key, default)
    for key in list(config):
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    if args.lr_schedule is not None:
        config['lr_schedule'] = args.lr_schedule
    for key in ['prefetch_factor', 'log_interval']:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if config['epochs'] < 1 or config['batch_size'] < 1 or config['workers'] < 0:
        parser.error('epochs and batch-size must be positive; workers must be nonnegative')
    if config['augmentation'] not in {'none', 'mixup', 'cutmix'}:
        parser.error('Unknown augmentation')
    if config['image_augmentation'] not in {'none', 'autoaugment'}:
        parser.error('Unknown image augmentation')
    if not math.isfinite(config['mix_alpha']) or config['mix_alpha'] <= 0:
        parser.error('mix-alpha must be finite and positive')
    if not 0 <= config['mix_probability'] <= 1 or not 0 <= config['label_smoothing'] < 1:
        parser.error('Invalid mix-probability or label-smoothing')
    if not 0 <= config['warmup_epochs'] <= config['epochs']:
        parser.error('warmup-epochs must be between 0 and epochs')
    if not 0 <= config['min_lr'] <= config['lr'] or config['lr'] <= 0:
        parser.error('Require 0 <= min-lr <= lr and lr > 0')
    if not 0 <= config['dropout'] < 1 or config['hidden_dim'] < 1:
        parser.error('Invalid dropout or hidden-dim')
    if config.get('lr_schedule', 'cosine') not in {'multistep', 'cosine'}:
        parser.error('Invalid lr_schedule')
    if config.get('lr_schedule') == 'multistep':
        milestones = config.get('milestones', [])
        if milestones != sorted(set(milestones)) or any(not isinstance(m, int) or m < 1 for m in milestones):
            parser.error('milestones must be positive, sorted, unique epoch counts')
        if not 0 < config.get('lr_gamma', 0) < 1:
            parser.error('lr_gamma must be in (0, 1)')
    for key in ['max_train_batches', 'max_val_batches']:
        if config[key] is not None and config[key] < 1:
            parser.error(key + ' must be positive')
    if config.get('prefetch_factor', 2) < 1 or config.get('log_interval', 50) < 0:
        parser.error('prefetch-factor must be positive; log-interval must be nonnegative')
    return args, config


def choose_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested, but unavailable; install a CUDA PyTorch build on the server')
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('Only cpu/cuda devices are supported')
    return device


def learning_rate(epoch, config):
    warmup, total = config['warmup_epochs'], config['epochs']
    if epoch < warmup:
        return config['lr'] * (epoch + 1) / warmup
    if config.get('lr_schedule', 'cosine') == 'multistep':
        decays = sum(epoch >= milestone for milestone in config['milestones'])
        return config['lr'] * config['lr_gamma'] ** decays
    progress = (epoch - warmup) / max(1, total - warmup - 1)
    return config['min_lr'] + (config['lr'] - config['min_lr']) * (1 + math.cos(math.pi * progress)) / 2


def loader_options(config, device):
    options = dict(batch_size=config['batch_size'], num_workers=config['workers'],
                   pin_memory=device.type == 'cuda', worker_init_fn=seed_worker)
    if config['workers'] > 0:
        options['prefetch_factor'] = config.get('prefetch_factor', 2)
    # Workers are recreated each epoch so seeded augmentation resumes consistently.
    return options


def epoch_pass(model, loader, device, optimizer=None, scaler=None, amp=False, max_batches=None,
               log_interval=0, epoch_number=None, augmentation_config=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct1, correct5, samples = 0., 0, 0, 0
    augmentation_config = augmentation_config or {}
    criterion = nn.CrossEntropyLoss(label_smoothing=
        augmentation_config.get('label_smoothing', 0.0) if training else 0.0)
    with torch.set_grad_enabled(training):
        for step, (images, labels) in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
                images, labels_a, labels_b, lam = mix_batch(images, labels,
                    mode=augmentation_config.get('augmentation', 'none'),
                    alpha=augmentation_config.get('mix_alpha', 0.2),
                    probability=augmentation_config.get('mix_probability', 1.0))
            else:
                labels_a, labels_b, lam = labels, labels, 1.0
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels_a)
                if lam != 1.0:
                    loss = lam * loss + (1.0 - lam) * criterion(logits, labels_b)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at batch {step}')
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            predictions = logits.detach().topk(5, dim=1).indices
            batch = labels.numel()
            samples += batch
            loss_sum += loss.item() * batch
            correct1 += lam * (predictions[:, 0] == labels_a).sum().item() + \
                (1.0 - lam) * (predictions[:, 0] == labels_b).sum().item()
            correct5 += lam * (predictions == labels_a[:, None]).any(dim=1).sum().item() + \
                (1.0 - lam) * (predictions == labels_b[:, None]).any(dim=1).sum().item()
            if log_interval and (step + 1) % log_interval == 0:
                print(json.dumps({'event': 'batch_progress', 'epoch': epoch_number,
                    'phase': 'train' if training else 'validation', 'batches': step + 1,
                    'samples': samples, 'loss': loss_sum / samples,
                    'top1': 100 * correct1 / samples}), flush=True)
    if samples == 0:
        raise RuntimeError('Empty dataset/loader')
    return {'loss': loss_sum / samples, 'top1': 100 * correct1 / samples,
            'top5': 100 * correct5 / samples, 'samples': samples}


def rng_state(generator):
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (state[0], state[1].tolist(), *state[2:]),
            'torch': torch.get_rng_state(), 'loader': generator.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_checkpoint(state, path):
    partial = path.with_suffix('.tmp')
    torch.save(state, partial)
    partial.replace(path)


def main():
    args, config = arguments()
    torch.set_num_threads(config['cpu_threads'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    if config['deterministic']:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    device = choose_device(config['device'])
    amp = config['amp'] and device.type == 'cuda'
    out = runtime.write_path(config['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if args.resume is None and any(out.iterdir()):
        raise FileExistsError('Output directory is not empty: choose a new --output-dir')
    if args.resume and args.resume.resolve() != (out / 'last.pt').resolve():
        raise ValueError('Resume must use last.pt in the selected output directory')
    root = restore(config['data_root'])
    train_set, val_set, classes, train_indices, val_indices = training_datasets(
        root, config['val_per_class'], config['seed'], config['image_augmentation'])
    generator = torch.Generator().manual_seed(config['seed'])
    loader_args = loader_options(config, device)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, generator=generator, **loader_args)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False,
                                            generator=torch.Generator().manual_seed(config['seed'] + 1),
                                            **loader_args)
    model_config = {k: config[k] for k in ['batch_norm', 'pool', 'hidden_dim', 'dropout']}
    model_config['num_classes'] = 100
    model = build_model(model_config).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'],
                                weight_decay=config['weight_decay'], nesterov=True)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    start, best, history = 0, -1., []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        legacy_defaults = {'image_augmentation': 'none', 'augmentation': 'none', 'mix_alpha': 0.2,
                           'mix_probability': 1.0, 'label_smoothing': 0.0}
        for key in config:
            if key not in {'device', 'workers', 'cpu_threads', 'data_root', 'output_dir', 'amp',
                           'prefetch_factor', 'log_interval'}:
                if checkpoint['config'].get(key, legacy_defaults.get(key)) != config[key]:
                    raise ValueError(f'Resume config mismatch: {key}')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if amp == checkpoint['amp_enabled']:
            scaler.load_state_dict(checkpoint['scaler'])
        start, best = checkpoint['epoch'], checkpoint['best_val_top1']
        history = checkpoint['history']
        restore_rng(checkpoint['rng'], generator)
    source_files = ['train.py', 'model.py', 'dataset.py', 'ann_augmentation.py',
                    'runtime.py', 'prepare_data.py', 'evaluate.py']
    source_record = {name: hashlib.sha256((runtime.ROOT / name).read_bytes()).hexdigest()
                     for name in source_files}
    if args.resume and checkpoint.get('source_sha256') and checkpoint['source_sha256'] != source_record:
        raise ValueError('Training source changed since checkpoint; use the original source snapshot')
    source_dir = out / 'source'
    source_dir.mkdir(exist_ok=True)
    for name in source_files:
        (source_dir / name).write_bytes((runtime.ROOT / name).read_bytes())
    (out / 'source_sha256.json').write_text(json.dumps(source_record, indent=2), encoding='utf-8')
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (out / 'split_indices.json').write_text(json.dumps({'train': train_indices, 'validation': val_indices}),
                                           encoding='utf-8')
    environment = {'python': sys.version.split()[0], 'torch': str(torch.__version__),
                   'train_accuracy_definition': 'label-weighted accuracy on augmented training batches',
                   'validation_accuracy_definition': 'ordinary top-k on unaugmented validation images',
                   'torchvision': str(torchvision.__version__), 'numpy': np.__version__,
                   'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                   'device': str(device), 'amp_enabled': amp,
                   'train_samples': len(train_set), 'validation_samples': len(val_set),
                   'parameters': sum(p.numel() for p in model.parameters()), 'model_config': model_config}
    if device.type == 'cuda':
        gpu = torch.cuda.get_device_properties(device)
        environment['gpu_name'] = gpu.name
        environment['gpu_total_gib'] = gpu.total_memory / 1024**3
    (out / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    print(json.dumps(environment), flush=True)
    debug = config['max_train_batches'] is not None or config['max_val_batches'] is not None
    end = min(config['epochs'], args.stop_after_epoch or config['epochs'])
    for epoch in range(start, end):
        begin = time.monotonic()
        lr = learning_rate(epoch, config)
        for group in optimizer.param_groups:
            group['lr'] = lr
        train_result = epoch_pass(model, train_loader, device, optimizer, scaler, amp,
            config['max_train_batches'], config.get('log_interval', 50), epoch + 1,
            augmentation_config=config)
        val_result = epoch_pass(model, val_loader, device, amp=amp, max_batches=config['max_val_batches'])
        row = {'epoch': epoch + 1, 'lr': lr, **{'train_' + k: v for k, v in train_result.items()},
               **{'val_' + k: v for k, v in val_result.items()}, 'seconds': time.monotonic() - begin}
        history.append(row)
        improved = val_result['top1'] > best
        best = max(best, val_result['top1'])
        state = {'format_version': 1, 'epoch': epoch + 1, 'model': model.state_dict(),
                 'source_sha256': source_record,
                 'model_config': model_config, 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                 'amp_enabled': amp, 'config': config, 'best_val_top1': best, 'class_names': classes,
                 'normalization': {'mean': MEAN, 'std': STD}, 'rng': rng_state(generator),
                 'history': history, 'debug_run': debug}
        atomic_checkpoint(state, out / 'last.pt')
        if improved:
            atomic_checkpoint(state, out / 'best.pt')
        with (out / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row), flush=True)
    (out / 'summary.json').write_text(json.dumps({'completed_epochs': len(history),
        'best_validation_top1': best, 'debug_run': debug,
        'test_set_used': False, 'full_schedule_completed': len(history) == config['epochs']}, indent=2),
        encoding='utf-8')


if __name__ == '__main__':
    main()
