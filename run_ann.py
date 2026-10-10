"""Train the verified ANN configuration and evaluate the official test set once."""
import runtime
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/vgg16_cifar100.json')
    parser.add_argument('--output-dir')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    config_path = runtime.ROOT / args.config
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    seed = config['seed'] if args.seed is None else args.seed
    workers = config['workers'] if args.workers is None else args.workers
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:6]
    out = runtime.write_path(args.output_dir or f'output/ann_seed{seed}_{tag}')
    if out.exists():
        raise FileExistsError(f'Use a new output directory: {out}')
    subprocess.run([sys.executable, '-B', str(runtime.ROOT / 'train.py'),
                    '--config', str(config_path), '--output-dir', str(out),
                    '--device', args.device, '--seed', str(seed),
                    '--workers', str(workers)], cwd=runtime.ROOT, check=True)
    subprocess.run([sys.executable, '-B', str(runtime.ROOT / 'evaluate.py'),
                    '--checkpoint', str(out / 'best.pt'), '--data-root', config['data_root'],
                    '--device', args.device, '--workers', str(workers),
                    '--output', str(out / 'official_test_metrics.json')],
                   cwd=runtime.ROOT, check=True)
    print(f'Completed ANN training and official test evaluation: {out}', flush=True)


if __name__ == '__main__':
    main()
