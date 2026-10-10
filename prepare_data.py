"""Restore committed data parts, verify official hashes, and safely extract."""
import runtime
import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

OFFICIAL_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
ARCHIVE_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'
FILE_MD5 = {'train': '16019d7e3df5f24257cddd939b257f8d',
            'test': 'f0ef6b0ae62326f3e7ffdfab6717acfc',
            'meta': '7973b15100ade9c7d40fb424638fde48'}


def digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def verify_extracted(root):
    return all((root / 'cifar-100-python' / name).is_file() and
               digest(root / 'cifar-100-python' / name, 'md5') == checksum
               for name, checksum in FILE_MD5.items())


def restore(root, download=False):
    root = runtime.write_path(root)
    root.mkdir(parents=True, exist_ok=True)
    if verify_extracted(root):
        print(f'All official train/test/meta hashes verified: {root}')
        return root
    archive = root / 'cifar-100-python.tar.gz'
    if not archive.exists() or digest(archive, 'md5') != ARCHIVE_MD5:
        partial = archive.with_suffix('.partial')
        manifest_path = runtime.ROOT / 'data' / 'cifar100_parts' / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
        if manifest and all((manifest_path.parent / entry['name']).is_file() for entry in manifest['parts']):
            with partial.open('wb') as output:
                for entry in manifest['parts']:
                    part = manifest_path.parent / entry['name']
                    if part.stat().st_size != entry['size'] or digest(part) != entry['sha256']:
                        raise ValueError(f'Data part failed verification: {part}')
                    with part.open('rb') as source:
                        shutil.copyfileobj(source, output)
            if digest(partial) != manifest['archive_sha256']:
                raise ValueError('Reassembled archive SHA256 mismatch')
        elif download:
            request = urllib.request.Request(OFFICIAL_URL, headers={'User-Agent': 'CIFAR100-VGG16'})
            with urllib.request.urlopen(request, timeout=60) as source, partial.open('wb') as output:
                shutil.copyfileobj(source, output)
        else:
            raise FileNotFoundError('No archive/data parts; use --download to fetch official CIFAR-100')
        if digest(partial, 'md5') != ARCHIVE_MD5:
            raise ValueError('Official archive MD5 mismatch')
        partial.replace(archive)
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            destination = runtime.write_path(root / member.name)
            if not destination.is_relative_to(root) or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsafe archive member: {member.name}')
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, destination.open('wb') as output:
                    shutil.copyfileobj(source, output)
    if not verify_extracted(root):
        raise ValueError('Extracted train/test/meta checksum mismatch')
    print(f'CIFAR-100 restored and verified: {root}')
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/raw')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    restore(args.data_root, args.download)
