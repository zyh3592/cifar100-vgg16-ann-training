"""Configure writable paths before importing training libraries."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def write_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if os.name == 'nt' and not path.is_relative_to(Path(r'D:\Codex_WorkSpace').resolve()):
        raise ValueError(f'Windows outputs must stay in D:\\Codex_WorkSpace: {path}')
    return path


def configure():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    cache = write_path('output/.cache')
    cache.mkdir(parents=True, exist_ok=True)
    temporary_root = write_path(r'D:\Codex_WorkSpace\cache') if os.name == 'nt' else cache / 'tmp'
    for key, suffix in {'TEMP': 'tmp', 'TMP': 'tmp', 'TMPDIR': 'tmp',
                        'TORCH_HOME': 'torch', 'TORCHINDUCTOR_CACHE_DIR': 'inductor',
                        'TRITON_CACHE_DIR': 'triton', 'CUDA_CACHE_PATH': 'cuda', 'XDG_CACHE_HOME': 'xdg',
                        'PIP_CACHE_DIR': 'pip'}.items():
        directory = temporary_root if key in {'TEMP', 'TMP', 'TMPDIR'} else cache / suffix
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    import sys
    import tempfile
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(temporary_root)


configure()
