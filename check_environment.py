"""Report actual runtime and optionally verify the dedicated CUDA environment."""
import runtime
import argparse
import importlib
import json
import platform
import shutil
import sys


def compatibility_issues(report, require_cuda=False, strict_server=False, min_free_gib=2):
    issues = []
    modules = report.get('modules', {})
    for name in ('torch', 'torchvision', 'numpy', 'PIL'):
        if name not in modules:
            issues.append(f'{name} cannot be imported: {report.get("import_errors", {}).get(name, "missing")}')
    if report.get('disk_free_gib', 0) < min_free_gib:
        issues.append(f'Project disk needs at least {min_free_gib} GiB free')
    if require_cuda and not report.get('cuda_available'):
        issues.append('CUDA unavailable: activate .conda-server and check the NVIDIA driver/GPU allocation')
    if require_cuda and report.get('cuda_test_error'):
        issues.append('GPU tensor test failed: ' + report['cuda_test_error'])
    if strict_server:
        if report.get('python_major_minor') != [3, 10]:
            issues.append('Dedicated server profile requires Python 3.10')
        for name, expected in {'torch': '2.7.1+cu118', 'torchvision': '0.22.1+cu118',
                               'numpy': '1.26.4'}.items():
            if modules.get(name) != expected:
                issues.append(f'{name}: expected {expected}, found {modules.get(name)}')
        if report.get('cuda_runtime') != '11.8':
            issues.append('Dedicated server profile requires the CUDA 11.8 PyTorch wheel')
    return issues


def collect_report():
    report = {'python': sys.version.split()[0], 'python_major_minor': list(sys.version_info[:2]),
              'python_executable': sys.executable, 'platform': platform.platform(),
              'disk_free_gib': shutil.disk_usage(runtime.ROOT).free / 1024**3,
              'modules': {}, 'import_errors': {}, 'cuda_available': False}
    loaded = {}
    for name in ('torch', 'torchvision', 'numpy', 'PIL'):
        try:
            loaded[name] = importlib.import_module(name)
            report['modules'][name] = str(loaded[name].__version__)
        except Exception as error:
            report['import_errors'][name] = f'{type(error).__name__}: {error}'
    torch = loaded.get('torch')
    if torch is not None:
        report['cuda_runtime'] = torch.version.cuda
        report['cudnn'] = torch.backends.cudnn.version()
        report['cuda_available'] = torch.cuda.is_available()
        report['gpus'] = []
        if report['cuda_available']:
            try:
                for index in range(torch.cuda.device_count()):
                    gpu = torch.cuda.get_device_properties(index)
                    report['gpus'].append({'visible_index': index, 'name': gpu.name,
                        'memory_gib': gpu.total_memory / 1024**3, 'compute_capability': [gpu.major, gpu.minor]})
                x = torch.ones((16, 16), device='cuda:0')
                report['cuda_tensor_test'] = bool((x @ x == 16).all().item())
                if not report['cuda_tensor_test']:
                    report['cuda_test_error'] = 'Unexpected GPU matrix multiplication result'
            except Exception as error:
                report['cuda_test_error'] = f'{type(error).__name__}: {error}'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--strict-server', action='store_true')
    parser.add_argument('--min-free-gib', type=float, default=2)
    parser.add_argument('--output')
    args = parser.parse_args()
    report = collect_report()
    report['issues'] = compatibility_issues(report, args.require_cuda, args.strict_server, args.min_free_gib)
    report['ok'] = not report['issues']
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = runtime.write_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding='utf-8')
    print(text, flush=True)
    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
