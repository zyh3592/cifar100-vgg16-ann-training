import runtime
import unittest
import numpy as np
import torch
from torch import nn
from dataset import stratified_split
from model import build_model
from train import learning_rate, loader_options
from check_environment import compatibility_issues


class CoreTests(unittest.TestCase):
    def test_loader_prefetch_only_with_workers(self):
        config = dict(batch_size=128, workers=0, prefetch_factor=2)
        self.assertNotIn('prefetch_factor', loader_options(config, torch.device('cpu')))
        config['workers'] = 4
        options = loader_options(config, torch.device('cuda:0'))
        self.assertEqual(options['prefetch_factor'], 2)
        self.assertTrue(options['pin_memory'])
        self.assertFalse(options.get('persistent_workers', False))

    def test_server_preflight_rejects_cpu_and_failed_gpu(self):
        report = dict(modules={'torch': '2.7.1+cu118', 'torchvision': '0.22.1+cu118',
                              'numpy': '1.26.4', 'PIL': '11.3.0'},
                      python_major_minor=[3, 10], disk_free_gib=100,
                      cuda_available=True, cuda_runtime='11.8')
        self.assertEqual(compatibility_issues(report, True, True), [])
        report['cuda_available'] = False
        self.assertTrue(compatibility_issues(report, True, True))
        report['cuda_available'] = True
        report['cuda_test_error'] = 'driver mismatch'
        self.assertTrue(compatibility_issues(report, True, True))
        del report['cuda_test_error']
        report['modules']['torch'] = '2.7.1+cpu'
        self.assertTrue(compatibility_issues(report, True, True))

    def test_stratified_split_excludes_validation_and_is_reproducible(self):
        labels = np.repeat(np.arange(100), 500)
        train, val = stratified_split(labels, 50, 42)
        self.assertEqual(len(train), 45000)
        self.assertEqual(len(val), 5000)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(len(set(train) | set(val)), 50000)
        self.assertTrue(np.all(np.bincount(labels[val]) == 50))
        self.assertEqual((train, val), stratified_split(labels, 50, 42))
        self.assertNotEqual(val, stratified_split(labels, 50, 43)[1])

    def test_model_has_16_weight_layers_and_real_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        model = build_model(dict(num_classes=100, batch_norm=True,
                                pool='max', hidden_dim=4096, dropout=0.5))
        self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in model.modules()), 13)
        self.assertEqual(sum(isinstance(m, nn.Linear) for m in model.modules()), 3)
        self.assertEqual(sum(isinstance(m, nn.BatchNorm2d) for m in model.modules()), 13)
        self.assertEqual(sum(isinstance(m, nn.MaxPool2d) for m in model.modules()), 5)
        self.assertEqual(sum(isinstance(m, nn.AvgPool2d) for m in model.modules()), 0)
        self.assertEqual(model.classifier[0].out_features, 4096)
        logits = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (2, 100))
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([4, 83]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.features[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.classifier[-1].weight.grad.abs().sum().item(), 0)
        model.eval()
        with torch.no_grad():
            inputs = torch.randn(1, 3, 32, 32)
            self.assertTrue(torch.equal(model(inputs), model(inputs)))

    def test_learning_rate_warmup_and_final_value(self):
        config = {'warmup_epochs': 2, 'epochs': 6, 'lr': 0.1, 'min_lr': 1e-5}
        self.assertAlmostEqual(learning_rate(0, config), 0.05)
        self.assertAlmostEqual(learning_rate(1, config), 0.1)
        self.assertAlmostEqual(learning_rate(5, config), 1e-5)
        rates = [learning_rate(i, config) for i in range(2, 6)]
        self.assertEqual(rates, sorted(rates, reverse=True))

    def test_conventional_multistep_boundaries(self):
        config = {'warmup_epochs': 0, 'epochs': 200, 'lr': 0.1,
                  'lr_schedule': 'multistep', 'milestones': [60, 120, 160], 'lr_gamma': 0.2}
        for epoch, expected in [(0, 0.1), (59, 0.1), (60, 0.02), (120, 0.004), (160, 0.0008)]:
            self.assertAlmostEqual(learning_rate(epoch, config), expected)


if __name__ == '__main__':
    unittest.main()
