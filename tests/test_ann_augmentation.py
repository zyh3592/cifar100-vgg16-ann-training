import runtime
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ann_augmentation import mix_batch
from train import epoch_pass


class AugmentationTests(unittest.TestCase):
    def test_mixup_labels_and_pixels_agree(self):
        images = torch.stack([torch.zeros(1, 4, 4), torch.ones(1, 4, 4)])
        labels = torch.tensor([0, 1])
        with patch('ann_augmentation.np.random.beta', return_value=0.25), \
             patch('ann_augmentation.torch.randperm', return_value=torch.tensor([1, 0])):
            mixed, a, b, lam = mix_batch(images, labels, 'mixup')
        self.assertEqual(lam, 0.25)
        self.assertTrue(torch.equal(b, labels.flip(0)))
        self.assertTrue(torch.allclose(mixed[0], torch.full_like(images[0], 0.75)))
        self.assertTrue(torch.equal(images[0], torch.zeros_like(images[0])))

    def test_cutmix_uses_actual_clipped_patch_area(self):
        images = torch.stack([torch.zeros(1, 4, 4), torch.ones(1, 4, 4)])
        with patch('ann_augmentation.np.random.beta', return_value=0.0), \
             patch('ann_augmentation.np.random.randint', return_value=0), \
             patch('ann_augmentation.torch.randperm', return_value=torch.tensor([1, 0])):
            mixed, _, _, lam = mix_batch(images, torch.tensor([0, 1]), 'cutmix')
        self.assertEqual(lam, 0.75)
        self.assertEqual(mixed[0].sum().item(), 4)
        self.assertEqual(images[0].sum().item(), 0)

    def test_rng_restore_reproduces_augmentation(self):
        images, labels = torch.randn(8, 3, 4, 4), torch.arange(8)
        np_state, torch_state = np.random.get_state(), torch.get_rng_state()
        expected = mix_batch(images, labels, 'mixup')
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        actual = mix_batch(images, labels, 'mixup')
        for x, y in zip(expected[:3], actual[:3]):
            self.assertTrue(torch.equal(x, y))
        self.assertEqual(expected[3], actual[3])

    def test_evaluation_never_uses_mixing_or_smoothing(self):
        model = torch.nn.Linear(3, 6)
        inputs, labels = torch.randn(4, 3), torch.tensor([0, 1, 2, 3])
        config = dict(augmentation='mixup', label_smoothing=0.1)
        with patch('train.mix_batch', side_effect=AssertionError('Must not augment evaluation')):
            result = epoch_pass(model, [(inputs, labels)], torch.device('cpu'),
                                augmentation_config=config)
        self.assertAlmostEqual(result['loss'], torch.nn.functional.cross_entropy(model(inputs), labels).item())


if __name__ == '__main__':
    unittest.main()
