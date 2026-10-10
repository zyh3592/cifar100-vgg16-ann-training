"""Training-only augmentation; the inference model remains ordinary ReLU."""
import numpy as np
import torch


def mix_batch(images, labels, mode='none', alpha=0.2, probability=1.0):
    """Return images, two label vectors and the actual first-label weight."""
    if mode == 'none' or images.shape[0] < 2 or np.random.random() >= probability:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    order = torch.randperm(images.shape[0], device=images.device)
    if mode == 'mixup':
        mixed = lam * images + (1.0 - lam) * images[order]
    elif mode == 'cutmix':
        height, width = images.shape[-2:]
        ratio = np.sqrt(1.0 - lam)
        cut_h, cut_w = int(height * ratio), int(width * ratio)
        cy, cx = np.random.randint(height), np.random.randint(width)
        y1, y2 = max(0, cy - cut_h // 2), min(height, cy + (cut_h + 1) // 2)
        x1, x2 = max(0, cx - cut_w // 2), min(width, cx + (cut_w + 1) // 2)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[order, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1)) / (height * width)
    else:
        raise ValueError(f'Unknown augmentation mode: {mode}')
    return mixed, labels, labels[order], lam
