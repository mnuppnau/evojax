"""Backward-compatible wrapper for BloodMNIST training script.

Use `examples/train_bloodmnist.py` for the canonical entry point.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from examples.train_bloodmnist import main, parse_args


if __name__ == '__main__':
    cfg = parse_args()
    if cfg.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    main(cfg)
