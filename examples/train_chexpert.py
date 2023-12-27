import argparse
import os
import shutil
import jax

from evojax import Trainer
from evojax.policy.densenet import DenseNetPolicy # Assuming DenseNetPolicy is properly implemented
from evojax.task.chexpert import CheXpert  # Assuming CheXpert task is properly implemented
from evojax.algo import PGPE
from evojax import util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pop-size', type=int, default=64, help='NE population size.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')  # Adjusted for CheXpert
    parser.add_argument('--data_path', default='./data', help='Location of train/valid datasets directory or path to test csv file.')
    parser.add_argument('--max-iter', type=int, default=1000, help='Max training iterations.')  # Adjust as needed
    parser.add_argument('--test-interval', type=int, default=100, help='Test interval.')
    parser.add_argument('--resize', type=int, help='Size of minimum edge to which to resize images.')
    parser.add_argument('--log-interval', type=int, default=10, help='Logging interval.')
    parser.add_argument('--restore', type=str, help='Path to a single model checkpoint to restore or folder of checkpoints to ensemble.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument('--center-lr', type=float, default=0.01, help='Center learning rate.')  # Adjust as needed
    parser.add_argument('--mini_data', type=int, help='Truncate dataset to this number of examples.')
    parser.add_argument('--std-lr', type=float, default=0.1, help='Std learning rate.')  # Adjust as needed
    parser.add_argument('--init-std', type=float, default=0.05, help='Initial std.')  # Adjust as needed
    parser.add_argument('--cuda', type=int, help='Which cuda device to use.')
    parser.add_argument('--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    config, _ = parser.parse_known_args()
    return config


def main(config):
    log_dir = './log/chexpert'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(name='CheXpert', log_dir=log_dir, debug=config.debug)
    logger.info('EvoJAX CheXpert Demo')
    logger.info('=' * 30)

    # Adjust the number of classes as per CheXpert's requirement
    num_classes = 5  # Update as needed based on CheXpert's classes
    policy = DenseNetPolicy(num_classes=num_classes, logger=logger)
    init_batch_stats = policy.init_batch_stats
    train_task = CheXpert(config, test=False, batch_stats=init_batch_stats)
    test_task = CheXpert(config, test=True, batch_stats=init_batch_stats)
    
    solver = PGPE(
        pop_size=config.pop_size,
        param_size=policy.num_params,
        optimizer='adam',
        center_learning_rate=config.center_lr,
        stdev_learning_rate=config.std_lr,
        init_stdev=config.init_std,
        logger=logger,
        seed=config.seed,
    )

    # Train.
    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=1,
        n_evaluations=1,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
    )
    trainer.run(demo_mode=False)

    # Test the final model.
    src_file = os.path.join(log_dir, 'best.npz')
    tar_file = os.path.join(log_dir, 'model.npz')
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)


if __name__ == '__main__':
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
    main(configs)

