import logging
import jax
import jax.numpy as jnp

from flax.training import train_state, checkpoints
from typing import Tuple
from flax import linen as nn
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn

densenet_kernel_init = nn.initializers.kaiming_normal()

class DenseLayer(nn.Module):
    bn_size : int  # Bottleneck size (factor of growth rate) for the output of the 1x1 convolution
    growth_rate : int  # Number of output channels of the 3x3 convolution
    act_fn : callable  # Activation function

    @nn.compact
    def __call__(self, x, train=False):
        z = nn.BatchNorm()(x, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.bn_size * self.growth_rate,
                    kernel_size=(1, 1),
                    kernel_init=densenet_kernel_init,
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.growth_rate,
                    kernel_size=(3, 3),
                    kernel_init=densenet_kernel_init,
                    use_bias=False)(z)
        x_out = jnp.concatenate([x, z], axis=-1)
        return x_out

class DenseBlock(nn.Module):
    num_layers : int  # Number of dense layers to apply in the block
    bn_size : int  # Bottleneck size to use in the dense layers
    growth_rate : int  # Growth rate to use in the dense layers
    act_fn : callable  # Activation function to use in the dense layers

    @nn.compact
    def __call__(self, x, train=False):
        for _ in range(self.num_layers):
            x = DenseLayer(bn_size=self.bn_size,
                           growth_rate=self.growth_rate,
                           act_fn=self.act_fn)(x, train=train)
        return x

class TransitionLayer(nn.Module):
    c_out : int  # Output feature size
    act_fn : callable  # Activation function

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = self.act_fn(x)
        x = nn.Conv(self.c_out,
                    kernel_size=(1, 1),
                    kernel_init=densenet_kernel_init,
                    use_bias=False)(x)
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        return x

class DenseNet(nn.Module):
    num_classes : int
    act_fn : callable = nn.relu
    num_layers : tuple = (6, 12, 24, 16)
    bn_size : int = 4
    growth_rate : int = 32

    @nn.compact
    def __call__(self, x, train=True):
        c_hidden = self.growth_rate * self.bn_size  # The start number of hidden channels
        
        x = jnp.pad(x, pad_width=((0, 0), (3, 3), (3, 3), (0, 0)))  # Zero padding
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding='VALID', 
                    kernel_init=densenet_kernel_init)(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = self.act_fn(x)
        x = jnp.pad(x, pad_width=((0, 0), (1, 1), (1, 1), (0, 0)))  # Zero padding for max pooling
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='VALID')

        for block_idx, num_layers in enumerate(self.num_layers):
            x = DenseBlock(num_layers=num_layers,
                           bn_size=self.bn_size,
                           growth_rate=self.growth_rate,
                           act_fn=self.act_fn)(x, train=train)
            c_hidden += num_layers * self.growth_rate
            if block_idx < len(self.num_layers)-1:  # Don't apply transition layer on last block
                x = TransitionLayer(c_out=c_hidden//2,
                                    act_fn=self.act_fn)(x, train=train)
                c_hidden //= 2

        x = nn.BatchNorm()(x, use_running_average=not train)
        x = self.act_fn(x)
        x = x.mean(axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        #x = nn.sigmoid(x)
        return x

class DenseNetPolicy(PolicyNetwork):
    """A DenseNet policy for classification tasks."""

    def __init__(self, num_classes: int, pretrained: bool, model_dir: None, logger: logging.Logger = None):
        if logger is None:
            self._logger = create_logger('DenseNetPolicy')
        else:
            self._logger = logger

        model = DenseNet(num_classes=num_classes)
        params = model.init(jax.random.PRNGKey(0), jnp.ones([1,320, 320,3]), train=False)  # Example input shape
        self.init_params, self.init_batch_stats = params['params'], params['batch_stats']

        self.num_params, format_params_fn = get_params_format_fn(self.init_params)
        self._logger.info('DenseNetPolicy.num_params = {}'.format(self.num_params))
        self._format_params_fn = jax.vmap(format_params_fn)

        #print('model dir : ', model_dir)
        state_dict = checkpoints.restore_checkpoint(ckpt_dir=model_dir, target=None)
        #print('state dict : ', state_dict)
        
        # Transfer all weights except for the final Dense layer ('Dense_0')
        self.transferred_params = {name: state_dict['params'][name] for name in self.init_params.keys() if 'Dense_0' not in name}

        jax.debug.print('init params shape : {}', len(self.init_params))
        jax.debug.print('trans params shape : {}',len(self.transferred_params))

        if num_classes is not None:
            self.transferred_params['Dense_0'] = self.init_params['Dense_0']
    
        # Use the batch stats from the checkpoint if available
        self.transferred_batch_stats = state_dict.get('batch_stats', self.init_batch_stats)

        # Step 1: Extract all parameters
        leaves, _ = jax.tree_util.tree_flatten(self.transferred_params)
        
        # Step 2: Concatenate all parameters into a single vector
        self.flat_transferred_params = jnp.concatenate([p.flatten() for p in leaves])
        
        def forward_fn_train(p, o, batch_stats):
            logits, updates = model.apply(
                {'params': p, 'batch_stats': batch_stats},
                o,
                train=True, mutable=['batch_stats']
            )
            return logits, updates['batch_stats']
        
        self._forward_fn_train = jax.vmap(forward_fn_train)

        def forward_fn(p, o, batch_stats):
            logits = model.apply(
                {'params': p, 'batch_stats': batch_stats},
                o,
                train=False
            )
            return logits, None
        
        self._forward_fn = jax.vmap(forward_fn)

    def get_actions(self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState, train: bool) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        if train:
            #jax.debug.print('train params shape : {}',params.shape)
            #jax.debug.print('train obs shape : {}',t_states.obs.shape)
            logits, batch_stats = self._forward_fn_train(params, t_states.obs, t_states.batch_stats)
            return logits, batch_stats, p_states
        else:
            #jax.debug.print('test params shape : {}',params.shape)
            #jax.debug.print('test obs shape : {}',t_states.obs.shape)
            logits, batch_stats = self._forward_fn(params, t_states.obs, t_states.batch_stats)
            return logits, batch_stats, p_states
        
