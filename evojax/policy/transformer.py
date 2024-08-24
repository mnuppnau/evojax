# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of transformer model.

The model is based on: https://github.com/google/flax/tree/main/examples/infogan
"""

import logging
import numpy as np
from typing import Tuple
from typing import Any
from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn

class ModuleList(Module, Generic[M]):
    """
    ## Module list

    This stores a list of modules.
    We needed this for transformer decoder to hold the list of transformer layers.
    """

    # For list of modules
    _submodules: List[M]

    def __init__(self, modules: List[M]):
        """
        Initialize with a list of modules.
        """
        super().__init__()
        self._submodules = modules

    def __getitem__(self, idx: int) -> M:
        """
        ### Get the `idx`-th module
        """
        return self._submodules[idx]

    def __setitem__(self, key, value):
        """
        This is not supported
        """
        raise NotImplementedError

    def __len__(self):
        """
        ### Number of modules
        """
        return len(self._submodules)

    def __getattr__(self, item):
        """
        Override `__getattr__` of `Module`
        """
        return self.__dict__[item]

    def __setattr__(self, key, value):
        """
        Override `__setattr__` of `Module`
        """
        self.__dict__[key] = value

    def _clear_params(self):
        """
        ### Clear all parameters
        """
        self._params = {}
        for sm in self._submodules:
            sm._clear_params()

    def get_params(self):
        """
        ### Get all parameters
        """
        params = self._params
        for i, sm in enumerate(self._submodules):
            for name, value in sm.get_params().items():
                params[f'{i}/{name}'] = value
        return params

    def _set_param(self, param_path: List[str], value: jnp.ndarray):
        """
        ### Set a parameter
        """
        self._submodules[int(param_path[0])]._set_param(param_path[1:], value)

class Embedding(Module):
    """
    <a id="Embedding"></a>

    ## Embedding layer

    This maintains embeddings by id.
    """

    def __init__(self, rnd_key: jax.random.PRNGKey, n_embeddings: int, n_dim: int):
        """
        * `rnd_key` is the PRNG state
        * `n_embeddings` is the number of embeddings
        * `n_dim` is the size of an embedding
        """
        super().__init__()
        # Embeddings are initialized from $\mathcal{N}(0, 1)$
        self.embeddings = jax.random.normal(rnd_key, (n_embeddings, n_dim))

    def __call__(self, ids: jnp.ndarray):
        """
        Return the embeddings for the given ids
        """
        return self.embeddings[ids, :]

class PositionalEncoding(Module):
    """
    Positional encoding layer
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = jnp.arange(max_len)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
        pe = jnp.zeros((max_len, d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe[jnp.newaxis, :, :]

    def __call__(self, x):
        return x + self.pe[:, :x.shape[1], :]

class Linear(Module):
    """
    <a id="Linear"></a>

    ## Linear Layer

    This is a simple linear layer with a weight matrix and a bias vector
    """

    def __init__(self, rnd_key: jax.random.PRNGKey, in_features: int, out_features: int):
        """
        * `rnd_key` is the PRNG state
        * `in_features` is the number of features in the input
        * `out_features` is the number of features in the output
        """
        super().__init__()
        #print('initializing linear layer')
        # Initialize weights to
        # $$\mathcal{U}\Bigg(-\frac{1}{\sqrt{d_{in}}}, \frac{1}{\sqrt{d_{in}}} \Bigg)$$
        rnd_range = 1 / in_features ** 0.5
        self.weight = jax.random.uniform(rnd_key, (in_features, out_features),
                                         minval=-rnd_range, maxval=rnd_range)
        #print('self weight shape in linear init : ', self.weight.shape)
        # Initialize the biases to $0$
        self.bias = jnp.zeros((out_features,))

    def __call__(self, x: jnp.ndarray):
        # Multiply by weights and add the bias
        #print('x shape in linear: ', x.shape)
        #print('weight shape in linear: ', self.weight.shape)
        return jnp.matmul(x, self.weight) + self.bias

class Dense(Module):
    """
    <a id="Dense"></a>

    ## Dense Layer

    This is a simple dense layer with a weight matrix and a bias vector
    """

    def __init__(self, rnd_key: jax.random.PRNGKey, in_features: int, out_features: int):
        """
        * `rnd_key` is the PRNG state
        * `in_features` is the number of features in the input
        * `out_features` is the number of features in the output
        """
        super().__init__()
        #print('initializing linear layer')
        # Initialize weights to
        # $$\mathcal{U}\Bigg(-\frac{1}{\sqrt{d_{in}}}, \frac{1}{\sqrt{d_{in}}} \Bigg)$$
        rnd_range = 1 / in_features ** 0.5
        self.weight = jax.random.uniform(rnd_key, (in_features, out_features),
                                         minval=-rnd_range, maxval=rnd_range)
        #print('self weight shape in linear init : ', self.weight.shape)
        # Initialize the biases to $0$
        self.bias = jnp.zeros((out_features,))

    def __call__(self, x: jnp.ndarray):
        # Multiply by weights and add the bias
        #print('x shape in linear: ', x.shape)
        #print('weight shape in linear: ', self.weight.shape)
        return jnp.matmul(x, self.weight) + self.bias

class LayerNorm(Module):
    r"""
    <a id="LayerNormalization"></a>

    ## Layer Normalization

    This implements the the layer normalization from the paper
    [Layer Normalization](https://papers.labml.ai/paper/1607.06450).

    When input $X \in \mathbb{R}^{L \times C}$ is a sequence of embeddings,
    where $C$ is the number of channels, $L$ is the length of the sequence.
    $\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$.
    $$\text{LN}(X) = \gamma
    \frac{X - \underset{C}{\mathbb{E}}[X]}{\sqrt{\underset{C}{Var}[X] + \epsilon}}
    + \beta$$

    This is based on
    [our PyTorch implementation](https://nn.labml.ai/normalization/layer_norm/index.html).
    """

    def __init__(self, normalized_shape: Union[Tuple[int], List[int]], *,
                 eps: float = 1e-5, elementwise_affine: bool = True):
        r"""
        * `normalized_shape` $S$ is the shape of the elements (except the batch).
         The input should then be
         $X \in \mathbb{R}^{* \times S[0] \times S[1] \times ... \times S[n]}$
        * `eps` is $\epsilon$, used in $\sqrt{Var[X] + \epsilon}$ for numerical stability
        * `elementwise_affine` is whether to scale and shift the normalized value
        """
        super().__init__()

        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.normalized_shape = tuple(normalized_shape)

        # Create parameters for $\gamma$ and $\beta$ for gain and bias
        if elementwise_affine:
            self.gain = jnp.ones(normalized_shape)
            self.bias = jnp.zeros(normalized_shape)

    def __call__(self, x: jnp.ndarray):
        # Sanity check to make sure the shapes match
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]

        # The exes to calculate the mean and variance on
        axes = [-(i + 1) for i in range(len(self.normalized_shape))]
        # Calculate the mean of all elements;
        # i.e. the means for each element $\mathbb{E}[X]$
        mean = x.mean(axis=axes, keepdims=True)
        # Calculate the squared mean of all elements;
        # i.e. the means for each element $\mathbb{E}[X^2]$
        mean_2 = (x ** 2).mean(axis=axes, keepdims=True)
        # Variance of all element $Var[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$
        var = mean_2 - mean ** 2
        # Normalize $$\hat{X} = \frac{X - \mathbb{E}[X]}{\sqrt{Var[X] + \epsilon}}$$
        x_norm = (x - mean) / (var + self.eps) ** 0.5

        # Scale and shift $$\text{LN}(x) = \gamma \hat{X} + \beta$$
        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias

        #
        return x_norm

class PrepareForMultiHeadAttention(Module):
    """
    <a id="PrepareMHA"></a>

    ## Prepare for multi-head attention

    This module does a linear transformation and splits the vector into given
    number of heads for multi-head attention.
    This is used to transform **key**, **query**, and **value** vectors.
    """

    def __init__(self, rnd_key: jax.random.PRNGKey, d_model: int, heads: int, d_k: int):
        super().__init__()
        # Linear layer for linear transform
        self.linear = Linear(rnd_key, d_model, heads * d_k)
        # Number of heads
        self.heads = heads
        # Number of dimensions in vectors in each head
        self.d_k = d_k

    def __call__(self, x: jnp.ndarray):
        # Input has shape `[seq_len, batch_size, d_model]` or `[batch_size, d_model]`.
        # We apply the linear transformation to the last dimension and split that into
        # the heads.
        head_shape = x.shape[:-1]

        # Linear transform
        x = self.linear(x)

        # Split last dimension into heads

        x = x.reshape(*head_shape, self.heads, self.d_k)

        # Output has shape `[seq_len, batch_size, heads, d_k]` or `[batch_size, d_model]`
        return x

class MultiHeadAttention(Module):
    r"""
    <a id="MHA"></a>

    ## Multi-Head Attention Module

    This computes scaled multi-headed attention from
    the paper [Attention Is All You Need](https://papers.labml.ai/paper/1706.03762)
    for given `query`, `key` and `value` vectors.

    $$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$

    In simple terms, it finds keys that matches the query, and gets the values of
     those keys.

    It uses dot-product of query and key as the indicator of how matching they are.
    Before taking the $softmax$ the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$.
    This is done to avoid large dot-product values causing softmax to
    give very small gradients when $d_k$ is large.

    Softmax is calculated along the axis of of the sequence (or time) for keys.

    This is based on
    [our PyTorch implementation](https://nn.labml.ai/transformers/mha.html#MHA).
    """

    def __init__(self, rnd_key: jax.random.PRNGKey, heads: int, d_model: int):
        """
        * `rnd_key` is the PRNG state
        * `heads` is the number of heads.
        * `d_model` is the number of features in the `query`, `key` and `value` vectors.
        """

        super().__init__()

        # Split the PRNG state
        _, *rnd_keys = jax.random.split(rnd_key, 5)

        # Number of features per head
        self.d_k = d_model // heads
        # Number of heads
        self.heads = heads

        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        self.query = PrepareForMultiHeadAttention(rnd_keys[0], d_model, heads, self.d_k)
        self.key = PrepareForMultiHeadAttention(rnd_keys[1], d_model, heads, self.d_k)
        self.value = PrepareForMultiHeadAttention(rnd_keys[2], d_model, heads, self.d_k)

        # Output layer
        self.output = Linear(rnd_keys[3], d_model, d_model)
        # Scaling factor before the softmax
        self.scale = 1 / self.d_k ** 0.5

    def __call__(self, *,
                 query: jnp.ndarray,
                 key: jnp.ndarray,
                 value: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None):
        """
        `query`, `key` and `value` are the tensors that store
        collection of *query*, *key* and *value* vectors.
        They have shape `[seq_len, d_model]`.

        `mask` has shape `[seq_len, seq_len]` and
        `mask[i, j]` indicates whether query at position `i` can see key-value at position `j`.
        """

        #print('query shape in MHA: ', len(query))
        # Get sequence length
        seq_len = 16 #len(query)

        #jax.debug.print('query shape in MHA {}: ', query.shape)
        #jax.debug.print('key shape in MHA {}: ', key.shape)
        if mask is not None:
            # Check mask shape
            assert mask.shape[0] == query.shape[0]
            assert mask.shape[1] == key.shape[0]

            # Same mask applied to all heads.
            mask = mask[:, :, None]


        # Prepare `query`, `key` and `value` for attention computation.
        # These will then have shape `[seq_len, heads, d_k]`.
        #query = jnp.squeeze(self.query(query), axis=0)
        #key = jnp.squeeze(self.key(key), axis=0)
        #value = jnp.squeeze(self.value(value), axis=0)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Compute attention scores $Q K^\top$.
        # This gives a tensor of shape `[seq_len, seq_len, heads]`.
        # $$S_{ijh} = \sum_d Q_{ihd} K_{jhd}$$
        scores = jnp.einsum('ihd,jhd->ijh', query, key)

        # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
        scores *= self.scale

        # Apply mask
        if mask is not None:
            scores = scores + (mask == 0) * float('-inf')

        # $softmax$ attention along the key sequence dimension
        # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = jax.nn.softmax(scores, axis=1)

        # Multiply by values
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
        x = jnp.einsum("ijh,jhd->ihd", attn, value)

        #print('x shape in MHA before concat: ', x.shape)
        #print('seq_len in MHA before concat: ', seq_len)
        # Concatenate multiple heads
        x = x.reshape(seq_len, -1)

        #print('x shape in MHA: ', x.shape)
        # Output layer
        return self.output(x)

class FeedForward(Module):
    """
    <a id="FFN"></a>

    ## Position-wise Feed-Forward layer

    This is based on
    [our PyTorch implementation](https://nn.labml.ai/transformers/feed_forward.html).
    """

    def __init__(self, rnd_key: jax.random.PRNGKey, d_model: int, d_ff: int,
                 activation=jax.nn.relu):
        """
        * `rnd_key` is the PRNG state
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `activation` is the activation function $f$
        """
        super().__init__()
        # Split the PRNG state
        _, *rnd_keys = jax.random.split(rnd_key, 5)

        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer1 = Linear(rnd_keys[0], d_model, d_ff)
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer2 = Linear(rnd_keys[1], d_ff, d_model)
        # Activation function $f$
        self.activation = activation

    def __call__(self, x: jnp.ndarray):
        # $f(x W_1 + b_1)$
        x = self.activation(self.layer1(x))
        # $f(x W_1 + b_1) W_2 + b_2$
        return self.layer2(x)

class TransformerLayer(Module):
    """
    <a id="TransformerLayer"></a>

    ## Transformer Layer

    This is a transformer layer with multi-head attention and a position-wise feed-forward layer.
    We use pre-layer layer normalization.
    """

    def __init__(self,
                 d_model: int,
                 self_attn: MultiHeadAttention,
                 feed_forward: FeedForward):
        """
        * `d_model` is the token embedding size
        * `self_attn` is the self attention module
        * `feed_forward` is the feed forward module
        """
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm_self_attn = LayerNorm([d_model])
        self.norm_ff = LayerNorm([d_model])

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray):
        # Normalize the vectors before doing self attention
        z = self.norm_self_attn(x)
        # Run through self attention, i.e. keys and values are from self
        self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
        x = x + self_attn

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Pass through the feed-forward network
        ff = self.feed_forward(z)
        # Add the feed-forward results
        x = x + ff
        #
        return x

class CrossEntropyLoss(Module):
    """
    <a id="CrossEntropyLoss"></a>

    ## Cross Entropy Loss
    """

    def __init__(self):
        super().__init__()

        # Use `jax.vmap` to vectorize the loss function
        self._loss_vmap = jax.vmap(self._loss, in_axes=(0, 0,))

    def _loss(self, output: jnp.ndarray, target: jnp.ndarray):
        # $$- \sum_k y_k \log \hat{y}_k$$
        return -jax.nn.log_softmax(output)[target]

    def __call__(self, output: jnp.ndarray, target: jnp.ndarray):
        """
        * `output` is the model outputs of shape `[seq_len, n_vocab]`
        * `target` is the target of shape `[seq_len]`
        """

        # Use the vectorized loss function and calculate the mean.
        #
        # We could have used a for loop to calculate the losses but using vmap is about 10X faster
        return self._loss_vmap(output, target).mean()

class TransformerGenerator(Module):
    """
    ## InfoGAN Generator with Transformer
    This is an adaptation of the autoregressive transformer for use as an InfoGAN generator.
    """
    layers: ModuleList[TransformerLayer]
    def __init__(self, rnd_key: jax.random.PRNGKey, latent_dim: int, d_model: int, n_layers: int, heads: int, d_ff: int, output_dim: int, seq_len: int):
        """
        * `rnd_key` is the PRNG state
        * `latent_dim` is the dimension of the input latent vector
        * `d_model` is the number of features in the transformer
        * `n_layers` is the number of transformer layers
        * `heads` is the number of attention heads
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `output_dim` is the dimension of the output (e.g., image size)
        * `seq_len` is the sequence length to generate
        """
        super().__init__()
        self.gen_d_model = d_model
        self.gen_seq_len = seq_len
        
        # Latent projection
        rnd_key, proj_key = jax.random.split(rnd_key)
        self.gen_latent_projection = Linear(proj_key, latent_dim, seq_len * d_model)
        
        # Positional encoding
        self.gen_positional_encoding = PositionalEncoding(d_model, seq_len)
        
        # Transformer layers
        layers = []
        for _ in range(n_layers):
            rnd_key, layer_key = jax.random.split(rnd_key)
            attn = MultiHeadAttention(layer_key, heads, d_model)
            ffn = FeedForward(layer_key, d_model, d_ff)
            layers.append(TransformerLayer(d_model, attn, ffn))
        self.gen_layers = ModuleList(layers)
        
        # Output projection
        rnd_key, out_key = jax.random.split(rnd_key)
        self.gen_norm = LayerNorm([d_model])
        self.gen_output = Linear(out_key, d_model, output_dim)
    
    def __call__(self, z: jnp.ndarray):
        # Project latent vector to sequence
        print('z shape in generator before projection: ', z.shape)
        x = self.gen_latent_projection(z)
        print('x shape in generator after projection: ', x.shape)
        x = x.reshape(-1, self.gen_seq_len, self.gen_d_model)
        print('x shape in generator after reshape: ', x.shape) 
        # Add positional encoding
        x = self.gen_positional_encoding(x)
       
        print('x shape in generator after positional encoding: ', x.shape)
        # No need for mask in generator
        #mask = None
        mask = jnp.tril(jnp.ones((self.gen_seq_len, self.gen_seq_len), bool))
        #print('x shape in generator before layers: ', x.shape)
        x = x.reshape(x.shape[1], x.shape[2])
        # Apply transformer layers
        for layer in self.gen_layers:
            x = layer(x, mask)

        # Final normalization
        x = self.gen_norm(x)
        # Final output projection
        print('x shape in generator before output: ', x.shape)
        x = self.gen_output(x)
        return x

class TransformerDiscriminator(Module):
    """
    ## InfoGAN Discriminator with Transformer
    This is an adaptation of the autoregressive transformer for use as an InfoGAN discriminator.
    """
    layers: ModuleList[TransformerLayer]
    def __init__(self, rnd_key: jax.random.PRNGKey, input_dim: int, d_model: int, n_layers: int, heads: int, d_ff: int, latent_dim: int, n_control_variables: int, seq_len: int):
        super().__init__()
        self.disc_d_model = d_model
        self.disc_n_control_variables = n_control_variables
        self.disc_seq_len = seq_len
       
        self.loss_func = CrossEntropyLoss()
        # Input projection
        rnd_key, proj_key = jax.random.split(rnd_key)
        self.disc_input_projection = Linear(proj_key, input_dim, d_model)
        
        # Positional encoding
        self.disc_positional_encoding = PositionalEncoding(d_model, seq_len)
        
        # Transformer layers
        layers = []
        for _ in range(n_layers):
            rnd_key, layer_key = jax.random.split(rnd_key)
            attn = MultiHeadAttention(layer_key, heads, d_model)
            ffn = FeedForward(layer_key, d_model, d_ff)
            layers.append(TransformerLayer(d_model, attn, ffn))
        self.disc_layers = ModuleList(layers)
        
        # Final normalization layer
        self.disc_norm = LayerNorm([d_model])
        
        # Output for real/fake classification
        rnd_key, out_key = jax.random.split(rnd_key)
        #self.disc_output = Linear(out_key, d_model, self.disc_seq_len)
        self.disc_output = Dense(out_key, input_dim, 1)
        # Q network for latent code prediction
        rnd_key, q_key1, q_key2 = jax.random.split(rnd_key, 3)
        self.disc_q_net1 = Linear(q_key1, d_model, d_model // 2)
        self.disc_q_net2 = Linear(q_key2, d_model // 2, self.disc_n_control_variables)
    
    def __call__(self, x: jnp.ndarray):
        #print('x shape in discriminator before projection: ', x.shape)
        #print('x shape in discriminator before projection: ', x.shape)
        # Project input to d_model dimension
        x = self.disc_input_projection(x)

        #print('x shape in discriminator after projection: ', x.shape) 
        #x = x.reshape(1,1,x.shape[0])
        x = jnp.atleast_2d(x)
        # Add positional encoding
        #print('x shape in discriminator before positional encoding: ', x.shape)
        x = self.disc_positional_encoding(x)
        #print('x shape in discriminator after positional encoding: ', x.shape) 
        # Apply transformer layers
        x = x.reshape(x.shape[1], x.shape[2])
        #print('x shape after flatten: ', x.shape)
        for layer in self.disc_layers:
            x = layer(x, mask=None)

        print('x shape in discriminator after layers: ', x.shape)
        # Final normalization
        x = self.disc_norm(x)
        print('x shape in discriminator after norm: ', x.shape) 
        # Global average pooling
        y = jnp.mean(x, axis=1)
       
        print('x shape in discriminator after pooling: ', x.shape)
        # Output for real/fake classification
        real_fake = self.disc_output(y)
        real_fake = jax.nn.sigmoid(real_fake)
        print('real_fake shape in discriminator: ', real_fake.shape)
        # Q network output for latent code prediction
        q = nn.leaky_relu(self.disc_q_net1(x), 0.2)
        #print('q shape in discriminator after q_net1: ', q.shape)
        q = self.disc_q_net2(q)
        #print('q shape in discriminator after q_net2: ', q.shape)
        q_mean = jnp.mean(q, axis=0)
        return real_fake, q_mean

class TransformerPolicy(PolicyNetwork):
    """A transformer policy that generates synthetic movie reviews."""

    def __init__(self,
                 model_name_gen : str,
                 model_name_disc : str,
                 model_class_gen : nn.Module,
                 model_class_disc : nn.Module,
                 optimizer_name : str,
                 optimizer_hparams_gen : dict,
                 optimizer_hparams_disc : dict,
                 sample_batch : Any,
                 seed=41):

        if logger is None:
            self._logger = create_logger('TransformerPolicy')
        else:
            self._logger = logger

        self.loss = {'generator': [], 'discriminator': []}
        self.seed = seed
        self.key, self.key_gen, self.key_disc, self.key_latent, self.latent, self.image_shape = self.generate_keys(sample_batch)
        self.model_name_gen = model_name_gen
        self.model_name_disc = model_name_disc
        self.model_class_gen = model_class_gen
        self.model_class_disc = model_class_disc
        self.optimizer_name = optimizer_name
        self.optimizer_hparams_gen = optimizer_hparams_gen
        self.optimizer_hparams_disc = optimizer_hparams_disc

        self.max_sequence_length = 16
        # Define model parameters
        num_control_variables = 3 + 5 + 3 + 5 + 4
        latent_dim = 16 + num_control_variables
        d_model = 32
        n_layers = 3
        heads = 4
        d_ff = 4*d_model
        output_dim = 30522
        seq_len = 16

        self.latent_dim = 16
        self.batch_size = 32
        
   
        self.model_gen = self.model_class_gen(
            rnd_key=self.key_gen,
            latent_dim=latent_dim,
            d_model=d_model,
            n_layers=n_layers,
            heads=heads,
            d_ff=d_ff,
            output_dim=output_dim,
            seq_len=seq_len
        )

        self.batch_forward_gen = jax.vmap(self.model_gen.purify(self.model_gen.__call__), in_axes=(None, 0))
        self.sample_input = jnp.array(sample_batch[0]['input_ids'], dtype=jnp.int32)

        #print('sample input shape : ', self.sample_input.shape)
        input_dim = tokenizer.vocab_size

        #print('input dim shape : ', input_dim)

        self.model_disc = self.model_class_disc(
            rnd_key=self.key_disc,
            input_dim=16,
            d_model=d_model,
            n_layers=n_layers,
            heads=heads,
            d_ff=d_ff,
            latent_dim=latent_dim,
            n_control_variables=num_control_variables,
            seq_len=seq_len
        )
                
        self.batch_forward_disc = jax.vmap(self.model_disc.purify(self.model_disc.__call__), in_axes=(None, 0))
       
        self.get_loss_disc_real = jax.vmap(self.model_disc.purify(self.model_disc.get_loss_real), in_axes=(None, 0))
        self.get_loss_disc_fake = jax.vmap(self.model_disc.purify(self.model_disc.get_loss_fake), in_axes=(None, 0))
        self.get_loss_disc = jax.vmap(self.model_disc.purify(self.model_disc.get_loss), in_axes=(None, 0))
        
        # Create jitted training and eval functions
        #self.create_functions()
        # Initialize model
        params_gen = self.init_model_gen()
        params_disc = self.init_model_disc()

        self.num_params_gen, self.format_params_fn_gen = get_params_format_fn(params_gen)
        self.num_params_disc, self.format_params_fn_disc = get_params_format_fn(params_disc)

        self._logger.info(
            'TransformerPolicy.num_params_gen = {}'.format(self.num_params_gen))
        self._logger.info(
            'TransformerPolicy.num_params_disc = {}'.format(self.num_params_disc))

        self._format_params_fn_gen = jax.vmap(self.format_params_fn_gen)
        self._format_params_fn_disc = jax.vmap(self.format_params_fn_disc)

        self._forward_fn_gen = forward_fn_gen
        self._forward_fn_disc = forward_fn_disc 


    def generate_keys(self, exmp_imgs):
        key  = jrandom.PRNGKey(seed=self.seed)

        key, key_gen, key_disc, key_latent = jax.random.split(key, 4)
        
        # Retrieve shapes for generator and discriminator input.
        noise = jax.random.normal(key, (100, 32))
        c = jnp.tile(jnp.arange(10), 10)
        c = jax.nn.one_hot(c, 10)
        latent = jnp.concatenate([noise, c], axis=-1)
        image_shape = len(exmp_imgs)

        return key, key_gen, key_disc, key_latent, latent, image_shape 

    def init_model_gen(self):
        key1, key2 = jax.random.split(self.key_latent, 2)
        # Latent vector z
        z = jnp.ones((self.batch_size, self.latent_dim), dtype=jnp.float32)
    
        # Control variables c
        # Assuming control variables consist of sentiment, genre, length, rating, and aspects
        # Sentiment: 3 classes, Genre: 5 classes, Length: 3 classes, Rating: 5 classes, Aspects: 4 binary
        num_control_variables = 3 + 5 + 3 + 5 + 4
        c = jnp.ones((self.batch_size, num_control_variables), dtype=jnp.float32)
      
        # For z, let's use values from a standard normal distribution
        #z = jax.random.normal(key1, shape=(self.batch_size, self.latent_dim), dtype=jnp.float32)

        # For c, let's use uniformly distributed values between -1 and 1
        #num_control_variables = 3 + 5 + 3 + 5 + 4 
        #c = jax.random.uniform(key2, shape=(self.batch_size, num_control_variables), 
        #               dtype=jnp.float32, minval=-1, maxval=1)

        # Combine z and c
        zc = jnp.concatenate([z, c], axis=-1)
   
        # Initialize the parameters
        _ = self.batch_forward_gen(self.model_gen.get_params(), zc)

        self.init_params_gen = self.model_gen.get_params()
        #jax.debug.print('init params gen : {}', self.init_params_gen)
        #total_params = 0
        #for key, value in self.init_params_gen.items():
        #    total_params += numpy.prod(value.shape) 

        return self.init_params_gen

    def init_model_disc(self):
        # Initialize the parameters
        _ = self.batch_forward_disc(self.model_disc.get_params(), self.sample_input)

        self.init_params_disc = self.model_disc.get_params()
        return self.init_params_disc 


    def forward_fn_gen(self, params_g, params_d):

    
    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        return self._forward_fn(params, t_states.obs), p_states
