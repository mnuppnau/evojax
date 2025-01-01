import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
import flax.linen as nn

import numpy as np
from jax.tree_util import tree_map
from torch.utils import data
from torchvision import datasets
from torchvision.datasets import MNIST
from flax.training import train_state
from flax.training import orbax_utils
import orbax.checkpoint as orbax_cp

import jax
import optax
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms


###############################################################################
#                              KEY HYPERPARAMS
###############################################################################

class BinaryMNISTClassifier(nn.Module):
    """CNN that expects flattened input, then reshapes to [28,28,1]."""
    @nn.compact
    def __call__(self, x, train: bool = True):
        # x has shape [batch, 784]
        # Reshape to [batch, 28, 28, 1]
        x = x.reshape((x.shape[0], 28, 28, 1))

        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(features=64)(x)
        x = nn.relu(x)

        # Output a single logit for binary classification
        x = nn.Dense(features=1)(x)
        return jnp.squeeze(x)  # [batch, 1]

def create_train_state(rng, learning_rate=1e-3):
    """Initialize the model and create the TrainState."""
    model = BinaryMNISTClassifier()
    
    # Initialize model parameters
    dummy_input = jnp.ones((1, 28, 28, 1), jnp.float32)  # Example shape for MNIST
    params = model.init(rng, dummy_input)['params']

    # Define optimizer
    tx = optax.adam(learning_rate=learning_rate)

    # Create a simple train state (model + optimizer)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    return state, model

def binary_cross_entropy_loss(logits, labels):
    """Compute binary cross-entropy from logits and {0,1} labels."""
    # logits: [batch, 1]
    # labels: [batch, 1]
    return optax.sigmoid_binary_cross_entropy(logits, labels).mean()

#def compute_metrics(logits, labels):
#    """Compute binary classification metrics: loss and accuracy."""
#    loss = binary_cross_entropy_loss(logits, labels)
#    preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
#    accuracy = jnp.mean(jnp.squeeze(preds) == jnp.squeeze(labels))
#    return {
#        'loss': loss,
#        'accuracy': accuracy
#    }

@jax.jit
def compute_metrics(logits, labels):
    loss = binary_cross_entropy_loss(logits, labels)
    #print('logits shape:', logits.shape)
    preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
    #print('preds shape:', preds.shape)
    #print('labels shape:', labels.shape)
    acc = jnp.mean(jnp.squeeze(preds) == labels)
    return {
        'loss': loss,
        'accuracy': acc
    }

@jax.jit
def train_step(state, batch_images, batch_labels):
    """Single training step."""
    def loss_fn(params):
        # add batch dimension to labels
        logits = state.apply_fn({'params': params}, batch_images)
        
        loss = binary_cross_entropy_loss(logits, batch_labels)
        #jax.debug.print('logits {} : ', logits)
        #jax.debug.print('labels {} : ', batch_labels)
        return loss

    # Compute gradients
    grads = jax.grad(loss_fn)(state.params)

    # Apply gradient updates
    state = state.apply_gradients(grads=grads)

    # Compute metrics
    logits = state.apply_fn({'params': state.params}, batch_images)
    metrics = compute_metrics(logits, batch_labels)
    return state, metrics


###############################################################################
#                            DATASET HELPERS
###############################################################################
def npy_files():
    return [
        f'/home/gh0st/projects/evojax/mnist-classification/images/{filename}'
        for filename in os.listdir('/home/gh0st/projects/evojax/mnist-classification/images/')
        if filename.endswith('.npy')
    ]

def load_npy_files(files):
    return jnp.array([np.load(f) for f in files])

def split_synthetic_data(data):
    # data of shape (s, b, 28, 28, 1) -> flatten
    data = data.reshape(data.shape[0] * data.shape[1], data.shape[2] * data.shape[3])
    # 80/20 split
    cutoff = int(0.8 * data.shape[0])
    return data[:cutoff], data[cutoff:]

def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))


class RealMNISTDataset(data.Dataset):
    def __init__(self, root, train=True, download=True, transform=None):
        super().__init__()
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform
        )
        # 1 = real
        self.label = 1

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, _ = self.mnist[idx]   # digit label not used
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        return image, self.label


class SyntheticMNISTDataset(data.Dataset):
    def __init__(self, folder_path, transform=None):
        super().__init__()
        self.transform = transform
        self.files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path) 
            if f.endswith('.npy')
        ]
        data_list = []
        for f in self.files:
            batch = np.load(f)  # shape: (batch_size, 28, 28, 1)
            data_list.append(batch)
        self.data = np.concatenate(data_list, axis=0)
        # 0 = synthetic
        self.labels = np.zeros(len(self.data), dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image) 
        label = self.labels[idx]
        return image, label


class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn
        )


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=np.float32))

def train_model(real_images, real_labels, fake_images, fake_labels, real_images_test, real_labels_test, fake_images_test, fake_labels_test, num_epochs=10, batch_size=64, learning_rate=1e-3):
    rng = jax.random.PRNGKey(42)

    # Create train state and model
    state, model = create_train_state(rng, learning_rate=learning_rate)

    # For demonstration, let's pretend we have a function that returns 
    # a batch of real MNIST and a batch of fake MNIST images:
    #
    #   real_images, real_labels = get_real_mnist_batch(batch_size)  # shape: (B,28,28,1), (B,1)
    #   fake_images, fake_labels = get_fake_mnist_batch(batch_size)  # shape: (B,28,28,1), (B,1)
    #
    # You’d combine them or run them separately. For simplicity, let's do separate runs.

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        # Pretend we have N batches per epoch
        for _ in range(200):
            
            
            half_batch = batch_size // 2
            # select a random batch of real and fake images
            real_idx = np.random.choice(len(real_images), half_batch, replace=False)
            fake_idx = np.random.choice(len(fake_images), half_batch, replace=False)

            real_batch_images = real_images[real_idx]
            real_batch_labels = real_labels[real_idx]

            fake_batch_images = fake_images[fake_idx]
            fake_batch_labels = fake_labels[fake_idx]

            batch_images = np.concatenate([real_batch_images, fake_batch_images], axis=0)
            batch_labels = np.concatenate([real_batch_labels, fake_batch_labels], axis=0)

            # Shuffle the batch
            idx = np.random.permutation(batch_size)
            batch_images = batch_images[idx]
            batch_labels = batch_labels[idx]

            state, train_metrics = train_step(state, batch_images, batch_labels)

            epoch_acc += train_metrics['accuracy']
            epoch_loss += train_metrics['loss']

        
        epoch_acc /= 200
        epoch_loss /= 200

        # ---- End of epoch: Validation step ----
        # Combine real and fake test sets
        images_test = jnp.concatenate([real_images_test, fake_images_test], axis=0)
        labels_test = jnp.concatenate([real_labels_test, fake_labels_test], axis=0)
        
        # Evaluate
        val_metrics = evaluate_model(state, images_test, labels_test)
        
        # Print both training and validation metrics
        print(f"Epoch {epoch+1:02d} | "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
          f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")    

    return state, model

def evaluate_model(state, images, labels):
    """Compute metrics on a given dataset (no gradient)."""
    logits = state.apply_fn({'params': state.params}, images)
    return compute_metrics(logits, labels)

def save_model(state, checkpoint_dir="my_checkpoint"):
    """
    Saves the entire TrainState to `checkpoint_dir` using Orbax.
    """
    # Create a Checkpointer that knows how to handle a PyTree (the TrainState).
    checkpointer = orbax_cp.PyTreeCheckpointer()

    save_args = orbax_utils.save_args_from_target(state)
    # Actually save the state (which is a PyTree).
    checkpointer.save(checkpoint_dir, state, save_args=save_args)

#def save_params(params, filename="params.npz"):
#    array_dict = {}
#    for i, (W, b) in enumerate(params):
#        array_dict[f"W_{i}"] = np.array(W)  # jax -> np
#        array_dict[f"b_{i}"] = np.array(b)
#    np.savez(filename, **array_dict)

def save_params(params, filename="params.npz"):                                                                                                                                                                                                                                              
    # params is a list of (W, b)                                                                                                                                                                                                                                                             
    # Each W has shape (n_out, n_in), each b has shape (n_out,)                                                                                                                                                                                                                              
    # (Assuming that's how you stored them in your code)                                                                                                                                                                                                                                     
    array_dict = {}                                                                                                                                                                                                                                                                          
    for i, (W, b) in enumerate(params):                                                                                                                                                                                                                                                      
        array_dict[f"W_{i}"] = np.array(W)  # convert jax array -> numpy                                                                                                                                                                                                                     
        array_dict[f"b_{i}"] = np.array(b)                                                                                                                                                                                                                                                   
    np.savez(filename, **array_dict)    

def load_params(filename="params.npz"):
    with np.load(filename) as data:
        params = [
            (data[f"W_{i}"], data[f"b_{i}"])
            for i in range(len(data.keys()) // 2)
        ]
    return params

###############################################################################
#                                  MAIN
###############################################################################
def main():
    #params = init_network_params(layer_sizes, random.PRNGKey(0))

    # ---------------------- REAL MNIST (Train + Test) ------------------------
    mnist_dataset_train = MNIST(
        '/tmp/mnist/', 
        download=True, 
        train=True, 
        transform=FlattenAndCast()
    )
    mnist_train_images = np.array(
        mnist_dataset_train.data.numpy().reshape(-1, 784), 
        dtype=jnp.float32
    )
    # integer label = 1 for real
    mnist_train_labels = jnp.ones((len(mnist_train_images),), dtype=jnp.int32)

    mnist_dataset_test = MNIST(
        '/tmp/mnist/',
        download=True,
        train=False,
        transform=FlattenAndCast()
    )
    mnist_test_images = np.array(
        mnist_dataset_test.data.numpy().reshape(-1, 784),
        dtype=jnp.float32
    )
    # integer label = 1 for real
    mnist_test_labels = jnp.ones((len(mnist_test_images),), dtype=jnp.int32)

    # ---------------------- SYNTHETIC MNIST (Train + Test) -------------------
    synthetic_files = npy_files()
    synthetic_images = load_npy_files(synthetic_files)  # shape (num_files, batch_size, 28, 28, 1)
    synthetic_train_images, synthetic_test_images = split_synthetic_data(synthetic_images)

    # integer label = 0 for synthetic
    synthetic_train_labels = jnp.zeros((len(synthetic_train_images),), dtype=jnp.int32)
    synthetic_test_labels = jnp.zeros((len(synthetic_test_images),), dtype=jnp.int32)

    print('Real MNIST Train:', mnist_train_images.shape, mnist_train_labels.shape)
    print('Real MNIST Test:', mnist_test_images.shape, mnist_test_labels.shape)
    print('Synthetic MNIST Train:', synthetic_train_images.shape, synthetic_train_labels.shape)
    print('Synthetic MNIST Test:', synthetic_test_images.shape, synthetic_test_labels.shape)

    # ---------------------- TRAINING -----------------------------------------     
    final_state, final_model = train_model(
        real_images=mnist_train_images,
        real_labels=mnist_train_labels,
        fake_images=synthetic_train_images,
        fake_labels=synthetic_train_labels,
        real_images_test=mnist_test_images,
        real_labels_test=mnist_test_labels,
        fake_images_test=synthetic_test_images,
        fake_labels_test=synthetic_test_labels,
        num_epochs=20,
        batch_size=64,
        learning_rate=0.003
    )
    
    save_model(final_state, checkpoint_dir="/home/gh0st/projects/evojax/mnist-classification/models/state")

if __name__ == '__main__':
    main()

