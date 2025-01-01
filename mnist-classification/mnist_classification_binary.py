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

import optax
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms


###############################################################################
#                              KEY HYPERPARAMS
###############################################################################
# For a 2-class classification (real vs. synthetic) we only need final size=2
layer_sizes = [784, 512, 512, 1]
step_size = 0.001
num_epochs = 15
batch_size = 64
n_targets = 1  # we have 2 classes: 0 (synthetic), 1 (real)


###############################################################################
#                            NETWORK INITIALIZATION
###############################################################################
def random_layer_params(m, n, key, scale=1e-2):
    """Return (W, b) for a dense layer with shape (n_out, n_in)."""
    w_key, b_key = random.split(key)
    return (
        scale * random.normal(w_key, (n, m)), 
        scale * random.normal(b_key, (n,))
    )

def init_network_params(sizes, key):
    """Build a list of (W, b) tuples for each layer."""
    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k) 
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]

def relu(x):
    return jnp.maximum(0, x)

def predict(params, image):
    """Forward pass for a single image. Returns log-softmax of shape (2,)."""
    activations = image
    # Hidden layers
    for (w, b) in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)
    # Final layer (2 units), produce log-softmax
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b  # shape (2,)
    return logits - logsumexp(logits)  # log-softmax over 2 classes

# Vectorize + jit-compile
_predict = jit(vmap(predict, in_axes=(None, 0)))


###############################################################################
#                                METRICS & LOSS
###############################################################################
def accuracy(params, images, targets):
    """
    If 'targets' is shape (N,), each entry = 0 or 1,
    then no need to do argmax on 'targets'. We just compare directly.
    """
    preds = _predict(params, images)            # shape (N, 2)
    # print predictions after converting from scientific notation
    preds_print = np.round(np.exp(preds), 4) 
    print('preds : ', preds_print)
    print('targets : ', targets)
    #predicted_class = jnp.argmax(preds, axis=1)  # in {0,1}
    return jnp.mean(jnp.argmax(preds, -1) == targets)

def loss(params, images, targets_onehot):
    """
    Standard cross-entropy:
    preds is log-softmax of shape (N, 2),
    targets_onehot is shape (N, 2).
    """
    preds = _predict(params, images)  # shape (N, 2)
    # cross-entropy = -sum(target * log_prob), then average
    return optax.softmax_cross_entropy(preds, targets_onehot).mean() 

@jit
def update(params, x, y_onehot):
    grads = grad(loss)(params, x, y_onehot)
    return [
        (w - step_size * dw, b - step_size * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]


def print_confusion(params, images, labels):
    preds = _predict(params, images)  # shape (N, 2)
    predicted_class = jnp.argmax(preds, axis=1)  # shape (N,)

    # Make sure shapes align
    print("predicted_class.shape:", predicted_class.shape)
    print("labels.shape:", labels.shape)

    combined = np.stack([labels, predicted_class], axis=1)  # shape (N,2)
    real_real = np.sum((combined[:, 0] == 1) & (combined[:, 1] == 1))
    fake_fake = np.sum((combined[:, 0] == 0) & (combined[:, 1] == 0))
    real_fake = np.sum((combined[:, 0] == 1) & (combined[:, 1] == 0))
    fake_real = np.sum((combined[:, 0] == 0) & (combined[:, 1] == 1))

    print("Confusion matrix:")
    print(f"  real -> real = {real_real}")
    print(f"  real -> fake = {real_fake}")
    print(f"  fake -> fake = {fake_fake}")
    print(f"  fake -> real = {fake_real}")

###############################################################################
#                            DATASET HELPERS
###############################################################################
def npy_files():
    return [
        f'/home/gh0st/projects/evojax/{filename}'
        for filename in os.listdir('/home/gh0st/projects/evojax/')
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
    params = init_network_params(layer_sizes, random.PRNGKey(0))

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
    print("synthetic_files:", synthetic_files)
    # select first 15 files
    synthetic_files = synthetic_files[30:]
    synthetic_images = load_npy_files(synthetic_files)  # shape (num_files, batch_size, 28, 28, 1)
    synthetic_train_images, synthetic_test_images = split_synthetic_data(synthetic_images)

    # integer label = 0 for synthetic
    synthetic_train_labels = jnp.zeros((len(synthetic_train_images),), dtype=jnp.int32)
    synthetic_test_labels = jnp.zeros((len(synthetic_test_images),), dtype=jnp.int32)

    # -------------------- Combine Real & Synthetic for Train/Test ------------
    train_images = np.concatenate(
        [mnist_train_images, synthetic_train_images], axis=0
    )
    train_labels = np.concatenate(
        [mnist_train_labels, synthetic_train_labels], axis=0
    )

    test_images = np.concatenate(
        [mnist_test_images, synthetic_test_images], axis=0
    )
    test_labels = np.concatenate(
        [mnist_test_labels, synthetic_test_labels], axis=0
    )

    # Torch dataset wrapper for training by batches:
    real_dataset_gen = RealMNISTDataset(
        root='/tmp/mnist/', 
        train=True, 
        download=True, 
        transform=FlattenAndCast()
    )
    synthetic_dataset_gen = SyntheticMNISTDataset(
        folder_path='/home/gh0st/projects/evojax/mnist-classification/images/', 
        transform=FlattenAndCast()
    )
    combined_dataset = ConcatDataset([real_dataset_gen, synthetic_dataset_gen])
    training_generator = NumpyLoader(
        combined_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )

    # select a subset of the training data for evaluation, 25 evenly spaced samples
    train_images = train_images[::int(len(train_images) / 25)]
    train_labels = train_labels[::int(len(train_labels) / 25)]

    test_images = test_images[::int(len(test_images) / 25)]
    test_labels = test_labels[::int(len(test_labels) / 25)]

    # select random order of indices to shuffle the training data
    indices_train = np.random.permutation(len(train_images))
    train_images = train_images[indices_train]
    train_labels = train_labels[indices_train]


    indices_test = np.random.permutation(len(test_images))
    test_images = test_images[indices_test]
    test_labels = test_labels[indices_test]
    print("mnist_train_images shape:", mnist_train_images.shape)
    print("synthetic_train_images shape:", synthetic_train_images.shape)
    print("train_labels shape:", train_labels.shape)

    unique_labels, counts = np.unique(train_labels, return_counts=True)
    print("Labels distribution in train set:", unique_labels, counts)
    
    # ------------------------------- TRAINING --------------------------------
    for epoch in range(num_epochs):
        start_time = time.time()

        for x_batch, y_batch in training_generator:
            # x_batch: shape (batch_size, 784)
            # y_batch: shape (batch_size,) with values in {0,1}

            # Convert integer labels to one-hot:
            y_onehot = nn.one_hot(y_batch, num_classes=n_targets)
            # Gradient update
            params = update(params, x_batch, y_onehot)

        epoch_time = time.time() - start_time

        print("train_images for accuracy:", train_images.shape)
        print("train_labels for accuracy:", np.unique(train_labels, return_counts=True))

        # Evaluate
        train_acc = accuracy(params, train_images, train_labels)
        
        print("test_images for accuracy:", test_images.shape)
        print("test_labels for accuracy:", np.unique(test_labels, return_counts=True))

        test_acc = accuracy(params, test_images, test_labels)

        print(f"Epoch {epoch} in {epoch_time:.2f} sec")
        print(f"Training set accuracy = {train_acc:.4f}")
    
        print(f"Test set accuracy     = {test_acc:.4f}")

        print_confusion(params, test_images, test_labels)
    # -------------------------- SAVE PARAMETERS ------------------------------
    save_params(params, filename="params.npz")

if __name__ == '__main__':
    main()

