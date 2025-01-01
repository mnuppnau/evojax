import os
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orbax_cp
import optax
import jax.tree_util as tree

from flax.training import train_state
from jax import jit, vmap
from flax import linen as nn
from flax.core.frozen_dict import freeze, unfreeze
from jax.scipy.special import logsumexp
# Disable TensorFlow GPU (optional, based on your environment)
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")


class MLP(nn.Module):
    """
    A simple 3-layer MLP for binary classification.
    features: list of hidden units for each layer.
       e.g. [512, 512, 1] => 3 layers total.
    """
    features: list  # e.g. [512, 512, 1]

    @nn.compact
    def __call__(self, x):
        """
        Args:
          x: shape (batch, 784) if the input is flattened MNIST images.

        Returns:
          A final output of shape (batch, 1), suitable for binary classification.
        """
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)  # kernel shape: (in_features, feat)
            # Apply ReLU on all but the last layer
            if i < len(self.features) - 1:
                x = nn.relu(x)
        return x  # shape (batch, 1)

def load_model(empty_state, checkpoint_dir="my_checkpoint"):
    """
    Loads TrainState from `checkpoint_dir` into `empty_state`.

    :param empty_state: A freshly created TrainState with the same structure as
                        your final trained state (same model, same optimizer).
    :param checkpoint_dir: The directory that contains the checkpoint files.
    :return: The restored TrainState with loaded params and optimizer state.
    """
    checkpointer = orbax_cp.PyTreeCheckpointer()

    # `restore` returns the exact PyTree that was saved.
    restored_state = checkpointer.restore(checkpoint_dir, item=empty_state)
    return restored_state

def load_params_into_flax(model: MLP, rng: jax.random.PRNGKey, paramfile: str):
    """
    Initializes the MLP with dummy data, then loads parameters from an .npz file.
    The .npz file is expected to contain arrays named W_0, b_0, W_1, b_1, etc.
    Each W_i is shape (n_out, n_in) in your manual code, but Flax expects
    (n_in, n_out), so we transpose them.
    """
    # 1. Init a dummy param dict to get the correct structure
    dummy_x = jnp.ones((1, 784), dtype=jnp.float32)
    variables = model.init(rng, dummy_x)  # => {'params': ...}

    # 2. Load arrays from .npz
    loaded = np.load(paramfile)  # => keys: W_0, b_0, W_1, b_1, ...

    # 3. Rebuild the param dict with these weights
    new_params = unfreeze(variables['params'])  # make it mutable

    # Example: if layer_sizes was [784, 512, 512, 1], that's 3 layers:
    #   i=0: 784->512, i=1: 512->512, i=2: 512->1
    # So we expect W_0..W_2, b_0..b_2.
    num_layers = len(model.features)

    for i in range(num_layers):
        # The Dense_i is "Dense_{i}" in the Flax param dict
        # kernel shape in Flax is (in_features, out_features).
        W = loaded[f"W_{i}"]
        b = loaded[f"b_{i}"]

        # Transpose if originally (out_features, in_features)
        W = W.T

        new_params[f"Dense_{i}"]["kernel"] = jnp.array(W)
        new_params[f"Dense_{i}"]["bias"]   = jnp.array(b)

    trained_params = freeze(new_params)
    return trained_params

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

def preprocess_image_batch(image_batch: np.ndarray) -> jnp.ndarray:
    """
    Preprocess a batch of images by:
      - Flattening (if shape is e.g. (N, 28, 28))
      - Normalizing pixel values to [0,1]
      - Casting to float32
    """
    # If images are (N, 28, 28, 1), remove the last dimension
    #if image_batch.ndim == 4 and image_batch.shape[-1] == 1:
    #    image_batch = image_batch.squeeze(-1)
    
    # If images are (N, 28, 28), flatten them to (N, 784).
    # If they're already (N, 784), this will still work correctly.
    #N = image_batch.shape[0]
    # Flatten everything from index 1 onward
    #image_batch = image_batch.reshape(N, -1)

    # Normalize
    #image_batch = image_batch / 255.0

    image_batch = image_batch.reshape(image_batch.shape[0], image_batch.shape[1] * image_batch.shape[2])
    
    # transform with flatten and cast to float32
    image_batch = image_batch.astype(jnp.float32)

    # shuffle the images
    image_batch = jax.random.permutation(jax.random.PRNGKey(0), image_batch)

    # normalize the images
    #image_batch = image_batch / 255.0
    #return jnp.array(image_batch, dtype=jnp.float32)
    return image_batch

def load_params(filename="params.npz"):
    with np.load(filename) as data:
        params = [
            (data[f"W_{i}"], data[f"b_{i}"])
            for i in range(len(data.keys()) // 2)
        ]
    return params

def predict_batch(image_batch: np.ndarray, model_dir: str):
    """
    Predict whether a batch of images are real or synthetic using the trained model.
    Expects:
      - 'params.npz' in model_dir
      - image_batch as a NumPy array of shape (N, 28, 28), (N, 28, 28, 1), or (N, 784).
    """
    # Preprocess images
    image_batch = preprocess_image_batch(image_batch)

    # Create the model
    #model = MLP(features=[512, 512, 2])  # Must match training's layer sizes

    # Load parameters
    rng = jax.random.PRNGKey(0)
    paramfile = os.path.join(model_dir, 'params.npz')
    params = load_params(paramfile)
    #params = load_params_into_flax(model, rng, paramfile)

    # Forward pass
    #logits = model.apply({'params': params}, image_batch)  # shape (N, 1)
    logits = _predict(params, image_batch)
    print('logits:', logits)
    probs = jax.nn.sigmoid(logits).reshape(-1)           # shape (N,)
    print('probs:', probs)

    # Print results
    for i in range(len(image_batch)):
        pred_val = float(probs[i])  # Convert from jnp.array to Python float
        print(f"Image {i} Prediction Value: {pred_val:.4f}")
        if pred_val > 0.5:
            print("  -> The image is predicted to be Real.\n")
        else:
            print("  -> The image is predicted to be Synthetic.\n")

    # Optional: Print overall distribution
    unique, counts = np.unique((probs > 0.5).astype(int), return_counts=True)
    print("Prediction Distribution:")
    for cls, cnt in zip(unique, counts):
        label = "Synthetic" if cls == 0 else "Real"
        print(f"  {label}: {cnt} predictions")


    # predict a single image from the batch and print the prediction
    image = image_batch[1]
    logits = predict(params, image)
    print('logits:', logits)
    probs = jax.nn.sigmoid(logits).reshape(-1)           # shape (N,)
    print('probs:', probs)

class BinaryMNISTClassifierMLP(nn.Module):
    """MLP for binary MNIST classification (real vs. fake)."""
    @nn.compact
    def __call__(self, x):
        # x has shape [batch, 784]
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        # Output a single logit
        x = nn.Dense(features=1)(x)
        return x

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

# ----------------------------
# 2) Create train state
# ----------------------------
def create_train_state(rng, learning_rate=1e-3):
    model = BinaryMNISTClassifier()
    # Dummy input: shape (1, 784)
    dummy_input = jnp.ones((1, 784), jnp.float32)
    params = model.init(rng, dummy_input)['params']
    
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    return state, model
# Example usage:
if __name__ == "__main__":
    model_dir = '/home/gh0st/projects/evojax/mnist-classification/models/state/'  # Directory where the model is saved (containing params.npz)
    #image_batch_path = '/home/gh0st/projects/evojax/mnist-classification/mnist/test_images.npy' 
    # Alternatively, use your own image batch path:
    image_batch_path = '/home/gh0st/projects/evojax/iteration-1000.npy'

    # Load the .npy file containing the batch of images (shape (N, 28, 28), (N, 28, 28, 1), or (N, 784))
    image_batch = np.load(image_batch_path)
    print(f"Loaded image batch of shape {image_batch.shape}")
    #predict_batch(image_batch, model_dir)
    rng = jax.random.PRNGKey(0)
    empty_state, model = create_train_state(rng, learning_rate=1e-3)

    loaded_state = load_model(empty_state, checkpoint_dir=model_dir)

    variables = {'params': loaded_state.params}
    print(tree.tree_structure(loaded_state.params))
    #print(loaded_state.params)
#    loaded_state = train_state.TrainState(
#        apply_fn=model.apply,
#        params=restored_data['params'],
#        tx=empty_state.tx,
#        opt_state=restored_data['opt_state'],
#        step=restored_data['step'],
#    )
#
    print('image batch shape:', image_batch.shape)
    logits = model.apply(variables, image_batch)
    print(logits)
