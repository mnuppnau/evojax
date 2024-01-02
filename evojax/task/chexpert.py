import os
import sys
import jax
import zipfile
import jax.numpy as jnp
import optax
import torch
import torchvision.transforms as T
import pandas as pd
import numpy as np

from collections.abc import Iterator
from jax import random
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils import data
from urllib import request
from typing import Tuple, Generator
from torch.utils.data import Dataset
from flax.struct import dataclass
from jax.tree_util import tree_map
from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState

#def numpy_collate(batch):
#  return tree_map(np.asarray, data.default_collate(batch))

# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=4,
                shuffle=True, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

class ChexpertSmall(Dataset):
    url = 'http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip'
    dir_name = os.path.splitext(os.path.basename(url))[0]  # folder to match the filename
    attr_all_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                      'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                      'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                      'Fracture', 'Support Devices']
    # select only the competition labels
    attr_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    def __init__(self, root, mode='train', transform=None, data_filter=None, mini_data=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        assert mode in ['train', 'valid', 'test']
        self.mode = mode

        # if mode is test; root is path to csv file (in test mode), construct dataset from this csv;
        # if mode is train/valid; root is path to data folder with `train`/`valid` csv file to construct dataset.
        if mode == 'test':
            self.data = pd.read_csv(self.root, keep_default_na=True)
            self.root = '.'  # base path; to be joined to filename in csv file in __getitem__
            self.data[self.attr_names] = pd.DataFrame(np.zeros((len(self.data), len(self.attr_names))))  # attr is vector of 0s under test
        else:
            self._maybe_download_and_extract()
            self._maybe_process(data_filter)

            data_file = os.path.join(self.root, self.dir_name, 'valid.pt' if mode in ['valid', 'vis'] else 'train.pt')
            self.data = torch.load(data_file)

        # store index of the selected attributes in the columns of the data for faster indexing
        self.attr_idxs = [self.data.columns.tolist().index(a) for a in self.attr_names]

    def __getitem__(self, idx):
        # 1. select and load image
        img_path = self.data.iloc[idx, 0]  # 'Path' column is 0
        img = Image.open(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        # 2. select attributes as targets
        attr = self.data.iloc[idx, self.attr_idxs].values.astype(np.float32)
        attr = torch.from_numpy(attr)

        # 3. save index for extracting the patient_id in prediction/eval results as 'CheXpert-v1.0-small/valid/patient64541/study1'
        #    performed using the extract_patient_ids function
        idx = self.data.index[idx]  # idx is based on len(self.data); if we are taking a subset of the data, idx will be relative to len(subset);
                                    # self.data.index(idx) pulls the index in the original dataframe and not the subset

        return img, attr, idx

    def __len__(self):
        return len(self.data)

    def _maybe_download_and_extract(self):
        fpath = os.path.join(self.root, os.path.basename(self.url))
        print('fpath : ', fpath)
        # if data dir does not exist, download file to root and unzip into dir_name
        if not os.path.exists(os.path.join(self.root, self.dir_name)):
            # check if zip file already downloaded
            if not os.path.exists(os.path.join(self.root, os.path.basename(self.url))):
                print('Downloading ' + self.url + ' to ' + fpath)
                def _progress(count, block_size, total_size):
                    sys.stdout.write('\r>> Downloading %s %.1f%%' % (fpath,
                        float(count * block_size) / float(total_size) * 100.0))
                    sys.stdout.flush()
                request.urlretrieve(self.url, fpath, _progress)
                print()
            print('Extracting ' + fpath)
            with zipfile.ZipFile(fpath, 'r') as z:
                z.extractall(self.root)
                if os.path.exists(os.path.join(self.root, self.dir_name, '__MACOSX')):
                    os.rmdir(os.path.join(self.root, self.dir_name, '__MACOSX'))
            os.unlink(fpath)
            print('Dataset extracted.')

    def _maybe_process(self, data_filter):
        # Dataset labels are: blank for unmentioned, 0 for negative, -1 for uncertain, and 1 for positive.
        # Process by:
        #    1. fill NAs (blanks for unmentioned) as 0 (negatives)
        #    2. fill -1 as 1 (U-Ones method described in paper)  # TODO -- setup options for uncertain labels
        #    3. apply attr filters as a dictionary {data_attribute: value_to_keep} e.g. {'Frontal/Lateral': 'Frontal'}

        # check for processed .pt files
        train_file = os.path.join(self.root, self.dir_name, 'train.pt')
        valid_file = os.path.join(self.root, self.dir_name, 'valid.pt')
        if not (os.path.exists(train_file) and os.path.exists(valid_file)):
            # load data and preprocess training data
            valid_df = pd.read_csv(os.path.join(self.root, self.dir_name, 'valid.csv'), keep_default_na=True)
            train_df = self._load_and_preprocess_training_data(os.path.join(self.root, self.dir_name, 'train.csv'), data_filter)

            # save
            torch.save(train_df, train_file)
            torch.save(valid_df, valid_file)

    def _load_and_preprocess_training_data(self, csv_path, data_filter):
        train_df = pd.read_csv(csv_path, keep_default_na=True)

        # 1. fill NAs (blanks for unmentioned) as 0 (negatives)
        # attr columns ['No Finding', ..., 'Support Devices']; note AP/PA remains with NAs for Lateral pictures
        train_df[self.attr_names] = train_df[self.attr_names].fillna(0)

        # 2. fill -1 as 1 (U-Ones method described in paper)  # TODO -- setup options for uncertain labels
        train_df[self.attr_names] = train_df[self.attr_names].replace(-1,1)

        if data_filter is not None:
            # 3. apply attr filters
            # only keep data matching the attribute e.g. df['Frontal/Lateral']=='Frontal'
            for k, v in data_filter.items():
                train_df = train_df[train_df[k]==v]

            with open(os.path.join(os.path.dirname(csv_path), 'processed_training_data_filters.json'), 'w') as f:
                json.dump(data_filter, f)

        return train_df


def fetch_dataloader(args, mode):
    assert mode in ['train', 'valid']

    transforms = T.Compose([
        T.Resize(args.resize) if args.resize else T.Lambda(lambda x: x),
        T.CenterCrop(320 if not args.resize else args.resize),
        lambda x: torch.from_numpy(np.array(x, copy=True)).float().div(255).unsqueeze(0),
        T.Normalize(mean=[0.5330], std=[0.0349]),
        lambda x: x.expand(3, -1, -1),
        T.Lambda(lambda x: x.permute(1, 2, 0))  # Permute dimensions to change to channels last format
    ])

    dataset = ChexpertSmall(args.data_path, mode, transforms, mini_data=args.mini_data)
    print('mode : ', mode)
    print('Inside fetch_dataloader, args batch size = ', args.batch_size)
    print('Inside fetch_dataloader, shuffle = ', mode=='train')

    return DataLoader(dataset, args.batch_size, collate_fn=numpy_collate,shuffle=(mode=='train'), pin_memory=(args.device.type=='cuda'),
                       num_workers=0 if mode=='valid' else 16)  # since evaluating the valid_dataloader is called inside the
                                                              # train_dataloader loop, 0 workers for valid_dataloader avoids
                                                              # forking (cf torch dataloader docs); else memory sharing gets clunky

@dataclass
class CheXpertState(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray
    batch_stats: any

def accumulate_data(generator,num_batches):
    all_data = []
    all_labels = []
    count = 0

    for data, labels, _ in generator:
        all_data.append(data)
        all_labels.append(labels)
        count += 1
        if count > num_batches:
            break

    # Concatenate all data and labels
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_data, all_labels

def sample_batch(key: jnp.ndarray,
                 data: jnp.ndarray,
                 labels: jnp.ndarray,
                 batch_size: int) -> Tuple:
    ix = random.choice(
        key=key, a=data.shape[0], shape=(batch_size,), replace=False)
    return (jnp.take(data, indices=ix, axis=0),
            jnp.take(labels, indices=ix, axis=0))

def load_subset_of_data(data_loader, num_records):
    images, labels, idxs = [], [], []
    count = 0

    for batch_images, batch_labels, batch_idxs in data_loader:
        # Append the data from the current batch to the lists
        images.append(batch_images)
        labels.append(batch_labels)
        idxs.append(batch_idxs)

        # Update the count and check if the desired number of records is reached
        count += len(batch_images)
        print('len batch images : ',len(batch_images))
        if count >= num_records:
            break

    # Concatenate the lists into arrays or tensors
    images = np.concatenate(images)[:num_records]
    labels = np.concatenate(labels)[:num_records]
    idxs = np.concatenate(idxs)[:num_records]

    return images, labels, idxs

def loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.float32:
    #jax.debug.print('targets : {}',targets)
    epsilon = 1e-7  # small constant to prevent log(0)
    clipped_predictions = jnp.clip(predictions, epsilon, 1 - epsilon)
    return -jnp.mean(targets * jnp.log(clipped_predictions) + (1 - targets) * jnp.log(1 - clipped_predictions))

def accuracy(prediction: jnp.ndarray, target: jnp.ndarray, threshold: float = 0.5) -> jnp.float32:
    # Assuming prediction is a probability or has passed through a sigmoid function
    # Threshold the predictions to get binary outputs
    predicted_classes = prediction > threshold

    # Compare the predicted classes with the targets
    correct_predictions = predicted_classes == target

    # Calculate accuracy as the mean of correct predictions
    return jnp.mean(correct_predictions)

def numpy_callback(x):
  # Need to forward-declare the shape & dtype of the expected output.
  result_shape = jax.core.ShapedArray(x.shape, x.dtype)
  return jax.pure_callback(np.sin, result_shape, x)

class CheXpert(VectorizedTask):
    """CheXpert classification task using PyTorch DataLoader."""
   
    def __init__(self, args, batch_stats: dict, test: bool = False):
        self.max_steps = 1
        self.num_batches = 25
        self.init_batch_stats = batch_stats
        self.batch_size = args.batch_size
        # Define observation and action shapes appropriately
        self.obs_shape = tuple([320, 320, 3])
        self.act_shape = tuple([5,])

        # Initialize PyTorch DataLoader
        args.device = torch.device('cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu')
        
        self.train_generator = fetch_dataloader(args, mode='train')
        num_records_to_load = 1400
        train_images, train_labels, train_idxs = load_subset_of_data(self.train_generator, num_records_to_load)

        for images, labels, idx in self.train_generator:
            # Print the shape of images and labels
            print("Shape of train images:", images.shape)
            print("train labels:", labels)
            # Optionally, break after the first batch to avoid printing shapes for all batches
            break

        self.valid_generator = fetch_dataloader(args, mode='valid')
        all_valid_data, all_valid_labels = accumulate_data(self.valid_generator,self.num_batches)
        
        for images, labels, idx in self.valid_generator:
            # Print the shape of images and labels
            print("Shape of valid images:", images.shape)
            print("valid labels:", labels)
            # Optionally, break after the first batch to avoid printing shapes for all batches
            break
       
        def reset_fn(key):
            if test:
                batch_data, batch_labels = all_valid_data, all_valid_labels
            else:
                batch_data, batch_labels = sample_batch(key,train_images,train_labels,self.batch_size)
               
            return CheXpertState(obs=batch_data, labels=batch_labels, batch_stats=self.init_batch_stats)
       
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            if test:
                reward = accuracy(action, state.labels)
            else:
                reward = loss(action, state.labels)
            return state, reward, jnp.ones(())

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> CheXpertState:
        return self._reset_fn(key)

    def step(self, state: TaskState, action: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
