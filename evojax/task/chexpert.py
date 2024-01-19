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
import torchvision.models as models

from sklearn.metrics import precision_recall_curve
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
        if count >= num_records:
            break

    # Concatenate the lists into arrays or tensors
    images = np.concatenate(images)[:num_records]
    labels = np.concatenate(labels)[:num_records]
    idxs = np.concatenate(idxs)[:num_records]

    return images, labels

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def loss(predictions: jnp.ndarray, targets: jnp.ndarray, weights: jnp.float32 = 1.2) -> jnp.float32:
    return -optax.sigmoid_binary_cross_entropy(predictions, targets).mean()

def precision_recall_curve_jax(y_true, y_scores):

    # Apply sigmoid to convert logits to probabilities
    y_scores = sigmoid(y_scores)

    jax.debug.print('y scores : {}',y_scores)
    # Sort scores and corresponding truth values
    descending_sort_indices = jnp.argsort(y_scores)[::-1]
    sorted_y_true = y_true[descending_sort_indices]
    sorted_y_scores = y_scores[descending_sort_indices]

    # Compute the number of positive labels
    n_positives = jnp.sum(sorted_y_true)

    # Create arrays for true positives and false positives
    tp = jnp.cumsum(sorted_y_true)
    fp = jnp.cumsum(1 - sorted_y_true)

    # Calculate precision and recall for each threshold
    precision = tp / (tp + fp)
    recall = tp / n_positives

    return precision, recall

def accuracy(predictions, targets):
    n_classes = targets.shape[1]
    average_precision_scores = []

    for i in range(n_classes):
        precision, recall = precision_recall_curve_jax(targets[:, i], predictions[:, i])

        # Compute the average precision
        average_precision = jnp.sum((recall[1:] - recall[:-1]) * precision[:-1])
        average_precision_scores.append(average_precision)

    # Return the mean of the average precisions
    return jnp.mean(jnp.array(average_precision_scores))

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

        if test:
            self.data_generator = fetch_dataloader(args, mode='valid')
            data, labels = accumulate_data(self.data_generator,self.num_batches)
        else: 
            self.data_generator = fetch_dataloader(args, mode='train')
            num_records_to_load = 1500
            data, labels = load_subset_of_data(self.data_generator, num_records_to_load)
 
        def reset_fn(key):
            if test:
                batch_data, batch_labels = data, labels
            else:
                batch_data, batch_labels = sample_batch(key,data,labels,self.batch_size)
               
            return CheXpertState(obs=batch_data, labels=batch_labels, batch_stats=self.init_batch_stats)
       
        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            if test:
                #jax.debug.print('Test action : {}',action)
                jax.debug.print('Test labels : {}',state.labels)
                jax.debug.print('Test action shape : {}',action.shape)
                jax.debug.print('Test labels shape : {}',state.labels.shape)
                reward = accuracy(action, state.labels)
            else:
                #jax.debug.print('Train action : {}',action)
                #jax.debug.print('Train labels : {}',state.labels)
                reward = loss(action, state.labels)
                #print('reward : ',reward)
            return state, reward, jnp.ones(())

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> CheXpertState:
        return self._reset_fn(key)

    def step(self, state: TaskState, action: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
