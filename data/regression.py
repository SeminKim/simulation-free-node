import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd


def onehot_encode_cat_feature(X, cat_var_idx_list):
    """
    Apply one-hot encoding to the categorical variable(s) in the feature set,
        specified by the index list.
    """
    # select numerical features
    X_num = np.delete(arr=X, obj=cat_var_idx_list, axis=1)
    # select categorical features
    X_cat = X[:, cat_var_idx_list]
    X_onehot_cat = []
    for col in range(X_cat.shape[1]):
        X_onehot_cat.append(pd.get_dummies(X_cat[:, col], drop_first=True))
    X_onehot_cat = np.concatenate(X_onehot_cat, axis=1).astype(np.float32)
    dim_cat = X_onehot_cat.shape[1]  # number of categorical feature(s)
    X = np.concatenate([X_num, X_onehot_cat], axis=1)
    return X, dim_cat


def _get_index_train_test_path(data_directory_path, split_num, train=True):
    """
       Method to generate the path containing the training/test split for the given
       split number (generally from 1 to 20).
       @param split_num      Split number for which the data has to be generated
       @param train          Is true if the data is training data. Else false.
       @return path          Path of the file containing the requried data
    """
    if train:
        return os.path.join(data_directory_path, "index_train_" + str(split_num) + ".txt")
    else:
        return os.path.join(data_directory_path, "index_test_" + str(split_num) + ".txt")


def preprocess_uci_feature_set(X, data_path):
    """
    Obtain preprocessed UCI feature set X (one-hot encoding applied for categorical variable)
        and dimension of one-hot encoded categorical variables.
    """
    dim_cat = 0
    task_name = data_path.split('/')[-1]
    if task_name == 'bostonHousing':
        X, dim_cat = onehot_encode_cat_feature(X, [3])
    elif task_name == 'energy':
        X, dim_cat = onehot_encode_cat_feature(X, [4, 6, 7])
    elif task_name == 'naval-propulsion-plant':
        X, dim_cat = onehot_encode_cat_feature(X, [0, 1, 8, 11])
    return X, dim_cat


class UCI(Dataset):
    def __init__(self, data_path, task, split=0, train_split='train', normalize_x=True, normalize_y=True, train_ratio=0.6):
        data_dir = os.path.join(data_path, task, 'data')
        data_file = os.path.join(data_dir, 'data.txt')
        index_feature_file = os.path.join(data_dir, 'index_features.txt')
        index_target_file = os.path.join(data_dir, 'index_target.txt')
        n_splits_file = os.path.join(data_dir, 'n_splits.txt')

        data = np.loadtxt(data_file)
        index_features = np.loadtxt(index_feature_file)
        index_target = np.loadtxt(index_target_file)

        X = data[:, [int(i) for i in index_features.tolist()]].astype(np.float32)
        y = data[:, int(index_target.tolist())].astype(np.float32)

        X, dim_cat = preprocess_uci_feature_set(X=X, data_path=data_path)
        self.dim_cat = dim_cat

        # load the indices of the train and test sets
        index_train = np.loadtxt(_get_index_train_test_path(data_dir, split, train=True))
        index_test = np.loadtxt(_get_index_train_test_path(data_dir, split, train=False))

        # read in data files with indices
        x_train = X[[int(i) for i in index_train.tolist()]]
        y_train = y[[int(i) for i in index_train.tolist()]].reshape(-1, 1)
        x_test = X[[int(i) for i in index_test.tolist()]]
        y_test = y[[int(i) for i in index_test.tolist()]].reshape(-1, 1)

        # split train set further into train and validation set for hyperparameter tuning
        num_training_examples = int(train_ratio * x_train.shape[0])
        x_val = x_train[num_training_examples:, :]
        y_val = y_train[num_training_examples:]
        x_train = x_train[0:num_training_examples, :]
        y_train = y_train[0:num_training_examples]

        self.x_train = x_train if type(x_train) is torch.Tensor else torch.from_numpy(x_train)
        self.y_train = y_train if type(y_train) is torch.Tensor else torch.from_numpy(y_train)
        self.x_test = x_test if type(x_test) is torch.Tensor else torch.from_numpy(x_test)
        self.y_test = y_test if type(y_test) is torch.Tensor else torch.from_numpy(y_test)
        self.x_val = x_val if type(x_val) is torch.Tensor else torch.from_numpy(x_val)
        self.y_val = y_val if type(y_val) is torch.Tensor else torch.from_numpy(y_val)

        self.train_n_samples = x_train.shape[0]
        self.train_dim_x = self.x_train.shape[1]  # dimension of training data input
        self.train_dim_y = self.y_train.shape[1]  # dimension of training regression output

        self.test_n_samples = x_test.shape[0]
        self.test_dim_x = self.x_test.shape[1]  # dimension of testing data input
        self.test_dim_y = self.y_test.shape[1]  # dimension of testing regression output

        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.scaler_x, self.scaler_y = None, None

        if self.normalize_x:
            self.normalize_train_test_x()
        if self.normalize_y:
            self.normalize_train_test_y()

        self.return_dataset(train_split)

    def normalize_train_test_x(self):
        """
        When self.dim_cat > 0, we have one-hot encoded number of categorical variables,
            on which we don't conduct standardization. They are arranged as the last
            columns of the feature set.
        """
        self.scaler_x = StandardScaler(with_mean=True, with_std=True)
        if self.dim_cat == 0:
            self.x_train = torch.from_numpy(
                self.scaler_x.fit_transform(self.x_train).astype(np.float32))
            self.x_test = torch.from_numpy(
                self.scaler_x.transform(self.x_test).astype(np.float32))
            self.x_val = torch.from_numpy(
                self.scaler_x.transform(self.x_val).astype(np.float32))
        else:  # self.dim_cat > 0
            x_train_num, x_train_cat = self.x_train[:, :-self.dim_cat], self.x_train[:, -self.dim_cat:]
            x_test_num, x_test_cat = self.x_test[:, :-self.dim_cat], self.x_test[:, -self.dim_cat:]
            x_val_num, x_val_cat = self.x_val[:, :-self.dim_cat], self.x_val[:, -self.dim_cat:]
            x_train_num = torch.from_numpy(
                self.scaler_x.fit_transform(x_train_num).astype(np.float32))
            x_test_num = torch.from_numpy(
                self.scaler_x.transform(x_test_num).astype(np.float32))
            x_val_num = torch.from_numpy(
                self.scaler_x.transform(x_val_num).astype(np.float32))
            self.x_train = torch.from_numpy(np.concatenate([x_train_num, x_train_cat], axis=1))
            self.x_test = torch.from_numpy(np.concatenate([x_test_num, x_test_cat], axis=1))
            self.x_val = torch.from_numpy(np.concatenate([x_val_num, x_val_cat], axis=1))

    def normalize_train_test_y(self):
        self.scaler_y = StandardScaler(with_mean=True, with_std=True)
        self.y_train = torch.from_numpy(
            self.scaler_y.fit_transform(self.y_train).astype(np.float32)
        )
        self.y_test = torch.from_numpy(
            self.scaler_y.transform(self.y_test).astype(np.float32)
        )
        self.y_val = torch.from_numpy(
            self.scaler_y.transform(self.y_val).astype(np.float32)
        )

    def return_dataset(self, split="train"):
        if split == "train":
            self.data = self.x_train.cuda()
            self.target = self.y_train.cuda()
        elif split == "val":
            self.data = self.x_val.cuda()
            self.target = self.y_val.cuda()
        else:
            self.data = self.x_test.cuda()
            self.target = self.y_test.cuda()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def get_UCI_datasets(data_path, task, split, keys=('train', 'val', 'test')):
    return (UCI(data_path, task, split, train_split=key) for key in keys)
