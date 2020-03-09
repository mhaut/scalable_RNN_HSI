import copy
import numpy as np
from operator import truediv
import scipy.io as sio
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix
from sru import *
import sys
import torch
import torch.utils.data as utils

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def read_simple_hsi_image(name):
    data_path = os.path.join(os.getcwd(),'data')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SV':
        data = sio.loadmat(os.path.join(data_path, 'salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'paviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
    else:
        print("NO DATASET")
        exit()
    return data, labels


def read_information_experiment(args):
    dataset, t_percent, num_dim_pca, rstate = args.dataset, args.tpercent, args.pca, args.random_state
    pixels, labels = read_simple_hsi_image(dataset)
    pixels = pixels.reshape(-1, pixels.shape[-1])
    labels = labels.reshape(-1)

    if num_dim_pca != -1: pixels = PCA(n_components=num_dim_pca).fit_transform(pixels)

    pixels = StandardScaler().fit_transform(pixels)
    pixels2 = pixels[labels!=0]
    labels2 = labels[labels!=0]
    labels2 -= 1
    num_class = len(np.unique(labels2))

    x_train, x_test, y_train, y_test = train_test_split(pixels2, labels2, test_size=(1-t_percent), stratify=labels2, random_state=rstate)
    if args.use_val:
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=(1-args.valpercent), stratify=y_test)
    else: 
        x_val = None; y_val = None
        
    return x_train, x_test, x_val, y_train, y_test, y_val, pixels, labels


def prepare_sequences(num_sequences, x_values):
    while x_values.shape[-1] % num_sequences != 0: x_values = x_values[:,:-1]
    x_values_f = copy.deepcopy(x_values); del x_values

    x_values = np.ones((x_values_f.shape[0], num_sequences, int(x_values_f.shape[1]/num_sequences)))*-1000.0
    batch = int(x_values_f.shape[-1]/num_sequences)

    for tstep in range(num_sequences): x_values[:,tstep,:] = x_values_f[:,batch*tstep:batch*(tstep+1)]

    return x_values



def get_loaders(args):
    x_train, x_test, x_val, y_train, y_test, y_val, image, labels = \
                        read_information_experiment(args)
    
    if args.numseq == -1: args.numseq = x_train.shape[-1]
    x_train, y_train = unison_shuffled_copies(x_train, y_train)
    x_test, y_test   = unison_shuffled_copies(x_test, y_test)
    x_train = prepare_sequences(args.numseq, x_train)
    x_test  = prepare_sequences(args.numseq, x_test)
    image   = prepare_sequences(args.numseq, image)

    tensor_x = torch.Tensor(x_train)
    tensor_y = torch.Tensor(y_train).view(-1).long()
    train_hyper = utils.TensorDataset(tensor_x,tensor_y)

    tensor_x = torch.Tensor(x_test)
    tensor_y = torch.Tensor(y_test).view(-1).long()
    test_hyper = utils.TensorDataset(tensor_x, tensor_y)

    if args.use_val:
        tensor_x = torch.Tensor(x_val)
        tensor_y = torch.Tensor(y_val).view(-1).long()
        val_hyper = utils.TensorDataset(tensor_x, tensor_y)
    else:
        val_hyper = None

    tensor_x = torch.Tensor(image)
    tensor_y = torch.Tensor(np.ones((image.shape[0])).astype("uint8")).view(-1).long()
    all_hyper = utils.TensorDataset(tensor_x, tensor_y) # create your datset

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=100, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_hyper, batch_size=2000, shuffle=False, **kwargs)
    val_loader   = torch.utils.data.DataLoader(val_hyper, batch_size=2000, shuffle=False, **kwargs)
    all_loader  = torch.utils.data.DataLoader(all_hyper, batch_size=2000, shuffle=False, **kwargs)

    return train_loader, test_loader, val_loader, all_loader, len(np.unique(labels))-1, image.shape[-1]



def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(y_pred, y_test, labels):
	classification = classification_report(y_test, y_pred, labels=labels)
	oa = accuracy_score(y_test, y_pred)
	confusion = confusion_matrix(y_test, y_pred)
	each_acc, aa = AA_andEachClassAccuracy(confusion)
	kappa = cohen_kappa_score(y_test, y_pred)

	return classification, confusion, list(np.round(np.array([oa, aa, kappa] + list(each_acc)) * 100, 2))
