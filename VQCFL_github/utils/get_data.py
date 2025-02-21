import os

import torch
from torch.utils.data import Dataset,DataLoader
import pickle
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms


# MNIST

def get_1shard(ds,row_0:int,digit:int,samples:int):
    row = row_0
    shard = list()
    while len(shard)<samples:
        if ds.train_labels[row]==digit:
            shard.append(ds.train_data[row].numpy())
        row +=1
    return row,shard


# 创建每个客户端的数据碎片
# 客户端数目、训练样本、测试样本
def create_MNIST_ds_1shard_per_client(n_clients, samples_train, samples_test):
    MNIST_train = datasets.MNIST(root="./data", train=True, download=True)
    MNIST_test = datasets.MNIST(root="./data", train=False, download=True)

    shards_train,shards_test=[],[]
    labels = []

    for i in range(10):
        row_train,row_test = 0,0
        for j in range(10):
            row_train, shard_train=get_1shard(
                MNIST_train, row_train, i, samples_train
            )
            row_test, shard_test = get_1shard(
                MNIST_test, row_test, i, samples_test
            )
            shards_train.append([shard_train])
            shards_test.append([shard_test])

            labels +=[[i]]
    #     数据
    X_train = np.array(shards_train)
    X_test = np.array(shards_test)
    # 对应标签
    y_train = labels
    y_test = y_train

    folder = "./data/"
    train_path = f"MNIST_shard_train_{n_clients}_{samples_train}.pkl"

    with open(folder + train_path, "wb") as output:
        pickle.dump((X_train, y_train), output)

    test_path = f"MNIST_shard_test_{n_clients}_{samples_test}.pkl"
    with open(folder + test_path, "wb") as output:
        pickle.dump((X_test, y_test), output)

# 数据集 每一类数字当前的索引  现在的主要数字是几 主要数字的个数 其他数字的个数
def get_actual_1shard_train(ds,idxs,cur_class,other_class,main_samples,samples):
    shard = list()
    labels= list()
    while len(shard)<main_samples:
        if ds.train_labels[idxs[cur_class]]==cur_class:
            shard.append(ds.train_data[idxs[cur_class]].numpy())
            labels.append(cur_class)
        idxs[cur_class] += 1
    # 每个类 取后两个类的各a张图片 即 (curclass+1)%10 和(curclass+2)%10
    a =(samples-main_samples)/2
    while len(shard)<main_samples+a:
        # next_class = (cur_class+1)%10
        next_class = (other_class + 1) % 10
        if ds.train_labels[idxs[next_class]]==next_class:
            shard.append(ds.train_data[idxs[next_class]].numpy())
            labels.append(next_class)
        idxs[next_class] += 1

    while len(shard)<samples:
        # next_class = (cur_class+2)%10
        next_class = (other_class+ 5) % 10
        if ds.train_labels[idxs[next_class]]==next_class:
            shard.append(ds.train_data[idxs[next_class]].numpy())
            labels.append(next_class)
        idxs[next_class] += 1

    return shard,labels

def get_actual_1shard_test(ds,idxs,cur_class,other_class,samples):
    shard = list()
    labels= list()

    # 每个类 取后两个类的各40张张图片 即 (curclass+1)%10 和(curclass+5)%10

    while len(shard)<samples/2:
        # next_class = (cur_class+1)%10
        next_class = (other_class + 1) % 10
        if ds.train_labels[idxs[next_class]]==next_class:
            shard.append(ds.train_data[idxs[next_class]].numpy())
            labels.append(next_class)
        idxs[next_class] += 1

    while len(shard)<samples:

        next_class = (other_class+ 5) % 10
        if ds.train_labels[idxs[next_class]]==next_class:
            shard.append(ds.train_data[idxs[next_class]].numpy())
            labels.append(next_class)
        idxs[next_class] += 1

    return shard,labels



def create_MNIST_same_size_per_client(n_clients:int, samples_train:int, samples_test:int,n_classes:int, alpha:float):
    MNIST_train = datasets.MNIST(root="./data", train=True, download=True)
    MNIST_test = datasets.MNIST(root="./data", train=False, download=True)

    print(f"len test:{len(MNIST_test)}")
    print(f"len train:{len(MNIST_train)}")



    shards_train, shards_test = [], []
    shards_train_labels,shards_test_labels = [],[]
    # 500*10
    matrix = np.zeros((n_clients, n_classes)).astype(float)
    matrix = np.random.dirichlet([alpha] * 10, size=n_clients)
    # 用来测试少量数据
    # test_matrix = np.zeros((n_clients, n_classes)).astype(float)
    # for i in range(n_clients):
    #
    #     for j in range(n_classes):
    #         # test_matrix[i][j] = 0.1
    #         if alpha < 0.000001:
    #             matrix[i][j] = 0.1
    #         elif j == int(t / 10):
    #             matrix[i][j] = alpha + (1 - alpha) / 10
    #         else:
    #             matrix[i][j] = (1 - alpha) / 10

# 给每个客户端分配数据

#    这里获得的是长度为100的列表 每一个元素 对应着数据和标签  按照 0 ，1 ，2 排序
#     每个数字到达的索引在哪里
    train_idxs = [0]*10
    test_idxs = [0]*10
    for i in range(n_clients):

        shard_train, shard_train_labels = get_actual_1shard(
            MNIST_train, train_idxs,i,matrix,samples_train,
        )
        shard_test, shard_test_labels = get_actual_1shard(
            MNIST_test, test_idxs, i, matrix, samples_test,
        )
        shards_train.append([shard_train])
        shards_train_labels.append([shard_train_labels])
        shards_test.append([shard_test])
        shards_test_labels.append([shard_test_labels])


    #数据
    X_train = np.array(shards_train)
    X_test = np.array(shards_test)
    # 对应标签
    y_train = np.array(shards_train_labels)
    y_test = np.array(shards_test_labels)

    folder = "./data/"

    train_path = f"MNIST_alpha{alpha}_train_{n_clients}_{samples_train}.pkl"

    with open(folder + train_path, "wb") as output:
        pickle.dump((X_train, y_train), output)

    test_path = f"MNIST_alpha{alpha}_test_{n_clients}_{samples_test}.pkl"

    with open(folder + test_path, "wb") as output:
        pickle.dump((X_test, y_test), output)


def get_actual_1shard(dataset, idxs,client,matrix,n_samples):
    shard = list()
    labels = list()
    sum = 0
    for i in range(len(matrix[0])):
        # 标签为i的图片需要的数目
        if i!= len(matrix[0])-1:
            num = int((matrix[client][i]+0.0000001) * n_samples)
            sum+= num
        else:
            num = max(0,n_samples-sum)
        for j in range(num):
            while dataset.train_labels[idxs[i]]!=i:
                idxs[i]+=1
                if idxs[i]>=len(dataset):
                    idxs[i] = 0
            shard.append(dataset.train_data[idxs[i]].numpy())
            labels.append(i)
            idxs[i]+=1
            if idxs[i] >= len(dataset):
                idxs[i] = 0

    return shard, labels




def clients_set_MNIST_shard(file_name, n_clients, batch_size=50, shuffle=True):
    """Download for all the clients their respective dataset"""
    print(file_name)

    list_dl = list()
    for k in range(n_clients):
        dataset_object = MnistShardDataset(file_name, k)
        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )
        list_dl.append(dataset_dl)

    return list_dl



# 将MNIST pkl 转化为 torch  dataset
class MnistShardDataset(Dataset):

    def __init__(self,file_path,k):
        with open(file_path,"rb") as pickle_file:
            dataset = pickle.load(pickle_file)
            self.features = np.vstack(dataset[0][k])
            self.labels = np.hstack(dataset[1][k])



    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 3D input 1x28x28
        x = torch.Tensor([self.features[idx]]) / 255
        y = torch.LongTensor([self.labels[idx]])[0]
        # print("feature len :",len(self.features))
        # print("labels len :", len(self.labels[0]))
        # y = torch.LongTensor([self.labels[0][idx]])
        return x, y















# cifar10 .....

def con_partition_dataset(
    dataset,
    file_name: str,
    matrix,
    n_clients: int,
    n_classes: int,
    train: bool,
):
    """Partition dataset into `n_clients`.
        Each client i has matrix[k, i] of data of class k"""

    list_clients_X = [[] for i in range(n_clients)]
    list_clients_y = [[] for i in range(n_clients)]
    if train:
        n_samples = [500] * n_clients

    elif not train:
        n_samples = [100] * n_clients

    list_idx = []
    for k in range(n_classes):
        idx_k = np.where(np.array(dataset.targets) == k)[0]
        list_idx += [idx_k]

    for idx_client, n_sample in enumerate(n_samples):

        clients_idx_i = []
        client_samples = 0

        for k in range(n_classes):

            if k < 9:
                samples_digit = int((matrix[idx_client, k] + 0.0000001) * n_sample)
            elif k == 9:
                samples_digit = n_sample - client_samples
            client_samples += samples_digit

            clients_idx_i = np.concatenate(
                (clients_idx_i, np.random.choice(list_idx[k], samples_digit, replace=False))
            )

        clients_idx_i = clients_idx_i.astype(int)

        for idx_sample in clients_idx_i:
            list_clients_X[idx_client] += [dataset.data[idx_sample]]
            list_clients_y[idx_client] += [dataset.targets[idx_sample]]

        list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

    folder = "./data/"
    with open(folder + file_name, "wb") as output:
        pickle.dump((list_clients_X, list_clients_y), output)



def con_fashion_partition_dataset(
    dataset,
    file_name: str,
    matrix,
    n_clients: int,
    n_classes: int,
    train: bool,
):
    """Partition dataset into `n_clients`.
        Each client i has matrix[k, i] of data of class k"""
    # image ,label = dataset[0]
    # print(image.shape)
    list_clients_X = [[] for i in range(n_clients)]
    list_clients_y = [[] for i in range(n_clients)]
    if train:
        n_samples = [500] * n_clients

    elif not train:
        n_samples = [100] * n_clients

    list_idx = []
    for k in range(n_classes):
        idx_k = np.where(np.array(dataset.targets) == k)[0]
        list_idx += [idx_k]
    for idx_client, n_sample in enumerate(n_samples):

        clients_idx_i = []
        client_samples = 0

        for k in range(n_classes):

            if k < 9:
                samples_digit = int((matrix[idx_client, k] + 0.0000001) * n_sample)
            elif k == 9:
                samples_digit = n_sample - client_samples
            client_samples += samples_digit

            clients_idx_i = np.concatenate(
                (clients_idx_i, np.random.choice(list_idx[k], samples_digit, replace=False))
            )

        clients_idx_i = clients_idx_i.astype(int)

        for idx_sample in clients_idx_i:
            image,label = dataset[idx_sample]

            list_clients_X[idx_client] +=[image]
            list_clients_y[idx_client] +=[label]
        # tensor 是1*28*28的转不了

        list_clients_X[idx_client] = np.stack(list_clients_X[idx_client])
    # print(list_clients_X[0].shape)
    print(len(list_clients_X[0]))
    print(list_clients_X[0].shape)

    folder = "./data/"
    with open(folder + file_name, "wb") as output:
        pickle.dump((list_clients_X, list_clients_y), output)

def partition_CIFAR_dataset(
    dataset,
    file_name: str,
    balanced: bool,
    matrix,
    n_clients: int,
    n_classes: int,
    train: bool,
):
    """Partition dataset into `n_clients`.
    Each client i has matrix[k, i] of data of class k"""

    list_clients_X = [[] for i in range(n_clients)]
    list_clients_y = [[] for i in range(n_clients)]
    client_label_count = [[] for i in range(n_clients)]
    n_samples = []
    if balanced and train:
        n_samples = [1000] * n_clients
        # for i in range(n_clients):
        #     n_samples.append((i+1)*100)
    elif balanced and not train:
        n_samples = [100] * n_clients
        # n_samples = [50] * n_clients

    list_idx = []
    for k in range(n_classes):

        idx_k = np.where(np.array(dataset.targets) == k)[0]
        list_idx += [idx_k]

    for idx_client, n_sample in enumerate(n_samples):

        clients_idx_i = []
        client_samples = 0

        for k in range(n_classes):

            if k < 9:
                samples_digit = int((matrix[idx_client, k]+0.0000001) * n_sample)
            elif k == 9:
                samples_digit = n_sample - client_samples

            client_label_count[idx_client].append(samples_digit)
            client_samples += samples_digit

            clients_idx_i = np.concatenate(
                (clients_idx_i, np.random.choice(list_idx[k], samples_digit,replace=False))
            )

        clients_idx_i = clients_idx_i.astype(int)

        for idx_sample in clients_idx_i:

            list_clients_X[idx_client] += [dataset.data[idx_sample]]
            list_clients_y[idx_client] += [dataset.targets[idx_sample]]

        list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

    folder = "./data/"
    with open(folder + file_name, "wb") as output:
        pickle.dump((list_clients_X, list_clients_y), output)
    return client_label_count


def partition_fashion_dataset_biye(
    dataset,
    file_name: str,
    cnt_strategy:int,
    matrix,
    n_clients: int,
    n_classes: int,
    train: bool,
):

    list_clients_X = [[] for i in range(n_clients)]
    list_clients_y = [[] for i in range(n_clients)]
    client_label_count = [[] for i in range(n_clients)]
    n_samples = []

    if cnt_strategy != 1 and train:
        if n_clients>=100:
            n_samples = [400]*n_clients
        else:
            n_samples = [1000]*n_clients
    elif cnt_strategy==1 and train:
        n_samples = []

        if n_clients>=100:
            for i in range(n_clients):
                if i<30:
                    n_samples.append(100)
                elif i<80:
                    n_samples.append(500)
                else:
                    n_samples.append(1000)
        else:
            for i in range(n_clients):
                n_samples.append((i + 1) * 100)

    elif cnt_strategy != 1 and not train:

        if n_clients>=100:
            n_samples = [50]*n_clients
        else:
            n_samples = [200]*n_clients
    elif cnt_strategy==1 and not train:
        n_samples = []
        if n_clients >= 100:
            for i in range(n_clients):
                if i < 30:
                    n_samples.append(50)
                elif i < 80:
                    n_samples.append(100)
                else:
                    n_samples.append(200)
        else:
            for i in range(n_clients):
                n_samples.append(100)

    list_idx = []
    for k in range(n_classes):
        idx_k = np.where(np.array(dataset.targets) == k)[0]
        list_idx += [idx_k]
    for idx_client, n_sample in enumerate(n_samples):

        clients_idx_i = []
        client_samples = 0

        for k in range(n_classes):

            if k < 9:
                samples_digit = int((matrix[idx_client, k] + 0.0000001) * n_sample)
            elif k == 9:
                samples_digit = n_sample - client_samples
            client_samples += samples_digit
            client_label_count[idx_client].append(samples_digit)
            clients_idx_i = np.concatenate(
                (clients_idx_i, np.random.choice(list_idx[k], samples_digit, replace=False))
            )

        clients_idx_i = clients_idx_i.astype(int)

        for idx_sample in clients_idx_i:
            image, label = dataset[idx_sample]

            list_clients_X[idx_client] += [image]
            list_clients_y[idx_client] += [label]
        # tensor 是1*28*28的转不了

        list_clients_X[idx_client] = np.stack(list_clients_X[idx_client])
    # print(list_clients_X[0].shape)
    print(len(list_clients_X[0]))
    print(list_clients_X[0].shape)

    folder = "./data/"
    with open(folder + file_name, "wb") as output:
        pickle.dump((list_clients_X, list_clients_y), output)

    return client_label_count


def partition_CIFAR_dataset_biye(
    dataset,
    file_name: str,
    cnt_strategy:int,
    matrix,
    n_clients: int,
    n_classes: int,
    samplesNum,
    train: bool,
):


    list_clients_X = [[] for i in range(n_clients)]
    list_clients_y = [[] for i in range(n_clients)]
    client_label_count = [[] for i in range(n_clients)]
    n_samples = []

    if cnt_strategy != 1 and train:
        if n_clients>=20:
            n_samples = [samplesNum]*n_clients
        else:
            n_samples = [1000]*n_clients
    elif cnt_strategy==1 and train:
        n_samples = []

        if n_clients>=100:
            for i in range(n_clients):
                if i<30:
                    n_samples.append(100)
                elif i<80:
                    n_samples.append(500)
                else:
                    n_samples.append(1000)
        else:
            for i in range(n_clients):
                n_samples.append((i + 1) * 100)

    elif cnt_strategy != 1 and not train:

        if n_clients>=100:
            n_samples = [int(samplesNum * 0.25)]*n_clients
        else:
            n_samples = [200]*n_clients
    elif cnt_strategy==1 and not train:
        n_samples = []
        if n_clients >= 100:
            for i in range(n_clients):
                if i < 30:
                    n_samples.append(50)
                elif i < 80:
                    n_samples.append(100)
                else:
                    n_samples.append(200)
        else:
            for i in range(n_clients):
                n_samples.append(100)


    list_idx = []
    for k in range(n_classes):

        idx_k = np.where(np.array(dataset.targets) == k)[0]
        list_idx += [idx_k]

    for idx_client, n_sample in enumerate(n_samples):

        clients_idx_i = []
        client_samples = 0

        for k in range(n_classes):

            if k < 9:
                samples_digit = int((matrix[idx_client, k]+0.0000001) * n_sample)
            elif k == 9:
                samples_digit = n_sample - client_samples

            client_label_count[idx_client].append(samples_digit)
            client_samples += samples_digit

            clients_idx_i = np.concatenate(
                (clients_idx_i, np.random.choice(list_idx[k], samples_digit,replace=False))
            )

        clients_idx_i = clients_idx_i.astype(int)

        for idx_sample in clients_idx_i:

            list_clients_X[idx_client] += [dataset.data[idx_sample]]
            list_clients_y[idx_client] += [dataset.targets[idx_sample]]

        list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

    folder = "./data/"
    with open(folder + file_name, "wb") as output:
        pickle.dump((list_clients_X, list_clients_y), output)
    return client_label_count

def partition_MNIST_dataset_biye(
    dataset,
    file_name: str,
    cnt_strategy:int,
    matrix,
    n_clients: int,
    n_classes: int,
    train: bool,
):


    list_clients_X = [[] for i in range(n_clients)]
    list_clients_y = [[] for i in range(n_clients)]
    client_label_count = [[] for i in range(n_clients)]
    n_samples = []

    if cnt_strategy != 1 and train:
        if n_clients>=20:
            n_samples = [400]*n_clients
        else:
            n_samples = [1000]*n_clients
    elif cnt_strategy==1 and train:
        n_samples = []

        if n_clients>=100:
            for i in range(n_clients):
                if i<30:
                    n_samples.append(100)
                elif i<80:
                    n_samples.append(500)
                else:
                    n_samples.append(1000)
        else:
            for i in range(n_clients):
                n_samples.append((i + 1) * 100)

    elif cnt_strategy != 1 and not train:

        if n_clients>=100:
            n_samples = [50]*n_clients
        else:
            n_samples = [200]*n_clients
    elif cnt_strategy==1 and not train:
        n_samples = []
        if n_clients >= 100:
            for i in range(n_clients):
                if i < 30:
                    n_samples.append(50)
                elif i < 80:
                    n_samples.append(100)
                else:
                    n_samples.append(200)
        else:
            for i in range(n_clients):
                n_samples.append(100)


    list_idx = []
    for k in range(n_classes):

        idx_k = np.where(np.array(dataset.targets) == k)[0]
        list_idx += [idx_k]

    for idx_client, n_sample in enumerate(n_samples):

        clients_idx_i = []
        client_samples = 0

        for k in range(n_classes):

            if k < 9:
                samples_digit = int((matrix[idx_client, k]+0.0000001) * n_sample)
            elif k == 9:
                samples_digit = n_sample - client_samples

            client_label_count[idx_client].append(samples_digit)
            client_samples += samples_digit

            clients_idx_i = np.concatenate(
                (clients_idx_i, np.random.choice(list_idx[k], samples_digit,replace=False))
            )

        clients_idx_i = clients_idx_i.astype(int)

        for idx_sample in clients_idx_i:

            list_clients_X[idx_client] += [dataset.data[idx_sample]]
            list_clients_y[idx_client] += [dataset.targets[idx_sample]]

        list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

    folder = "./data/"
    with open(folder + file_name, "wb") as output:
        pickle.dump((list_clients_X, list_clients_y), output)
    return client_label_count



def create_CIFAR10_dirichlet(
    dataset_name: str,
    balanced: bool,
    alpha: float,
    n_clients: int,
    n_classes: int,
):
    """Create a CIFAR dataset partitioned according to a
    dirichilet distribution Dir(alpha)"""

    from numpy.random import dirichlet

    matrix = dirichlet([alpha] * n_classes, size=n_clients)


    CIFAR10_train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    CIFAR10_test = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    file_name_train = f"{dataset_name}_train_alpha{alpha}_{n_clients}.pkl"
    client_label_count = partition_CIFAR_dataset(
        CIFAR10_train,
        file_name_train,
        balanced,
        matrix,
        n_clients,
        n_classes,
        True,
    )

    file_name_test = f"{dataset_name}_test_alpha{alpha}_{n_clients}.pkl"
    partition_CIFAR_dataset(
        CIFAR10_test,
        file_name_test,
        balanced,
        matrix,
        n_clients,
        n_classes,
        False,
    )
    return client_label_count


def create_CIFAR10_biye(
    dataset: str,
    alpha: float,
    n_clients: int,
    cnt_strategy:int,
    n_classes:int,
    samplesNum,
        p,
):
    """Create a CIFAR dataset partitioned according to a
    dirichilet distribution Dir(alpha)"""

    from numpy.random import dirichlet


    matrix = dirichlet([alpha] * n_classes, size=n_clients)

    if cnt_strategy == 2:
        import random
        matrix = np.zeros((n_clients, n_classes))
        for i in range(n_clients):
            k1 = random.randint(0, 9)
            k2 = random.randint(0, 9)
            matrix[i][k1] = 0.5
            matrix[i][k2] = 0.5


    CIFAR10_train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    CIFAR10_test = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    file_name_train = f"{dataset}_train_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}.pkl"
    client_label_count = partition_CIFAR_dataset_biye(
        CIFAR10_train,
        file_name_train,
        cnt_strategy,
        matrix,
        n_clients,
        n_classes,
        samplesNum,
        True,
    )

    file_name_test1 = f"{dataset}_test_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}_test_{0}.pkl"
    partition_CIFAR_dataset_biye(
        CIFAR10_test,
        file_name_test1,
        cnt_strategy,
        matrix,
        n_clients,
        n_classes,
        samplesNum,
        False,
    )
    matrix2 = np.full((n_clients, n_classes), 0.1)
    file_name_test2 = f"{dataset}_test_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}_test_{1}.pkl"
    partition_CIFAR_dataset_biye(
        CIFAR10_test,
        file_name_test2,
        cnt_strategy,
        matrix2,
        n_clients,
        n_classes,
        samplesNum,
        False,
    )

    return client_label_count

def create_MNIST_biye(
    dataset: str,
    alpha: float,
    n_clients: int,
    cnt_strategy:int,
    n_classes:int,
        p,
):
    """Create a CIFAR dataset partitioned according to a
    dirichilet distribution Dir(alpha)"""

    from numpy.random import dirichlet

    matrix = dirichlet([alpha] * n_classes, size=n_clients)

    if cnt_strategy == 2:
        import random
        matrix = np.zeros((n_clients, n_classes))
        for i in range(n_clients):
            k1 = random.randint(0, 9)
            k2 = random.randint(0, 9)
            matrix[i][k1] = 0.5
            matrix[i][k2] = 0.5

    CIFAR10_train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    CIFAR10_test = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    file_name_train = f"{dataset}_train_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}.pkl"
    client_label_count = partition_MNIST_dataset_biye(
        CIFAR10_train,
        file_name_train,
        cnt_strategy,
        matrix,
        n_clients,
        n_classes,
        True,
    )

    file_name_test1 = f"{dataset}_test_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}_test_{0}.pkl"
    partition_MNIST_dataset_biye(
        CIFAR10_test,
        file_name_test1,
        cnt_strategy,
        matrix,
        n_clients,
        n_classes,
        False,
    )
    matrix2 = np.full((n_clients, n_classes), 0.1)
    file_name_test2 = f"{dataset}_test_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}_test_{1}.pkl"
    partition_MNIST_dataset_biye(
        CIFAR10_test,
        file_name_test2,
        cnt_strategy,
        matrix2,
        n_clients,
        n_classes,
        False,
    )

    return client_label_count



def create_CIFAR10_iid(
    dataset_name: str,
    balanced: bool,
    n_clients: int,
    n_classes: int,
):
    """Create a CIFAR dataset iid """


    # matrix = dirichlet([alpha] * n_classes, size=n_clients)
    matrix = np.zeros((n_clients,n_classes)).astype(float)
    for i in range(n_clients):
        for j in range(n_classes):
            matrix[i][j] = 1/n_classes


    CIFAR10_train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    CIFAR10_test = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    file_name_train = f"{dataset_name}_train_{n_clients}.pkl"


    partition_CIFAR_dataset(
        CIFAR10_train,
        file_name_train,
        balanced,
        matrix,
        n_clients,
        n_classes,
        True,
    )

    file_name_test = f"{dataset_name}_test_{n_clients}.pkl"
    partition_CIFAR_dataset(
        CIFAR10_test,
        file_name_test,
        balanced,
        matrix,
        n_clients,
        n_classes,
        False,
    )

def create_CIFAR10_pathological(
    dataset_name: str,
    balanced: bool,
    alpha: float,
    n_clients: int,
    n_classes: int,
):
    """Create a CIFAR dataset partitioned according to a
    pathological alpha : (1-alpha)"""


    # matrix = dirichlet([alpha] * n_classes, size=n_clients)
    matrix = np.zeros((n_clients,n_classes)).astype(float)

    # 用来测试少量数据
    test_matrix = np.zeros((n_clients, n_classes)).astype(float)
    for i in range(n_clients):
        if i>=100:
            t = i-100
        else:
            t = i
        for j in range(n_classes):

            test_matrix[i][j] = 0.1

            if alpha < 0.000001:
                matrix[i][j] = 0.1
            elif j == int(t% 10):
                matrix[i][j] = alpha + (1-alpha)/10
            else:
                matrix[i][j] = (1 - alpha)/10

    CIFAR10_train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    CIFAR10_test = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    file_name_train = f"{dataset_name}_train_alpha{alpha}_{n_clients}.pkl"
    balanced=True
    partition_CIFAR_dataset(
        CIFAR10_train,
        file_name_train,
        balanced,
        matrix,
        n_clients,
        n_classes,
        True,
    )

    file_name_test = f"{dataset_name}_test_alpha{alpha}_{n_clients}.pkl"
    if dataset_name[:8] == "difCIFAR":
        matrix = test_matrix
    partition_CIFAR_dataset(
        CIFAR10_test,
        file_name_test,
        balanced,
        matrix,
        n_clients,
        n_classes,
        False,
    )

class CIFARDataset(Dataset):
    """Convert the CIFAR pkl file into a Pytorch Dataset"""

    def __init__(self, file_path: str, k: int):

        dataset = pickle.load(open(file_path, "rb"))

        self.X = dataset[0][k]
        self.y = np.array(dataset[1][k])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):

        # 3D input 32x32x3
        x = torch.Tensor(self.X[idx]).permute(2, 0, 1) / 255
        x = (x - 0.5) / 0.5
        y = self.y[idx]

        return x, y

class MNISTDataset(Dataset):
    """Convert the CIFAR pkl file into a Pytorch Dataset"""

    def __init__(self, file_path: str, k: int):

        dataset = pickle.load(open(file_path, "rb"))

        self.X = dataset[0][k]
        self.y = np.array(dataset[1][k])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):

        # 3D input 32x32x1
        x = torch.Tensor(self.X[idx]).permute(2, 0, 1) / 255
        x = (x - 0.5) / 0.5
        y = self.y[idx]

        return x, y

class DigitsLoadDataset(Dataset):

    def __init__(self,file_path,k):
        with open(file_path,"rb") as pickle_file:
            dataset = pickle.load(pickle_file)
            self.features = dataset[0][k]
            self.labels = dataset[1][k]



    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 3D input 3x28x28
        x = torch.Tensor(self.features[idx]) / 255
        # y = torch.LongTensor([self.labels[idx]])[0]
        return x, self.labels[idx]


class featureDataset(Dataset):
    """Convert the feature list into a Pytorch Dataset"""

    def __init__(self, feature_list: [], label_list:[]):

        self.X = feature_list
        self.y = label_list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx][0], self.y[idx]

class FashionDataset(Dataset):
    """Convert the Fashion pkl file into a Pytorch Dataset"""

    def __init__(self, file_path: str, k: int):
        dataset = pickle.load(open(file_path, "rb"))

        self.X = dataset[0][k]
        self.y = np.array(dataset[1][k])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        # 3D input 28*28
        # x = torch.Tensor(self.X[idx]).permute(1,2,0) / 255
        # x = (x - 0.5) / 0.5
        # y = self.y[idx]
        # return x, y

        # x = torch.Tensor([self.features[idx]]) / 255
        # y = torch.LongTensor([self.labels[idx]])[0]
        x = torch.Tensor(self.X[idx])
        x = (x-0.5)/0.5
        y = self.y[idx]
        return x, y





def clients_set_CIFAR(
    file_name: str, n_clients: int, batch_size: int, shuffle=True
):
    """Download for all the clients their respective dataset"""
    print(file_name)

    list_dl = list()

    for k in range(n_clients):
        dataset_object = CIFARDataset(file_name, k)

        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )

        list_dl.append(dataset_dl)

    return list_dl

def clients_set_MNIST(
    file_name: str, n_clients: int, batch_size: int, shuffle=True
):
    """Download for all the clients their respective dataset"""
    print(file_name)

    list_dl = list()

    for k in range(n_clients):
        dataset_object = MNISTDataset(file_name, k)

        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )

        list_dl.append(dataset_dl)

    return list_dl

def clients_set_digits(
    file_name: str, n_clients: int, batch_size: int, shuffle=True
):

    print(file_name)
    list_dl = list()
    for k in range(n_clients):
        dataset_object = DigitsLoadDataset(file_name, k)

        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )

        list_dl.append(dataset_dl)

    return list_dl




def get_dataloaders(dataset,n_clients, batch_size: int, alpha: float, shuffle=True):
    # current_directory = os.getcwd()
    # print("当前工作目录:", current_directory)

    folder = "./data/"
    client_label_count = []
    if dataset[:5] == "CIFAR":
        n_classes = 10
        # balanced = dataset[8:12] == "bbal"
        balanced = True
        # alpha = float(dataset[13:])

        file_name_train = f"{dataset}_train_alpha{alpha}_{n_clients}.pkl"
        # "./data/cifar/train_alpha0.5_10.pkl"
        path_train = folder + file_name_train

        file_name_test = f"{dataset}_test_alpha{alpha}_{n_clients}.pkl"
        path_test = folder + file_name_test

        if not os.path.isfile(path_train):
            print("creating pathological dataset alpha:", alpha)
            client_label_count = create_CIFAR10_dirichlet(
                dataset, balanced, alpha, n_clients, n_classes
            )


        list_dls_train = clients_set_CIFAR(
            path_train, n_clients, batch_size, True
        )

        list_dls_test = clients_set_CIFAR(
            path_test, n_clients, batch_size, True
        )
    elif dataset[:5]=="MNIST":

        samples_train, samples_test = 500,100
        file_name_train = f"MNIST_alpha{alpha}_train_{n_clients}_{samples_train}.pkl"
        path_train = folder + file_name_train
        file_name_test = f"MNIST_alpha{alpha}_test_{n_clients}_{samples_test}.pkl"
        path_test = folder + file_name_test
        n_classes = 10
        if not os.path.isfile(path_train):
            create_MNIST_same_size_per_client(
                n_clients,samples_train,samples_test,n_classes,alpha
            )

        list_dls_train = clients_set_MNIST_shard(
            path_train, n_clients, batch_size=batch_size, shuffle=shuffle
        )

        list_dls_test = clients_set_MNIST_shard(
            path_test, n_clients, batch_size=batch_size, shuffle=shuffle
        )
    elif dataset[:6]=="digit":

        samples_train, samples_test =200, 100

        file_name_train = f"digit_train_alpha{alpha}_{n_clients}.pkl"
        path_train = folder + file_name_train
        file_name_test = f"digit_test_alpha{alpha}_{n_clients}.pkl"
        path_test = folder + file_name_test
        n_classes = 10
        balanced = False
        if not os.path.isfile(path_train):
            print("creating dirle dataset alpha:", alpha)
            create_digits_dirichlet(
                dataset, balanced, alpha, n_clients, n_classes
            )

        list_dls_train = clients_set_digits(
            path_train, n_clients, batch_size, True
        )

        list_dls_test = clients_set_digits(
            path_test, n_clients, batch_size, True
        )


    return list_dls_train, list_dls_test,client_label_count


def get_biye_dataloaders(
    dataset: str,
    alpha:float,
    N:int,
    cnt_strategy:int,
    B:int,
    samplesNum,
    p,
):
    folder = "./data/"
    client_label_count = None
    n_classes = 10
    if dataset[:5]=="CIFAR":
        file_name_train = f"{dataset}_train_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}.pkl"
        path_train = folder + file_name_train
        # 一个训练集会对应两个测试集

        if not os.path.isfile(path_train):
            print("creating pathological dataset alpha:", alpha)
            client_label_count = create_CIFAR10_biye(
                dataset,alpha, N,cnt_strategy, n_classes,samplesNum,p
            )
        list_dls_train = clients_set_CIFAR(
            path_train, N, B, True
        )

        file_name_test = f"{dataset}_test_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}_test_{0}.pkl"
        path_test = folder + file_name_test

        list_dls_test0 = clients_set_CIFAR(
            path_test, N, B, True
        )

        file_name_test = f"{dataset}_test_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}_test_{1}.pkl"
        path_test = folder + file_name_test

        list_dls_test1 = clients_set_CIFAR(
            path_test, N, B, True
        )

    if dataset[:5]=="MNIST":
        file_name_train = f"{dataset}_train_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}.pkl"
        path_train = folder + file_name_train
        # 一个训练集会对应两个测试集

        if not os.path.isfile(path_train):
            print("creating pathological dataset alpha:", alpha)
            client_label_count = create_MNIST_biye(
                dataset,alpha, N,cnt_strategy, n_classes,p
            )
        list_dls_train = clients_set_MNIST(
            path_train, N, B, True
        )

        file_name_test = f"{dataset}_test_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}_test_{0}.pkl"
        path_test = folder + file_name_test

        list_dls_test0 = clients_set_MNIST(
            path_test, N, B, True
        )

        file_name_test = f"{dataset}_test_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}_test_{1}.pkl"
        path_test = folder + file_name_test

        list_dls_test1 = clients_set_MNIST(
            path_test, N, B, True
        )

    elif dataset[:7]=="fashion":

        file_name_train = f"{dataset}_train_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}.pkl"
        path_train = folder + file_name_train
        n_classes = 10
        if not os.path.isfile(path_train):
            client_label_count = create_fashion_biye(
                dataset, alpha, N, cnt_strategy, n_classes,p
            )
        print("数据集创建完毕！")
        list_dls_train = clients_set_Fashion(
            path_train, N, B, True
        )

        file_name_test = f"{dataset}_test_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}_test_{0}.pkl"
        path_test = folder + file_name_test

        list_dls_test0 = clients_set_Fashion(
            path_test, N, B, True
        )

        file_name_test = f"{dataset}_test_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}_test_{1}.pkl"
        path_test = folder + file_name_test

        list_dls_test1 = clients_set_Fashion(
            path_test, N, B, True
        )

    elif dataset[:8]=="R_FMNIST":
        file_name_train = f"{dataset}_train_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}.pkl"
        path_train = folder + file_name_train
        n_classes = 10
        if not os.path.isfile(path_train):
            client_label_count = create_R_FMNIST_biye(
                dataset, alpha, N, cnt_strategy, n_classes,samplesNum,p
            )
        print("数据集创建完毕！")
        list_dls_train = clients_set_digits(
            path_train, N, B, True
        )

        file_name_test = f"{dataset}_test_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}_test{0}.pkl"
        path_test = folder + file_name_test

        list_dls_test0 = clients_set_digits(
            path_test, N, B, True
        )

        file_name_test = f"{dataset}_test_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}_test{1}.pkl"
        path_test = folder + file_name_test

        list_dls_test1 = clients_set_digits(
            path_test, N, B, True
        )


    elif "digit" in dataset[:7]:
        if dataset == "R_digit":
            use_rotations = True
        else:
            use_rotations = False

        file_name_train = f"{dataset}_train_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}.pkl"
        path_train = folder + file_name_train
        n_classes = 10
        if not os.path.isfile(path_train):
            client_label_count = create_digits_biye(
                dataset, alpha, N, cnt_strategy, n_classes,samplesNum,p,use_rotations = use_rotations
            )
        print("数据集创建完毕！")
        list_dls_train = clients_set_digits(
            path_train, N, B, True
        )


        file_name_test = f"{dataset}_test_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}_test{0}.pkl"
        path_test = folder + file_name_test

        list_dls_test0 = clients_set_digits(
            path_test, N, B, True
        )

        file_name_test = f"{dataset}_test_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}_test{1}.pkl"
        path_test = folder + file_name_test

        list_dls_test1 = clients_set_digits(
            path_test, N, B, True
        )

    elif dataset[:7]=="R_MNIST":
        file_name_train = f"{dataset}_train_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}.pkl"
        path_train = folder + file_name_train
        n_classes = 10
        if not os.path.isfile(path_train):
            client_label_count = create_R_MNIST_biye(
                dataset, alpha, N, cnt_strategy, n_classes,samplesNum,p
            )
        print("数据集创建完毕！")
        list_dls_train = clients_set_digits(
            path_train, N, B, True
        )

        file_name_test = f"{dataset}_test_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}_test{0}.pkl"
        path_test = folder + file_name_test

        list_dls_test0 = clients_set_digits(
            path_test, N, B, True
        )

        file_name_test = f"{dataset}_test_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}_test{1}.pkl"
        path_test = folder + file_name_test

        list_dls_test1 = clients_set_digits(
            path_test, N, B, True
        )

    elif dataset[:7]=="R_CIFAR":
        file_name_train = f"{dataset}_train_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}.pkl"
        path_train = folder + file_name_train
        n_classes = 10
        if not os.path.isfile(path_train):
            client_label_count = create_R_CIFAR_biye(
                dataset, alpha, N, cnt_strategy, n_classes,samplesNum,p
            )
        print("数据集创建完毕！")
        list_dls_train = clients_set_digits(
            path_train, N, B, True
        )

        file_name_test = f"{dataset}_test_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}_test{0}.pkl"
        path_test = folder + file_name_test

        list_dls_test0 = clients_set_digits(
            path_test, N, B, True
        )

        file_name_test = f"{dataset}_test_alpha{alpha}_N{N}_cnt_{cnt_strategy}_p{p}_test{1}.pkl"
        path_test = folder + file_name_test

        list_dls_test1 = clients_set_digits(
            path_test, N, B, True
        )


    return list_dls_train, list_dls_test0,list_dls_test1, client_label_count


def create_fashion_biye(
    dataset, alpha, n_clients, cnt_strategy, n_classes,p
):
    Fashion_train = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
    )

    Fashion_test = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
    )

    image, label = Fashion_train[0]
    print(image.shape)
    from numpy.random import dirichlet

    matrix = dirichlet([alpha] * n_classes, size=n_clients)
    if cnt_strategy == 2:
        import random
        matrix = np.zeros((n_clients, n_classes))
        for i in range(n_clients):
            k1 = random.randint(0, 9)
            k2 = random.randint(0, 9)
            matrix[i][k1] = 0.5
            matrix[i][k2] = 0.5


    file_name_train = f"{dataset}_train_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}.pkl"
    client_label_count = partition_fashion_dataset_biye(
        Fashion_train,
        file_name_train,
        cnt_strategy,
        matrix,
        n_clients,
        n_classes,
        True,
    )

    file_name_test1 = f"{dataset}_test_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}_test_{0}.pkl"
    partition_fashion_dataset_biye(
        Fashion_test,
        file_name_test1,
        cnt_strategy,
        matrix,
        n_clients,
        n_classes,
        False,
    )
    matrix2 = np.full((n_clients, n_classes), 0.1)
    file_name_test2 = f"{dataset}_test_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}_test_{1}.pkl"
    partition_fashion_dataset_biye(
        Fashion_test,
        file_name_test2,
        cnt_strategy,
        matrix2,
        n_clients,
        n_classes,
        False,
    )

    return client_label_count


    con_fashion_partition_dataset(
        Fashion_train,
        file_name_train,
        matrix,
        N,
        n_classes,
        True,
    )



#不加噪声
def create_digits_biye(
    dataset, alpha, n_clients, cnt_strategy, n_classes,samplesNum,p,use_rotations = True
):


    from numpy.random import dirichlet
    matrix = dirichlet([alpha] * n_classes, size=n_clients)
    if cnt_strategy == 2:
        import random
        matrix = np.zeros((n_clients, n_classes))
        for i in range(n_clients):
            k1 = random.randint(0, 9)
            k2 = random.randint(0, 9)
            matrix[i][k1] = 0.5
            matrix[i][k2] = 0.5

    if use_rotations:
        rotations = [0, 90, 180, 270]
        rotations_transform = [transforms.RandomRotation(degrees=(angle, angle)) for angle in rotations]
    else:
        rotations_transform = [None,None,None,None]

    transform_mnist = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_svhn = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_usps = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_synth = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_mnistm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    from utils.data_util import DigitsDataset
    # MNIST
    # MNIST
    mnist_trainset = DigitsDataset(data_path="./data/MNIST", channels=1, train=True,
                                   transform=transform_mnist,rotations_transform=rotations_transform[0])
    mnist_testset = DigitsDataset(data_path="./data/MNIST", channels=1, train=False,
                                  transform=transform_mnist,rotations_transform=rotations_transform[0])

    # SVHN
    svhn_trainset = DigitsDataset(data_path='./data/SVHN', channels=3, train=True,
                                  transform=transform_svhn,rotations_transform=rotations_transform[1])
    svhn_testset = DigitsDataset(data_path='./data/SVHN', channels=3, train=False,
                                 transform=transform_svhn,rotations_transform=rotations_transform[1])

    # # USPS
    usps_trainset = DigitsDataset(data_path='./data/USPS', channels=1, train=True,
                                  transform=transform_usps,rotations_transform=rotations_transform[2])
    usps_testset = DigitsDataset(data_path='./data/USPS', channels=1, train=False,
                                 transform=transform_usps,rotations_transform=rotations_transform[2])

    # # # Synth Digits
    synth_trainset = DigitsDataset(data_path='./data/SynthDigits/', channels=3, train=True,
                                   transform=transform_synth,rotations_transform=rotations_transform[3])
    synth_testset = DigitsDataset(data_path='./data/SynthDigits/', channels=3, train=False,
                                  transform=transform_synth,rotations_transform=rotations_transform[3])
    # #
    # # # MNIST-M
    mnistm_trainset = DigitsDataset(data_path='./data/MNIST_M/', channels=3, train=True,
                                    transform=transform_mnistm,rotations_transform=rotations_transform[0])
    mnistm_testset = DigitsDataset(data_path='./data/MNIST_M/', channels=3, train=False,
                                   transform=transform_mnistm,rotations_transform=rotations_transform[0])

    digits_train = [mnist_trainset, svhn_trainset, usps_trainset, synth_trainset, mnistm_trainset]
    digits_test = [mnist_testset, svhn_testset, usps_testset, synth_testset, mnistm_testset]
    file_name_train = f"{dataset}_train_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}.pkl"
    partition_digits_dataset_biye(
        digits_train,
        file_name_train,
        cnt_strategy,
        matrix,
        n_clients,
        n_classes,
        True,
        samplesNum
    )

    file_name_test0 = f"{dataset}_test_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}_test{0}.pkl"
    partition_digits_dataset_biye(
        digits_test,
        file_name_test0,
        cnt_strategy,
        matrix,
        n_clients,
        n_classes,
        False,
        samplesNum
    )
    matrix1 = np.full((n_clients,n_classes),0.1)
    file_name_test1 = f"{dataset}_test_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}_test{1}.pkl"
    partition_digits_dataset_biye(
        digits_test,
        file_name_test1,
        cnt_strategy,
        matrix1,
        n_clients,
        n_classes,
        False,
        samplesNum
    )


# 加噪声
# def create_digits_biye(
#         dataset, alpha, n_clients, cnt_strategy, n_classes, samplesNum, p, use_rotations=True
# ):
#     from numpy.random import dirichlet
#     import torch
#     import random
#     from torchvision import transforms
#     import numpy as np
#
#     matrix = dirichlet([alpha] * n_classes, size=n_clients)
#     if cnt_strategy == 2:
#         matrix = np.zeros((n_clients, n_classes))
#         for i in range(n_clients):
#             k1 = random.randint(0, 9)
#             k2 = random.randint(0, 9)
#             matrix[i][k1] = 0.5
#             matrix[i][k2] = 0.5
#
#     if use_rotations:
#         rotations = [0, 90, 180, 270]
#         rotations_transform = [transforms.RandomRotation(degrees=(angle, angle)) for angle in rotations]
#     else:
#         rotations_transform = [None, None, None, None]
#
#     # 自定义噪声转换类
#     class AddGaussianNoise(object):
#         def __init__(self, mean=0., std=1.):
#             self.std = std
#             self.mean = mean
#
#         def __call__(self, tensor):
#             return tensor + torch.randn(tensor.size()) * self.std + self.mean
#
#     class AddSaltPepperNoise(object):
#         def __init__(self, prob=0.05):
#             self.prob = prob
#
#         def __call__(self, tensor):
#             noise_tensor = torch.rand(tensor.size())
#             salt = noise_tensor < self.prob / 2
#             pepper = noise_tensor > (1 - self.prob / 2)
#             tensor[salt] = 1
#             tensor[pepper] = 0
#             return tensor
#
#     # MNIST使用高斯噪声
#     transform_mnist = transforms.Compose([
#         transforms.Grayscale(num_output_channels=3),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         AddGaussianNoise(0., 0.1)
#     ])
#
#     # SVHN使用椒盐噪声
#     transform_svhn = transforms.Compose([
#         transforms.Resize([28, 28]),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         AddSaltPepperNoise(0.05)
#     ])
#
#     # USPS使用随机擦除
#     transform_usps = transforms.Compose([
#         transforms.Resize([28, 28]),
#         transforms.Grayscale(num_output_channels=3),
#         transforms.ToTensor(),
#         transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     # SynthDigits使用混合噪声
#     transform_synth = transforms.Compose([
#         transforms.Resize([28, 28]),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         AddGaussianNoise(0., 0.05),
#         AddSaltPepperNoise(0.02)
#     ])
#
#     # MNIST-M保持原样
#     transform_mnistm = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     from utils.data_util import DigitsDataset
#
#     # 创建数据集
#     mnist_trainset = DigitsDataset(data_path="./data/MNIST", channels=1, train=True,
#                                    transform=transform_mnist, rotations_transform=rotations_transform[0])
#     mnist_testset = DigitsDataset(data_path="./data/MNIST", channels=1, train=False,
#                                   transform=transform_mnist, rotations_transform=rotations_transform[0])
#
#     svhn_trainset = DigitsDataset(data_path='./data/SVHN', channels=3, train=True,
#                                   transform=transform_svhn, rotations_transform=rotations_transform[1])
#     svhn_testset = DigitsDataset(data_path='./data/SVHN', channels=3, train=False,
#                                  transform=transform_svhn, rotations_transform=rotations_transform[1])
#
#     usps_trainset = DigitsDataset(data_path='./data/USPS', channels=1, train=True,
#                                   transform=transform_usps, rotations_transform=rotations_transform[2])
#     usps_testset = DigitsDataset(data_path='./data/USPS', channels=1, train=False,
#                                  transform=transform_usps, rotations_transform=rotations_transform[2])
#
#     synth_trainset = DigitsDataset(data_path='./data/SynthDigits/', channels=3, train=True,
#                                    transform=transform_synth, rotations_transform=rotations_transform[3])
#     synth_testset = DigitsDataset(data_path='./data/SynthDigits/', channels=3, train=False,
#                                   transform=transform_synth, rotations_transform=rotations_transform[3])
#
#     mnistm_trainset = DigitsDataset(data_path='./data/MNIST_M/', channels=3, train=True,
#                                     transform=transform_mnistm, rotations_transform=rotations_transform[0])
#     mnistm_testset = DigitsDataset(data_path='./data/MNIST_M/', channels=3, train=False,
#                                    transform=transform_mnistm, rotations_transform=rotations_transform[0])
#
#     digits_train = [mnist_trainset, svhn_trainset, usps_trainset, synth_trainset, mnistm_trainset]
#     digits_test = [mnist_testset, svhn_testset, usps_testset, synth_testset, mnistm_testset]
#
#     # 保存数据集
#     file_name_train = f"{dataset}_train_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}.pkl"
#     partition_digits_dataset_biye(
#         digits_train,
#         file_name_train,
#         cnt_strategy,
#         matrix,
#         n_clients,
#         n_classes,
#         True,
#         samplesNum
#     )
#
#     file_name_test0 = f"{dataset}_test_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}_test{0}.pkl"
#     partition_digits_dataset_biye(
#         digits_test,
#         file_name_test0,
#         cnt_strategy,
#         matrix,
#         n_clients,
#         n_classes,
#         False,
#         samplesNum
#     )
#
#     matrix1 = np.full((n_clients, n_classes), 0.1)
#     file_name_test1 = f"{dataset}_test_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}_test{1}.pkl"
#     partition_digits_dataset_biye(
#         digits_test,
#         file_name_test1,
#         cnt_strategy,
#         matrix1,
#         n_clients,
#         n_classes,
#         False,
#         samplesNum
#     )

def create_R_MNIST_biye(
    dataset, alpha, n_clients, cnt_strategy, n_classes,samplesNum,p
):
    from numpy.random import dirichlet
    matrix = dirichlet([alpha] * n_classes, size=n_clients)
    if cnt_strategy == 2:
        import random
        matrix = np.zeros((n_clients, n_classes))
        for i in range(n_clients):
            k1 = random.randint(0, 9)
            k2 = random.randint(0, 9)
            matrix[i][k1] = 0.5
            matrix[i][k2] = 0.5

    rotations = [0, 90, 180, 270]
    transform_mnist = [[] for _ in range(len(rotations))]
    rotations_transform = [transforms.RandomRotation(degrees=(angle, angle)) for angle in rotations]

    from utils.data_util import R_MnistDataset
    # MNIST
    mnist_trainset = [[] for _ in range(len(rotations))]
    mnist_testset = [[] for _ in range(len(rotations))]

    for idx in range(len(rotations)):
        mnist_trainset[idx] = R_MnistDataset(data_path="./data/MNIST", channels=1, train=True,
                                       rotations_transform=rotations_transform[idx])
        mnist_testset[idx] = R_MnistDataset(data_path="./data/MNIST", channels=1, train=False,
                                  rotations_transform=rotations_transform[idx])



    digits_train = [mnist_train for mnist_train in mnist_trainset]
    digits_test = [mnist_test for mnist_test in mnist_testset]
    file_name_train = f"{dataset}_train_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}.pkl"
    partition_digits_dataset_biye(
        digits_train,
        file_name_train,
        cnt_strategy,
        matrix,
        n_clients,
        n_classes,
        True,
        samplesNum
    )

    file_name_test0 = f"{dataset}_test_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}_test{0}.pkl"
    partition_digits_dataset_biye(
        digits_test,
        file_name_test0,
        cnt_strategy,
        matrix,
        n_clients,
        n_classes,
        False,
        samplesNum
    )
    matrix1 = np.full((n_clients,n_classes),0.1)
    file_name_test1 = f"{dataset}_test_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}_test{1}.pkl"
    partition_digits_dataset_biye(
        digits_test,
        file_name_test1,
        cnt_strategy,
        matrix1,
        n_clients,
        n_classes,
        False,
        samplesNum
    )

def create_R_FMNIST_biye(
    dataset, alpha, n_clients, cnt_strategy, n_classes,samplesNum,p
):
    from numpy.random import dirichlet
    matrix = dirichlet([alpha] * n_classes, size=n_clients)

    if cnt_strategy == 2:
        import random
        matrix = np.zeros((n_clients, n_classes))
        for i in range(n_clients):
            k1 = random.randint(0, 9)
            k2 = random.randint(0, 9)
            matrix[i][k1] = 0.5
            matrix[i][k2] = 0.5

    rotations = [0, 90, 180, 270]
    transform_mnist = [[] for _ in range(len(rotations))]
    rotations_transform = [transforms.RandomRotation(degrees=(angle, angle)) for angle in rotations]

    # 自定义噪声转换类
    class AddGaussianNoise(object):
        def __init__(self, mean=0., std=1.):
            self.std = std
            self.mean = mean

        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean

    class AddSaltPepperNoise(object):
        def __init__(self, prob=0.05):
            self.prob = prob

        def __call__(self, tensor):
            noise_tensor = torch.rand(tensor.size())
            salt = noise_tensor < self.prob / 2
            pepper = noise_tensor > (1 - self.prob / 2)
            tensor[salt] = 1
            tensor[pepper] = 0
            return tensor

    transform = transforms.Compose([transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5, ))])

    from utils.data_util import R_MnistDataset
    # MNIST
    mnist_trainset = [[] for _ in range(len(rotations))]
    mnist_testset = [[] for _ in range(len(rotations))]

    for idx in range(len(rotations)):
        mnist_trainset[idx] = R_MnistDataset(data_path="./data/FMNIST", channels=1, train=True,
                                       rotations_transform=rotations_transform[idx],transform = None)
        mnist_testset[idx] = R_MnistDataset(data_path="./data/FMNIST", channels=1, train=False,
                                  rotations_transform=rotations_transform[idx],transform = None)



    digits_train = [mnist_train for mnist_train in mnist_trainset]
    digits_test = [mnist_test for mnist_test in mnist_testset]
    file_name_train = f"{dataset}_train_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}.pkl"
    partition_digits_dataset_biye(
        digits_train,
        file_name_train,
        cnt_strategy,
        matrix,
        n_clients,
        n_classes,
        True,
        samplesNum
    )

    file_name_test0 = f"{dataset}_test_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}_test{0}.pkl"
    partition_digits_dataset_biye(
        digits_test,
        file_name_test0,
        cnt_strategy,
        matrix,
        n_clients,
        n_classes,
        False,
        samplesNum
    )
    matrix1 = np.full((n_clients,n_classes),0.1)
    file_name_test1 = f"{dataset}_test_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}_test{1}.pkl"
    partition_digits_dataset_biye(
        digits_test,
        file_name_test1,
        cnt_strategy,
        matrix1,
        n_clients,
        n_classes,
        False,
        samplesNum
    )

def create_R_CIFAR_biye(
    dataset, alpha, n_clients, cnt_strategy, n_classes,samplesNum,p
):
    from numpy.random import dirichlet
    matrix = dirichlet([alpha] * n_classes, size=n_clients)
    if cnt_strategy == 2:
        import random
        matrix = np.zeros((n_clients, n_classes))
        for i in range(n_clients):
            k1 = random.randint(0, 9)
            k2 = random.randint(0, 9)
            matrix[i][k1] = 0.5
            matrix[i][k2] = 0.5

    rotations = [0, 90, 180,270]
    transform_mnist = [[] for _ in range(len(rotations))]
    rotations_transform = [transforms.RandomRotation(degrees=(angle, angle)) for angle in rotations]



    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    from utils.data_util import R_CifarDataset
    # MNIST
    mnist_trainset = [[] for _ in range(len(rotations))]
    mnist_testset = [[] for _ in range(len(rotations))]

    for idx in range(len(rotations)):
        mnist_trainset[idx] = R_CifarDataset(data_path="./data/CIFAR10", channels=3, train=True,
                                       rotations_transform=rotations_transform[idx],transform = None)
        mnist_testset[idx] = R_CifarDataset(data_path="./data/CIFAR10", channels=3, train=False,
                                  rotations_transform=rotations_transform[idx],transform = None)



    digits_train = [mnist_train for mnist_train in mnist_trainset]
    digits_test = [mnist_test for mnist_test in mnist_testset]
    file_name_train = f"{dataset}_train_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}.pkl"
    partition_digits_dataset_biye(
        digits_train,
        file_name_train,
        cnt_strategy,
        matrix,
        n_clients,
        n_classes,
        True,
        samplesNum
    )

    file_name_test0 = f"{dataset}_test_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}_test{0}.pkl"
    partition_digits_dataset_biye(
        digits_test,
        file_name_test0,
        cnt_strategy,
        matrix,
        n_clients,
        n_classes,
        False,
        samplesNum
    )
    matrix1 = np.full((n_clients,n_classes),0.1)
    file_name_test1 = f"{dataset}_test_alpha{alpha}_N{n_clients}_cnt_{cnt_strategy}_p{p}_test{1}.pkl"
    partition_digits_dataset_biye(
        digits_test,
        file_name_test1,
        cnt_strategy,
        matrix1,
        n_clients,
        n_classes,
        False,
        samplesNum
    )


def create_digits_dirichlet(
        dataset,balanced,alpha,n_clients,n_classes
):
    from numpy.random import dirichlet
    matrix = dirichlet([alpha]*n_classes,size=n_clients)

    transform_mnist = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_svhn = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_usps = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_synth = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_mnistm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    from utils.data_util import DigitsDataset
    # MNIST
    # MNIST

    # from torchvision.datasets import MNIST, SVHN, USPS
    # MNIST(root='./data',  download=True)
    # # USPS(root='./data',  download=True)
    # # SVHN(root='./data', download=True)
    # # SynthDigits(root='./data', download=True)
    # # MNIST_M(root='./data', download=True)

    mnist_trainset = DigitsDataset(data_path="./data/MNIST", channels=1,  train=True,
                                              transform=transform_mnist)
    mnist_testset = DigitsDataset(data_path="./data/MNIST", channels=1, train=False,
                                             transform=transform_mnist)

    # SVHN
    svhn_trainset = DigitsDataset(data_path='./data/SVHN', channels=3,  train=True,
                                             transform=transform_svhn)
    svhn_testset = DigitsDataset(data_path='./data/SVHN', channels=3,  train=False,
                                            transform=transform_svhn)



    # # USPS
    usps_trainset = DigitsDataset(data_path='./data/USPS', channels=1, train=True,
                                  transform=transform_usps)
    usps_testset = DigitsDataset(data_path='./data/USPS', channels=1,  train=False,
                                 transform=transform_usps)

    # # # Synth Digits
    synth_trainset = DigitsDataset(data_path='./data/SynthDigits/', channels=3,  train=True,
                                   transform=transform_synth)
    synth_testset = DigitsDataset(data_path='./data/SynthDigits/', channels=3, train=False,
                                  transform=transform_synth)
    # #
    # # # MNIST-M
    mnistm_trainset = DigitsDataset(data_path='./data/MNIST_M/', channels=3, train=True,
                                    transform=transform_mnistm)
    mnistm_testset = DigitsDataset(data_path='./data/MNIST_M/', channels=3, train=False,
                                   transform=transform_mnistm)


    digits_train = [mnist_trainset,svhn_trainset,usps_trainset,synth_trainset,mnistm_trainset]
    digits_test = [ mnist_testset,svhn_testset,usps_testset,synth_testset,mnistm_testset]
    file_name_train = f"{dataset}_train_alpha{alpha}_{n_clients}.pkl"
    partition_digits_dataset(
        digits_train,
        file_name_train,
        balanced,
        matrix,
        n_clients,
        n_classes,
        True
    )

    file_name_test = f"{dataset}_test_alpha{alpha}_{n_clients}.pkl"
    partition_digits_dataset(
        digits_test,
        file_name_test,
        balanced,
        matrix,
        n_clients,
        n_classes,
        False
    )


#  A+B  c% 一个类别 -- B (1-c)%  全部类别

def partition_digits_dataset_biye(
        dataset,
        file_name: str,
        cnt_strategy: int,
        matrix,
        n_clients: int,
        n_classes: int,
        train: bool,
        samplesNum:int,
):
    #     每个客户端i有[k,i]个数据 对于类别k
    list_clients_X = [[] for i in range(n_clients)]
    list_clients_y = [[] for i in range(n_clients)]
    client_label_count = [[] for i in range(n_clients)]
    n_samples = []

    if cnt_strategy != 1 and train:
        if n_clients >= 20:
            n_samples = [samplesNum] * n_clients
        else:
            n_samples = [samplesNum] * n_clients
    elif cnt_strategy == 1 and train:
        n_samples = []
        if n_clients >= 100:
            for i in range(n_clients):
                if i < 30:
                    n_samples.append(100)
                elif i < 80:
                    n_samples.append(300)
                else:
                    n_samples.append(500)
        else:
            for i in range(n_clients):
                n_samples.append((i + 1) * 100)

    elif cnt_strategy != 1 and not train:
        if n_clients >= 100:
            n_samples = [int(samplesNum * 0.25)] * n_clients
        else:
            n_samples = [100] * n_clients
    elif cnt_strategy == 1 and not train:
        n_samples = []
        if n_clients >= 100:
            for i in range(n_clients):
                if i < 30:
                    n_samples.append(50)
                elif i < 80:
                    n_samples.append(50)
                else:
                    n_samples.append(100)
        else:
            for i in range(n_clients):
                n_samples.append(100)

    n_features = len(dataset)

    list_idx = [[] for i in range(n_features)]
    # 第i个数据集的第k类数据样本的索引
    for i in range(n_features):
        for k in range(n_classes):
            idx_k = np.where(np.array(dataset[i].labels) == k)[0]
            list_idx[i] += [idx_k]
    # list_idx[i][j] 表示第i个数据集中标签为j的所有数据列表
    for idx_client, n_sample in enumerate(n_samples):
        clients_idx_i = []
        client_samples = 0
        # 当前客户端应该分配哪个数据集
        idx_feature = idx_client % n_features
        for k in range(n_classes):

            if k < 9:
                samples_digit = int((matrix[idx_client, k] + 0.0000001) * n_sample)
            elif k == 9:
                samples_digit = n_sample - client_samples
            client_samples += samples_digit
            client_label_count[idx_client].append(samples_digit)
            clients_idx_i = np.concatenate(
                (clients_idx_i, np.random.choice(list_idx[idx_feature][k], samples_digit, replace=True))
            )

        clients_idx_i = clients_idx_i.astype(int)

        for idx_sample in clients_idx_i:

            image, label = dataset[idx_feature][idx_sample]
            image = np.array(image)
            label = np.array(label)
            list_clients_X[idx_client] += [image]
            list_clients_y[idx_client] += [label]

        import random
        list_clients_X[idx_client] = np.array(list_clients_X[idx_client])
        list_clients_y[idx_client] = np.array(list_clients_y[idx_client])

        index = [i for i in range(len(list_clients_X[idx_client]))]
        random.shuffle(index)
        list_clients_X[idx_client] = list_clients_X[idx_client][index]
        list_clients_y[idx_client] = list_clients_y[idx_client][index]

    print("shape test")

    folder = "./data/"
    with open(folder + file_name, "wb") as output:
        pickle.dump((list_clients_X, list_clients_y), output)

    return client_label_count



def partition_digits_dataset(
    dataset,
    file_name: str,
    balanced: bool,
    matrix,
    n_clients: int,
    n_classes: int,
    train: bool,
):
#     每个客户端i有[k,i]个数据 对于类别k
    list_clients_X = [[] for i in range(n_clients)]
    list_clients_y = [[] for i in range(n_clients)]
    if train:
        n_samples = [200]*n_clients
    else:
        n_samples = [50]*n_clients
    n_features = len(dataset)

    list_idx = [[] for i in range(n_features)]
    # 第i个数据集的第k类数据样本的索引
    for i in range(n_features):
        for k in range(n_classes):
            idx_k = np.where(np.array(dataset[i].labels)==k)[0]
            list_idx[i] += [idx_k]

    for idx_client,n_sample in enumerate(n_samples):
        clients_idx_i = []
        client_samples = 0
        # 当前客户端应该分配哪个数据集
        idx_feature = idx_client%n_features
        for k in range(n_classes):

            if k < 9:
                samples_digit = int((matrix[idx_client, k] + 0.0000001) * n_sample)
            elif k == 9:
                samples_digit = n_sample - client_samples
            client_samples += samples_digit

            clients_idx_i = np.concatenate(
                (clients_idx_i, np.random.choice(list_idx[idx_feature][k], samples_digit, replace=False))
            )

        clients_idx_i = clients_idx_i.astype(int)

        for idx_sample in clients_idx_i:
            image,label = dataset[idx_feature][idx_sample]
            list_clients_X[idx_client] += [np.array(image)]
            list_clients_y[idx_client] += [np.array(label)]
            # list_clients_X[idx_client] += [dataset[idx_feature].images[idx_sample]]
            # list_clients_y[idx_client] += [dataset[idx_feature].labels[idx_sample]]

        list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

    print("shape test")
    # a = dataset[0].images[0]
    # b = dataset[1].images[0]
    # c = dataset[2].images[0]
    # d = dataset[3].images[0]
    # e = dataset[4].images[0]
    folder = "./data/"
    with open(folder + file_name, "wb") as output:
        pickle.dump((list_clients_X, list_clients_y), output)


def create_Fashion_condition1(dataset, balanced, N, n_classes):

    Fashion_train = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    Fashion_test = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    image,label = Fashion_train[0]
    print(image.shape)
    # matrix = dirichlet([alpha] * n_classes, size=n_clients)
    # 客户端i第j类数据分配的比例
    matrix = np.zeros((N, n_classes)).astype(float)

    # 每个客户端取四个类
    for i in range(N):
        for j in range(i, i + 4):
            type = j % 10
            matrix[i][type] = 0.25

    file_name_train = f"con1_{dataset}_train_{N}.pkl"
    balanced = True
    con_fashion_partition_dataset(
        Fashion_train,
        file_name_train,
        matrix,
        N,
        n_classes,
        True,
    )

    file_name_test = f"con1_{dataset}_test_{N}.pkl"

    con_fashion_partition_dataset(
        Fashion_test,
        file_name_test,
        matrix,
        N,
        n_classes,
        False,
    )


def create_Fashion_condition2(dataset, balanced, N, n_classes):

    Fashion_train = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    Fashion_test = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    image,label = Fashion_train[0]
    print(image.shape)
    # matrix = dirichlet([alpha] * n_classes, size=n_clients)
    # 客户端i第j类数据分配的比例
    matrix = np.zeros((N, n_classes)).astype(float)

    # 每个客户端取两个类各40% 其他的0.2/8 = 0.025
    for i in range(N):
        for j in range(i, i + 9):
            if j == i or j % 10 == (i + 1) % 10:
                matrix[i][j % 10] = 0.4
            else:
                matrix[i][j % 10] = 0.025

    file_name_train = f"con2_{dataset}_train_{N}.pkl"
    balanced = True
    con_fashion_partition_dataset(
        Fashion_train,
        file_name_train,
        matrix,
        N,
        n_classes,
        True,
    )

    file_name_test = f"con2_{dataset}_test_{N}.pkl"

    con_fashion_partition_dataset(
        Fashion_test,
        file_name_test,
        matrix,
        N,
        n_classes,
        False,
    )



def clients_set_Fashion(
    file_name: str, n_clients: int, batch_size: int, shuffle=True
):
    """Download for all the clients their respective dataset"""
    print(file_name)
    list_dl = list()

    for k in range(n_clients):

        dataset_object = FashionDataset(file_name, k)

        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )

        list_dl.append(dataset_dl)

    return list_dl




