import torch
import numpy as np
import copy
import torch.nn as nn
from torch.nn import functional as F
def produce_Ew(train_ds_local_set,num_classes,device):
    # 返回Ew_set 和Mask_set 其中  每一个Ew 表示每个客户端每个类所占权重， Mask 有为1 没有为0
    Ew_set = []
    Mask_set = []

    for net_id in range(len(train_ds_local_set)):
        train_ds = train_ds_local_set[net_id]
        # try:
        #
        #     ds = train_ds.dataset
        #     indices = train_ds.indices
        #     targets = np.array(ds.targets)[indices]
        # except:
        #     targets = train_ds.targets
        targets = train_ds.dataset.labels
        label = torch.tensor(targets)
        label.requires_grad = False

        uni_label, count = torch.unique(label, return_counts=True)
        num_samples = len(label)

        Ew = torch.zeros(num_classes)
        Mask = torch.zeros(num_classes)
        for idx, class_id in enumerate(uni_label):
            Ew[class_id] = 1 * count[idx] / num_samples
            Mask[class_id] = 1
        Ew = Ew * num_classes
        Ew = Ew.to(device)
        Mask = Mask.to(device)
        Ew_set.append(Ew)
        Mask_set.append(Mask)

    return Mask_set, Ew_set

def get_eft(feat_in,num_classes):
    a = np.random.random(size=(feat_in, num_classes))
    P, _ = np.linalg.qr(a)
    P = torch.tensor(P).float()
    I = torch.eye(num_classes)
    one = torch.ones(num_classes, num_classes)
    M = np.sqrt(num_classes / (num_classes - 1)) * torch.matmul(P, I - ((1 / num_classes) * one))
    return M

def gela_train(train_model,client_id,Ew_set,Mask_set,ETF,n_SGD, loss_f, train_data, mu, temperature, lr,device):
    model_0 = copy.deepcopy(train_model)

    bias_p = []
    weight_p = []

    for name, p in train_model.named_parameters():
        # print(f"name :{name} p :{p}")
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    optimizer = torch.optim.SGD(
        [
            {'params': weight_p, 'weight_decay': 0.0001},
            {'params': bias_p, 'weight_decay': 0}
        ],
        lr=lr,
        momentum=0.5
    )
    criterion = nn.NLLLoss().to(device)
    sfm = nn.LogSoftmax(dim=1)


    len_batch = len(train_data)
    for i in range(n_SGD):
        for j in range(len_batch):
            features, labels = next(iter(train_data))

            labels = labels.long()
            if torch.cuda.is_available():
                h, predictions = train_model(features.to(device))
            else:
                h, predictions = train_model(features)

            learned_norm = Ew_set[client_id]
            cur_M = learned_norm *ETF
            out_g = torch.matmul(h,cur_M)
            logits = sfm(out_g/temperature)
            logits = logits.to(device)
            labels = labels.to(device)

            loss_global = criterion(logits, labels)
            loss = loss_global
            loss.backward()
            nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

    return train_model.state_dict()


def loss_gela_dataset(model, train_data, loss_f,ETF,Ew_set,client,device):
    model.eval()
    loss = 0.0
    sfm = nn.LogSoftmax(dim=1)
    for idx,(features, labels) in enumerate(train_data):
        if torch.cuda.is_available():
            h,predictions = model(features.to(device))
        else:
            h,predictions = model(features)
        labels=labels.long()
        learned_norm = Ew_set[client]
        cur_M = learned_norm*ETF
        out_g = torch.matmul(h,cur_M)
        logits = sfm(out_g)
        loss +=loss_f(logits,labels)
    loss /= idx+1

    return loss


# 返回训练出来的模型的正确率  dataset 是训练集  features是训练数据 label是对应标签
def accuracy_gela_dataset(model,dataset,ETF,Ew_set,client,device):
    model.eval()
    correct = 0
    for features, labels in dataset:
        if torch.cuda.is_available():
            h,predictions = model(features.to(device))
        else:
            h,predictions = model(features)

        labels = labels.long()
        learned_norm = Ew_set[client]
        cur_M = learned_norm * ETF
        out_g = torch.matmul(h, cur_M)


        y_pred = out_g.data.max(1,keepdim=True)[1]
        # print(f"label:{labels}  y :{y_pred}")
        if torch.cuda.is_available():
            correct += y_pred.eq(labels.data.view_as(y_pred).to(device)).long().sum()
        else:
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().sum()

    accuracy = 100 * correct / len(dataset.dataset)

    return accuracy




def load_model(dataset,seed):

    # 为cpu设置种子，生成随机数，使得每次实验生成的随机数一致
    # torch.manual_seed(seed)
    #     gpu
    torch.manual_seed(520)
    torch.cuda.manual_seed(520)
    torch.cuda.manual_seed(520)
    if dataset=="CIFAR" or dataset == "R_CIFAR":
        model = CNN_CIFAR()
    elif dataset[:8] == "cifar100":
        model = CNN_CIFAR100()
    elif dataset[:8] == "R_FMNIST":
        model = CNN_FMNIST()
    elif dataset[:7] == "R_MNIST":
        model = CNN_MNIST()
    elif dataset == "digit":
        model = DigitModel()
    return model

class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.proxy = proxies(10, 50, "etf")
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bn1(F.relu(F.max_pool2d(self.conv1(x), 2)))
        x = self.bn2(F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        a = self.proxy(x)
        return a, x

    def feature2logit(self, x):
        return self.fc2(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CNN_FMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_FMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.proxy = proxies(10, 128, "etf")
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes, bias=True)


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bn1(F.relu(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        a = self.proxy(x)
        return a, x
    def feature2logit(self, x):
        return self.fc2(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class DigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits.
    """

    def __init__(self, num_classes=10, **kwargs):
        super(DigitModel, self).__init__()

        # self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        # # self.bn1 = nn.BatchNorm2d(10)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        # # self.bn2 = nn.BatchNorm2d(20)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(500, 50)
        # self.fc2 = nn.Linear(50, 10)

        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)


        self.proxy = proxies(10,50,"etf")

    def forward(self, x):

        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # a = self.proxy(x)
        # return a,x

        x = self.bn1(F.relu(F.max_pool2d(self.conv1(x), 2)))
        x = self.bn2(F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        a = self.proxy(x)

        return a,x


class CNN_CIFAR100(nn.Module):
    def __init__(self, out_dim=100):
        super(CNN_CIFAR100, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.6)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 1600)
        self.fc2 = nn.Linear(1600, 512)
        self.fc3 = nn.Linear(512, out_dim)
        self.cls = out_dim
        self.proxy = proxies(100, 512, "etf")
        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.drop((F.relu(self.fc2(x))))
        a = self.proxy(x)
        return a,x


# class CNN_CIFAR(torch.nn.Module):
#   """Model Used by the paper introducing FedAvg"""
#   def __init__(self):
#         super(CNN_CIFAR, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3,out_channels=64, kernel_size=(3,3))
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3))
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3))
#         self.bn3 = nn.BatchNorm2d(32)
#
#         self.fc1 = nn.Linear(4*4*32, 64)
#         self.fc2 = nn.Linear(64, 10)
#         self.last_feature = None
#         self.proxy = proxies(10, 64, "etf")
#         self.weight_keys = [['fc1.weight', 'fc1.bias'],
#                       ['fc2.weight', 'fc2.bias'],
#                       ['conv3.weight', 'conv3.bias'],
#                       ['conv2.weight', 'conv2.bias'],
#                       ['conv1.weight', 'conv1.bias'],
#                       ]
#
#   def forward(self, x):
#        x = F.relu(self.conv1(x))
#        x = self.bn1(F.max_pool2d(x, 2, 2))
#
#        x = F.relu(self.conv2(x))
#        x = self.bn2(F.max_pool2d(x, 2, 2))
#
#        x = self.bn3(F.relu(self.conv3(x)))
#        self.last_feature = x.detach().clone()
#
#        x = x.view(-1, 4*4*32)
#        x = F.relu(self.fc1(x))
#
#        y = self.proxy(x)
#
#        return y,x
#
#   def feature2logit(self, x):
#       return self.fc2(x)
#
#   def num_flat_features(self, x):
#       size = x.size()[1:]
#       num_features = 1
#       for s in size:
#           num_features *= s
#       return num_features
class CNN_CIFAR(torch.nn.Module):
  """Model Used by the paper introducing FedAvg"""
  def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3))
        self.fc1 = nn.Linear(4*4*64, 64)
        self.fc2 = nn.Linear(64, 10)
        self.last_feature = None
        self.proxy = proxies(10, 64, "etf")
        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                      ['fc2.weight', 'fc2.bias'],
                      ['conv3.weight', 'conv3.bias'],
                      ['conv2.weight', 'conv2.bias'],
                      ['conv1.weight', 'conv1.bias'],
                      ]

  def forward(self, x):
       x = F.relu(self.conv1(x))
       x = F.max_pool2d(x, 2, 2)

       x = F.relu(self.conv2(x))
       x = F.max_pool2d(x, 2, 2)

       x=self.conv3(x)
       self.last_feature = x.detach().clone()

       x = x.view(-1, 4*4*64)
       x = F.relu(self.fc1(x))


       y = self.proxy(x)

       return y,x

  def feature2logit(self, x):
      return self.fc2(x)

  def num_flat_features(self, x):
      size = x.size()[1:]
      num_features = 1
      for s in size:
          num_features *= s
      return num_features


class proxies(nn.Module):
    def __init__(self,class_num=100,feat_dim=128,type="etf"):
        super(proxies,self).__init__()
        if type =="etf":
            self.proxy=nn.BatchNorm1d(feat_dim)

    def forward(self,feature):
        return self.proxy(feature)
