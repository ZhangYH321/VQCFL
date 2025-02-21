import torch
import torch.nn as nn
import torch.nn.functional as F
import copy



class NN(nn.Module):
    def __init__(self,layer_1,layer_2):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(784,layer_1)

        self.fc3 = nn.Linear(layer_1,10)


    def forward(self,x):

        # x = F.relu(self.fc1(x.view(-1,784).cuda()))
        x = F.relu(self.fc1(x.view(-1,784)))

        x = self.fc3(x)
        return x
# 三个卷积层 两个全连接层
class CNN_CIFAR_dropout(torch.nn.Module):
    """Model Used by the paper introducing FedAvg"""

    def __init__(self):
        super(CNN_CIFAR_dropout, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3)
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3)
        )

        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.dropout = nn.Dropout(p=0.2)
        # 记录最后的特征输出
        self.last_feature = None
        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['conv3.weight', 'conv3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.dropout(x)

        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




class Net_cifar(nn.Module):
    def __init__(self):
        super(Net_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # return F.log_softmax(x, dim=1)
        return x

class CNNCifar(nn.Module):

    def __init__(self):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)
        self.cls = 10
        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        # self.conv1_represention = x
        x = self.pool(F.relu(self.conv2(x)))
        # self.conv2_represention=x
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        # self.fc1_represention=x
        x = F.relu(self.fc2(x))
        # self.fc2_represention = x
        x = self.fc3(x)
        # self.fc3_represention = x
        return F.log_softmax(x, dim=1)


class CNN_CIFAR(torch.nn.Module):
    """Model Used by the paper introducing FedAvg"""

    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(4 * 4 * 32, 64)
        self.fc2 = nn.Linear(64, 10)
        self.last_feature = None

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['conv3.weight', 'conv3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.bn1(F.relu(F.max_pool2d(self.conv1(x), 2, 2)))

        x = self.bn2(F.relu(F.max_pool2d(self.conv2(x), 2, 2)))

        x = self.bn3(F.relu(self.conv3(x)))

        # f1 = x.detach().clone()
        # self.last_feature = x.detach().clone()
        # self.last_feature = x.clone()

        x = x.view(-1, 4 * 4 * 32)
        x = F.relu(self.fc1(x))

        y = self.fc2(x)
        # self.last_feature = x.detach().clone()
        # f2 = y.detach().clone()
        # return x,f1,f2
        return x, F.log_softmax(y, dim=1)

    def feature2logit(self, x):
        return self.fc2(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class AS_CNN_CIFAR(torch.nn.Module):
  """Model Used by the paper introducing FedAvg"""
  def __init__(self):
        super(AS_CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3))
        self.fc1 = nn.Linear(4*4*32, 64)
        self.fc2 = nn.Linear(64, 10)
        self.last_feature = None

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
       # f1 = x.detach().clone()
       self.last_feature = x.detach().clone()
       # self.last_feature = x.clone()
       x = x.view(-1, 4*4*32)
       x = F.relu(self.fc1(x))
       y = self.fc2(x)
       # self.last_feature = x.detach().clone()
       f2 = y.detach().clone()
       return x,F.log_softmax(y, dim=1)
  def feature2logit(self, x):
      return self.fc2(x)




class CNN_CIFAR_Autoencoder(nn.Module):
    def __init__(self):
        super(CNN_CIFAR_Autoencoder, self).__init__()

        # Encoder
        self.encoder_conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1)
        self.encoder_fc1 = nn.Linear(32 * 4 * 4, 128)

        # Decoder
        self.decoder_fc1 = nn.Linear(128, 32 * 4 * 4)
        self.decoder_conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        # Encoder
        # print(x.shape)
        x = F.relu(self.encoder_conv1(x))
        # print(x.shape)
        x = x.view(-1, 32 * 4 * 4)
        # print(x.shape)

        x = F.relu(self.encoder_fc1(x))
        # print(x.shape)

        # Decoder
        x = F.relu(self.decoder_fc1(x))
        # print(x.shape)

        x = x.view(-1, 32, 4, 4)
        # print(x.shape)
        x = F.relu(self.decoder_conv1(x))
        # print(x.shape)

        return x



class Net_mnist(nn.Module):
    def __init__(self):
        super(Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bn1(F.relu(F.max_pool2d(self.conv1(x), 2)))
        x = self.bn2(F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        feature1 = x
        x = self.fc2(x)
        # return x,feature1,feature2
        # return x
        return feature1,x


class VQ_Net_mnist(nn.Module):
    def __init__(self):
        super(VQ_Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(20)
        from test.vq_vae import VQVAE
        self.vq_vae = VQVAE(20, 20, 100)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bn1(F.relu(F.max_pool2d(self.conv1(x), 2)))
        x = self.bn2(F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)))


        recon, input, vq_loss = self.vq_vae(x)
        x = 0. * recon + x * 1

        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        y = self.fc2(x)

        return recon, input, vq_loss,F.log_softmax(y, dim=1)

class DigitModel(nn.Module):

    def __init__(self):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.bn1(F.relu(F.max_pool2d(self.conv1(x), 2)))
        x = self.bn2(F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        feature1 = x
        x = self.fc2(x)
        feature2 = x
        # return x,feature1,feature2
        # return x
        return feature1,F.log_softmax(x, dim=1)

class AS_DigitModel(nn.Module):
    def __init__(self):
        super(AS_DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        feature1 = x
        x = self.fc2(x)
        # return x,feature1,feature2
        # return x
        return feature1,F.log_softmax(x, dim=1)



class CifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes, bias=True)
        self.base_weight_keys = [
            'conv1.weight', 'conv1.bias',
            'conv2.weight', 'conv2.bias',
            'conv3.weight', 'conv3.bias',
            'fc1.weight', 'fc1.bias',
        ]
        self.classifier_weight_keys = [
            'fc2.weight', 'fc2.bias',
        ]

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        y = self.fc2(x)
        return x, y

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
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes, bias=True)
        self.base_weight_keys = ['conv1.weight', 'conv1.bias',
                                 'conv2.weight', 'conv2.bias',
                                 'fc1.weight', 'fc1.bias', ]
        self.classifier_weight_keys = ['fc2.weight', 'fc2.bias', ]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.bn1(F.max_pool2d(x, 2))
        x = F.relu(self.conv2(x))
        x = self.bn2(F.max_pool2d(x, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        return x,F.log_softmax(y, dim=1)

    def feature2logit(self, x):
        return self.fc2(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class VQ_CNN_FMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(VQ_CNN_FMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)

        from test.vq_vae import VQVAE
        self.vq_vae = VQVAE(32,32,100)

        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes, bias=True)
        self.base_weight_keys = ['conv1.weight', 'conv1.bias',
                                 'conv2.weight', 'conv2.bias',
                                 'fc1.weight', 'fc1.bias', ]
        self.classifier_weight_keys = ['fc2.weight', 'fc2.bias', ]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(F.max_pool2d(x, 2))

        recon, input, vq_loss = self.vq_vae(x)
        x = 0. * recon + x * 1

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        y = self.fc2(x)

        return recon, input, vq_loss, F.log_softmax(y, dim=1)

    def feature2logit(self, x):
        return self.fc2(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Digits_AutoEncoder(nn.Module):
    def __init__(self):
        super(Digits_AutoEncoder, self).__init__()

        # Encoder
        self.encoder_conv1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), padding=1)
        # self.encoder_bn1 = nn.BatchNorm2d(32)

        self.encoder_fc1 = nn.Linear(32 * 7 * 7, 128)

        # Decoder
        self.decoder_fc1 = nn.Linear(128, 32 * 7 * 7)
        self.decoder_conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=128, kernel_size=(3, 3), padding=1)
        # self.decoder_bn1 = nn.BatchNorm2d(128)
    def forward(self, x):
        # Encoder
        # print(x.shape)
        # x = F.relu(self.encoder_bn1(self.encoder_conv1(x)))
        x = F.relu(self.encoder_conv1(x))
        # print(x.shape)
        x = x.view(-1, 32 * 7 * 7)
        # print(x.shape)

        x = F.relu(self.encoder_fc1(x))
        # print(x.shape)

        # Decoder
        x = F.relu(self.decoder_fc1(x))
        # print(x.shape)

        x = x.view(-1, 32, 7, 7)
        # print(x.shape)
        # x = F.relu(self.decoder_bn1(self.decoder_conv1(x)))
        x = F.relu(self.decoder_conv1(x))

        # print(x.shape)

        return x


def load_autoEncoder(dataset,seed):
    torch.cuda.manual_seed(seed)
    if dataset[:5] == "CIFAR":
        autoEncoder = CNN_CIFAR_Autoencoder()
    elif dataset[:6] == "digits":
        autoEncoder = Digits_AutoEncoder()
    return autoEncoder





class CNN_VQ_CIFAR(torch.nn.Module):


    def __init__(self):
        super(CNN_VQ_CIFAR, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64, kernel_size=(3,3))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3))
        self.bn3 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(4*4*32, 64)
        from test.vq_vae import VQVAE
        self.vq_vae = VQVAE(32,32,100)
        self.fc2 = nn.Linear(64, 10)

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                      ['fc2.weight', 'fc2.bias'],
                      ['conv3.weight', 'conv3.bias'],
                      ['conv2.weight', 'conv2.bias'],
                      ['conv1.weight', 'conv1.bias'],
                      ]
    def forward(self, x):

        x = self.bn1(F.relu(F.max_pool2d(self.conv1(x), 2, 2)))

        x = self.bn2(F.relu(F.max_pool2d(self.conv2(x), 2, 2)))

        x = self.bn3(F.relu(self.conv3(x)))

        recon, input, vq_loss = self.vq_vae(x)


        x = 0.5 * recon + x * 1


        x = x.view(-1, 4*4*32)
        x = F.relu(self.fc1(x))
        y = self.fc2(x)

        return recon, input, vq_loss,F.log_softmax(y, dim=1)


class VQ_DigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits.
    """


    def __init__(self):
        super(VQ_DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(20)
        from test.vq_vae import VQVAE
        self.vq_vae = VQVAE(20, 20, 100)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.bn1(F.relu(F.max_pool2d(self.conv1(x), 2)))
        x = self.bn2(F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)))


        recon, input, vq_loss = self.vq_vae(x)

        x = 0.0 * recon + x * 1

        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        y = self.fc2(x)

        return recon, input, vq_loss,F.log_softmax(y, dim=1)


def load_model(dataset,seed,*args):

    # 为cpu设置种子，生成随机数，使得每次实验生成的随机数一致
    # torch.manual_seed(seed)
    #     gpu
    torch.manual_seed(520)
    torch.cuda.manual_seed(520)
    torch.cuda.manual_seed(520)
    if "digit" in dataset[:7]:
        if args[0] == "FedVQ":
            model = VQ_DigitModel()
        # elif args[0] == "FedAS":
        #     model = AS_DigitModel()
        else:
            model = DigitModel()

    if dataset[:5]=="CIFAR":
        # model = CifarCNN()
        if args[0] == "FedVQ":
            model = CNN_VQ_CIFAR()
        elif args[0] == "FedAS":
            model = AS_CNN_CIFAR()
        else:
            model = CNN_CIFAR()

    if dataset[:7]=="R_CIFAR":
        if args[0] == "FedVQ":
            model = CNN_VQ_CIFAR()
        elif args[0] == "FedAS":
            model = AS_CNN_CIFAR()
        else:
            model = CNN_CIFAR()

    if dataset[:7] == "fashion":
        model = CNN_FMNIST()

    if dataset[:8] == "R_FMNIST":
        if args[0] == "FedVQ":
            model = VQ_CNN_FMNIST()
        else:
            model = CNN_FMNIST()


    if dataset[:5]=="MNIST" :
        model = Net_mnist()

    if dataset[:7]=="R_MNIST":
        if args[0] == "FedVQ":
            model = VQ_Net_mnist()
        else:
            model = Net_mnist()

    return model
