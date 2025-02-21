import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import sys
from itertools import product
from copy import deepcopy
from utils.NCloss import *
import math
from torch.nn import functional as F
import os
from utils.pactools import *
import argparse

# 将模型的所有参数置为0  即初始训练情况
from torch.utils.data import DataLoader


def set_to_zero_model_weights(model):

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)


#  加权聚合 返回聚合后的模型
def FedAvg_agregation_process(model, clients_models_hist: list, weights: list):
    new_model = deepcopy(model)
    set_to_zero_model_weights(new_model)
    for k, client_hist in enumerate(clients_models_hist):
        for idx, layer_weights in enumerate(new_model.parameters()):
            contribution = client_hist[idx].data * weights[k]
            layer_weights.data.add_(contribution)
    return new_model

def agregation_process(model, clients_models_hist: list, weights: list):
    new_model = deepcopy(model)
    set_to_zero_model_weights(new_model)
    param = model.state_dict()

    for key in param.keys():
        for i in range(len(clients_models_hist)):
            param[key] += clients_models_hist[i].state_dict()[key]*weights[i]
    new_model.load_state_dict(param)
    return new_model


def agregation_process_bn(model, clients_models_hist: list, weights: list,glob_keys,head_keys):
    new_model = deepcopy(model)
    set_to_zero_model_weights(new_model)
    param = model.state_dict()

    for key in param.keys():
        if key in glob_keys or head_keys:
            for i in range(len(clients_models_hist)):
                param[key] += clients_models_hist[i].state_dict()[key]*weights[i]
    new_model.load_state_dict(param)
    return new_model



# 返回两个模型之间的norm2 距离
def difference_models_norm_2(model_1,model_2):
    tensor_1 = list(model_1.parameters())
    tensor_2 = list(model_2.parameters())

    norm = sum(
        [
            torch.sum((tensor_1[i] - tensor_2[i]) ** 2)
            for i in range(len(tensor_1))
        ]
    )
    return norm


# 返回一个计算预测值和标签值之间的损失函数
def loss_classifier(predictions, labels):

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        return criterion(predictions.cuda(), labels.cuda())
    else:
        return criterion(predictions, labels)















# 返回训练出来的模型的正确率  dataset 是训练集  features是训练数据 label是对应标签
def accuracy_dataset(model,dataset):
    model.eval()
    correct = 0
    for features, labels in dataset:
        if torch.cuda.is_available():
            predictions = model(features.cuda())[-1]
        else:
            predictions = model(features)[-1]

        # _, predicted = predictions.max(1, keepdim=True)

        # correct += torch.sum(predicted.view(-1, 1).cuda() == labels.view(-1, 1).cuda()).item()
        # correct += torch.sum(predicted.view(-1, 1) == labels.view(-1, 1)).item()
        y_pred = predictions.data.max(1,keepdim=True)[1]
        # print(f"label:{labels}  y :{y_pred}")
        if torch.cuda.is_available():
            correct += y_pred.eq(labels.data.view_as(y_pred).cuda()).long().sum()
        else:
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().sum()

    accuracy = 100 * correct / len(dataset.dataset)

    return accuracy

def pac_accuracy_dataset(model,dataset):
    model.eval()
    correct = 0
    for features, labels in dataset:
        if torch.cuda.is_available():
            protos,predictions = model(features.cuda())
        else:
            protos,predictions = model(features)

        # _, predicted = predictions.max(1, keepdim=True)

        # correct += torch.sum(predicted.view(-1, 1).cuda() == labels.view(-1, 1).cuda()).item()
        # correct += torch.sum(predicted.view(-1, 1) == labels.view(-1, 1)).item()
        y_pred = predictions.data.max(1,keepdim=True)[1]
        # print(f"label:{labels}  y :{y_pred}")
        if torch.cuda.is_available():
            correct += y_pred.eq(labels.data.view_as(y_pred).cuda()).long().sum()
        else:
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().sum()

    accuracy = 100 * correct / len(dataset.dataset)

    return accuracy

# 返回模型在训练集上的损失
def loss_dataset(model, train_data, loss_f):
    model.eval()
    loss = 0
    for idx,(features, labels) in enumerate(train_data):
        if torch.cuda.is_available():
            predictions = model(features.cuda(),)[-1]
        else:
            predictions = model(features)[-1]
        labels = labels.long()
        loss += loss_f(predictions,labels)
    loss /= idx+1

    return loss

def loss_pac_dataset(model, train_data, loss_f):
    model.eval()
    loss = 0
    for idx,(features, labels) in enumerate(train_data):
        if torch.cuda.is_available():
            protos,predictions = model(features.cuda())
        else:
            protos,predictions = model(features)
        labels=labels.long()
        loss +=loss_f(predictions,labels)
    loss /= idx+1
    return loss



# 返回模型中的参数个数
def n_params(model):
    n_params = sum(
        [
            np.prod([tensor.size()[k] for k in range(len(tensor.size()))])
            for tensor in list(model.parameters())
        ]
    )

    return n_params

# 客户端本地训练

def local_learning(model, mu: float, optimizer, train_data, n_SGD: int, loss_f):
    model_0 = deepcopy(model)
    for _ in range(n_SGD):
        features, labels = next(iter(train_data))
        optimizer.zero_grad()
        predictions = model(features.cuda())
        # labels
        batch_loss = loss_f(predictions, labels)
        # mu != 0 表示prox
        batch_loss += mu / 2 * difference_models_norm_2(model_0, model)

        batch_loss.backward()
        optimizer.step()


def personal_train(train_model,glob_keys,n_SGD,loss_f,train_data, mu,isPersonal ,lr,device):

    model_0 = deepcopy(train_model)
    train_model.train()
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

    # optimizer = torch.optim.Adam(train_model.parameters(), lr=1e-3)

    #     共享层和 个性化层 单独梯度下降
    head_eps = n_SGD-1
    from tqdm import tqdm
    len_batch = len(train_data)
    # tqdm_bar = tqdm(range(n_SGD), desc="Training")
    for i in range(n_SGD):
        if(i<head_eps) and isPersonal:
            for name,param in train_model.named_parameters():
                if name in glob_keys:
                    # 将这一层屏蔽
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        elif i==head_eps and isPersonal:
            for name,param in train_model.named_parameters():
                if name in glob_keys:
                    # 将这一层屏蔽
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif not isPersonal:
            for name, param in train_model.named_parameters():
                param.requires_grad = True
        for j in range(len_batch):
            features, labels = next(iter(train_data))

            labels = labels.long()
            if torch.cuda.is_available():
                _,predictions = train_model(features.cuda())
            else:
                _,predictions = train_model(features)

            batch_loss = loss_f(predictions,labels)

            back_loss = mu/2 * difference_models_norm_2(model_0,train_model)

            batch_loss += back_loss

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # if j % 20 == 0:
            #     tqdm_bar.set_description(
            #         'total_loss: {}'.format(batch_loss))

    return train_model.state_dict()



def lg_train(train_model,glob_keys,n_SGD,loss_f,train_data,lr):

    model_0 = deepcopy(train_model)

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

    n_SGD = int(n_SGD/2)+1
    len_batch = len(train_data)
    for i in range(n_SGD):

        for name,param in train_model.named_parameters():
            if name in glob_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for j in range(len_batch):
            features, labels = next(iter(train_data))

            labels = labels.long()
            if torch.cuda.is_available():
                _,predictions = train_model(features.cuda())
            else:
                _,predictions = train_model(features)

            batch_loss = loss_f(predictions,labels)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            for name, param in train_model.named_parameters():
                if name in glob_keys:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            if torch.cuda.is_available():
                _,predictions = train_model(features.cuda())
            else:
                _,predictions = train_model(features)

            batch_loss = loss_f(predictions,labels)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()






    return train_model.state_dict()

def pac_train(train_model,glob_keys,head_keys,n_SGD,loss_f,train_data, mu ,lr,global_protos):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mse_loss = nn.MSELoss()
    model_0 = deepcopy(train_model)
    model_0.train()
    bias_p = []
    weight_p = []
    model_0.zero_grad()
    for name, p in train_model.named_parameters():
        # print(f"name :{name} p :{p}")
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    local_protos1 = get_local_protos(train_model,train_data,device)

    local_ep_rep = n_SGD
    epoch_classifier = 1
    local_epoch = int(epoch_classifier + local_ep_rep)

    if local_epoch>0:
        for name,param in train_model.named_parameters():
            if name in glob_keys:
                param.requires_grad = False
            elif name in head_keys:
                param.requires_grad = True
        lr_g = 0.1
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, train_model.parameters()), lr=lr_g,
                                    momentum=0.5, weight_decay=0.0005)
        for ep in range(epoch_classifier):
            # local training for 1 epoch
            data_loader = iter(train_data)
            iter_num = len(data_loader)
            for it in range(iter_num):
                images, labels = next(data_loader)
                labels = labels.long()
                images,labels = images.to(device),labels.to(device)
                train_model.zero_grad()

                protos, output = train_model(images)

                loss = loss_f(output, labels)

                loss.backward()

                optimizer.step()

        for name, param in train_model.named_parameters():
            if name in glob_keys:
                param.requires_grad = True
            elif name in head_keys:
                param.requires_grad = False

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, train_model.parameters()), lr=lr,
                                    momentum=0.5, weight_decay=0.0005)
        round = 0
        for ep in range(local_ep_rep):
            data_loader = iter(train_data)
            iter_num = len(data_loader)
            for it in range(iter_num):
                images, labels = next(data_loader)
                labels = labels.long()
                images,labels = images.to(device),labels.to(device)
                train_model.zero_grad()
                protos, output = train_model(images)
                loss0 = loss_f(output, labels)
                loss1 = 0
                if round > 0:
                    loss1 = 0
                    protos_new = protos.clone().detach()
                    for i in range(len(labels)):
                        yi = labels[i].item()
                        if yi in global_protos:
                            protos_new[i] = global_protos[yi].detach()
                        else:
                            protos_new[i] = local_protos1[yi].detach()
                    loss1 = mse_loss(protos_new, protos)
                round+=1
                loss = loss0 + 1.0 * loss1
                loss.backward()
                optimizer.step()
    local_protos2 = get_local_protos(train_model,train_data,device)
    return train_model.state_dict(),local_protos2



def new_train(train_model,n_SGD,loss_f,train_data,lr,train_autoEncoder_models,device,mu):
    model_0 = deepcopy(train_model)

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
    feature_list =[]
    label_list = []

    for i in range(n_SGD):

        for name, param in train_model.named_parameters():
            param.requires_grad = True
        for k in range(len(train_autoEncoder_models)):
            for name, param in train_autoEncoder_models[k].named_parameters():
                param.requires_grad = False
        #开始训练
        # 当前batch的数据
        features, labels = next(iter(train_data))
        labels = labels.long()
        train_model.last_feature = None
        if device.type=="cpu":
            predictions = train_model(features)
        else:
            predictions = train_model(features.cuda())

        # 记录当前batch保留的feature
        teach_loss = 0.0
        # batch_size* 特征维度
        last_features = train_model.last_feature
        if last_features != None:
            for j in range(len(last_features)):
                feature = last_features[j]
                label = labels[j]

                # print(feature.size())
                feature = feature.unsqueeze(0)
                # print(feature.size())
                if i>=n_SGD-10:
                    feature_list.append(feature)
                    label_list.append(label)
                new_feature = train_autoEncoder_models[label](feature)
                teach_loss += torch.norm(feature-new_feature,p=2).sum()

        batch_loss = loss_f(predictions,labels)+mu*teach_loss
        # batch_loss = loss_f(predictions, labels)
        # print(f"第{i}次batch的loss :{batch_loss}，teach_loss : {teach_loss}")
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    for k in range(len(train_autoEncoder_models)):
        for name, param in train_autoEncoder_models[k].named_parameters():
            param.requires_grad = True

    return train_model.state_dict(),feature_list,label_list



def cw_train(train_model, n_SGD, loss_f, train_data, lr,device):
    model_0 = deepcopy(train_model)

    bias_p = []
    weight_p = []

    for name, p in train_model.named_parameters():
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
    len_batch = len(train_data)
    for i in range(n_SGD):

        for name, param in train_model.named_parameters():
            param.requires_grad = True
        for j in range(len_batch):
            features, labels = next(iter(train_data))
            labels = labels.long()

            if device.type == "cpu":
                _,predictions = train_model(features)
            else:
                _,predictions = train_model(features.cuda())

            batch_loss = loss_f(predictions, labels)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

    return train_model.state_dict()

def cfa_train(train_model,glob_keys,head_keys,n_SGD,loss_f,train_data, lr,global_protos,lambda1,lambda2,label_count):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mse_loss = nn.MSELoss()
    model_0 = deepcopy(train_model)
    model_0.train()
    bias_p = []
    weight_p = []
    model_0.zero_grad()
    for name, p in train_model.named_parameters():
        # print(f"name :{name} p :{p}")
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    local_protos1 = get_local_protos(train_model, train_data)

    local_ep_rep = n_SGD
    epoch_classifier = 2
    local_epoch = int(epoch_classifier + local_ep_rep)


    if local_epoch > 0:
        for name, param in train_model.named_parameters():
            if name in glob_keys:
                param.requires_grad = False
            elif name in head_keys:
                param.requires_grad = True
        lr_g = 0.1
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, train_model.parameters()), lr=lr_g,
                                    momentum=0.5, weight_decay=0.0005)
        # step1： 对分类器进行微调
        for ep in range(epoch_classifier):
            # local training for 1 epoch
            data_loader = iter(train_data)
            iter_num = len(data_loader)
            for it in range(iter_num):
                images, labels = next(data_loader)
                labels = labels.long()
                images, labels = images.to(device), labels.to(device)
                train_model.zero_grad()
                protos, output = train_model(images)
                loss = loss_f(output, labels)
                loss.backward()
                optimizer.step()


        # step2 一起训练  添加类间分离损失和类内聚合损失
        for name, param in train_model.named_parameters():
            if name in glob_keys:
                param.requires_grad = True
            elif name in head_keys:
                param.requires_grad = True

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, train_model.parameters()), lr=lr,
                                    momentum=0.5, weight_decay=0.0005)

        class_weights = label_count / sum(label_count)
        inverse_weights = [1.0 / weight for weight in class_weights]
        # 计算每个向量的权重 用于平衡 loss1
        weights = torch.tensor([inverse_weights[label] for label in labels])
        weights = weights/sum(weights)
        weights = weights.to(device)
        round = 0
        for ep in range(local_ep_rep):
            data_loader = iter(train_data)
            iter_num = len(data_loader)
            for it in range(iter_num):
                images, labels = next(data_loader)
                labels = labels.long()
                images, labels = images.to(device), labels.to(device)
                train_model.zero_grad()
                protos, output = train_model(images)
                loss0 = loss_f(output, labels)
                # 类内对齐损失
                loss1 = 0.0
                # 类间分离损失
                loss2 = 0.0
                # 计算nk

                if round > 0:
                    protos_new = protos.clone().detach()
                    for i in range(len(labels)):
                        yi = labels[i].item()
                        if yi in global_protos:
                            protos_new[i] = global_protos[yi].detach()
                        else:
                            protos_new[i] = local_protos1[yi].detach()
                    # nc1 1/nk || x - c ||
                    # losst = mse_loss(protos_new, protos)
                    # loss1 = losst * weights
                    loss1 = NC1Loss(protos_new,protos,labels,weights)
                    loss2,max_cos = NC2Loss(protos)

                round += 1

                loss = loss0 + lambda1 * loss1 + lambda2 *loss2
                loss.backward()
                optimizer.step()

    local_protos2 = get_local_protos(train_model, train_data)
    return train_model.state_dict(), local_protos2




def autoEncoder_train(
    encoder_model,
    train_list,
    label
):
    model = deepcopy(encoder_model)
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    criterion = nn.MSELoss()
    epochs = 10
    from utils.get_data import featureDataset
    if len(train_list) == 0:
        return model.state_dict()
    dataset_object = featureDataset(train_list,[label]*len(train_list))
    dataset_dl = DataLoader(
        dataset_object, batch_size=50, shuffle=True
    )


    for epoch in range(epochs):
        data, labels = next(iter(dataset_dl))

        optimizer.zero_grad()

        outputs = model(data)

        loss = criterion(outputs,data)
        loss.backward()



        optimizer.step()

    return model.state_dict()


def meta_train(params_list,glob_param,local_metaModel,train_data,glob_keys,head_keys):
    local_metaModel.train()
    len_batch = len(train_data)
    for i in range(len_batch):
        features, labels = next(iter(train_data))
        features = features.to('cuda:0')
        labels = labels.to('cuda:0')
        # labels = labels.long()
        local_metaModel(params_list,glob_param,features,labels,glob_keys,head_keys)

    classifier_param = local_metaModel.get_final_classifier(params_list,glob_param,head_keys)
    final_train_param = deepcopy(glob_param)
    for key in head_keys:
        final_train_param[key] = classifier_param[key]
    return final_train_param,local_metaModel.state_dict()

def train(train_model,n_SGD,loss_f,train_data, mu,lr):
    model_0 = deepcopy(train_model)

    bias_p = []
    weight_p = []
    train_model.train()
    for name, p in train_model.named_parameters():
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
    len_batch = len(train_data)
    for i in range(n_SGD):
        for j in range(len_batch):
            features, labels = next(iter(train_data))
            labels = labels.long()
            train_model.zero_grad()
            if torch.cuda.is_available():
                _,predictions = train_model(features.cuda())
            else:
                _,predictions = train_model(features)

            batch_loss = loss_f(predictions, labels)
            back_loss = mu / 2 * difference_models_norm_2(model_0, train_model)
            # print(f"模型变化量{difference_models_norm_2(model_0, train_model)}")
            batch_loss += back_loss

            batch_loss.backward()
            optimizer.step()
    return train_model.state_dict()



def vq_loss_function(recon, input, vq_loss):

    recons_loss = F.mse_loss(recon, input)

    loss = recons_loss + vq_loss
    return {'loss': loss,
            'Reconstruction_Loss': recons_loss,
            'VQ_Loss': vq_loss}

from torch.nn.utils import clip_grad_norm_

def as_train(train_model,n_SGD,loss_f,train_data,lr,fim_trace_history,device):

    train_model.to(device)
    train_model.train()
    mse_loss = nn.MSELoss()
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
    is_selected = True
    if is_selected == True:
        local_epoch = n_SGD
        for e in range(local_epoch):
            data_loader = iter(train_data)
            iter_num = len(data_loader)
            for it in range(iter_num):
                images, labels = next(data_loader)
                labels = labels.long()
                images, labels = images.to(device), labels.to(device)

                protos, output = train_model(images)

                loss = loss_f(output, labels)
                optimizer.zero_grad()
                loss.backward()
                # clip_grad_norm_(train_model.parameters(), max_norm=1.0)  # max_norm 是梯度的最大范数
                optimizer.step()

        # compute FIM and its trace
        train_model.eval()
        data_loader = iter(train_data)
        iter_num = len(data_loader)
        fim_trace_sum = 0
        for it in range(iter_num):
            images, labels = next(data_loader)
            labels = labels.long()
            images, labels = images.to(device), labels.to(device)

            protos, output = train_model(images)

            # nll = -torch.nn.functional.log_softmax(output, dim=1)[range(len(labels)), labels].mean()
            nll = -output[range(len(labels)), labels].mean()

            from torch.autograd import grad
            params_to_grad = []
            for name, param in train_model.named_parameters():
                if "bn" not in name:  # 检查是否为 BN 层
                    if param.requires_grad:
                        params_to_grad.append(param)
            grads = grad(nll,  params_to_grad)
            for g in grads:
                fim_trace_sum += torch.sum(g ** 2).detach()

        fim_trace_history.append(fim_trace_sum.item())

    return train_model.state_dict(),fim_trace_history



def vq_train(train_model,n_SGD,loss_f,train_data,lr,anchors,PretrainIter,device,train_iter):
    model_0 = deepcopy(train_model)

    bias_p = []
    weight_p = []
    train_model.train()
    for name, p in train_model.named_parameters():
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
        momentum=0.9

    )

    # optimizer = torch.optim.Adam(train_model.parameters(), lr=1e-3)
    from tqdm import tqdm
    len_batch = len(train_data)
    # tqdm_bar = tqdm(range(n_SGD), desc="Training")
    for epoch in range(n_SGD):
        for j in range(len_batch):
            features, labels = next(iter(train_data))
            labels = labels.long().to(device)
            # train_model.zero_grad()
            optimizer.zero_grad()
            features = features.to(device)
            recon, input, vq_loss, predictions = train_model(features)

            cls_loss = nn.CrossEntropyLoss()(predictions, labels)

            loss = train_model.vq_vae.loss_function(recon, input, vq_loss)['loss']
            embedding1 = train_model.vq_vae.vq_layer.embedding.weight
            anchor_loss = achorLoss(embedding1, anchors)

            if train_iter < PretrainIter:
                total_loss = loss * (0.8 **(train_iter/2)) + anchor_loss * 0.8 + cls_loss
                # total_loss = loss * 0.8 + anchor_loss * 0.8 + cls_loss
            else:
                total_loss = cls_loss

            total_loss.backward()
            optimizer.step()

            # if j % 20 == 0:
            #     tqdm_bar.set_description(
            #         'total_loss: {},loss : {},anchor_loss :  {},cls_loss : {}'.format(total_loss, loss,
            #                                                                           anchor_loss,
            #                                                                           cls_loss))


    return train_model.state_dict()



def achorLoss(codebook,anchors):
    total_loss = 0
    for i in range(len(codebook)):
        distance = torch.maximum(torch.norm(codebook[i] - anchors[i]) - 0.1,torch.tensor(0.))
        total_loss += distance

    return total_loss

def proto_train(train_model,n_SGD,loss_f,train_data, mu ,lr,global_protos,device):

    mse_loss = nn.MSELoss()
    model_0 = deepcopy(train_model)
    model_0.train()
    bias_p = []
    weight_p = []
    model_0.zero_grad()
    for name, p in train_model.named_parameters():
        # print(f"name :{name} p :{p}")
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    local_protos1 = get_local_protos(train_model,train_data,device)
    optimizer = torch.optim.SGD(
        [
            {'params': weight_p, 'weight_decay': 0.0001},
            {'params': bias_p, 'weight_decay': 0}
        ],
        lr=lr,
        momentum=0.5

    )
    round = 0
    for ep in range(n_SGD):
        data_loader = iter(train_data)
        iter_num = len(data_loader)
        for it in range(iter_num):
            images, labels = next(data_loader)
            labels = labels.long()
            images, labels = images.to(device), labels.to(device)
            train_model.zero_grad()
            protos, output = train_model(images)
            loss0 = loss_f(output, labels)
            loss1 = 0.0
            if round > 0:

                protos_new = protos.clone().detach()
                for i in range(len(labels)):
                    yi = labels[i].item()
                    if yi in global_protos:
                        protos_new[i] = global_protos[yi].detach()
                    else:
                        protos_new[i] = local_protos1[yi].detach()
                loss1 = mse_loss(protos_new, protos)
            round += 1
            loss = loss0 + 1.0 * loss1
            loss.backward()
            optimizer.step()
    local_protos2 = get_local_protos(train_model, train_data,device)
    return train_model.state_dict(), local_protos2


def layer_wised_train(train_model, model_keys,pweight,idx_client, n_SGD, loss_f, train_data, mu,  lr):
    # model_keys[i]  表示 [weight,bias]


    model_0 = deepcopy(train_model)

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

    for i in range(n_SGD):

        # 开始训练咯
        # for name, param in train_model.named_parameters():
        #     print(f"{name}: {param.requires_grad}")
        features, labels = next(iter(train_data))

        train_model.zero_grad()

        predictions = train_model(features.cuda())
        batch_loss = loss_f(predictions, labels)
        # 每一层的二范数损失
        back_loss = 0.0

        for k in range(len(model_keys)):
        #     model_keys[k][0] 表示weight  model_keys[k][1]表示bias
        #     求train_model 和model_0 在第k层 即key时的二范数损失
        #     lamda[i][k]表示客户端i第k层的重要程度

            weight1 = model_0.state_dict()[model_keys[k][0]]
            bias1 = model_0.state_dict()[model_keys[k][1]]

            weight2 = train_model.state_dict()[model_keys[k][0]]
            bias2 = train_model.state_dict()[model_keys[k][1]]

            param1 = torch.cat([weight1.view(-1),bias1.view(-1)])
            param2 = torch.cat([weight2.view(-1), bias2.view(-1)])
            t = pweight[idx_client][k]*mu/2 * get_tensor_l2(param1,param2)
            back_loss += t


        # print(f"前面的loss : {batch_loss}")
        # print(f"后面的loss :{back_loss}")

        batch_loss += back_loss

        batch_loss.backward()

        optimizer.step()
    # print("-----------------------------------------------------------------------------------------------------")
    return train_model.state_dict()





 # 保存文件
def save_pkl(dictionnary, directory, file_name):
    flord = os.getcwd()
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)

def save_model(model,directory,file_name):
    flord = os.getcwd()
    os.makedirs(directory,exist_ok=True)
    torch.save(model, f"{flord}/{directory}/{file_name}")

# 获取两个梯度之间的模型相似度
def get_similarity(grad_1,grad_2, distance_type="cosine"):
    # L1 两向量差的绝对值求和  L2 方差  cosine 余弦相似度
    if distance_type =="L1":
        norm = 0
        for g_1,g_2 in zip(grad_1,grad_2):
            norm +=np.sum(np.abs(g_1-g_2))
        return norm

    elif distance_type=="L2":
        norm =0
        for g_1,g_2 in zip(grad_1,grad_2):
            norm +=np.sum((g_1-g_2)**2)
        return norm
    # a * b / (|a|*|b|)
    elif distance_type =="cosine":
        norm,norm_1,norm_2=0,0,0
        for i in range(len(grad_1)):
            norm += np.sum(grad_1[i]*grad_2[i])
            norm_1 += np.sum(grad_1[i] ** 2)
            norm_2 += np.sum(grad_2[i] ** 2)

        if norm_1 == 0.0 or norm_2 == 0.0:
            return 0.0
        else:
            norm /= np.sqrt(norm_1 * norm_2)

            return np.arccos(norm)



def get_distance_by_model(model_1,model_2,distance_type='cosine'):
    model_1.cpu()
    model_2.cpu()
    # 两个模型的参数
    param1 = model_1.state_dict()
    param2 = model_2.state_dict()
    res = 0.0
    list1 = []
    list2 = []
    for key in param1.keys():
        list1 += param1[key].numpy().flatten().tolist()
        list2 += param2[key].numpy().flatten().tolist()
    #
    a = 0.0
    b = 0.0
    c = 0.0
    for i in range(len(list1)):
        a += list1[i]*list2[i]
        b += list1[i]*list1[i]
        c += list2[i]*list2[i]

    res = a/np.sqrt(b*c)
    return res


def get_distance_by_model_2(model_1,model_2,distance_type='cosine'):

    params1 = []
    for p in model_1.parameters():
        params1.append(p)
    m_v1 = torch.cat([p.view(-1) for p in params1], dim=0)
    m_v1 = m_v1.detach().cpu().numpy()

    params2 = []
    for p in model_2.parameters():
        params2.append(p)
    m_v2 = torch.cat([p.view(-1) for p in params2], dim=0)
    m_v2 = m_v2.detach().cpu().numpy()

    cosine_similarity = np.dot(m_v1, m_v2) / (np.linalg.norm(m_v1) * np.linalg.norm(m_v2))


    return cosine_similarity


def get_distance_by_model_not_glob_keys(model_1, model_2, glob_keys,distance_type='cosine'):
    params1 = []

    for key in model_1.state_dict().keys():
        if key not in glob_keys:
            # print(f"添加了{key}")
            params1.append(model_1.state_dict()[key])

    m_v1 = torch.cat([p.view(-1) for p in params1], dim=0)
    m_v1 = m_v1.detach().cpu().numpy()

    params2 = []
    for key in model_2.state_dict().keys():
        if key not in glob_keys:
            params2.append(model_1.state_dict()[key])
    m_v2 = torch.cat([p.view(-1) for p in params2], dim=0)
    m_v2 = m_v2.detach().cpu().numpy()

    cosine_similarity = np.dot(m_v1, m_v2) / (np.linalg.norm(m_v1) * np.linalg.norm(m_v2))

    return cosine_similarity

def get_distance_by_model_in_glob_keys(model_1, model_2, glob_keys,distance_type='cosine'):
    params1 = []

    for key in model_1.state_dict().keys():
        if key  in glob_keys:
            # print(f"添加了{key}")
            params1.append(model_1.state_dict()[key])

    m_v1 = torch.cat([p.view(-1) for p in params1], dim=0)
    m_v1 = m_v1.detach().cpu().numpy()

    params2 = []
    for key in model_2.state_dict().keys():
        if key  in glob_keys:
            params2.append(model_1.state_dict()[key])
    m_v2 = torch.cat([p.view(-1) for p in params2], dim=0)
    m_v2 = m_v2.detach().cpu().numpy()

    cosine_similarity = np.dot(m_v1, m_v2) / (np.linalg.norm(m_v1) * np.linalg.norm(m_v2))

    return cosine_similarity


# 返回局部模型和全局模型之间的代表性梯度
def get_gradients(global_m,local_models):

    local_model_params =[]
    for model in local_models:
        model.cpu()
        local_model_params +=[
            [tens.detach().numpy() for tens in list(model.parameters())]
        ]
        model.cuda()
    # model 不能这么遍历 要改成参数 然后更新model
    global_m.cpu()
    global_model_params = [
        tens.detach().numpy() for tens in list(global_m.parameters())
    ]
    global_m.cuda()
    local_model_grads = []

    for local_params in local_model_params:
        local_model_grads += [
            [
                local_weights - global_weights
                for local_weights, global_weights in zip(
                local_params, global_model_params
                )
            ]
        ]

    return local_model_grads

# 返回单个模型和全局模型之间的梯度
def get_gradient(global_m,local_model):
    local_model_params = []
    model = deepcopy(local_model)
    model.cpu()
    local_model_param = [tens.detach().numpy() for tens in list(model.parameters())]
    model.cuda()
    global_m.cpu()
    global_model_params = [
        tens.detach().numpy() for tens in list(global_m.parameters())
    ]
    global_m.cuda()
    local_model_grad = [
            local_weights - global_weights
            for local_weights, global_weights in zip(
            local_model_param, global_model_params
        )
    ]
    return local_model_grad


# 根据模型的梯度数组 返回相似度矩阵
def get_matrix_similarity_from_grads(local_model_grads, distance_type):

    n_clients = len(local_model_grads)

    metric_matrix = np.zeros((n_clients, n_clients))

    for i,j in product(range(n_clients),range(n_clients)):
        metric_matrix[i,j] = get_similarity(
            local_model_grads[i],local_model_grads[j],distance_type
        )
    return metric_matrix

# 根据全局模型和局部模型数组 返回相似度矩阵
def get_matrix_similarity(global_m,local_models,distance_type):
    n_clients = len(local_models)

    local_model_grads = get_gradients(global_m,local_models)

    metric_matrix = np.zeros((n_clients, n_clients))
    for i, j in product(range(n_clients), range(n_clients)):
        metric_matrix[i, j] = get_similarity(
            local_model_grads[i], local_model_grads[j], distance_type
        )

    return metric_matrix


def inter_group_aggregation(group_models,k,weights,glob_keys):
    # 根据权重聚合模型
    model = deepcopy(group_models[k])
    param = model.state_dict()
    for key in param.keys():
        param[key] = param[key]*weights[k]

    for i in range(len(group_models)):
        if i==k:
            continue
        else:
            for key in param.keys():
                param[key] += group_models[i].state_dict()[key] * weights[i]
    model.load_state_dict(param)

    return model


def inter_group_aggregation_not_glob_keys(group_models,weights,cur_class,glob_keys):
    # 根据权重聚合模型
    model = deepcopy(group_models[cur_class])
    w_model = model.state_dict()
    param = {}
    for i in range(len(group_models)):
        if(len(param)==0):
            param = deepcopy(group_models[i].state_dict())
            for key in param.keys():
                param[key] = param[key]*weights[i]
        else:
            for key in param.keys():
                param[key] += group_models[i].state_dict()[key] * weights[i]

    for key in model.state_dict().keys():
        if key not in glob_keys:
            w_model[key] = param[key]

    model.load_state_dict(w_model)

    return model



def inter_group_aggregation_head_keys(group_models,weights,cur_class,head_keys):
    # 根据权重聚合模型
    model = deepcopy(group_models[cur_class])
    w_model = model.state_dict()
    param = {}
    for i in range(len(group_models)):
        if(len(param)==0):
            param = deepcopy(group_models[i].state_dict())
            for key in param.keys():
                param[key] = param[key]*weights[i]
        else:
            for key in param.keys():
                param[key] += group_models[i].state_dict()[key] * weights[i]

    for key in model.state_dict().keys():
        if key  in head_keys:
            w_model[key] = param[key]

    model.load_state_dict(w_model)

    return model

import argparse

def get_sys_params(args):
    # print("datasetname-dirichlet-N-数量策略-algrithm-E-eta-decay-T-B-p-测试集评判标准-mu")

    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('--datasetname', type=str, default=args['datasetname'], help="选定数据集")
    parser.add_argument('--alpha', type=float, default=args['alpha'], help="数据标签分布情况")
    parser.add_argument('--N', type=int, default=args['N'], help="客户端数量")
    parser.add_argument('--samplesNum', type=int, default=args['samplesNum'], help="客户端数据数量")
    parser.add_argument('--cnt_strategy', type=int, default=args['cnt_strategy'], help="数量策略(客户端分配的数据量是否一样)")
    parser.add_argument('--algrithm', type=str, default=args['algrithm'], help="使用的算法")
    parser.add_argument('--E', type=int, default=args['E'], help="本地训练轮次")
    parser.add_argument('--eta', type=float, default=args['eta'], help="学习率")
    parser.add_argument('--decay', type=float, default=args['decay'], help="学习率的衰减")
    parser.add_argument('--T', type=float, default=args['T'], help="训练轮次")
    parser.add_argument('--B', type=float, default=args['B'], help="batchsize")
    parser.add_argument('--p', type=float, default=args['p'], help="每轮选择客户端的比例")
    parser.add_argument('--test_strategy', type=int, default=args['test_strategy'],
                        help="测试集评判标准(0 表示测试集训练集相同分布  1表示不同分布)")
    parser.add_argument('--seed', type=int, default=args['seed'], help="随机种子")
    parser.add_argument('--mu', type=float, default=args['mu'], help="prox正则项系数")
    parser.add_argument('--gama', type=float, default=args['gama'], help="特征锚点跟新系数")
    parser.add_argument('--n_groups', type=int, default=args['n_groups'], help="聚类数")
    parser.add_argument('--pathological', type=bool, default=args['pathological'], help="病理分布")


    #CFL
    parser.add_argument('--sampling', type=str, default="random", help="采样方式")
    parser.add_argument('--sim_type', type=str, default="cosine", help="相似度类型（L1  L2 cosine）")

    # ap.add_argument('--delta', type=float, default=0.8, help="损失函数系数")

    return parser.parse_args()

def get_biye_filename(
    run_name:str,
    datasetname: str,
    alpha: float,
    N: int,
    cnt_strategy: int,
    algrithm: str,
    E: int,
    eta: float,
    decay: float,
    T: int,
    B: int,
    p: float,
    test_strategy:int,
    seed:int,
    mu:float,


):

    filename = (f"{run_name}_dataset_{datasetname}_alpha_{alpha}_N_{N}_"
                f"cntstgy_{cnt_strategy}_algrithm_{algrithm}_E_{E}_eta_{eta}_decay_{decay}"+
                f"_T_{T}_B_{B}_p_{p}_teststgy_{test_strategy}_seed_{seed}_mu_{mu}")

    return filename


# 将模型字典转化为tensor  model_keys[i] 表示第i层的参数  有两个tensor weight和bias
#
def get_model_matrix(models_dict,model_matrix,model_keys,sampled_clients,N,n):

    # model_matrix = np.empty((N,n),dtype=object)
    # print(f"model_keys = {model_keys}")
    # [['fc1.weight', 'fc1.bias'], ['fc2.weight', 'fc2.bias'], ['fc3.weight', 'fc3.bias'], ['conv2.weight', 'conv2.bias'],
    #  ['conv1.weight', 'conv1.bias']]
    #  1、获取weight 和bias的参数
    # 2、 连接在一起

    for idx,i in enumerate(sampled_clients):
        for k in range(len(model_keys)):
            # model_keys[k] 包括weight和bias
            # print("------")
            # print(model_keys[k][0])
            # print(model_keys[k][1])
            weight1 = models_dict[i][model_keys[k][0]]
            bias1 = models_dict[i][model_keys[k][1]]
            param = torch.cat([weight1.view(-1),bias1.view(-1)])
            model_matrix[i][k] = param


    return model_matrix

def get_tensor_cos(t1,t2):

    cos_similarity = F.cosine_similarity(t1,t2,dim=0)
    return cos_similarity.item()

def get_tensor_l2(t1,t2):

    l2_distance = torch.norm(t1 - t2)
    return l2_distance.item()
# 返回alpha矩阵
def get_matrix_alpha(models_matrix,base_alpha,sampled_clients,N,n):
    # models_matrix[i][j] 表示客户端i在第j层的tensor
    # base_alpha 表示每个客户端的自关注度 N*n  base[i][j]表示客户端i在第j层的自关注度
    # 返回一个n*N*N的矩阵 matrix[k][i][j]表示客户端i和客户端j在第k层时的聚合权重 alpha_{i,j}^{lk}
    matrix_alpha = np.zeros((n,N,N),dtype=float)
    # 系数
    sigmod = 100
    for k in range(n):

        for idx,i in enumerate(sampled_clients):
            sum = 0.0
            for idx2,j in enumerate(sampled_clients):
                if(i==j):
                    continue
                # cos = get_tensor_cos(models_matrix[i][k], models_matrix[j][k])
                l2 = get_tensor_l2(models_matrix[i][k], models_matrix[j][k])
                # print(f"{i} 和{j} 在d第{k}层的 维度 {len(models_matrix[i][k])}")
                # print(f"{i} 和{j} 在d第{k}层的余弦相似度为 :{math.exp(sigmod*get_tensor_cos(models_matrix[i][k],models_matrix[j][k]))}")
                # print(f"{i} 和{j} 在d第{k}层的l2为 :{l2}")
                sum += math.exp(-sigmod*l2)

            for j in range(N):
                if i==j:
                    matrix_alpha[k][i][i] = base_alpha[i][k];
                    continue;
                l2 = get_tensor_l2(models_matrix[i][k], models_matrix[j][k])

                matrix_alpha[k][i][j] = (1-base_alpha[i][k])* math.exp(-sigmod*l2)/sum
                # print(f"{i}和{j} 在第{k}层的协作参数 {matrix_alpha[k][i][j]}")

    return matrix_alpha
# 聚合客户端i 的第k 层 并返回参数
def layer_wised_agg(local_params,alpha,model_keys,i,k,idx,N):
    # 训练后的参数集
    # alpha 矩阵
    # model_keys[i] 表示 [weight,bias]
    # 客户端i
    # 第k层
    #  idx = 0 weight更新， bias 更新
    # 客户端总数
    key = model_keys[k][idx]
    param = deepcopy(local_params[i][key])

    param *= alpha[k][i][i]

    for j in range(N):
        if(j==i):
            continue
        param = param+ local_params[j][key]* alpha[k][i][j]

    return param


def update_base_alpha(alpha,base_alpha,mu):
    n = len(alpha)
    N = len(alpha[0])
    # T_{i}^{lk} = t_{i}^{lk} - (1-base_alpha_{i}^{lk})
    T = deepcopy(base_alpha)

    for i in range(N):
        for k in range(n):
            t = 0.0
            for j in range(N):
                if(i==j):
                    continue
                t+= alpha[k][j][i]
            T[i][k] = t - (1-base_alpha[i][k])
    for i in range(N):
        for k in range(n):
            base_alpha[i][k] -= mu * T[i][k]
            if base_alpha[i][k]<0:
                base_alpha[i][k] = 0.1
            elif base_alpha[i][k] >1:
                base_alpha[i][k] = 0.9

    return base_alpha



# pac function

def get_local_protos(train_model,train_data,device):

    local_protos_list = {}
    for inputs, labels in train_data:
        inputs,labels = inputs.to(device),labels.to(device)
        features, outputs = train_model(inputs)
        protos = features.clone().detach()
        for i in range(len(labels)):
            if labels[i].item() in local_protos_list.keys():
                local_protos_list[labels[i].item()].append(protos[i, :])
            else:
                local_protos_list[labels[i].item()] = [protos[i, :]]
    local_protos = get_protos(local_protos_list)
    return local_protos

def statistics_extraction(model,personal_keys,train_data,probs_label,num_classes,datasize,device):

    cls_keys = personal_keys
    g_params = model.state_dict()[cls_keys[0]] if isinstance(cls_keys, list) else model.state_dict()[cls_keys]
    d = g_params[0].shape[0]
    feature_dict = {}
    with torch.no_grad():
        for inputs, labels in train_data:
            inputs, labels = inputs.to(device),labels.to(device)
            features, outputs = model(inputs)
            feat_batch = features.clone().detach()
            for i in range(len(labels)):
                yi = labels[i].item()
                if yi in feature_dict.keys():
                    feature_dict[yi].append(feat_batch[i, :])
                else:
                    feature_dict[yi] = [feat_batch[i, :]]
    for k in feature_dict.keys():
        feature_dict[k] = torch.stack(feature_dict[k])

    py = probs_label
    py2 = py.mul(py)
    v = 0
    h_ref = torch.zeros((num_classes, d))
    for k in range(num_classes):
        if k in feature_dict.keys():
            feat_k = feature_dict[k]
            num_k = feat_k.shape[0]
            feat_k_mu = feat_k.mean(dim=0)
            h_ref[k] = py[k] * feat_k_mu
            v += (py[k] * torch.trace((torch.mm(torch.t(feat_k), feat_k) / num_k))).item()
            v -= (py2[k] * (torch.mul(feat_k_mu, feat_k_mu))).sum().item()

    v = v / datasize.item()

    return v, h_ref



def prior_label(dataset,n_classes):
    py = torch.zeros(n_classes)
    total = len(dataset.dataset)
    data_loader = iter(dataset)
    iter_num = len(data_loader)
    for it in range(iter_num):
        images, labels = next(data_loader)
        for i in range(n_classes):
            py[i] = py[i] + (i == labels).sum()
    py = py / (total)
    return py

def size_label(dataset,num_classes):
    py = torch.zeros(num_classes)
    total = len(dataset.dataset)
    data_loader = iter(dataset)
    iter_num = len(data_loader)
    for it in range(iter_num):
        images, labels = next(data_loader)
        for i in range(num_classes):
            py[i] = py[i] + (i == labels).sum()
    py = py / (total)
    size_label = py * total
    return size_label


def compute_weights(samples,clients):
    total = 0
    for c in clients:
        total += samples[c]
    weight = []
    for c in clients:
        weight.append(samples[c]/total)
    return weight

def get_head_glob_keys(dataset,model_keys):
    head_keys = []
    glob_keys = []
    if dataset == "fashion":
        for i in range(6):
            glob_keys.append(model_keys[i])
        print(f"global keys :{glob_keys}")

        for key in model_keys:
            if key not in glob_keys:
                head_keys.append(key)
        print(f"head keys :{head_keys}")
    elif dataset == "CIFAR":
        for i in range(8):
            glob_keys.append(model_keys[i])
        print(f"global keys :{glob_keys}")

        for key in model_keys:
            if key not in glob_keys:
                head_keys.append(key)
        print(f"head keys :{head_keys}")
    else:
        for i in range(len(model_keys)):
            if model_keys[i].startswith('bn'):
                continue
            elif model_keys[i].startswith('fc2'):
                head_keys.append(model_keys[i])
            else:
                glob_keys.append(model_keys[i])

    return head_keys,glob_keys