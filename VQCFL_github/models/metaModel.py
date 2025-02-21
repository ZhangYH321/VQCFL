import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from models.create_model import *

# 第一层注意力网络，输入为N个客户端的模型参数
class attention_net1(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(attention_net1,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.key = nn.Linear(input_dim,output_dim)
        self.query = nn.Linear(input_dim,output_dim)
        self.value = nn.Linear(input_dim,output_dim)
        self.score = None

    def forward(self,inputs):
        inputs_tensor = torch.stack(inputs, dim=0)  # (20, 650)

        queries = self.query(inputs_tensor)
        keys = self.key(inputs_tensor)
        values = self.value(inputs_tensor)

        attention_scores = torch.matmul(queries, keys.transpose(0, 1))  # (20, 20)
        attention_scores = F.softmax(attention_scores, dim=-1)  # (20, 20)
        self.score = attention_scores
        weighted_sum = torch.matmul(attention_scores, values)  # (20, 650)
        # 训练时已经将当前客户端的参数放到第一个了
        output = weighted_sum[0]
        return output

    def get_socre(self):
        return self.score



class Meta_Cifar(nn.Module):
    # 分类器层的维度
    def __init__(self,input_dim):
        super(Meta_Cifar,self).__init__()
        # 用于分类的网络
        self.net = CNN_CIFAR()
        self.att_net1 = attention_net1(input_dim,input_dim)
        self.att_net2 = attention_net1(input_dim,input_dim)
        self.meta_optim = torch.optim.Adam(
            list(self.att_net1.parameters()) + list(self.att_net2.parameters()),
            lr=0.05
        )


    def forward(self,params_list,glob_param,x,y,glob_keys,head_keys):
        head_params = self.get_head_params(params_list,head_keys)

        glob_head = self.get_glob_head(glob_param,head_keys)
        for param in self.att_net1.parameters():
            param.requires_grad = True
        for param in self.att_net2.parameters():
            param.requires_grad = True
        self.att_net1.cuda()
        self.att_net2.cuda()
        self.net.cuda()

        #更新两个注意力层的参数
        layer1_classifier = self.att_net1(head_params)
        inputs2 = [layer1_classifier,glob_head]
        layer2_classifier = self.att_net2(inputs2)
        classifier_params = self.restore_classifier_params(layer2_classifier,head_keys)

        # self.update_classifier_params(classifier_params)
        #组合模型  更新net
        net_param = self.net.state_dict()
        for key in glob_keys:
            net_param[key] = glob_param[key]
        for key in head_keys:
            net_param[key] = classifier_params[key]

        self.net.load_state_dict(net_param)


        _,output = self.net(x)
        loss = F.cross_entropy(output, y)
        # 固定分类网络参数
        for param in self.net.parameters():
            param.requires_grad = False
        self.meta_optim.zero_grad()
        loss.backward()
        self.meta_optim.step()

        for param in self.net.parameters():
            param.requires_grad = True


    def restore_classifier_params(self, layer2_classifier, head_keys):
        classifier_params = {}
        start = 0
        for key in head_keys:
            param_size = self.net.state_dict()[key].size()
            param_flat_size = np.prod(param_size)
            classifier_params[key] = layer2_classifier[start:start + param_flat_size].view(param_size)
            start += param_flat_size
        return classifier_params


    def update_classifier_params(self, classifier_params):
        net_param = self.net.state_dict()
        for key, value in classifier_params.items():
            net_param[key] = value
        self.net.load_state_dict(net_param)

    def get_head_params(self,params_list,head_keys):
        head_params = []
        for params in params_list:
            extracted_params = [params[key] for key in head_keys]
            flattened_params = torch.cat([param.view(-1) for param in extracted_params])
            head_params.append(flattened_params)
        return head_params

    def get_glob_head(self,glob_param,head_keys):
        extracted_params = [glob_param[key] for key in head_keys]
        flattened_params = torch.cat([param.view(-1) for param in extracted_params])
        return flattened_params

    def get_final_classifier(self,params_list,glob_param,head_keys):
        head_params = self.get_head_params(params_list, head_keys)

        glob_head = self.get_glob_head(glob_param, head_keys)

        layer1_classifier = self.att_net1(head_params)
        inputs2 = [layer1_classifier, glob_head]
        layer2_classifier = self.att_net2(inputs2)

        classifier_params = self.restore_classifier_params(layer2_classifier, head_keys)
        return classifier_params


