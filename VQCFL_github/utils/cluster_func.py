
import numpy as np
from copy import deepcopy
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from numpy.random import choice
from utils.cka_util import *
import math
import torch
import copy

def  get_group_sharing_weights(cluster_models,n_classes,cur_class,a,b):
    from utils.comm_func import get_distance_by_model_2

    delta = 100
    cooperation_weights = np.zeros(n_classes).astype(float)
    for i in range(n_classes):
        if(i!=cur_class):
            b = b + math.exp(delta * get_distance_by_model_2(deepcopy(cluster_models[i]),deepcopy(cluster_models[cur_class]),'cosine'))


    for i in range(n_classes):
        if(i!=cur_class):
            cooperation_weights[i] = math.exp(delta * get_distance_by_model_2(deepcopy(cluster_models[i]),deepcopy(cluster_models[cur_class]),'cosine'))/b*(1-a)
        else:
            cooperation_weights[i] = a
    return cooperation_weights


def  get_personal_weights(cluster_models,n_classes,cur_class,a,b,globs_keys):
    from utils.comm_func import get_distance_by_model_not_glob_keys

    delta = 100
    cooperation_weights = np.zeros(n_classes).astype(float)
    for i in range(n_classes):
        if(i!=cur_class):
            b = b + math.exp(delta * get_distance_by_model_not_glob_keys(deepcopy(cluster_models[i]),deepcopy(cluster_models[cur_class]),globs_keys,'cosine'))
            # cos = get_distance_by_model(deepcopy(cluster_models[i]), deepcopy(cluster_models[cur_class]), 'cosine')
            # b = b + cos
            # print(f" 组{i} 和 组 {cur_class} 的相似度 :{cos}")

    for i in range(n_classes):
        if(i!=cur_class):
            cooperation_weights[i] = math.exp(delta * get_distance_by_model_not_glob_keys(deepcopy(cluster_models[i]),deepcopy(cluster_models[cur_class]),globs_keys,'cosine'))/b*(1-a)
            # cooperation_weights[i] = get_distance_by_model(deepcopy(cluster_models[i]),deepcopy(cluster_models[cur_class]),'cosine')/b*(1-a)
        else:
            cooperation_weights[i] = a
    return cooperation_weights


def get_cka_repre_weights(cluster_models,n_classes,cur_class,data):
    delta = 100
    cooperation_weights = np.zeros(n_classes).astype(float)
    sim = np.zeros(n_classes).astype(float)
    for i in range(n_classes):
        sim[i] = get_linear_repre_CKA(deepcopy(cluster_models[i]),deepcopy(cluster_models[cur_class]),data)
        b = b + math.exp(delta * sim[i] )

    for i in range(n_classes):
        cooperation_weights[i] = math.exp(delta * sim[i])/b

    return cooperation_weights



def get_cka_cla_weights(cluster_models,n_classes,cur_class,data):
    delta = 100
    cooperation_weights = np.zeros(n_classes).astype(float)
    sim = np.zeros(n_classes).astype(float)
    for i in range(n_classes):
        sim[i] = get_linear_fc3_CKA(deepcopy(cluster_models[i]), deepcopy(cluster_models[cur_class]), data)
        b = b + math.exp(delta * sim[i])

    for i in range(n_classes):
        cooperation_weights[i] = math.exp(delta * sim[i]) / b

    return cooperation_weights

def get_cos_repre_weights(cluster_models,n_classes,cur_class,globs_keys):
    from utils.comm_func import get_distance_by_model_not_glob_keys
    from utils.comm_func import get_distance_by_model_in_glob_keys
    delta = 100
    a = 0.8
    b=0.0
    cooperation_weights = np.zeros(n_classes).astype(float)
    sim = np.zeros(n_classes).astype(float)
    sim[cur_class] = a
    for i in range(n_classes):
        if i !=cur_class:
            sim[i] = get_distance_by_model_in_glob_keys(deepcopy(cluster_models[i]),deepcopy(cluster_models[cur_class]),globs_keys,'cosine')
            b = b + math.exp(-delta * sim[i])


    for i in range(n_classes):
        if i!=cur_class:
            cooperation_weights[i] = math.exp(-delta * sim[i])*(1-a)/b
    cooperation_weights[cur_class] = a
    return cooperation_weights

def get_cos_cla_weights(cluster_models,n_classes,cur_class,globs_keys):
    from utils.comm_func import get_distance_by_model_not_glob_keys
    b=0.0
    delta = 100
    cooperation_weights = np.zeros(n_classes).astype(float)
    sim = np.zeros(n_classes).astype(float)
    for i in range(n_classes):
        sim[i] = get_distance_by_model_not_glob_keys(deepcopy(cluster_models[i]), deepcopy(cluster_models[cur_class]),
                                                     globs_keys, 'cosine')
        b = b + math.exp(delta * sim[i])

    for i in range(n_classes):
        cooperation_weights[i] = math.exp(delta * sim[i]) / b

    return cooperation_weights



def get_group_sharing_weights_by_not_glob_keys(model,group_params,n_classes,cur_class,a,b,glob_keys):

    model.cpu()
    model_0 = deepcopy(model)
    model_0.load_state_dict(group_params[cur_class])

    cooperation_weights = np.zeros(n_classes).astype(float)
    total_cos = 0.0
    for i in range(n_classes):
        if i!=cur_class:
    #         计算 cur_class 的model 和i 的model 的余弦相似度
            tmpa = []
            tmpb = []
            norm1 = 0.0
            norm2 = 0.0
            norm3 = 0.0
            model.load_state_dict(group_params[i])
            for key in group_params[i].keys():
                if key not in glob_keys:
                    tmpa += model.state_dict()[key].numpy().flatten().tolist()

                    tmpb += model_0.state_dict()[key].numpy().flatten().tolist()
            for j in range(len(tmpa)):
                norm1 += tmpa[j]*tmpb[j]
                norm2 += tmpa[j]*tmpa[j]
                norm3 +=  tmpb[j]*tmpb[j]

            cos = norm1/np.sqrt(norm2*norm3)
            cooperation_weights[i] = cos
            total_cos += cos

    cooperation_weights = cooperation_weights/total_cos*(1-a)
    cooperation_weights[cur_class] = a
    return cooperation_weights


# def get_group_sharing_weights_by_head_keys(model, group_params, n_classes, cur_class, a, b, head_keys):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     if torch.cuda.is_available():
#         model.to(device)
#     model_0 = deepcopy(model)
#     model_0.load_state_dict(group_params[cur_class])
#     model_0.to(device)
#
#     cooperation_weights = torch.zeros(n_classes, dtype=torch.float32, device=device)
#     total_cos = torch.tensor(0.0, dtype=torch.float32, device=device)
#
#     for i in range(n_classes):
#         if i != cur_class:
#             # 计算 cur_class 的model 和i 的model 的余弦相似度
#             tmpa = []
#             tmpb = []
#
#             model.load_state_dict(group_params[i])
#             if torch.cuda.is_available():
#                 model.to(device)
#
#             for key in group_params[i].keys():
#                 if key in head_keys:
#                     tmpa.extend(model.state_dict()[key].cpu().numpy().flatten().tolist())
#                     tmpb.extend(model_0.state_dict()[key].cpu().numpy().flatten().tolist())
#
#             tmpa = torch.tensor(tmpa, dtype=torch.float32, device=device)
#             tmpb = torch.tensor(tmpb, dtype=torch.float32, device=device)
#
#             norm1 = torch.sum(tmpa * tmpb)
#             norm2 = torch.sum(tmpa * tmpa)
#             norm3 = torch.sum(tmpb * tmpb)
#
#             cos = norm1 / torch.sqrt(norm2 * norm3)
#             cooperation_weights[i] = cos
#             total_cos += cos
#
#     cooperation_weights = cooperation_weights / total_cos * (1 - a)
#     cooperation_weights[cur_class] = a
#
#     return cooperation_weights.cpu().numpy()  # 将最终结果移到 CPU 上

def are_models_equal(model1, model2):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(param1, param2):
            return False
    return True

def are_params_equal(params1, params2):
    for key in params1:
        param1 = params1[key]
        param2 = params2[key]

        # 将非张量参数转换为张量再进行比较
        if not isinstance(param1, torch.Tensor):
            param1 = torch.tensor(param1)
        if not isinstance(param2, torch.Tensor):
            param2 = torch.tensor(param2)

        if not torch.equal(param1, param2):
            return False

    return True

def get_all_norm2(params,head_keys):

    num_models = len(params)
    norms = np.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(num_models):
            if i>=j:
                norms[i][j] = norms[j][i]
                continue
            norms[i][j] = get_norm2(params[i],params[j],head_keys)

    return norms


def get_norm2(params1,params2,head_keys):
    head_params1 = torch.cat([params1[key].view(-1) for key in head_keys])
    head_params2 = torch.cat([params2[key].view(-1) for key in head_keys])
    param_diff = head_params1 - head_params2
    norm2 = torch.norm(param_diff, p=2).item()
    return norm2


def get_amp_weights(local_params,keys,lam,attention):
    n_clients = len(local_params)
    all_cos = np.zeros((n_clients, n_clients))
    for i in range(n_clients):
        for j in range(n_clients):
            if i > j:
                all_cos[i][j] = all_cos[j][i]
                continue
            all_cos[i][j] = get_cos(local_params[i], local_params[j], keys)

    weights = np.zeros((n_clients,n_clients))
    for i in range(n_clients):
        cur_sum = 0.0
        for j in range(n_clients):
            if i==j:
                continue
            cur_sum+=get_cw_exp(all_cos[i][j],lam)
        for j in range(n_clients):
            if i ==j:
                weights[i][j] = attention
                continue
            weights[i][j] = get_cw_exp(all_cos[i][j],lam)/cur_sum * (1-attention)
    return weights




def get_cw_weights(avg_weights,local_params,head_keys,lam,gama):
    n_clients = len(local_params)

    all_cos = np.zeros((n_clients,n_clients))
    for i in range(n_clients):
        for j in range(n_clients):
            if i>j:
                all_cos[i][j] = all_cos[j][i]
                continue
            all_cos[i][j] = get_norm2(local_params[i],local_params[j],head_keys)
    weights = np.zeros((n_clients,n_clients))


    for i in range(n_clients):
        cur_sum = 0.0
        for j in range(n_clients):
            cur_sum +=get_cw_exp(all_cos[i][j],lam)
        for j in range(n_clients):
            weights[i][j] = get_cw_exp(all_cos[i][j],lam)/cur_sum * gama + (1-gama)*avg_weights[j]
    # tmp = []
    # for i in range(len(weights)):
    #     tmp.append(sum(weights[i]))
    return weights

def get_cw_exp(x,lam):
    return np.exp(-lam *x)

def get_cos(params1,params2,head_keys):

    head_params1 = torch.cat([params1[key].view(-1) for key in head_keys])
    head_params2 = torch.cat([params2[key].view(-1) for key in head_keys])
    dot_product = torch.dot(head_params1, head_params2)
    norm1 = torch.norm(head_params1, p=2)
    norm2 = torch.norm(head_params2, p=2)

    cosine_similarity = dot_product / (norm1 * norm2)
    return cosine_similarity.item()



def get_alpha(norms,lamda):
    n = len(norms)
    alphas = np.zeros((n, n))
    for i in range(n):
        cur_sum = 0.0
        for j in range(n):
            cur_sum+= get_exp(norms[i][j],lamda)
        for j in range(n):
            alphas[i][j] = get_exp(norms[i][j],lamda)/cur_sum

    return alphas

def get_exp(x,lamda):
    return np.exp(-lamda*np.cos(x))

def get_group_sharing_weights_by_head_keys(model, group_params, n_classes, cur_class, a, b, head_keys):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model_0 = deepcopy(model)
    model_0.load_state_dict(group_params[cur_class])
    model_0.to(device)

    cooperation_weights = torch.zeros(n_classes, dtype=torch.float32, device=device)
    total_cos = torch.tensor(0.0, dtype=torch.float32, device=device)

    for i in range(n_classes):
        if i != cur_class:
            # 计算 cur_class 的model 和i 的model 的余弦相似度
            tmpa = []
            tmpb = []

            model.load_state_dict(group_params[i])
            model.to(device)

            for key in head_keys:
                tmpa.extend(model.state_dict()[key].view(-1).tolist())
                tmpb.extend(model_0.state_dict()[key].view(-1).tolist())

            tmpa = torch.tensor(tmpa, dtype=torch.float32, device=device)
            tmpb = torch.tensor(tmpb, dtype=torch.float32, device=device)

            norm1 = torch.sum(tmpa * tmpb)
            norm2 = torch.sum(tmpa * tmpa)
            norm3 = torch.sum(tmpb * tmpb)

            cos = norm1 / torch.sqrt(norm2 * norm3)
            cooperation_weights[i] = cos
            total_cos += cos

    cooperation_weights = cooperation_weights / total_cos * (1 - a)
    cooperation_weights[cur_class] = a

    return cooperation_weights.cpu().numpy()


# 选着sampled个客户端参与本轮训练
def FedGS_sample(distri_clusters, n_sampled):
    n_clients = len(distri_clusters[0])
    n_classes = len(distri_clusters)
    sampled_clients = np.zeros(n_sampled)
    for k in range(n_sampled):
        sampled_clients[k] = int(choice(n_clients, 1, p=distri_clusters[k%n_classes]))

    return sampled_clients


def Cluster_grad(group,cur_group_params,glob_model,glob_keys,n_samples,num_group):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = deepcopy(glob_model)
    glob_model.cpu()
    # 根据模型更新的第一轮梯度进行层次聚类
    glob_param = glob_model.state_dict()
    # 每个客户端都有一个列表
    X = [[] for i in range(len(group))]

    for i in range(len(group)):
        # model.load_state_dict(cur_group_params[i])
        # model.cpu()
        # cur = model.state_dict()
        for j in cur_group_params[i].keys():
            a = cur_group_params[i][j].cpu().numpy().flatten().tolist()
            b = glob_param[j].cpu().numpy().flatten().tolist()
            X[i] += [local - glob
                     for local, glob in zip(a,b)
                     ]
    glob_model.to(device)

    X = np.array(X)
    np.save("./CFL_data.npy",X)
    Z = linkage(X,'average')
    # 最多10个类 cluster[i] 表示i在第几个类中 从1开始
    clusters = fcluster(Z,num_group,criterion='maxclust')

    new_groups = [[] for i in range(max(clusters))]

    for i in range(len(group)):
        new_groups[clusters[i]-1].append(group[i])

    new_group_params = []

    for i in range(max(clusters)):
    #     对每一个类求共享模型
        cur_param = {}
        total_samples = 0
        for j in range(len(clusters)):
            # 如果当前客户端j是第i组的 参与聚合
            if clusters[j] == i+1:
                total_samples += n_samples[j]
                if len(cur_param)==0:
                    cur_param = deepcopy(cur_group_params[j])
                    for key in cur_group_params[j].keys():
                        cur_param[key] = cur_param[key] * n_samples[j]
                else:
                    for key in cur_group_params[j].keys():
                        cur_param[key] += cur_group_params[j][key] * n_samples[j]

        for key in glob_model.state_dict().keys():
            cur_param[key] = torch.div(cur_param[key], total_samples)

        new_group_params.append(deepcopy(cur_param))


    return new_groups,new_group_params,clusters



def FedAvg(w):
    w_avg = w[0]
    for i in w_avg.keys():
        for j in range(1, len(w)):
            w_avg[i] += w[j][i]
        w_avg[i] = torch.div(w_avg[i], len(w))
    return w_avg


def L2(old_params, old_w, param, rel):
    w = []
    for p in param:
        w.append(p)
    for i in range(len(w)):
        w[i] = torch.flatten(w[i])
    for i in range(len(old_w)):
        old_w[i] = torch.flatten(old_w[i])
    for i in range(len(old_params)):
        for j in range(len(old_params[i])):
            old_params[i][j] = torch.flatten(old_params[i][j])

    _w = w[0]
    for i in range(1, len(w)):
        _w = torch.cat([_w, w[i]])
    _old_w = old_w[0]
    for i in range(1, len(old_w)):
        _old_w = torch.cat([_old_w, old_w[i]])
    _old_params = []
    for i in range(len(old_params)):
        _old_param = old_params[i][0]
        for j in range(1, len(old_params[i])):
            _old_param = torch.cat([_old_param, old_params[i][j]])
        _old_params.append(_old_param)

    x = torch.sub(_w, _old_w)
    x = torch.norm(x, 'fro')
    x = torch.pow(x, 2)
    loss = x

    for i in range(len(_old_params)):
        _param = _old_params[i]
        x = torch.sub(_w, _param)
        x = torch.linalg.norm(x)
        x = torch.pow(x, 2)
        # L 的值
        L = 0.1
        x = torch.mul(x, L)
        x = torch.mul(x, rel[i])
        loss = torch.add(loss, x)

    return loss

def Prox(model,w_groups, rel):
    w = L2_Prox(model,w_groups, rel)
    return w


def L2_Prox(model,w_groups, rel):


    old_params = []
    for w in w_groups:
        net = deepcopy(model)
        net.load_state_dict(w)
        old_params.append([])
        for p in net.parameters():
            old_params[-1].append(copy.deepcopy(p))

    w_n = []
    for i in range(len(w_groups)):
        w = w_groups[i]
        net = deepcopy(model)
        net.load_state_dict(w)
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        for iter in range(10):
            loss = L2(old_params[:i] + old_params[i + 1:], old_params[i], net.parameters(), rel[i])
            if iter == 0:
                loss_start = copy.deepcopy(loss.item())
            if iter == 10 - 1:
                loss_end = copy.deepcopy(loss.item())
                # percent = (loss_end - loss_start) / loss_start * 100
                # print("Percent: {:.2f}%".format(percent))
            opt.zero_grad()
            loss.backward()
            opt.step()
        w_n.append(copy.deepcopy(net.state_dict()))

    return w_n

def Cluster_CFTML(group,w_local,glob_model,n_samples,num_group):
    model = deepcopy(glob_model)
    model.cpu()
    X = [[] for i in range(len(group))]
    ww_local = []
    for i in range(len(group)):
        model.load_state_dict(w_local[i])
        ww_local.append(model.state_dict())
        for j in w_local[i].keys():

            X[i] += ww_local[i][j].numpy().flatten().tolist()
    X = np.array(X)
    Z = linkage(X, 'average')

    clusters = fcluster(Z,num_group,criterion='maxclust')

    new_groups = [[] for i in range(max(clusters))]
    new_w_groups = []

    for i in range(len(group)):
        new_groups[clusters[i] - 1].append(group[i])

    X_groups = []
    for i in range(max(clusters)):
        new_w_local = []
        X_local = []
        for j in range(len(clusters)):
            if clusters[j] == i + 1:
                new_w_local.append(w_local[j])
                X_local.append(X[j])
        new_w_group = FedAvg(new_w_local)
        new_w_groups.append(new_w_group)
        X_group = X_local[0]
        for j in range(1, len(X_local)):
            X_group += X_local[j]
        X_group /= len(X_local)
        X_groups.append(copy.deepcopy(X_group))

    rel = []
    for i in range(len(X_groups)):
        rel.append([])
        for j in range(len(X_groups)):
            if j != i:
                a = np.linalg.norm(X_groups[i])
                b = np.linalg.norm(X_groups[j])
                dist = 1 - np.dot(X_groups[i], X_groups[j].T) / (a * b)
                rel[-1].append(dist)

    return new_groups, new_w_groups, rel,clusters

