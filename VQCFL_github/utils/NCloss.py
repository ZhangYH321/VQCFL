import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# class NC1Loss(nn.Module):
#     '''
#     Modified Center loss, 1 / n_k ||h-miu||
#     '''
#     def __init__(self, num_classes=10, feat_dim=128, use_gpu=True):
#         super(NC1Loss, self).__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.use_gpu = use_gpu
#
#         if self.use_gpu:
#             self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
#         else:
#             self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
#
#     def forward(self, x, labels):
#         """
#         Args:
#             x: feature matrix with shape (batch_size, feat_dim).
#             labels: ground truth labels with shape (batch_size).
#         """
#         batch_size = x.size(0)
#         distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
#                   torch.pow(self.means, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
#         distmat.addmm_(x, self.means.t(), beta=1, alpha=-2)
#
#         classes = torch.arange(self.num_classes).long()
#         if self.use_gpu: classes = classes.cuda()
#         labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
#         mask = labels.eq(classes.expand(batch_size, self.num_classes))
#
#         dist = distmat * mask.float()
#         D = torch.sum(dist, dim=0)
#         N = mask.float().sum(dim=0) + 1e-10
#         loss = (D / N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes
#
#         return loss, self.means

def NC1Loss(pre_proto,cur_proto,labels,weights):
    loss = 0.0
    for i in range(len(pre_proto)):
        p1 = pre_proto[i].unsqueeze(0)
        p2 = cur_proto[i].unsqueeze(0)
        loss += torch.norm(p1-p2,p=2).sum() * weights[i]

    return loss



def NC2Loss(means):
    '''
    NC2 loss v0: maximize the average minimum angle of each centered class mean
    '''
    g_mean = means.mean(dim=0)
    centered_mean = means - g_mean
    means_ = F.normalize(centered_mean, p=2, dim=1)
    cosine = torch.matmul(means_, means_.t())
    # make sure that the diagnonal elements cannot be selected
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))
    max_cosine = cosine.max().clamp(-0.99999, 0.99999)
    # print('min angle:', min_angle)
    # maxmize the minimum angle
    # dim=1 means the maximum angle of the other class to each class
    loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()
    # loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean() + 1. / (means.size(0)-1)
    return loss, max_cosine

def NC2Loss_v1(means):
    '''
    NC2 loss v1: maximize the minimum angle of centered class means
    '''
    g_mean = means.mean(dim=0)
    centered_mean = means - g_mean
    means_ = F.normalize(centered_mean, p=2, dim=1)
    cosine = torch.matmul(means_, means_.t())
    # make sure that the diagnonal elements cannot be selected
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))
    max_cosine = cosine.max().clamp(-0.99999, 0.99999)
    # print('min angle:', min_angle)
    # maxmize the minimum angle
    # dim=1 means the maximum angle of the other class to each class
    loss = -torch.acos(max_cosine)
    min_angle = math.degrees(torch.acos(max_cosine.detach()).item())
    # loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean() + 1. / (means.size(0)-1)

    return loss, max_cosine