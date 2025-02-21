# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import torch
from copy import deepcopy
import random
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist,squareform
from utils.comm_func import *
from utils.cluster_func import *
from utils.get_data import *
from models.create_model import load_model



def plot_heatmap(data, linkage_matrix, training_sets, iter, datasetname):
    """Plot clustering heatmap and dendrogram"""
    # 计算标签分布矩阵
    clientNum = len(training_sets)
    matrix = np.zeros(shape=(clientNum, 10))
    for i, training_set in enumerate(training_sets):
        img_label = training_set.dataset.labels
        unique_elements, counts = np.unique(img_label, return_counts=True)
        for u, c in zip(unique_elements, counts):
            matrix[i, u] = c / len(img_label)

    # 绘制标签分布热图
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(matrix, cmap="viridis",
                xticklabels=np.arange(0, 10),
                yticklabels=np.arange(0, 20),
                cbar_kws={"shrink": .8})
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel("Class", fontsize=18)
    plt.ylabel("Clients", fontsize=18)
    plt.savefig(f"./pic/{datasetname}_class.png", dpi=300)

    # 绘制聚类结果
    fig = plt.figure(figsize=(18, 10))

    # 绘制树状图
    ax_dendrogram = fig.add_axes([0.05, 0.1, 0.1, 0.8])
    dendrogram_result = dendrogram(linkage_matrix, orientation='left', ax=ax_dendrogram)
    leaves_order = dendrogram_result['leaves']
    ax_dendrogram.set_xticks([])
    ax_dendrogram.set_yticks([])
    ax_dendrogram.axis('off')

    # 绘制相似度热图
    ax_heatmap = plot_similarity_heatmap(fig, data, leaves_order)

    # 绘制分布热图
    ax_heatmap1 = plot_distribution_heatmap(fig, matrix, leaves_order)

    # 保存图像
    plt.savefig(f"./pic/{datasetname}_Similarity_{iter}.png", dpi=300)
    plt.show()


def plot_similarity_heatmap(fig, data, leaves_order):
    """Plot similarity heatmap"""
    ax_heatmap = fig.add_axes([0.15, 0.1, 0.4, 0.8])
    sorted_data = data[leaves_order]
    similarity_matrix = 1 - squareform(pdist(sorted_data, metric='cosine'))

    sns.heatmap(similarity_matrix, cmap="viridis",
                xticklabels=leaves_order,
                yticklabels=leaves_order,
                ax=ax_heatmap,
                cbar_kws={"shrink": .8})

    ax_heatmap.set_xlabel("Clients", fontsize=20)
    ax_heatmap.tick_params(axis='both', which='major', labelsize=15)
    ax_heatmap.invert_yaxis()
    ax_heatmap.yaxis.set_visible(False)
    return ax_heatmap


def plot_distribution_heatmap(fig, matrix, leaves_order):
    """Plot label distribution heatmap"""
    ax_heatmap1 = fig.add_axes([0.6, 0.1, 0.4, 0.8])
    sns.heatmap(matrix[leaves_order], cmap="viridis",
                xticklabels=np.arange(0, 10),
                yticklabels=np.flip(leaves_order),
                ax=ax_heatmap1,
                cbar_kws={"shrink": .8})

    ax_heatmap1.set_xlabel("Class", fontsize=20)
    ax_heatmap1.set_ylabel("Clients", fontsize=20)
    ax_heatmap1.tick_params(axis='both', which='major', labelsize=15)
    return ax_heatmap1


def plot_cluster_metrics(mean_diffs, std_diffs, PretrainIter):
    """Plot clustering metrics"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(PretrainIter), mean_diffs, label="Mean Difference")
    plt.fill_between(
        range(PretrainIter),
        np.array(mean_diffs) - np.array(std_diffs),
        np.array(mean_diffs) + np.array(std_diffs),
        alpha=0.3,
        label="Std Deviation"
    )
    plt.xlabel("Number of Clusters")
    plt.ylabel("Difference in Distances")
    plt.title("Average Distance Difference vs. Number of Clusters")
    plt.legend()
    plt.show()


# federated_system.py


def calculate_distances(data, current_cluster_labels):
    distances_to_cluster_centers = []
    min_distances_to_other_clusters = []

    # 获取所有的类中心
    unique_labels = np.unique(current_cluster_labels)
    cluster_centers = {label: np.mean(data[current_cluster_labels == label], axis=0) for label in unique_labels}

    for i, point in enumerate(data):
        # 获取当前点的标签
        label = current_cluster_labels[i]
        # 计算客户端到其所属类中心的距离
        # dist_to_center = np.linalg.norm(point - cluster_centers[label])  # 使用欧几里得距离
        dist_to_center = pdist([point, cluster_centers[label]], metric='cosine')[0]  # 或者使用余弦距离
        distances_to_cluster_centers.append(dist_to_center)

        # 计算客户端到其他类中心的最小距离
        other_centers = [cluster_centers[l] for l in unique_labels if l != label]
        # min_dist_to_other = min([np.linalg.norm(point - oc) for oc in other_centers])  # 欧几里得距离
        min_dist_to_other = min([pdist([point, oc], metric='cosine')[0] for oc in other_centers])  # 余弦距离
        min_distances_to_other_clusters.append(min_dist_to_other)

    return distances_to_cluster_centers, min_distances_to_other_clusters


class FederatedSystem:
    def __init__(self, args):
        """Initialize Federated Learning System"""
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_name = "FedVQ"
        self.PretrainIter = 5
        self.setup_system()


    def setup_system(self):
        """Setup system parameters and load data/model"""
        # 加载数据和模型 - 使用外部函数
        self.train_loaders, self.test_loaders0, self.test_loaders1, _ = get_biye_dataloaders(
            self.args.datasetname, self.args.alpha, self.args.N,
            self.args.cnt_strategy, self.args.B, self.args.samplesNum, self.args.p
        )

        self.model = load_model(self.args.datasetname, self.args.seed,self.run_name).to(self.device)

        # 设置系统参数
        self.setup_parameters()
        self.initialize_models()

    def setup_parameters(self):
        """Setup system parameters"""
        self.n_clients = len(self.train_loaders)
        self.n_samples = np.array([len(dl.dataset) for dl in self.train_loaders])
        self.weights = self.n_samples / np.sum(self.n_samples)
        self.n_sampled = int(self.n_clients * self.args.p)

        # 初始化追踪变量
        self.mean_diffs = []
        self.std_diffs = []

    def initialize_models(self):
        """Initialize models and parameters"""
        # 初始化全局模型
        self.glob_model = deepcopy(self.model)
        self.anchors = self.glob_model.vq_vae.vq_layer.embedding.weight.clone()

        # 初始化本地模型
        self.local_models = [deepcopy(self.glob_model) for _ in range(self.n_clients)]
        self.local_params = {}
        self.embedding_params = {}

        for client in range(self.n_clients):
            self.local_params[client] = dict(self.glob_model.state_dict())
            self.embedding_params[client] = (
                self.glob_model.vq_vae.vq_layer.embedding.weight.cpu().detach().numpy()
            )

        # 初始化组模型
        self.cluster_labels = np.ones(self.n_clients, dtype=int)
        self.cluster_groups = [[] for _ in range(self.args.n_groups)]
        self.group_models = [deepcopy(self.glob_model) for _ in range(self.args.n_groups)]

    def select_active_clients(self, iter):
        """
        Select active clients for current iteration
        """
        if iter < self.PretrainIter:
            # 预训练阶段，所有客户端参与
            return list(range(self.n_clients))
        else:
            # 后续轮次，从每个组中随机选择客户端
            active_clients = []
            for group in range(self.args.n_groups):
                group_size = len(self.cluster_groups[group])
                if group_size > 0:
                    n_sample_group = max(1, int(self.n_sampled * group_size / self.n_clients))
                    active_clients.extend(random.sample(self.cluster_groups[group], n_sample_group))
            return active_clients

    def aggregate_vq_codebook(self, active_clients, total_samples):
        """
        Aggregate VQ codebook across all groups
        """
        total_vq_codebook = {
            key: torch.zeros_like(param)
            for key, param in self.group_models[0].state_dict().items()
            if key == "vq_vae.vq_layer.embedding.weight"
        }

        # 计算所有活跃客户端的VQ codebook加权平均
        for idx in active_clients:
            for key, param in self.local_models[idx].state_dict().items():
                if key == "vq_vae.vq_layer.embedding.weight":
                    weight = self.n_samples[idx] / total_samples
                    total_vq_codebook[key] += weight * param

        return total_vq_codebook

    def aggregate_group_model(self, group_idx, active_clients):
        """
        Aggregate model parameters within a group
        """
        # 获取当前组的客户端索引
        indices = np.where(self.cluster_labels == group_idx + 1)[0]
        sampled_indices = [idx for idx in indices if idx in active_clients]

        if not sampled_indices:
            return

        # 计算组内总样本数
        group_samples = sum(self.n_samples[idx] for idx in sampled_indices)

        # 初始化聚合字典
        current_local_dict = {
            key: torch.zeros_like(param).float()
            for key, param in self.group_models[group_idx].state_dict().items()
        }

        # 聚合参数
        for idx in sampled_indices:
            weight = self.n_samples[idx] / group_samples
            for key, param in self.local_models[idx].state_dict().items():
                if "vae" not in key:  # 非VQ-VAE层的参数
                    current_local_dict[key] += weight * param

        # 更新组模型
        self.group_models[group_idx].load_state_dict(current_local_dict, strict=False)

    def update_linear_weights(self):
        """
        Update linear layer weights using similarity-based aggregation
        """
        # 准备线性层权重
        linear_weights = []
        similarity_weights = [[] for _ in range(self.args.n_groups)]
        linear_keys = [key for key in self.model.state_dict().keys() if 'fc' in key]

        # 计算每个组的线性层权重
        for group in range(self.args.n_groups):
            for key, param in self.group_models[group].state_dict().items():
                if key in linear_keys:
                    similarity_weights[group] += param.cpu().numpy().flatten().tolist()

        # 计算组间相似度
        similarities = cosine_similarity(similarity_weights)

        # 基于相似度更新每个组的线性层权重
        for group in range(self.args.n_groups):
            total_linear_weights = {}
            for key, param in self.group_models[group].state_dict().items():
                if key in linear_keys:
                    # 初始化权重
                    if key not in total_linear_weights:
                        total_linear_weights[key] = torch.zeros_like(param).float()

                    # 根据相似度计算加权平均
                    for j in range(self.args.n_groups):
                        total_linear_weights[key] += (
                                self.group_models[j].state_dict()[key] * similarities[group, j]
                        )
                    total_linear_weights[key] /= sum(similarities[group, :])

            linear_weights.append(total_linear_weights)

        return linear_weights

    def update_group_model(self, group_model, client_idx):
        """
        Update group model parameters from client model
        """
        for key in group_model.state_dict().keys():
            if "vae" in key and key != "vq_vae.vq_layer.embedding.weight":
                group_model.load_state_dict(
                    {key: self.local_models[client_idx].state_dict()[key]},
                    strict=False
                )

    def evaluate(self, loss_hist, acc0_hist, acc1_hist, iter):
        """
        Evaluate models and log results
        """
        # 计算每个客户端的损失和准确率
        for k in range(self.n_clients):
            group_idx = self.cluster_labels[k] - 1
            model = self.group_models[group_idx]

            # 计算训练损失
            loss_hist[iter + 1, k] = float(
                loss_dataset(model, self.train_loaders[k], loss_classifier).detach()
            )

            # 计算测试准确率
            acc0_hist[iter + 1, k] = accuracy_dataset(model, self.test_loaders0[k])
            acc1_hist[iter + 1, k] = accuracy_dataset(model, self.test_loaders1[k])

        # 计算加权平均指标
        loss = np.dot(self.weights, loss_hist[iter + 1])
        acc0 = np.dot(self.weights, acc0_hist[iter + 1])
        acc1 = np.dot(self.weights, acc1_hist[iter + 1])

        print(f"====> local train Loss: {loss:.4f} personal Accuracy: {acc0:.4f} global acc {acc1:.4f}")

    def save_results(self, loss_hist, acc0_hist, acc1_hist):
        """
        Save training results
        """
        loss_hist = loss_hist.mean(axis=1)
        acc0_hist = acc0_hist.mean(axis=1)
        acc1_hist = acc1_hist.mean(axis=1)

        save_pkl(loss_hist, f"./acc/revise/{self.run_name}_loss", f"0.1_{self.args.datasetname}_K{self.args.n_groups}")
        save_pkl(acc0_hist, f"./acc/revise/{self.run_name}_acc0", f"0.1_{self.args.datasetname}_K{self.args.n_groups}")
        save_pkl(acc1_hist, f"./acc/revise/{self.run_name}_acc1", f"0.1_{self.args.datasetname}_K{self.args.n_groups}")

    def perform_clustering(self, iter):
        """Perform hierarchical clustering"""
        # 准备数据
        number, embedding_dim = self.anchors.shape
        data = np.zeros(shape=(self.n_clients, number * embedding_dim))

        for k in range(data.shape[0]):
            data[k, :] = self.embedding_params[k].flatten()

        # PCA降维
        pca = PCA(n_components=self.n_clients)
        data = pca.fit_transform(data)
        # np.save("revise_plot/VQCFL_data.npy", data)
        # 层次聚类
        linkage_matrix = linkage(data, method='average', metric="cosine")
        self.cluster_labels = fcluster(linkage_matrix, self.args.n_groups, criterion='maxclust')

        # 可视化聚类结果
        # if iter % 1 == 0:
            # plot_heatmap(data, linkage_matrix, self.train_loaders, iter, self.args.datasetname)
            # self.update_cluster_metrics(data)

        # 更新聚类组
        self.update_cluster_groups()

    def update_cluster_metrics(self, data):
        """Update clustering metrics"""
        distances_center, distances_other = calculate_distances(data, self.cluster_labels)
        diffs = np.array(distances_other) - np.array(distances_center)
        self.mean_diffs.append(np.mean(diffs))
        self.std_diffs.append(np.std(diffs))

    def update_cluster_groups(self):
        """Update cluster groups"""
        self.cluster_groups = [[] for _ in range(self.args.n_groups)]
        for i, cluster in enumerate(self.cluster_labels, 0):
            self.cluster_groups[cluster - 1].append(i)
        print(self.cluster_groups)

    def train_clients(self, active_clients, iter, lr):
        """Train selected clients"""
        total_samples = 0
        for idx in active_clients:
            current_group = self.cluster_labels[idx]
            group_model = self.group_models[current_group - 1]

            # 更新组模型参数
            if iter > 0:
                self.update_group_model(group_model, idx)

            total_samples += self.n_samples[idx]

            # 训练客户端模型
            train_model = deepcopy(group_model)
            train_model.train()
            train_param = vq_train(
                train_model, self.args.E, loss_classifier,
                self.train_loaders[idx], lr, self.anchors,
                self.PretrainIter, device=self.device,
                train_iter=iter
            )

            self.local_models[idx].load_state_dict(train_param)
            self.embedding_params[idx] = (
                self.local_models[idx].vq_vae.vq_layer.embedding.weight.cpu().detach().numpy()
            )

        return total_samples

    def aggregate_models(self, active_clients, total_samples):
        """Aggregate models within groups"""
        # 聚合VQ codebook
        total_vq_codebook = self.aggregate_vq_codebook(active_clients, total_samples)

        # 聚合每个组的模型
        for group in range(self.args.n_groups):
            self.aggregate_group_model(group, active_clients)

        # 更新线性层权重
        linear_weights = self.update_linear_weights()

        # 更新每个组的模型
        for group in range(self.args.n_groups):
            self.group_models[group].load_state_dict(total_vq_codebook, strict=False)
            self.group_models[group].load_state_dict(linear_weights[group], strict=False)

    def train(self):
        """Main training loop"""
        loss_hist = np.zeros((self.args.T + 1, self.n_clients))
        acc0_hist = np.zeros((self.args.T + 1, self.n_clients))
        acc1_hist = np.zeros((self.args.T + 1, self.n_clients))

        lr = self.args.eta

        for iter in range(self.args.T):
            print(f"Iteration {iter}")

            # 选择活跃客户端
            active_clients = self.select_active_clients(iter)

            # 训练客户端
            total_samples = self.train_clients(active_clients, iter, lr)

            # 预训练阶段进行聚类
            if iter < self.PretrainIter:
                self.perform_clustering(iter)

            # 如果是预训练结束，绘制指标图
            # if iter == self.PretrainIter:
            #     plot_cluster_metrics(self.mean_diffs, self.std_diffs, self.PretrainIter)

            # 聚合模型
            self.aggregate_models(active_clients, total_samples)

            # 评估
            self.evaluate(loss_hist, acc0_hist, acc1_hist, iter)

            # 更新学习率
            lr *= self.args.decay

        # 保存结果
        self.save_results(loss_hist, acc0_hist, acc1_hist)
        return loss_hist, acc0_hist, acc1_hist


def main(param):
    """Main function"""
    # 解析参数
    args = get_sys_params(param)

    # 创建系统实例
    system = FederatedSystem(args)

    # 训练系统
    loss_hist, acc0_hist, acc1_hist = system.train()

    return loss_hist, acc0_hist, acc1_hist


if __name__ == '__main__':
    from paramater import param
    main(param)