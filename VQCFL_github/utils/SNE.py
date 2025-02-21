import copy
import os
import  numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from utils.load_feature import *
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import cv2

def SNE(model,paths,datasets):
    features_list = []
    labels_list = []
    for i in range(10):

        fea,lab = get_features(model,paths[i],datasets[i])
        # fea,lab = get_features_origin(datasets[i])
        features_list.append(fea)
        labels_list.append(lab)
    features_list = [np.array(features) for features in features_list]
    labels_list = [np.array(labels) for labels in labels_list]

    visualize_tsne_with_hog_all(features_list, labels_list)
#
# def visualize_tsne(features_list, labels_list):
#     markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h']
#
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')  # 使用三维投影
#
#
#     # 将列表转换为数组
#     markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h']
#     color_map = plt.cm.get_cmap('tab10', 10)  # 使用 'tab10' 调色板，其中颜色差异较大
#
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#
#     # 对每个类别的特征进行标准化
#     for i, (features, labels) in enumerate(zip(features_list, labels_list)):
#         tsne = TSNE(n_components=3, init='pca', perplexity=5, random_state=2024, method='exact')
#         features = features.reshape(features.shape[0], -1)
#
#         # 标准化数据
#         scaler = StandardScaler()
#         features_standardized = scaler.fit_transform(features)
#
#         transformed_features = tsne.fit_transform(features_standardized)
#
#         # 将 transformed_features 映射到 [0, 1] 范围
#         transformed_features = (transformed_features - transformed_features.min()) / (
#                 transformed_features.max() - transformed_features.min())
#
#         marker = markers[i]
#         for j in range(10):
#             label_indices = [index for index, label in enumerate(labels) if label == j]
#             ax.scatter(transformed_features[label_indices, 0], transformed_features[label_indices, 1],
#                        transformed_features[label_indices, 2],
#                        s=100,  # 设置一个合适的标量值作为散点大小
#                        label=f'Class {j}', marker=marker, color=color_map(j))
#
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1])
#     ax.set_zlim([0, 1])
#
#     # plt.legend()
#     plt.show()

#
# def visualize_tsne_with_hog(features_list, labels_list):
#
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#     markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h']
#     color_map = plt.cm.get_cmap('tab10', 10)
#
#     for i, (features, labels) in enumerate(zip(features_list, labels_list)):
#         tsne = TSNE(n_components=3, init='pca', perplexity=5, random_state=2024, method='exact')
#
#         # HOG处理
#         hog_features = []
#         for img in features:
#             hog_feature = hog(img.reshape((28, 28)), block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2))
#             hog_features.append(hog_feature)
#
#         hog_features = np.array(hog_features)
#
#         # 标准化数据
#         scaler = StandardScaler()
#         features_standardized = scaler.fit_transform(hog_features)
#
#         transformed_features = tsne.fit_transform(features_standardized)
#
#         transformed_features = (transformed_features - transformed_features.min()) / (
#                 transformed_features.max() - transformed_features.min())
#
#         marker = markers[i]
#         for j in range(10):
#             label_indices = [index for index, label in enumerate(labels) if label == j]
#             ax.scatter(transformed_features[label_indices, 0], transformed_features[label_indices, 1],
#                        transformed_features[label_indices, 2],
#                        s=200,
#                        label=f'Class {j}', marker=marker, color=color_map(j))
#
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1])
#     ax.set_zlim([0, 1])
#
#     plt.show()
#

def visualize_tsne_with_hog_all(features_list, labels_list, alpha=0.4):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    markers = ['o', 's', '^', 'v', '<', '>', 'D', 'x', '*', '+']
    color_map = plt.cm.get_cmap('tab10', 10)

    all_features = np.concatenate(features_list, axis=0)  # 合并所有特征
    all_labels = np.concatenate(labels_list, axis=0)  # 合并所有标签

    tsne = TSNE(n_components=3, init='pca', perplexity=8, random_state=2024, method='exact')

    # HOG处理
    # hog_features = []
    # for img in all_features:
    #     hog_feature = process_color_image(img)
    #     hog_features.append(hog_feature)

    # hog_features = np.array(hog_features)
    hog_features = all_features

    # 标准化数据
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(hog_features)

    transformed_features = tsne.fit_transform(features_standardized)

    transformed_features = (transformed_features - transformed_features.min()) / (
            transformed_features.max() - transformed_features.min())

    for j in range(10):
        label_indices = [index for index, label in enumerate(all_labels) if label == j]
        ax.scatter(transformed_features[label_indices, 0], transformed_features[label_indices, 1],
                   transformed_features[label_indices, 2],
                   s=50,
                   label=f'Class {j}', marker=markers[j], color=color_map(j, alpha=alpha))

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    plt.show()



#
# def visualize_tsne_with_model(features_list, labels_list, alpha=0.6):
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#     markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h']
#     color_map = plt.cm.get_cmap('tab10', 10)
#
#     all_features = np.concatenate(features_list, axis=0)  # 合并所有特征
#     all_labels = np.concatenate(labels_list, axis=0)  # 合并所有标签
#
#     tsne = TSNE(n_components=3, init='pca', perplexity=10, random_state=100, method='exact')
#
#     # HOG处理
#     hog_features = []
#     for img in all_features:
#         hog_feature = process_color_image(img)
#         hog_features.append(hog_feature)
#
#     hog_features = np.array(hog_features)
#
#     # 标准化数据
#     scaler = StandardScaler()
#     features_standardized = scaler.fit_transform(hog_features)
#
#     transformed_features = tsne.fit_transform(features_standardized)
#
#     transformed_features = (transformed_features - transformed_features.min()) / (
#             transformed_features.max() - transformed_features.min())
#
#     for j in range(10):
#         label_indices = [index for index, label in enumerate(all_labels) if label == j]
#         ax.scatter(transformed_features[label_indices, 0], transformed_features[label_indices, 1],
#                    transformed_features[label_indices, 2],
#                    s=100,
#                    label=f'Class {j}', marker=markers[j], color=color_map(j, alpha=alpha))
#
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1])
#     ax.set_zlim([0, 1])
#
#     plt.show()
#
def process_color_image(img):
    # 将通道维度移到最后，并转换为8位无符号整数图像
    img_uint8 = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

    # 将彩色图像转换为灰度图像
    gray_img = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    # 进行HOG处理
    hog_feature = hog(gray_img, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2))

    return hog_feature



if __name__ == '__main__':
    matplotlib.use("Qt5Agg")
    project_directory = "E:\实验部分\源代码\FedGS\FedGS"  # 请将此路径替换为你的项目路径
    os.chdir(project_directory)
    from models.create_model import load_model
    model = load_model("CIFAR", 0)
    paths = []
    for i in range(10):
        paths.append(f"./gapsave/model/rep_model_{i}_i_100.pth")

    from  utils.get_data import get_dataloaders
    list_dls_train, list_dls_test, client_lable_count = get_dataloaders("digits", 50, 0.1)


    SNE(copy.deepcopy(model),paths,list_dls_train)