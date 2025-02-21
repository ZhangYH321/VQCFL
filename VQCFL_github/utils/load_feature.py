import torch
import copy
import numpy as np

def get_features(model_0,path,data):
    model = copy.deepcopy(model_0)

    s = torch.load(path,map_location = torch.device('cpu'))
    # model.load_state_dict(s)
    model = copy.deepcopy(s)
    feature_list = []
    label_list = []

    batch_limit = 4  # 设置你想要取的 batch 数量

    for batch_idx, (features, labels) in enumerate(data):
        if batch_idx >= batch_limit:
            break  #
        labels = labels.long()
        model.last_feature = None
        predictions = model(features)

        last_features = model.last_feature
        if last_features is not None:
            for j in range(len(last_features)):
                feature = last_features[j].numpy().flatten()
                label = labels[j]

                # feature = feature.unsqueeze(0)
                feature_list.append(feature)
                label_list.append(label)
        model.last_feature = None
    return feature_list,label_list


def get_features_origin(data):
    feature_list = []
    label_list = []

    batch_limit = 3  # 设置你想要取的 batch 数量

    for batch_idx, (features, labels) in enumerate(data):
        if batch_idx >= batch_limit:
            break  #
        # if batch_idx != batch_limit:
        #     continue
        labels = labels.long()
        for sample_features in features:
            feature_list.append(sample_features.tolist())

            # 将对应的标签添加到列表中
        label_list.extend(labels.tolist())
    # feature_array = np.array(feature_list)
    # reshaped_array = feature_array.reshape(feature_array.shape[0], -1, 28, 28)[:, 0, :, :]
    # feature_list = reshaped_array.tolist()
    return feature_list, label_list