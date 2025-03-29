# -*- coding: UTF-8 -*-
'''
@Time    : 2025/3/21 17:00
@Author  : AadSama
@Software: Pycharm
'''
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import pandas as pd
import random
from imblearn.over_sampling import RandomOverSampler

from dataset import tmb_dataset
from model.MT_Clip import MT_Clip
from evaluation.K_M import K_M
import warnings
warnings.filterwarnings("ignore")


seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载参数
L = torch.load("data/L.pt")
S = torch.load("data/S.pt")
subgroup_num = torch.load("data/subgroup_num.pt")
clone_num = torch.load("data/clone_num.pt")
print("S:", S)
print("L:", L)


# 超参数
param_gamma = 0.1
pos_weights = [
    [0.9, 2.7, 2.5, 3],
    [1, 3, 3, 3],
    [1, 2.5, 2.5, 3]
]
epochs = [2, 14, 56]
oversample_rates = [1, 1, 1]
lrs = [[0.001, 0.0005, 0.001, 0.001],
       [0.01, 0.005, 0.01, 0.01],
       [0.001, 0.0005, 0.001, 0.001]]


# 训练结果
study_cv = []
true_label_cv = []
PFS_cv = []
Status_cv = []
score_cv = []
group_cv = []


def MT_Clip_Train(train_loader, models, optimizers, fold):
    '''
    sorted_data: [['TMB_sum', 'AF_avg', 'CCF_clone']]
    response: ['Study ID', 'ORR', 'PFS', 'Status']
    '''
    train_loss = torch.zeros(subgroup_num)
    train_error = torch.zeros(subgroup_num)
    batch_count = torch.zeros(subgroup_num)

    for i in range(subgroup_num):
        models[i].train()

    for batch_idx, (features, response) in enumerate(train_loader):
        features, response = features.squeeze(0), response.squeeze(0)
        #print("features", features)
        #print("response", response)
        study_id = response[0]
        study_index = int(study_id - 1)
        label = response[1]

        loss = []
        error = []
        theta_X = []
        for i in range(subgroup_num):
            optimizers[i].zero_grad()
            loss_i, predicted_prob_i, error_i, _, _ = models[i].calculate(features, label, pos_weights[fold][i])
            loss.append(loss_i)
            error.append(error_i)
            theta_X.append(predicted_prob_i)
        '''
        loss_all = sum(S[study_index][i] * loss[i] for i in range(subgroup_num)) + \
                   (param_gamma) * sum(theta_X[i] ** 2 * L[i][i] for i in range(subgroup_num))
        '''
        loss_all = (loss[0] * S[study_index][0] + \
                    loss[1] * S[study_index][1] + \
                    loss[2] * S[study_index][2] + \
                    loss[3] * S[study_index][3] + \
                    (param_gamma) * theta_X[0] ** 2 * L[0][0] + \
                    theta_X[1] ** 2 * L[1][1] + \
                    theta_X[2] ** 2 * L[2][2] + \
                    theta_X[3] ** 2 * L[3][3])
        loss_all.backward()

        for i in range(subgroup_num):
            optimizers[i].step()

        batch_count[study_index] += 1
        train_loss[study_index] = train_loss[study_index] + loss[study_index].item()
        train_error[study_index] = train_error[study_index] + error[study_index]

    for i in range(subgroup_num):
        if batch_count[i] > 0:
            train_loss[i] = train_loss[i] / batch_count[i]
            train_error[i] = train_error[i] / batch_count[i]
            print(f'Study ID: {i + 1}, Train Loss: {train_loss[i]:.4f}, Train Error: {train_error[i]:.4f}')


def MT_Clip_Val(val_loader, models, save_flag):
    for i in range(subgroup_num):
        models[i].eval()

    test_loss = torch.zeros(subgroup_num)
    test_error = torch.zeros(subgroup_num)
    batch_count = torch.zeros(subgroup_num)
    test_loss_all = 0.
    test_error_all = 0.
    prediction_list = []

    with torch.no_grad():
        for batch_idx, (features, response) in enumerate(val_loader):
            features, response = features.squeeze(0), response.squeeze(0)
            study_id = response[0]
            study_index = int(study_id - 1)
            label = response[1]
            PFS = response[2]
            Status = response[3]

            loss, predicted_prob, error, predicted_label, A = models[study_index].calculate(features, label)

            test_loss[study_index] = test_loss[study_index] + loss.data[0]
            test_loss_all = test_loss_all + loss.data[0]
            test_error[study_index] = test_error[study_index] + error
            test_error_all = test_error_all + error
            batch_count[study_index] = batch_count[study_index] + 1

            prediction_list.append([label.item(), predicted_prob.item(), predicted_label.item(), study_id.item(), PFS.item(), Status.item()])

    for i in range(subgroup_num):
        if batch_count[i] > 0:
            test_loss[i] = test_loss[i] / batch_count[i]
            test_error[i] = test_error[i] / batch_count[i]
            print(f'Study ID: {i + 1}, Test Loss: {test_loss[i]:.4f}, Test error: {test_error[i]:.4f}')

    test_loss_all = test_loss_all / len(val_loader)
    test_error_all = test_error_all / len(val_loader)
    print(f'Test Loss: {test_loss_all.item():.4f}, Test error: {test_error_all:.4f}')

    true_labels = [label for label, _, _, _, _, _ in prediction_list]
    predicted_probs = [predicted_prob for _, predicted_prob, _, _, _, _ in prediction_list]
    predicted_labels = [predicted_label for _, _, predicted_label, _, _, _ in prediction_list]
    auc = roc_auc_score(true_labels, predicted_probs)
    acc = accuracy_score(true_labels, predicted_labels)
    print('ACC: {:.4f}'.format(acc))
    print('AUC: {:.4f}'.format(auc))

    if(save_flag ==1):
        # [label, predicted_prob, predicted_label, study_id, PFS, Status]
        for i in range(len(prediction_list)):
            study_cv.append(prediction_list[i][3])
            true_label_cv.append(prediction_list[i][0])
            PFS_cv.append(prediction_list[i][4])
            Status_cv.append(prediction_list[i][5])
            score_cv.append(prediction_list[i][1])
            group_cv.append(prediction_list[i][2])


def Oversampling(oversample_rate, train_subset_raw, clone_num):
    '''
    sorted_data: [['TMB_sum', 'AF_avg', 'CCF_clone']]
    response: ['Study ID', 'ORR', 'PFS', 'Status']
    '''
    if (oversample_rate != 0):
        features_list = []
        labels_list = []
        response_dict = {}

        for i in range(len(train_subset_raw)):
            features, response = train_subset_raw[i]
            k, feature_dim = features.shape

            if k < clone_num:
                pad_size = (clone_num - k, feature_dim)
                padded_features = torch.cat([features, torch.zeros(pad_size)], dim=0)
            else:
                padded_features = features[:clone_num]

            flat_features = padded_features.numpy().flatten()
            features_list.append(flat_features)
            labels_list.append(response[1].item())

            response_dict[tuple(flat_features)] = response

        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        #print("features_array", features_array[:3])
        #print("labels_array", labels_array[:3])

        ros = RandomOverSampler(sampling_strategy=oversample_rate, random_state=seed)
        features_resampled, labels_resampled = ros.fit_resample(features_array, labels_array)

        resampled_dataset = []
        for i in range(len(features_resampled)):
            resampled_features = torch.tensor(features_resampled[i].reshape(clone_num, 3), dtype=torch.float)

            non_zero_rows = (resampled_features.sum(dim=1) != 0)
            resampled_features = resampled_features[non_zero_rows]

            original_response = response_dict.get(tuple(features_resampled[i]),
                                                  torch.tensor([0, labels_resampled[i], 0, 0], dtype=torch.float))
            resampled_dataset.append((resampled_features, original_response))
    else:
        resampled_dataset = train_subset_raw

    #print(resampled_dataset[:3])
    return resampled_dataset


def Cross_Validation(raw_data):
    '''
    raw_data: [['Study ID', 'ORR', 'PFS', 'Status', ['TMB_sum', 'AF_avg', 'CCF_clone']]]
    '''
    dataset = tmb_dataset(raw_data)
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    labels = [patient[1] for patient in raw_data]
    #print("labels", labels)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(raw_data, labels)):
        print(f"Fold {fold + 1} ############################################################")

        train_subset_raw = Subset(dataset, train_idx)
        train_subset = Oversampling(oversample_rates[fold], train_subset_raw, clone_num)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

        # 初始化模型
        models = {}
        optimizers = {}
        for i in range(subgroup_num):
            torch.manual_seed(42)
            model = MT_Clip().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lrs[fold][i], betas=(0.9, 0.999), weight_decay=10e-5)
            models[i] = model
            optimizers[i] = optimizer

        for epoch in range(epochs[fold]):
            print(f"Epoch {epoch + 1}")
            MT_Clip_Train(train_loader, models, optimizers, fold)
            MT_Clip_Val(val_loader, models, epoch==epochs[fold]-1)


    y_true = np.array(true_label_cv)
    y_pred_label = np.array(group_cv)
    y_pred_score = np.array(score_cv)
    acc = accuracy_score(y_true, y_pred_label)
    print(f"CV ACC: {acc:.4f}")
    auc = roc_auc_score(y_true, y_pred_score)
    print(f"CV AUC: {auc:.4f}")


    result_cv = pd.DataFrame({
        "study": study_cv,
        "true_label": true_label_cv,
        "PFS": PFS_cv,
        "Status": pd.Series(Status_cv).astype(int),
        "score": score_cv,
        "group": pd.Series(group_cv).astype(int)
    })
    result_cv.to_csv('results/MT_Clip_find_all.csv', index=False)

    K_M('results/MT_Clip_find_all.csv')


if __name__ == "__main__":
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 加载数据
    raw_data = torch.load("data/raw_data.pt")

    Cross_Validation(raw_data)