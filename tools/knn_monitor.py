from tqdm import tqdm
import torch.nn.functional as F
import torch
import numpy as np
import copy
from utils.metrics import mask_classes


def knn_monitor(net, dataset, memory_data_loader, test_data_loader, task_id, k=200, t=0.1):
    net.eval()
    try:
        classes = len(memory_data_loader.dataset.classes)
    except:
        classes = 200
    total_top1 = total_top1_mask = total_top5 = total_num = 0.0
    feature_bank = []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=True):
            feature = net(data.cuda(non_blocking=True), return_features=True)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        # feature_labels = torch.tensor(memory_data_loader.dataset.targets - np.amin(memory_data_loader.dataset.targets), device=feature_bank.device)
        feature_labels = torch.tensor(
            memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc='kNN', disable=True)
        for data, target in test_bar:
            data, target = data.cuda(
                non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data, return_features=True)
            feature = F.normalize(feature, dim=1)
            pred_scores = knn_predict(
                feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.shape[0]
            _, preds = torch.max(pred_scores.data, 1)
            total_top1 += torch.sum(preds == target).item()

            pred_scores_mask = mask_classes(
                copy.deepcopy(pred_scores), dataset, task_id)
            _, preds_mask = torch.max(pred_scores_mask.data, 1)
            total_top1_mask += torch.sum(preds_mask == target).item()

    return total_top1 / total_num * 100, total_top1_mask / total_num * 100


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(
        feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(
        0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(
        0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    return pred_scores
