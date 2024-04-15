"""
Evaluation script
"""
import os
import torch
from tqdm import tqdm
import numpy as np
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import knn_monitor
from datasets import get_dataset
from utils.loggers import *
from utils.metrics import forgetting
from utils import get_valid_keys_for_backbone
from datasets import BACKBONES


def evaluate_single(model, dataset, test_loader, memory_loader, device, k, last=False):
    accs, accs_mask_classes = [], []
    knn_accs, knn_accs_mask_classes = [], []
    correct = correct_mask_classes = total = 0
    knn_acc, knn_acc_mask = knn_monitor(
        model.net.module.backbone,
        dataset,
        memory_loader,
        test_loader,
        task_id=k,
        k=min(args.train.knn_k, len(dataset.memory_loaders[k].dataset))
    )

    return knn_acc


def evaluate(model, dataset, device, classifier=None, last=False, image_size=255):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.training
    model.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders[image_size]):
        if last and k < len(dataset.test_loaders[image_size]) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if classifier is not None:
                outputs = classifier(outputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.train(status)
    return accs, accs_mask_classes


def main(device, args):
    dataset = get_dataset(args)

    results, results_mask_classes = [], []
    for t in tqdm(range(0, dataset.N_TASKS), desc='Evaluating'):
        train_loader, memory_loader, test_loader = dataset.get_data_loaders(
            args, t)
        image_size = 255
        state_dict_substring = 'backbone.'
        if args.dataset == "dcase19":
            if t >= 1:
                image_size = 224
        else:
            if t >= 2:
                image_size = 224

        model_path = os.path.join(
            "./checkpoints", f"distl_{args.dataset}_{t}.pth")
        save_dict = torch.load(model_path, map_location='cpu')
        model = get_model(
            device, dataset, dataset.get_transform(), args, task_id=t)

        backbone = BACKBONES[args.dataset][t]
        valid_keys = get_valid_keys_for_backbone(backbone)
        msg = model.net.module.backbone.load_state_dict(
            {
                k[k.find(state_dict_substring) + len(state_dict_substring):].lstrip('.'): v for k, v in save_dict['state_dict'].items() if k[k.find(state_dict_substring) + len(state_dict_substring):].lstrip('.') in valid_keys
            },
            strict=True)
        model = model.to(args.device)

        accs = evaluate(model.net.module.backbone,
                        dataset, device, image_size=image_size)
        results.append(accs[0])
        results_mask_classes.append(accs[1])
        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

    ci_mean_fgt = forgetting(results)
    ti_mean_fgt = forgetting(results_mask_classes)
    print(f'CI Forgetting: {ci_mean_fgt} \t TI Forgetting: {ti_mean_fgt}')


if __name__ == "__main__":
    args = get_args()
    main(device=args.device, args=args)
