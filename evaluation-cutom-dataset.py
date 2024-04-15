"""
Evaluation script For Custom Datasets made using adversarial attacks
"""
import os
import h5py
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from arguments import get_args
from augmentations import get_aug
from models import get_model
from datasets import get_dataset
from utils.loggers import print_mean_accuracy
from utils.metrics import forgetting
from utils import get_valid_keys_for_backbone
from datasets import BACKBONES
from datasets.my_dcase19 import MYDCASE19
from datasets.my_esc10 import MYESC10
from datasets.utils.validation import ValidationDataset


def getData(attack, backbone, dataset, task_id, args):
    """
    Gets the dataset for the specific backbone and performs transformations on the dataset to fit to model parameters
    :param attack: the attack performed to obtain the dataset
    :param backbone: the backbone for the model
    :param dataset: the dataset being used
    :param task_id: id of the task being performed
    :param args: additional arguments
    """
    image_file = os.path.join("customdatasets", attack,
                              dataset, backbone, "image_data.h5")
    target_file = os.path.join(
        "customdatasets", attack, dataset, backbone, "target_data.h5")

    image_size = 255
    if dataset == "esc10":
        if task_id >= 3:
            image_size = 224
    elif dataset == "dcase19":
        if task_id >= 1:
            image_size = 224
    esc_norm = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]
    test_transform = get_aug(train=False, train_classifier=False,
                             mean_std=esc_norm, name="simsiam", image_size=image_size, cl_default=True)

    h5image = h5py.File(image_file, "r")
    image_loaders = h5image[dataset][:]
    h5image.close()
    h5target = h5py.File(target_file, "r")
    target_loaders = h5target[dataset][:]
    h5target.close()

    image_data, target_data = [], []
    num_data_samples = len(image_loaders)
    for i in range(num_data_samples):
        image = image_loaders[i]
        target = target_loaders[i]
        for data_sample_index in range(128):
            S_img = Image.fromarray(image[data_sample_index, 0].astype(np.uint8)).convert(
                'RGB').resize((image_size, image_size))
            image_data.append(S_img)
            target_data.append(target[data_sample_index])

    if dataset == "esc10":
        test_dataset = MYESC10(image_data, target_data, test_transform)
    elif dataset == "dcase19":
        test_dataset = MYDCASE19(image_data, target_data, test_transform)
    else:
        raise Exception

    test_dataset = ValidationDataset(
        test_dataset.data, test_dataset.targets, transform=test_transform, image_size=image_size)
    test_loader = DataLoader(
        test_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    return test_loader


def mask_classes(outputs, classes, tasks, k):
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the number of classes per task
    :param tasks: the total number of tasks
    :param k: the task index
    """
    outputs[:, 0:k * classes] = -float('inf')
    outputs[:, (k + 1) * classes:tasks * classes] = -float('inf')

    return outputs


def evaluate(model, dataset, device, classifier=None, last=False, dataset_name=""):
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
    correct, correct_mask_classes, total = 0.0, 0.0, 0.0
    for k, data in enumerate(dataset):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        if classifier is not None:
            outputs = classifier(outputs)

        _, pred = torch.max(outputs.data, 1)
        correct += torch.sum(pred == labels).item()
        total += labels.shape[0]

        classes, tasks = 0, 0
        if dataset_name == "esc10":
            classes = 2
            tasks = 5
        if dataset_name == "dcase19":
            classes = 5
            tasks = 2

        mask_classes(outputs, classes, tasks, k)
        _, pred = torch.max(outputs.data, 1)
        correct_mask_classes += torch.sum(pred == labels).item()

    accs.append(correct / total * 100)
    accs_mask_classes.append(correct_mask_classes / total * 100)

    model.train(status)
    return accs, accs_mask_classes


def main(device, args):
    attacks = [
        "fgsm",
        "transfer"
    ]

    for attack in attacks:
        print(f"Benchmarking for {attack} attack")
        dataset = get_dataset(args)

        results, results_mask_classes = [], []
        for t in tqdm(range(0, dataset.N_TASKS), desc='Evaluating'):
            state_dict_substring = 'backbone.'

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

            test_dataloader = getData(attack, backbone, args.dataset, t, args)
            accs = evaluate(model.net.module.backbone,
                            test_dataloader, device, dataset_name=args.dataset)
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
