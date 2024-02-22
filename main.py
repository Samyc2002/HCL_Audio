import os
from tqdm import tqdm
import numpy as np
import wandb
import copy
import torch
from pytorch_model_summary import summary

from arguments import get_args
from datasets import BACKBONES, get_dataset
from models import get_model
from utils.loggers import *


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
                outputs = classifier(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == "class-il":
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct/total * 100)
        accs_mask_classes.append(correct_mask_classes/total * 100)

    model.train(status)
    return accs, accs_mask_classes


def main(device, args):
    dataset = get_dataset(args)
    dataset_copy = get_dataset(args)

    # train_loader, test_loader, memory_loader = dataset_copy.get_data_loaders(
    #     args, 0)
    # wandb.init(project="hcl_audio", sync_tensorboard=True)
    # wandb.run.name = f"{args.model.cl_model}_{args.dataset.name}_n_alpha_{args.alpha}"

    global_model = get_model(device,
                             dataset_copy, dataset.get_transform(), args, global_model=None)
    model = get_model(device, dataset_copy, dataset.get_transform(), args,
                      global_model=global_model)

    accuracy = 0
    results, results_mask_classes = [], []

    for t in range(dataset.N_TASKS):
        train_loader, memory_loader, test_loader = dataset.get_data_loaders(
            args, t)

        global_progress = tqdm(range(0, args.num_epochs), desc=f'Training')

        prev_mean_acc = 0.
        best_epoch = 0.

        if BACKBONES[args.dataset][t] != BACKBONES[args.dataset][t-1]:
            model = get_model(device, dataset_copy, dataset.get_transform(), args,
                              task_id=t, global_model=global_model)
            image_size = 255
            if "mixer" in BACKBONES[args.dataset][t] or "vit" in BACKBONES[args.dataset][t]:
                image_size = 224

            print(summary(model.net.module.backbone, torch.zeros(
                (1, 3, image_size, image_size)).to(device), show_input=True))

        if hasattr(model, "begin_task"):
            model.begin_task(t, dataset)

        if t:
            accs = evaluate(model, dataset, device, last=True,
                            image_size=224 if "mixer" in BACKBONES[args.dataset][t] or "vit" in BACKBONES[args.dataset][t] else 255)
            results[t-1] = results[t-1] + accs[0]
            results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        for epoch in global_progress:
            model.train()

            local_progress = tqdm(
                train_loader, desc=f'Epoch {epoch}/{args.num_epochs}', disable=True)
            for idx, data in enumerate(local_progress):
                (images1, images2, notaug_images), labels = data
                data_dict = model.observe(
                    images1, labels, images2, notaug_images, t)

            global_progress.set_postfix(data_dict)

            accs = evaluate(model.net.module.backbone, dataset, device,
                            image_size=224 if "mixer" in BACKBONES[args.dataset][t] or "vit" in BACKBONES[args.dataset][t] else 255)
            mean_acc = np.mean(accs, axis=1)

            epoch_dict = {"epoch": epoch, "accuracy": mean_acc}

            if sum(mean_acc)/2. - prev_mean_acc < 0.2:
                continue
            best_model = copy.deepcopy(model.net.module.backbone)
            prev_mean_acc = sum(mean_acc)/2.
            best_epoch = epoch

        accs = evaluate(best_model, dataset, device,
                        image_size=224 if "mixer" in BACKBONES[args.dataset][t] or "vit" in BACKBONES[args.dataset][t] else 255)
        results.append(accs[0])
        results_mask_classes.append(accs[1])
        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t+1, "class-il")

        model.global_model.net.backbone = copy.deepcopy(best_model)
        print(
            f"Updated global model at epoch {best_epoch} with accuracy {prev_mean_acc}")

        model_path = os.path.join(
            "./checkpoints", f"distl_{args.dataset}_{t}.pth")
        torch.save({
            'epoch': best_epoch+1,
            'state_dict': model.global_model.net.state_dict(),
        }, model_path)
        print(f"Task Model saved to {model_path}")
        with open(os.path.join("./logs", f"{args.dataset}/checkpoint_path.txt"), 'w+') as f:
            f.write(f'../.{model_path}')
        with open(os.path.join("./logs", f"{args.dataset}/benchmarking.txt"), "a") as f:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            f.write('Accuracy for {} task(s): \t [Class-IL]: {} % \t [Task-IL]: {} %\n'.format(
                t+1, round(mean_acc_class_il, 2), round(mean_acc_task_il, 2)))

        if hasattr(model, "end_task"):
            model.end_task(dataset)


if __name__ == "__main__":
    args = get_args()
    main(args.device, args)
