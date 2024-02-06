import os
from tqdm import tqdm
import numpy as np
import wandb
import copy
import torch
from pytorch_model_summary import summary

from datasets import BACKBONES, get_dataset
from models import get_model
from utils.loggers import *


def evaluate(model, dataset, device, classifier=None, last=False):
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
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
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


def main():
    num_epochs = 2

    dataset = get_dataset()
    dataset_copy = get_dataset()

    train_loader, test_loader, memory_loader = dataset_copy.get_data_loaders()
    # wandb.init(project="hcl_audio", sync_tensorboard=True)
    # wandb.run.name = f"{args.model.cl_model}_{args.dataset.name}_n_alpha_{args.alpha}"

    global_model = get_model(
        dataset_copy, dataset.get_transform(), global_model=None)
    model = get_model(dataset_copy, dataset.get_transform(),
                      global_model=global_model)

    accuracy = 0
    results, results_mask_classes = [], []

    for t in range(dataset.N_TASKS):
        train_loader, memory_loader, test_loader = dataset.get_data_loaders()

        global_progress = tqdm(range(0, num_epochs), desc=f'Training')

        prev_mean_acc = 0.
        best_epoch = 0.

        if BACKBONES["esc50"][t] != BACKBONES["esc50"][t-1]:
            model = get_model(dataset_copy, dataset.get_transform(),
                              task_id=t, global_model=global_model)
            print(summary(model.net.module.backbone, torch.zeros(
                (1, 3, 32, 32)).to("cuda:0"), show_input=True))

        if hasattr(model, "begin_task"):
            model.begin_task(t, dataset)

        if t:
            accs = evaluate(model, dataset, "cuda:0", last=True)
            results[t-1] = results[t-1] + accs[0]
            results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        for epoch in global_progress:
            model.train()

            local_progress = tqdm(
                train_loader, desc=f'Epoch {epoch}/{num_epochs}', disable=True)
            for idx, data in enumerate(local_progress):
                (images1, images2, notaug_images), labels = data
                data_dict = model.observe(
                    images1, labels, images2, notaug_images, t)

            global_progress.set_postfix(data_dict)

            accs = evaluate(model.net.module.backbone, dataset, "cuda:0")
            mean_acc = np.mean(accs, axis=1)

            epoch_dict = {"epoch": epoch, "accuracy": mean_acc}

            if sum(mean_acc)/2. - prev_mean_acc < 0.2:
                continue
            best_model = copy.deepcopy(model.net.module.backbone)
            prev_mean_acc = sum(mean_acc)/2.
            best_epoch = epoch

        accs = evaluate(best_model, dataset, "cuda:0")
        results.append(accs[0])
        results_mask_classes.append(accs[1])
        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t+1, "class-il")

        model.global_model.net.backbone = copy.deepcopy(best_model)
        print(
            f"Updated global model at epoch {best_epoch} with accuracy {prev_mean_acc}")

        model_path = os.path.join(".\checkpoints", f"distl_esc50_{t}.pth")
        torch.save({
            'epoch': best_epoch+1,
            'state_dict': model.global_model.net.state_dict(),
        }, model_path)
        print(f"Task Model saved to {model_path}")
        with open(os.path.join(".\logs", f"checkpoint_path.txt"), 'w+') as f:
            f.write(f'{model_path}')

        if hasattr(model, "end_task"):
            model.end_task(dataset)


if __name__ == "__main__":
    main()
