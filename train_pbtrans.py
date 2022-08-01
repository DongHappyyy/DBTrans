import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import json
import copy
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from models.pbtrans import PBTrans



class Config:
    image_resize = 256
    image_crop = 224                # 256 for 4D-Light, 224 for others
    batch_size = 128                # 64 for MIT-Indoor, 128 for others
    num_epochs = 200
    learning_ratio = 0.0001
    feature_extract = True
    backbone = "resnet50"           # vgg19 resnet18 resnet50 resnet101 densenet161
    dataset_name = "FMD"            #  "DTD" "FMD" "KTH-TIPS2-b" "4D_Light" "MIT-Indoor"
    model_name = "PBTrans"
    data_dir = f"datasets/{dataset_name}/splits"
    output_dir = f"outputs/{model_name}/{dataset_name}"


DataTransforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(Config.image_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    'test' : transforms.Compose([
        transforms.Resize(Config.image_resize),
        transforms.CenterCrop(Config.image_crop),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


# function for training model
def trainModel(split, run, model, dataloaders, criterion, optimizer, save_dir, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    tra_acc_history = []
    best_acc = 0.0
    best_epoch = 0
    best_model_path = ""

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print(f'Start Time : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print('-' * 35)
        t1 = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                def closure():
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    return loss

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step(closure)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test':
                val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:

                    old_model_path = save_dir / f"{best_epoch}_{best_acc:.6f}.pth.tar"
                    # old_model_path.unlink(missing_ok=True)  # python 3.8
                    if old_model_path.exists():
                        old_model_path.unlink()

                    best_acc = epoch_acc
                    best_epoch = epoch + 1

                    best_model_path = save_dir / f"{best_epoch}_{best_acc:.6f}.pth.tar"
                    state = {"epoch": epoch + 1,
                             "accuracy": epoch_acc,
                             "state_dict": model.state_dict(),
                             "optimizer_dict": optimizer.state_dict()}
                    torch.save(state, best_model_path)
            else:
                tra_acc_history.append(epoch_acc)

        t2 = time.time()
        print(f'Best Acc : {best_acc:.6f} Epoch : {best_epoch}')
        print(f'Split : {split} Run : {run}')
        print('-' * 35)
        print(f'End Time : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'Epoch Time : {(t2-t1)/60:.4f} minute  ({(t2-t1):.4f}s)')
        print('=' * 39, '\n')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Best Test Acc: {best_acc:6f}')

    return val_acc_history, tra_acc_history, best_acc, best_epoch, best_model_path



def freezeModelParameters(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



run_count = 1
split_count = 10
dtstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
output_root = Path(Config.output_dir)
for split_idx in range(1, split_count+1):

    split_dir = f'split_{split_idx}'
    data_dir = Path(Config.data_dir) / split_dir
    save_root = output_root / split_dir
    print(f"data_dir : {data_dir}")
    print(f"save_root : {save_root}")

    Experiment_acc_history = []
    for experiment_idx in range(1, run_count+1):
        print(f"RUN : {experiment_idx}")
        save_dir = save_root / f"RUN_{experiment_idx}"
        save_dir.mkdir(parents=True, exist_ok=True)

        image_datasets = {x: datasets.ImageFolder(data_dir / x,
                          transform=DataTransforms[x])
                          for x in ['train', 'test']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                          batch_size=Config.batch_size, shuffle=True, num_workers=4)
                          for x in ['train', 'test']}
        class_names = image_datasets['train'].classes
        num_classes = len(class_names)
        print(f'dataset_sizes : {dataset_sizes}')
        print(f'num_classes : {num_classes}')
        print(f'class_names : {class_names}')

        net = eval(Config.model_name)(Config.backbone, num_classes)
        feature_extract = Config.feature_extract
        freezeModelParameters(net.backbone_1, feature_extract)
        freezeModelParameters(net.backbone_2, feature_extract)

        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_ft = net.to(device)
        print(f'device : {device}')

        
        print("Params to learn:")
        params_to_update = [] if feature_extract else model_ft.parameters()
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
                if feature_extract:
                    params_to_update.append(param)

                    
        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(params_to_update, lr=Config.learning_ratio)
        criterion = nn.CrossEntropyLoss()

        
        # training part
        hist_val, hist_train, best_acc, best_epoch, best_model_path = trainModel(
            split_idx, experiment_idx, model_ft, dataloaders, criterion,
            optimizer_ft, save_dir, num_epochs=Config.num_epochs, is_inception=False)


        # checking if there is a file with this nam
        middle = "_".join([f'{Config.backbone}',
                           f'{Config.image_crop}',
                           f'epoch{Config.num_epochs}',
                           f'lr{Config.learning_ratio}',
                           f'batchsize{Config.batch_size}',
                           f'{Config.dataset_name}',
                           f'split{split_idx}',
                           f'run{experiment_idx}',
                           f'bestEpoch{best_epoch}',
                           f'bestAcc{best_acc:.6f}'])
        path_model = save_dir / f"{middle}.pth.tar"
        path_save_fig = save_dir / f"{middle}.png"
        path_save_txt = save_dir / f"{middle}.txt"
        path_save_json = save_dir / f"{middle}.json"


        # save best model
        shutil.copy(best_model_path, path_model)


        # draw learning curve and save the results (accuracy and corresponding model)
        tra_hist = [float(h.cpu().numpy()) for h in hist_train]
        tes_hist = [float(h.cpu().numpy()) for h in hist_val]
        plt.title("Accuracy vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Accuracy")
        plt.plot(range(1, Config.num_epochs+1), tra_hist, label="train")
        plt.plot(range(1, Config.num_epochs+1), tes_hist, label='test')
        plt.ylim((0, 1.))
        plt.xticks(np.arange(0, Config.num_epochs+1, 25))
        plt.legend()
        fig = plt.gcf()
        fig.savefig(path_save_fig, dpi=600, bbox_inches='tight', pad_inches=0)
        fig.clear()

        
        # save acc history
        acc_history = {
            'test' : tes_hist,
            'train' : tra_hist
        }
        with open(path_save_json, 'w') as handle:
            json.dump(acc_history, handle)

            
        # save best acc history
        Experiment_acc_history.append(best_acc)
        Experiment_acc_history_float = torch.FloatTensor(Experiment_acc_history)
        np.savetxt(path_save_txt, Experiment_acc_history_float)

        
        print('Best Accracy History list :\n', Experiment_acc_history_float)
        Experiment_acc_history_np = Experiment_acc_history_float.numpy()
        print(f"Average accuracy in {str(experiment_idx)} runs is")
        print(str(torch.mean(torch.FloatTensor(Experiment_acc_history))))
        print("var : ", np.var(Experiment_acc_history_np))
        print('\n', '=' * 16, ' Finish Training ', '=' * 16, '\n\n')
