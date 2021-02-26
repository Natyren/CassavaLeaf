import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from losses import *
from cutmix import cutmix
from sklearn.metrics import accuracy_score as accuracy, f1_score as f1
from tqdm import tqdm
import time

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class Leaf_Dataset(Dataset):
    def __init__(self, data, transforms, split_type, train_path, test_path):
        self.data = data
        self.transform = transforms
        self.type = split_type
        self.train_path = train_path
        self.test_path = test_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        lbl = self.data.iloc[idx].label
        im_name = self.data.iloc[idx].image_id

        if self.type == 'train' or self.type == 'val':
            image_path = self.train_path + im_name
        elif self.type == 'test':
            image_path = self.test_path + im_name

        img = Image.open(image_path).convert('RGB')
        img = np.array(img)

        if self.transform:
            img = self.transform(image=img)

        return {'img': img['image'], 'lbl': lbl}


def train_val(model, epochs, fold, dataloader, criterion, optimizer, scheduler, device):
    accs = {'train': [], 'val': []}
    losses = {'train': [], 'val': []}
    lrs = []
    for epoch in range(epochs):

        for phase in ['train', 'val']:
            print('Epoch: {}'.format(epoch))
            print('Phase: {}'.format(phase))
            time.sleep(0.3)
            if phase == 'train':
                model.train()
            else:
                model.eval()

            y_preds, y_trues = [], []
            tr_loss = 0
            tk = tqdm(dataloader[phase], total=len(dataloader[phase]), position=0, leave=True)
            l = len(dataloader[phase])

            for step, batch in enumerate(tk):

                optimizer.zero_grad()

                labels_ = batch['lbl'].to(device, dtype=torch.long)
                inputs = batch['img'].to(device, dtype=torch.float)
                mix_decision = np.random.rand()
                if phase == 'train' and mix_decision < 0.5:
                    inputs, labels = cutmix(inputs, labels_, 1.)

                with torch.set_grad_enabled(phase == 'train'):

                    preds = model(inputs)
                    if phase == 'train' and mix_decision < 0.5:
                        loss = criterion(preds, labels[0]) * labels[2] + criterion(preds, labels[1]) * (1. - labels[2])
                    else:
                        loss = criterion(preds, labels_)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                y_preds.extend(torch.argmax(preds.cpu(), axis=1))
                y_trues.extend(labels_.cpu())

                tr_loss += loss.item()

                # if phase == 'val':
                if phase == 'train':
                    scheduler.step(epoch + step / l)
                    lrs.append(scheduler.get_last_lr())

            acc = accuracy(y_preds, y_trues)
            f1_score = f1(y_preds, y_trues, average='macro')
            epoch_loss = tr_loss / l
            print()
            print(phase + ' loss: {:.4f}'.format(epoch_loss))
            print(phase + ' accuracy: {:.4f}'.format(acc))
            print(phase + ' f1: {:.4f}'.format(f1_score))
            print()
            time.sleep(0.3)

            accs[phase].append(acc)
            losses[phase].append(epoch_loss)

            if phase == 'val':
                plt.title('Accuracy')
                plt.plot(accs['train'])
                plt.plot(accs['val'])
                plt.show()
                plt.title('Loss')
                plt.plot(losses['train'])
                plt.plot(losses['val'])
                plt.show()
                plt.title('Learning_rates')
                plt.plot(lrs)
                plt.show()

        # scheduler.step()
        # torch.save(model.state_dict(), 'weight_epoch#{}_fold#{}_ver#{}_effnet4.pt'.format(epoch, fold, CFG.version))
    if epoch == epochs - 1:
        return (f1_score, epoch_loss, acc)