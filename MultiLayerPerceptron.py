import datetime

from sklearn.metrics import classification_report

from AnalysisToolkit import AnalysisToolkit
from pyprobar import probar
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import wandb

class MultiLayerPerceptron(AnalysisToolkit):

    def __init__(self, route, vis_deg, rot_deg, train_path, test_path):
        AnalysisToolkit.__init__(self, route, vis_deg, rot_deg)
        self.model_name = 'MLP'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"RUNNING ON: {self.device}")

        # MLP hyper-parameters
        self.INPUT_SIZE = 360
        self.HIDDEN_SIZES = [360, 360]
        self.TRAIN_VAL_SPLIT = 0.2
        self.EPOCHS = 50
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.005

        # Data loading
        dataloaders = self.get_dataloaders(train_path, test_path)
        self.trainloader = dataloaders['TRAIN']
        self.valloader = dataloaders['VAL']
        self.testloader = dataloaders['TEST']

    def gen_data(self, angle, train_val_split=0.2):
        print('Generating data...')
        dp = []
        for filename in probar(self.route_filenames):
            view = self.preprocess(cv2.imread(self.route_path + filename))
            dp.append([f'{filename.strip(".png")}_0', view, 1])
            dp.append([f'{filename.strip(".png")}_{angle}', self.rotate(view, angle), 0])
            dp.append([f'{filename.strip(".png")}_-{angle}', self.rotate(view, -angle), 0])
        df = pd.DataFrame(dp, columns=['FILENAME', 'VIEW', 'LABEL'])
        train, test = train_test_split(df, test_size=train_val_split, random_state=0)
        for filename, view, label in train.values:
            plt.imsave(f"./ANN_DATA/{angle}_DEGREES/TRAIN/{label}/{filename}.png",
                       cv2.cvtColor(view.astype(np.uint8), cv2.COLOR_BGR2RGB))
        for filename, view, label in test.values:
            plt.imsave(f"./ANN_DATA/{angle}_DEGREES/TEST/{label}/{filename}.png",
                       cv2.cvtColor(view.astype(np.uint8), cv2.COLOR_BGR2RGB))

    def get_dataloaders(self, train_path, test_path):
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor()])
        train_dataset = datasets.ImageFolder(train_path, transform=transform)
        test_dataset = datasets.ImageFolder(test_path, transform=transform)
        train_dataset_indices = list(range(len(train_dataset)))
        np.random.seed(101)
        np.random.shuffle(train_dataset_indices)
        train_sampler = SubsetRandomSampler(train_dataset_indices[int(np.floor(self.TRAIN_VAL_SPLIT * len(train_dataset))):])
        val_sampler = SubsetRandomSampler(train_dataset_indices[:int(np.floor(self.TRAIN_VAL_SPLIT * len(train_dataset)))])
        return {"TRAIN": DataLoader(train_dataset, batch_size=self.BATCH_SIZE, sampler=train_sampler, shuffle=False, drop_last=True),
                "VAL": DataLoader(train_dataset, batch_size=1, sampler=val_sampler, shuffle=False, drop_last=True),
                "TEST": DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)}

    def multi_acc(self, y_pred, y_test):
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
        correct_pred = (y_pred_tags == y_test).float()
        return torch.round(correct_pred.sum() / len(correct_pred)*100)

    def train_model(self, save_path="MLP_MODELS", save_model=True):
        wandb.init(project='routenavigation-mlp')

        model = Model(self.INPUT_SIZE, self.HIDDEN_SIZES)
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=self.LEARNING_RATE)

        wandb.watch(model)

        accuracy_stats = {'train': [], 'val': []}
        loss_stats = {'train': [], 'val': []}

        print("Beginning training")
        for epoch in range(self.EPOCHS):
            model.train()
            train_epoch_loss, train_epoch_acc = 0, 0
            for X_train_batch, y_train_batch in self.trainloader:
                X_train_batch, y_train_batch = X_train_batch.to(self.device), y_train_batch.to(self.device)
                optimizer.zero_grad()

                y_train_pred = model(X_train_batch)

                train_loss = criterion(y_train_pred, y_train_batch)
                train_acc = self.multi_acc(y_train_pred, y_train_batch)

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()
            with torch.no_grad():
                model.eval()
                val_epoch_loss, val_epoch_acc = 0, 0
                for X_val_batch, y_val_batch in self.valloader:
                    X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)

                    y_val_pred = model(X_val_batch)

                    val_loss = criterion(y_val_pred, y_val_batch)
                    val_acc = self.multi_acc(y_val_pred, y_val_batch)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()

            loss_stats['train'].append(train_epoch_loss / len(self.trainloader))
            loss_stats['val'].append(val_epoch_loss / len(self.valloader))
            accuracy_stats['train'].append(train_epoch_acc / len(self.trainloader))
            accuracy_stats['val'].append(val_epoch_acc / len(self.valloader))

            print(f"Epoch {(epoch+1)+0:02}: | Train Loss: {loss_stats['train'][-1]:.5f} | Val Loss: {loss_stats['val'][-1]:.5f} | "
                  f"Train Acc: {accuracy_stats['train'][-1]:.3f} | Val Acc: {accuracy_stats['val'][-1]:.3f}")
            wandb.log({'Train Loss': loss_stats['train'][-1], 'Val Loss': loss_stats['val'][-1],
                       'Train Acc': accuracy_stats['train'][-1], 'Val Acc': accuracy_stats['val'][-1]})
        print("Finished Training")
        if save_model:
            torch.save(model.state_dict(), f"{save_path}/{wandb.run.name}.pth")

    def test_model(self, model_path):
        model = Model(self.INPUT_SIZE, self.HIDDEN_SIZES)
        model.to(self.device)
        model.load_state_dict(torch.load(model_path))
        y_pred, y_ground_truth = [], []
        with torch.no_grad():
            for X_test_batch, y_test_batch in self.testloader:
                X_test_batch, y_test_batch = X_test_batch.to(self.device), y_test_batch.to(self.device)

                y_test_pred = model(X_test_batch)
                _, y_pred_tag = torch.max(y_test_pred, dim=1)

                y_pred.append(y_pred_tag.cpu().numpy())
                y_ground_truth.append(y_test_batch.cpu().numpy())
        print(classification_report(y_ground_truth, y_pred))

class Model(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZES):
        nn.Module.__init__(self)

        self.HIDDEN_SIZES = HIDDEN_SIZES

        # Model layers
        self.fc_input = nn.Linear(INPUT_SIZE, HIDDEN_SIZES[0])
        self.fc_hidden = nn.ModuleList([nn.Linear(i, j) for i, j in zip(self.HIDDEN_SIZES, self.HIDDEN_SIZES[1:])])
        self.fc_output = nn.Linear(HIDDEN_SIZES[-1], 2)
        self.dropout = nn.Dropout(0.5)

        self.activation = nn.ReLU()

    def forward(self, inputs):
        x = inputs.view(inputs.size(0), -1)
        # x = self.dropout(x)
        x = self.activation(self.fc_input(x))
        for layer in self.fc_hidden:
            x = self.activation(layer(x))
        return self.fc_output(x)

if __name__ == '__main__':
    mlp = MultiLayerPerceptron(route="ant1_route1", vis_deg=360, rot_deg=2,
                               train_path="ANN_DATA/60_DEGREES_DATA/TRAIN", test_path="ANN_DATA/60_DEGREES_DATA/TEST")
    # mlp.train_model(save_path= "MLP_MODELS/TRAINED_ON_60_DEGREES_DATA", save_model=True)
    mlp.test_model("MLP_MODELS/TRAINED_ON_60_DEGREES_DATA/bright-water-32.pth")