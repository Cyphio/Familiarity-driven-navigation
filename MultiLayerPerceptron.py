from sklearn.metrics import classification_report

from AnalysisToolkit import AnalysisToolkit
from pyprobar import probar
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

class MultiLayerPerceptron(AnalysisToolkit):

    def __init__(self, route, vis_deg, rot_deg):
        AnalysisToolkit.__init__(self, route, vis_deg, rot_deg)
        self.model_name = 'MLP'

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"RUNNING ON: {self.device}")

        # MLP hyper-parameters
        self.INPUT_SIZE = 360
        self.HIDDEN_SIZE = 360
        self.TEST_SIZE = 0.33
        self.EPOCHS = 5
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.001
        self.MOMENTUM = 0.9

        # Data generation
        # self.gen_data(angle=60)

        # Data loading
        dataloaders = self.get_dataloaders("ANN_DATA/60")
        self.trainloader = dataloaders['TRAIN']
        self.testloader = dataloaders['TEST']

    def gen_data(self, angle):
        print('Generating data...')
        dp = []
        for filename in probar(self.route_filenames):
            view = self.preprocess(cv2.imread(self.route_path + filename))
            dp.append([f'{filename.strip(".png")}_0', view, 1])
            dp.append([f'{filename.strip(".png")}_{angle}', self.rotate(view, angle), 0])
            dp.append([f'{filename.strip(".png")}_-{angle}', self.rotate(view, -angle), 0])
        df = pd.DataFrame(dp, columns=['FILENAME', 'VIEW', 'LABEL'])
        for filename, view, label in df.values:
            plt.imsave(f"./ANN_DATA/{angle}/{label}/{filename}.png", cv2.cvtColor(view.astype(np.uint8), cv2.COLOR_BGR2RGB))

    def binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        return torch.round(acc * 100)

    def get_dataloaders(self, path):
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor()])
        dataset = datasets.ImageFolder(path, transform=transform)
        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=self.TEST_SIZE, random_state=0)
        dataset_dict = {"TRAIN": Subset(dataset, train_idx), "TEST": Subset(dataset, test_idx)}
        return {"TRAIN": torch.utils.data.DataLoader(dataset_dict["TRAIN"], batch_size=self.BATCH_SIZE, shuffle=True),
                "TEST": torch.utils.data.DataLoader(dataset_dict["TEST"], batch_size=1)}

    def train_model(self, val_flags=False, train_flags=False):
        model = Model(self.INPUT_SIZE, self.HIDDEN_SIZE)
        model.to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.SGD(params=model.parameters(), lr=self.LEARNING_RATE, momentum=self.MOMENTUM)

        model.eval()
        running_y_pred, running_y_test = [], []
        with torch.no_grad():
            for X_test, y_test in self.testloader:
                X_test, y_test = X_test.to(self.device), y_test.to(self.device)
                y_pred = model(X_test.view(1, -1))
                y_pred_tag = torch.round(y_pred)
                running_y_pred.append(y_pred_tag.cpu().numpy())
                running_y_test.append(y_test.float())
        running_y_pred = [a.squeeze().tolist() for a in running_y_pred]
        running_y_test = [a.squeeze().tolist() for a in running_y_test]
        if val_flags:
            print(running_y_pred)
            print(running_y_test)
            print(f"VALIDATION BEFORE TRAINING:\n{classification_report(running_y_test, running_y_pred)}")

        model.train()
        for epoch in range(self.EPOCHS):
            running_loss, running_acc = 0.0, 0.0
            for batch_idx, (X_train, y_train) in enumerate(self.trainloader):
                X_train, y_train = X_train.to(self.device), y_train.to(self.device)
                optimizer.zero_grad()
                y_pred = model(X_train.view(self.BATCH_SIZE, -1))
                loss = criterion(y_pred.squeeze(), y_train.float())
                acc = self.binary_acc(y_pred.squeeze(), y_train.float())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_acc += acc.item()
                if train_flags:
                    print(f'Epoch {(epoch+1) + 0:03}: | Loss: {running_loss / len(self.trainloader):.5f} '
                          f'| Acc: {running_acc / len(self.trainloader):.3f}')
        print('Finished Training')

        model.eval()
        running_y_pred, running_y_test = [], []
        with torch.no_grad():
            for X_test, y_test in self.testloader:
                X_test, y_test = X_test.to(self.device), y_test.to(self.device)
                y_pred = model(X_test.view(1, -1))
                y_pred_tag = torch.round(y_pred)
                running_y_pred.append(y_pred_tag.cpu().numpy())
                running_y_test.append(y_test.float())
        running_y_pred = [a.squeeze().tolist() for a in running_y_pred]
        running_y_test = [a.squeeze().tolist() for a in running_y_test]
        if val_flags:
            print(running_y_pred)
            print(f"VALIDATION AFTER TRAINING:\n{classification_report(running_y_test, running_y_pred)}")

class Model(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE):
        nn.Module.__init__(self)

        # Model layers
        self.fc_input = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc_hidden = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc_output = nn.Linear(HIDDEN_SIZE, 1)

        # Model activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        fc_input = self.fc_input(input)
        relu = self.relu(fc_input)
        fc_hidden = self.fc_hidden(relu)
        relu = self.relu(fc_hidden)
        fc_output = self.fc_output(relu)
        return self.sigmoid(fc_output)

if __name__ == '__main__':
    mlp = MultiLayerPerceptron(route="ant1_route1", vis_deg=360, rot_deg=2)
    mlp.train_model(val_flags=True, train_flags=True)