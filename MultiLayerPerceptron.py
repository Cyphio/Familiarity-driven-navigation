from AnalysisToolkit import AnalysisToolkit
from pyprobar import probar
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

class MultiLayerPerceptron(AnalysisToolkit, nn.Module):

    def __init__(self, route, vis_deg, rot_deg):
        AnalysisToolkit.__init__(self, route, vis_deg, rot_deg)
        nn.Module.__init__(self)
        self.model_name = 'MLP'

        # MLP hyper-parameters
        self.INPUT_SIZE = 45
        self.HIDDEN_SIZE = 1536
        self.TEST_SIZE = 0.33
        self.EPOCHS = 50
        self.BATCH_SIZE = 64
        self.LEARNING_RATE = 0.001
        self.MOMENTUM = 0.9

        # Data generation
        # self.gen_data(angle=60)

        # Data loading
        dataloaders = self.get_dataloaders("ANN_DATA/60")
        self.X_train, self.y_train = next(iter(dataloaders['TRAIN']))
        self.X_test, self.y_test = next(iter(dataloaders['TEST']))

        # MLP layers
        self.fc_input = nn.Linear(self.INPUT_SIZE, self.HIDDEN_SIZE)
        self.fc_hidden = nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE)
        self.fc_output = nn.Linear(self.HIDDEN_SIZE, 1)

        # MLP activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # MLP loss and optimizer
        self.criterion = nn.BCELoss()
        params = list(self.fc_input.parameters()) + list(self.fc_hidden.parameters()) + list(self.fc_output.parameters())
        self.optimizer = optim.SGD(params=params, lr=self.LEARNING_RATE, momentum=self.MOMENTUM)

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

    def get_dataloaders(self, path):
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.ImageFolder(path, transform=transform)
        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=self.TEST_SIZE, random_state=0)
        dataset_dict = {"TRAIN": Subset(dataset, train_idx), "TEST": Subset(dataset, test_idx)}
        return {tag: torch.utils.data.DataLoader(dataset_dict[tag], batch_size=self.BATCH_SIZE, shuffle=True)
                for tag in ["TRAIN", "TEST"]}

    def forward(self, input):
        fc_input = self.fc_input(input)
        relu = self.relu(fc_input)
        fc_hidden = self.fc_hidden(relu)
        relu = self.relu(fc_hidden)
        fc_output = self.fc_output(relu)
        return self.sigmoid(fc_output)

    def train_model(self):
        self.eval()
        y_pred = self.forward(self.X_test)
        before_train = self.criterion(y_pred.squeeze(), self.y_test)
        print(f"Test loss before training: {before_train.item()}")

        self.train()
        for epoch in range(self.EPOCHS):
            self.optimizer.zero_grad()
            y_pred = self.forward(self.X_train)
            loss = self.criterion(y_pred.squeeze(), self.y_train)
            print(f"Epoch: {epoch}, Loss: {loss}" )
            loss.backward()
            self.optimizer.step()

        self.eval()
        y_pred = self.forward(self.X_test)
        after_train = self.criterion(y_pred.squeeze(), self.y_test)
        print(f"Test loss after training: {after_train.item()}")

if __name__ == '__main__':
    mlp = MultiLayerPerceptron(route="ant1_route1", vis_deg=360, rot_deg=2)
    mlp.train_model()
