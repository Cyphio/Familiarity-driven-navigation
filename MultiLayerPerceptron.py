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

class MultiLayerPerceptron(AnalysisToolkit):

    def __init__(self, route, vis_deg, rot_deg):
        AnalysisToolkit.__init__(self, route, vis_deg, rot_deg)
        self.model_name = 'MLP'

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"RUNNING ON: {self.device}")

        # MLP hyper-parameters
        self.INPUT_SIZE = 360
        self.HIDDEN_SIZE = 45
        self.TEST_SIZE = 0.33
        self.EPOCHS = 50
        self.BATCH_SIZE = 64
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

    def get_dataloaders(self, path):
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor()])
        dataset = datasets.ImageFolder(path, transform=transform)
        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=self.TEST_SIZE, random_state=0)
        dataset_dict = {"TRAIN": Subset(dataset, train_idx), "TEST": Subset(dataset, test_idx)}
        return {tag: torch.utils.data.DataLoader(dataset_dict[tag], batch_size=self.BATCH_SIZE, shuffle=True)
                for tag in ["TRAIN", "TEST"]}

    def train_model(self):
        model = Model(self.INPUT_SIZE, self.HIDDEN_SIZE)
        model.to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.SGD(params=model.parameters(), lr=self.LEARNING_RATE, momentum=self.MOMENTUM)

        # self.eval()
        # y_pred = self.forward(self.X_test)
        # before_train = self.criterion(y_pred.squeeze(), self.y_test)
        # print(f"Test loss before training: {before_train.item()}")

        # for epoch in range(self.EPOCHS):
        #     for X_train, y_train in self.train_generator:
        #         self.optimizer.zero_grad()
        #         y_pred = self.forward(X_train)
        #         loss = self.criterion(y_pred.squeeze(), y_train)
        #         print(f"Epoch: {epoch}, Loss: {loss}" )
        #         loss.backward()
        #         self.optimizer.step()
        for epoch in range(self.EPOCHS):  # loop over the dataset multiple times
            running_loss = 0.0
            for idx, (X_train, y_train) in enumerate(self.trainloader, 0):
                X_train, y_train = X_train.to(self.device), y_train.to(self.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                y_pred = model(X_train.view(self.BATCH_SIZE, -1))
                print(f"Y_PRED TYPE: {y_pred.squeeze()}\nY_TRAIN TYPE: {y_train}")
                loss = criterion(y_pred.squeeze(), y_train)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        print('Finished Training')

        # self.eval()
        # y_pred = self.forward(self.X_test)
        # after_train = self.criterion(y_pred.squeeze(), self.y_test)
        # print(f"Test loss after training: {after_train.item()}")

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
    mlp.train_model()
