from AnalysisToolkit import AnalysisToolkit
from pyprobar import probar
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class MultiLayerPerceptron(AnalysisToolkit, nn.Module):

    def __init__(self, route, vis_deg, rot_deg):
        AnalysisToolkit.__init__(self, route, vis_deg, rot_deg)
        self.model_name = 'MLP'

        self.X_train, self.X_test, self.y_train, self.y_test = self.data_loader(angle=60, test_size=0.33)

        # MLP hyper-parameters
        self.INPUT_SIZE = 360
        self.HIDDEN_SIZE = 360
        self.EPOCHS = 50
        self.BATCH_SIZE = 64
        self.LEARNING_RATE = 0.001
        self.MOMENTUM = 0.9

        # MLP loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=self.LEARNING_RATE, momentum=self.MOMENTUM)

        # MLP layers
        self.fc_input = nn.Linear(self.INPUT_SIZE, self.HIDDEN_SIZE)
        self.fc_hidden = nn.Linear(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE)
        self.fc_output = nn.Linear(self.HIDDEN_LAYER_SIZE, 1)

        # MLP activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def data_loader(self, angle, test_size):
        print('Generating training data...')
        dp = []
        for filename in probar(self.route_filenames):
            view = self.preprocess(cv2.imread(self.route_path + filename))
            dp.append([view, 1])
            dp.append([self.rotate(view, angle), 0])
            dp.append([self.rotate(view, -angle), 0])
        df = pd.DataFrame(dp, columns=["VIEW", "LABEL"])
        # sns.countplot(x='LABEL', data=df)
        # plt.show()
        return train_test_split(df.iloc[:, 0:-1], df.iloc[:, -1], test_size=test_size, random_state=0)

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
    print(mlp.X_train['VIEW'][0].shape)
