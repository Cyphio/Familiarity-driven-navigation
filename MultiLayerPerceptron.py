from AnalysisToolkit import AnalysisToolkit
from pyprobar import probar
import cv2
import numpy as np
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

class MultiLayerPerceptron(AnalysisToolkit):

    def __init__(self, route, vis_deg, rot_deg):
        AnalysisToolkit.__init__(self, route, vis_deg, rot_deg)
        self.model_name = 'MLP'
        self.X_train, self.X_test, self.y_train, self.y_test = self.gen_train_data(angle=60, test_size=0.33)

    def gen_train_data(self, angle, test_size):
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
        return train_test_split(df.iloc[:, 0:-1], df.iloc[:, -1], test_size=test_size, random_state=69)

if __name__ == '__main__':
    mlp = MultiLayerPerceptron(route="ant1_route1", vis_deg=360, rot_deg=2)