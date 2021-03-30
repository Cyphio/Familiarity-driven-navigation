from PIL import Image
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
        self.HIDDEN_SIZES = [(360, 360)]
        self.TRAIN_VAL_SPLIT = 0.2
        self.EPOCHS = 50
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.005

        # Preprocess transforms
        self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

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
        train_dataset = datasets.ImageFolder(train_path, transform=self.transform)
        test_dataset = datasets.ImageFolder(test_path, transform=self.transform)
        train_dataset_indices = list(range(len(train_dataset)))
        np.random.seed(101)
        np.random.shuffle(train_dataset_indices)
        train_sampler = SubsetRandomSampler(train_dataset_indices[int(np.floor(self.TRAIN_VAL_SPLIT * len(train_dataset))):])
        val_sampler = SubsetRandomSampler(train_dataset_indices[:int(np.floor(self.TRAIN_VAL_SPLIT * len(train_dataset)))])
        return {"TRAIN": DataLoader(train_dataset, batch_size=self.BATCH_SIZE, sampler=train_sampler, shuffle=False, drop_last=True),
                "VAL": DataLoader(train_dataset, batch_size=1, sampler=val_sampler, shuffle=False),
                "TEST": DataLoader(test_dataset, batch_size=1, shuffle=False)}

    def multi_acc(self, y_pred, y_test):
        _, y_pred_tags = torch.max(torch.log_softmax(y_pred, dim=1), dim=1)
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

    def load_model(self, model_path):
        model = Model(self.INPUT_SIZE, self.HIDDEN_SIZES)
        model.to(self.device)
        model.load_state_dict(torch.load(model_path))
        self.model = model

    def test_model(self, model):
        y_pred, y_ground_truth = [], []
        with torch.no_grad():
            for X_test_batch, y_test_batch in self.testloader:
                X_test_batch, y_test_batch = X_test_batch.to(self.device), y_test_batch.to(self.device)

                y_test_pred = model(X_test_batch)
                _, y_pred_tag = torch.max(y_test_pred, dim=1)

                y_pred.append(y_pred_tag.cpu().numpy())
                y_ground_truth.append(y_test_batch.cpu().numpy())
        print(classification_report(y_ground_truth, y_pred))

    def get_route_rFF(self, view, view_heading=0):
        view_preprocessed = self.preprocess(view)
        rFF = {}
        for i in np.arange(0, self.vis_deg, step=self.rot_deg, dtype=int):
            view = self.rotate(view_preprocessed, i)
            tensor = self.transform(Image.fromarray(view)).float().to(self.device).view(1, self.INPUT_SIZE)
            pos_tag_val = torch.index_select(torch.log_softmax(self.model(tensor), dim=1), dim=1, index=torch.tensor([1]).to(self.device))
            # pos_tag_val = torch.index_select(self.model(tensor), dim=1, index=torch.tensor([1]).to(self.device))
            rFF[(i + view_heading) % self.vis_deg] = pos_tag_val.item()
        return rFF

    # Need to implement this properly - placeholder
    def get_view_rFF(self, view_1, view_2, view_1_heading=0):
        view_preprocessed = self.preprocess(view_1)
        rFF = {}
        for i in np.arange(0, self.vis_deg, step=self.rot_deg, dtype=int):
            view = self.rotate(view_preprocessed, i)
            tensor = self.transform(Image.fromarray(view)).float().to(self.device).view(1, self.INPUT_SIZE)
            pos_tag_val = torch.index_select(torch.log_softmax(self.model(tensor), dim=1), dim=1,
                                             index=torch.tensor([1]).to(self.device))
            # pos_tag_val = torch.index_select(self.model(tensor), dim=1, index=torch.tensor([1]).to(self.device))
            rFF[(i + view_1_heading) % self.vis_deg] = pos_tag_val.item()
        return rFF

    # Get the most familiar heading given an rIDF for a view
    def get_most_familiar_heading(self, rFF):
        return max(rFF, key=rFF.get)

    # Calculates the signal strength of an rFF
    def get_signal_strength(self, rFF):
        return max(rFF.values()) / np.array(list(rFF.values())).mean()

    # Need to implement this properly - placeholder
    def get_matched_route_view_idx(self, view, view_heading=0):
        x = {}
        for i, route_view in enumerate(self.route_views):
            view_tensor = self.transform(Image.fromarray(view)).float().to(self.device).view(1, self.INPUT_SIZE)
            route_view_tensor = self.transform(Image.fromarray(route_view)).float().to(self.device).view(1, self.INPUT_SIZE)
            x[(i + view_heading) % self.vis_deg] = torch.cdist(torch.log_softmax(self.model(route_view_tensor), dim=1),
                                                               torch.log_softmax(self.model(view_tensor), dim=1), p=2)
        return min(x, key=x.get)



class Model(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZES):
        nn.Module.__init__(self)

        # Model layers
        self.fc_input = nn.Linear(INPUT_SIZE, HIDDEN_SIZES[0][0])
        self.fc_hidden = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in HIDDEN_SIZES])
        self.fc_output = nn.Linear(HIDDEN_SIZES[-1][1], 2)

        self.activation = nn.ReLU()

    def forward(self, inputs):
        x = inputs.view(inputs.size(0), -1)
        x = self.activation(self.fc_input(x))
        for layer in self.fc_hidden:
            x = self.activation(layer(x))
        return self.fc_output(x)

if __name__ == '__main__':
    mlp = MultiLayerPerceptron(route="ant1_route1", vis_deg=360, rot_deg=2,
                               train_path="ANN_DATA/60_DEGREES_DATA/TRAIN", test_path="ANN_DATA/60_DEGREES_DATA/TEST")
    # mlp.train_model(save_path= "MLP_MODELS/TRAINED_ON_60_DEGREES_DATA", save_model=True)
    mlp.load_model("MLP_MODELS/TRAINED_ON_60_DEGREES_DATA/bright-water-32.pth")

    # mlp.test_model(model)

    # Database analysis
    mlp.database_analysis(spacing=20, save_data=False)
    # mlp.database_analysis(spacing=10, bounds=[[490, 370], [550, 460]], save_data=True)
    # mlp.database_analysis(spacing=20, corridor=30, save_data=False)

    # Off-route view analysis
    # grid_view = mlp.grid_views.get((500, 500))
    # mlp.view_analysis(view_1=grid_view, view_2=grid_view, save_data=False)

    # On-route view analysis
    # idx = 0
    # route_view = mlp.route_views[idx]
    # route_heading = mlp.route_headings[idx]
    # pm.view_analysis(view_1=route_view, view_2=route_view, view_1_heading=route_heading, save_data=False)
    # rFF = mlp.get_route_rFF(view=route_view, view_heading=route_heading)
    # mlp.rFF_plot(rFF=rFF,title="rFF", ylim=None, save_data=False)
    # print(mlp.get_most_familiar_heading(rFF))