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
import os
import random

class ANN(AnalysisToolkit):
    def __init__(self, route, vis_deg, rot_deg, ANN_flag, train_path, test_path):
        AnalysisToolkit.__init__(self, route, vis_deg, rot_deg)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"RUNNING ON: {self.device}")

        # Seeds
        np.random.seed(101)
        random.seed(101)

        # ANN hyper-parameters
        self.INPUT_SIZE = 360
        self.TRAIN_VAL_SPLIT = 0.4
        self.EPOCHS = 50
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.001

        if ANN_flag == 'MLP':
            self.model_name = 'MLP'
            self.model_class = MLPModel(self.INPUT_SIZE)
        if ANN_flag == 'RBFNN':
            self.model_name = 'RBFNN'
            self.model_class = RBFNNModel(self.INPUT_SIZE)

        # Preprocess transforms
        self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

        # Data loading
        dataloaders = self.get_dataloaders(train_path, test_path)
        self.trainloader = dataloaders['TRAIN']
        self.valloader = dataloaders['VAL']
        self.testloader = dataloaders['TEST']

        # Arbitrary initialization
        self.model = None
        self.route_view_layton_spaces = None

    def gen_data(self, angle, is_random=False, split=0.2):
        print('Generating data...')
        dp = []
        for on_route_filename in probar(self.route_filenames):
            if is_random:
                off_route_filename = self.grid_filenames[(random.choice(self.grid_X)), random.choice(self.grid_Y)]
                on_route_view = self.preprocess(cv2.imread(self.route_path + on_route_filename))
                off_route_view = self.preprocess(cv2.imread(self.grid_path + off_route_filename))
                dp.append([f'{on_route_filename.strip(".png")}_0', on_route_view, 1])
                angle = random.randint(0, 360)
                dp.append([f'{off_route_filename.strip(".png")}_{angle}', self.rotate(off_route_view, angle), 0])
            else:
                view = self.preprocess(cv2.imread(self.route_path + on_route_filename))
                dp.append([f'{on_route_filename.strip(".png")}_0', view, 1])
                dp.append([f'{on_route_filename.strip(".png")}_{angle}', self.rotate(view, angle), 0])
                dp.append([f'{on_route_filename.strip(".png")}_-{angle}', self.rotate(view, -angle), 0])
        df = pd.DataFrame(dp, columns=['FILENAME', 'VIEW', 'LABEL'])
        train, test = train_test_split(df, test_size=split)
        for dataset in ["TRAIN", "TEST"]:
            for label in ["0", "1"]:
                if is_random:
                    path = f"ANN_DATA/{self.route_name}/RAND_DATA/{dataset}/{label}"
                else:
                    path = f"./ANN_DATA/{self.route_name}/{angle}_DEGREES_DATA/{dataset}/{label}"
                if not os.path.isdir(path):
                    os.makedirs(path)
        for filename, view, label in train.values:
            if is_random:
                plt.imsave(f"./ANN_DATA/{self.route_name}/RAND_DATA/TRAIN/{label}/{filename}.png",
                           cv2.cvtColor(view.astype(np.uint8), cv2.COLOR_BGR2RGB))
            else:
                plt.imsave(f"./ANN_DATA/{self.route_name}/{angle}_DEGREES_DATA/TRAIN/{label}/{filename}.png",
                           cv2.cvtColor(view.astype(np.uint8), cv2.COLOR_BGR2RGB))
        for filename, view, label in test.values:
            if is_random:
                plt.imsave(f"./ANN_DATA/{self.route_name}/RAND_DATA/TEST/{label}/{filename}.png",
                           cv2.cvtColor(view.astype(np.uint8), cv2.COLOR_BGR2RGB))
            else:
                plt.imsave(f"./ANN_DATA/{self.route_name}/{angle}_DEGREES_DATA/TEST/{label}/{filename}.png",
                           cv2.cvtColor(view.astype(np.uint8), cv2.COLOR_BGR2RGB))

    def get_dataloaders(self, train_path, test_path):
        train_dataset = datasets.ImageFolder(train_path, transform=self.transform)
        test_dataset = datasets.ImageFolder(test_path, transform=self.transform)
        train_dataset_indices = list(range(len(train_dataset)))
        np.random.shuffle(train_dataset_indices)
        train_sampler = SubsetRandomSampler(train_dataset_indices[int(np.floor(self.TRAIN_VAL_SPLIT * len(train_dataset))):])
        val_sampler = SubsetRandomSampler(train_dataset_indices[:int(np.floor(self.TRAIN_VAL_SPLIT * len(train_dataset)))])
        return {"TRAIN": DataLoader(train_dataset, batch_size=self.BATCH_SIZE, sampler=train_sampler, drop_last=True),
                "VAL": DataLoader(train_dataset, batch_size=self.BATCH_SIZE, sampler=val_sampler, drop_last=True),
                "TEST": DataLoader(test_dataset, batch_size=1)}

    def multi_acc(self, y_pred, y_test):
        _, y_pred_tags = torch.max(torch.log_softmax(y_pred, dim=1), dim=1)
        correct_pred = (y_pred_tags == y_test).float()
        return torch.round(correct_pred.sum() / len(correct_pred)*100)

    def train_model(self, save_path="ANN_MODELS", save_model=True):
        model = self.model_class
        model.to(self.device)

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=self.LEARNING_RATE)

        if save_model:
            if self.model_name == "MLP":
                wandb.init(project='routenavigation-mlp')
            if self.model_name == 'RBFNN':
                wandb.init(project='routenavigation-rbfnn')
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
                # print(y_train_pred)

                train_loss = loss_func(y_train_pred, y_train_batch)
                train_acc = self.multi_acc(y_train_pred, y_train_batch)

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()

                if save_model and epoch % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss}, f"{save_path}/{wandb.run.name}_epoch{epoch}.pth")
            with torch.no_grad():
                model.eval()
                val_epoch_loss, val_epoch_acc = 0, 0
                for X_val_batch, y_val_batch in self.valloader:
                    X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)

                    y_val_pred = model(X_val_batch)

                    val_loss = loss_func(y_val_pred, y_val_batch)
                    val_acc = self.multi_acc(y_val_pred, y_val_batch)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()

            loss_stats['train'].append(train_epoch_loss / len(self.trainloader))
            loss_stats['val'].append(val_epoch_loss / len(self.valloader))
            accuracy_stats['train'].append(train_epoch_acc / len(self.trainloader))
            accuracy_stats['val'].append(val_epoch_acc / len(self.valloader))

            print(f"Epoch {(epoch+1)+0:02}: | Train Loss: {loss_stats['train'][-1]:.5f} | Val Loss: {loss_stats['val'][-1]:.5f} | "
                  f"Train Acc: {accuracy_stats['train'][-1]:.3f} | Val Acc: {accuracy_stats['val'][-1]:.3f}")
            if save_model:
                wandb.log({'Train Loss': loss_stats['train'][-1], 'Val Loss': loss_stats['val'][-1],
                           'Train Acc': accuracy_stats['train'][-1], 'Val Acc': accuracy_stats['val'][-1]})
        print("Finished Training")
        if save_model:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            torch.save({
                'epoch': self.EPOCHS,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_stats['train'][-1]}, f"{save_path}/{wandb.run.name}_epoch{self.EPOCHS}.pth")

    def load_model(self, model_path):
        print("Loading model...")
        model = self.model_class
        model.to(self.device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.model = model
        print("Calculating training route view latent spaces...")
        self.route_view_layton_spaces = []
        for filename in probar(self.route_filenames):
            self.model(self.transform(Image.fromarray(self.preprocess(cv2.imread(self.route_path + filename)))).float().to(self.device).view(1, self.INPUT_SIZE))
            self.route_view_layton_spaces.append(self.model.get_latent_space())

    def test_model(self):
        y_pred, y_ground_truth = [], []
        with torch.no_grad():
            for X_test_batch, y_test_batch in self.testloader:
                X_test_batch, y_test_batch = X_test_batch.to(self.device), y_test_batch.to(self.device)

                y_test_pred = self.model(X_test_batch)
                # print(y_test_pred)
                _, y_pred_tag = torch.max(y_test_pred, dim=1)

                y_pred.append(y_pred_tag.cpu().numpy())
                y_ground_truth.append(y_test_batch.cpu().numpy())
        print(classification_report(y_ground_truth, y_pred, zero_division=0))

    # Rotational Familiarity Function of a view against an ANN route representation
    def get_route_rFF(self, view, view_heading=0):
        view_preprocessed = self.preprocess(view)
        rFF = {}
        for i in np.arange(0, self.vis_deg, step=self.rot_deg, dtype=int):
            view = self.rotate(view_preprocessed, i)
            tensor = self.transform(Image.fromarray(view)).float().to(self.device).view(1, self.INPUT_SIZE)
            pos_tag_val = torch.index_select(torch.log_softmax(self.model(tensor), dim=1), dim=1,
                                             index=torch.tensor([1]).to(self.device))
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
            rFF[(i + view_1_heading) % self.vis_deg] = pos_tag_val.item()
        return rFF

    # Get the most familiar heading given an rFF for a view
    def get_most_familiar_heading(self, rFF):
        # return max(rFF, key=rFF.get)
        # x = [k for k, v in rFF.items() if 0.8*max(rFF.values()) <= v <= 1.2*max(rFF.values())]
        x = [k for k, v in rFF.items() if self.is_within_prcnt(v, max(rFF.values()), 1.0)]
        return np.median(x).astype(np.int64)

    # Calculates the signal strength of an rFF
    def get_signal_strength(self, rFF):
        return max(rFF.values()) / np.array(list(rFF.values())).mean()

    # get the index of the best matching route view to a view
    def get_matched_route_view_idx(self, view, view_heading=0):
        view_preprocessed = self.preprocess(self.rotate(view, view_heading))
        view_tensor = self.transform(Image.fromarray(view_preprocessed)).float().to(self.device).view(1, self.INPUT_SIZE)
        self.model(view_tensor)
        view_latent_space = self.model.get_latent_space()
        x = {i: torch.cdist(self.route_view_layton_spaces[i], view_latent_space) for i in range(len(self.route_view_layton_spaces))}
        return min(x, key=x.get)

class MLPModel(nn.Module):
    def __init__(self, INPUT_SIZE):
        nn.Module.__init__(self)

        LAYER_WIDTHS = [INPUT_SIZE, 360, 2]

        self.linear_layers = nn.ModuleList()
        for i in range(len(LAYER_WIDTHS) - 1):
            self.linear_layers.append(nn.Linear(LAYER_WIDTHS[i], LAYER_WIDTHS[i + 1]))

        self.activation = nn.ReLU()
        self.latent_space = None

    def forward(self, inputs):
        x = inputs.view(inputs.size(0), -1)
        for i in range(len(self.linear_layers)-1):
            x = self.activation(self.linear_layers[i](x))
        self.latent_space = x
        return self.linear_layers[-1](x)

    def get_latent_space(self):
        return self.latent_space

class RBFNNModel(nn.Module):
    def __init__(self, INPUT_SIZE):
        nn.Module.__init__(self)

        LAYER_WIDTHS = [INPUT_SIZE, 180, 2]
        LAYER_CENTRES = [180, 1]
        BASIS_FUNC_FLAG = "GAUSSIAN"

        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i in range(len(LAYER_WIDTHS)-1):
            self.rbf_layers.append(RBF(LAYER_WIDTHS[i], LAYER_CENTRES[i], BASIS_FUNC_FLAG))
            self.linear_layers.append(nn.Linear(LAYER_CENTRES[i], LAYER_WIDTHS[i+1]))

        self.latent_space = None

    def forward(self, inputs):
        x = inputs.view(inputs.size(0), -1)
        for i in range(len(self.rbf_layers)-1):
            x = self.rbf_layers[i](x)
            x = self.linear_layers[i](x)
        self.latent_space = x
        x = self.rbf_layers[-1](x)
        return self.linear_layers[-1](x)

    def get_latent_space(self):
        return self.latent_space

class RBF(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE, BASIS_FUNC_FLAG):
        nn.Module.__init__(self)
        self.in_features = INPUT_SIZE
        self.out_features = OUTPUT_SIZE
        self.centres = nn.Parameter(torch.Tensor(OUTPUT_SIZE, INPUT_SIZE))
        self.log_sigmas = nn.Parameter(torch.Tensor(OUTPUT_SIZE))
        self.basis_func_flag = BASIS_FUNC_FLAG.upper()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, inputs):
        size = (inputs.size(0), self.out_features, self.in_features)
        x = inputs.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        # distances = torch.sqrt(torch.sum((x - c).pow(2))).unsqueeze(0)
        return self.basis_func(distances, self.basis_func_flag)

    def basis_func(self, alpha, BASIS_FUNC_FLAG):
        if BASIS_FUNC_FLAG == "GAUSSIAN":
            phi = torch.exp(-1 * alpha.pow(2))
            return phi
        if BASIS_FUNC_FLAG == "LINEAR":
            return alpha
        if BASIS_FUNC_FLAG == "QUADRATIC":
            phi = alpha.pow(2)
            return phi
        if BASIS_FUNC_FLAG == "INVERSE_QUADRATIC":
            phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
            return phi



if __name__ == '__main__':
    ANN_flag = "MLP"
    route_name = "ant1_route1"
    data_path = "90_DEGREES_DATA"
    model_name = "blooming-glitter-79_epoch50"

    ann = ANN(route=route_name, vis_deg=360, rot_deg=8, ANN_flag=ANN_flag,
              train_path=f"ANN_DATA/{route_name}/{data_path}/TRAIN",
              test_path=f"ANN_DATA/{route_name}/{data_path}/TEST")

    # ann.gen_data(angle=90, is_random=False, split=0.2)
    # ann.train_model(save_path=f"ANN_MODELS/{ANN_flag}/{route_name}/TRAINED_ON_{data_path}", save_model=True)

    # ann.load_model(f"ANN_MODELS/{ANN_flag}/{route_name}/TRAINED_ON_{data_path}/{model_name}.pth")
    # ann.test_model()


    # ann.scatter_confidence_against_error(spacing=20, corridor=30, save_data=True)


    # Database analysis
    # ann.database_analysis(spacing=20,
    #                       save_path=f"DATABASE_ANALYSIS/{ANN_flag}/{route_name}/TRAINED_ON_{data_path}", save_data=True)
    # ann.database_analysis(spacing=10, bounds=[[490, 370], [550, 460]],
    #                       save_path=f"DATABASE_ANALYSIS/{ANN_flag}/{route_name}/TRAINED_ON_{data_path}", save_data=True)
    # ann.database_analysis(spacing=20, corridor=30,
    #                       save_path=f"DATABASE_ANALYSIS/{ANN_flag}/{route_name}/TRAINED_ON_{data_path}", save_data=True)

    # csv_file_path = "DATABASE_ANALYSIS/MLP/ant1_route1/TRAINED_ON_90_DEGREES_DATA/tough-paper-74-ant1_route1.csv"
    # ann.show_database_analysis_plot(data_path=csv_file_path,
    #                                 spacing=20, locationality=True,
    #                                 save_path=f"DATABASE_ANALYSIS/{ANN_flag}/{route_name}/TRAINED_ON_{data_path}", save_data=True)





    # indexes = [f"neg examples rotated {deg}° off-route" for deg in [0, 10, 20, 45, 60, 90, 120, 180]]
    # indexes.append("random neg examples")
    # ann.error_boxplot(["DATABASE_ANALYSIS/MLP/ant1_route1/TRAINED_ON_0_DEGREES_DATA/22-4-2021_14-18-46_ant1_route1_140x740_20.csv",
    #                    "DATABASE_ANALYSIS/MLP/ant1_route1/TRAINED_ON_10_DEGREES_DATA/22-4-2021_14-19-36_ant1_route1_140x740_20.csv",
    #                    "DATABASE_ANALYSIS/MLP/ant1_route1/TRAINED_ON_20_DEGREES_DATA/22-4-2021_14-20-16_ant1_route1_140x740_20.csv",
    #                    "DATABASE_ANALYSIS/MLP/ant1_route1/TRAINED_ON_45_DEGREES_DATA/22-4-2021_14-20-59_ant1_route1_140x740_20.csv",
    #                    "DATABASE_ANALYSIS/MLP/ant1_route1/TRAINED_ON_60_DEGREES_DATA/22-4-2021_14-21-28_ant1_route1_140x740_20.csv",
    #                    "DATABASE_ANALYSIS/MLP/ant1_route1/TRAINED_ON_90_DEGREES_DATA/blooming-glitter-79_epoch50-ant1_route1.csv",
    #                    "DATABASE_ANALYSIS/MLP/ant1_route1/TRAINED_ON_120_DEGREES_DATA/22-4-2021_14-21-56_ant1_route1_140x740_20.csv",
    #                    "DATABASE_ANALYSIS/MLP/ant1_route1/TRAINED_ON_180_DEGREES_DATA/22-4-2021_14-22-34_ant1_route1_140x740_20.csv",
    #                    "DATABASE_ANALYSIS/MLP/ant1_route1/TRAINED_ON_RAND_DATA/22-4-2021_14-23-11_ant1_route1_140x740_20.csv"],
    #                   indexes, locationality=False, save_data=True)


    # indexes = [f"MLP, neg examples taken 90° off-route", "PM, 8° resolution"]
    # ann.error_boxplot(["DATABASE_ANALYSIS/MLP/ant1_route1/TRAINED_ON_90_DEGREES_DATA/blooming-glitter-79_epoch50-ant1_route1.csv",
    #                    "DATABASE_ANALYSIS/PERFECTMEMORY/ant1_route1/8_deg_px_res/PM-ant1_route1.csv"],
    #                   indexes, save_data=False)


    # Off-route view analysis
    # coors = (630, 590)
    # grid_name = ann.grid_filenames.get(coors)
    # grid_view = cv2.imread(ann.grid_path + grid_name)
    # mlp.view_analysis(view_1=grid_view, view_2=grid_view, save_data=False)
    # save_data = True
    # rff = ann.get_route_rFF(grid_view)
    # ybound = [-27, 0]
    # ann.rFF_plot(ann.normalize(rFF, min_=ybound[0], max_=ybound[1]), ylim=[0, 1],
    #              ybound=ybound, title=f"rFF of test view at ({coors[0]}, {coors[1]}) vs route memories at epoch 100",
    #              save_path=save_path, save_data=save_data)
    # min_ = min(rff.values())
    # max_ = max(rff.values())
    # ann.rFF_plot(ann.normalize(rff, min_, max_), ylim=[0, 1], ybound=[round(min_), round(max_)],
    #              title=f"MLP rFF of view at ({coors[0]}, {coors[1]}) vs route memories",
    #              save_data=save_data)

    # On-route view analysis
    # idx = 400
    # route_name = ann.route_filenames[idx]
    # route_heading = ann.route_headings[idx]
    # print(f"FILE: {route_name}, TRUE HEADING: {route_heading}")
    # route_view = cv2.imread(ann.route_path + route_name)
    # print(mlp.get_matched_route_view_idx(route_view))
    # mlp.view_analysis(view_1=route_view, view_2=route_view, view_1_heading=route_heading, save_data=False)

    # save_data = True
    # save_path = f"VIEW_ANALYSIS/RFF_OVER_EPOCH/{data_path}"
    # ybound = [-20, 0]
    # ann.rFF_plot(ann.normalize(ann.get_route_rFF(route_view, route_heading), min=ybound[0], max=ybound[1]), ylim=[0, 1],
    #              ybound=ybound, title=f"rFF of training view at ({ann.route_X[idx]}, {ann.route_Y[idx]}) vs route memories at epoch 70",
    #              save_path=save_path, save_data=save_data)

    # Off-route best matched view analysis
    # mlp.best_matched_view_analysis(view_x=610, view_y=810, save_data=False)




    # save_data = False
    # coor = [530, 270]
    # info_loss_ybound = [-20, -0]
    # save_path = f"VIEW_ANALYSIS/INFO_LOSS_TEST/({coor[0]}, {coor[1]})/{ANN_flag}/TRAINED_ON_{data_path}"

    # ann.gen_info_loss_data(coor=coor, ybound=info_loss_ybound, model_name=ANN_flag,
    #                        save_path=save_path, save_data=save_data)

    # x = "original"
    # image = cv2.imread(f"VIEW_ANALYSIS/INFO_LOSS_TEST/({coor[0]}, {coor[1]})/IMAGES/{x}.png")
    # ann.rFF_plot(ann.normalize(ann.get_route_rFF(image), min=info_loss_ybound[0], max=info_loss_ybound[1]), ylim=[0, 1],
    #              ybound=info_loss_ybound, title=f"{model_name} rFF of view ({x}) at ({coor[0]}, {coor[1]}) vs route memories",
    #              save_path=save_path, save_data=save_data)


    # image = ann.preprocess(cv2.imread("VIEW_ANALYSIS/INFO_LOSS_TEST/(540, 390)/IMAGES/missing ground information.png"))
    # plt.imshow(image, cmap=plt.cm.binary)
    # plt.axis('off')
    # plt.show()


    # save_data = True
    # coor = [510, 250]
    #
    # image = cv2.imread(ann.grid_path + ann.grid_filenames[(coor[0], coor[1])])
    #
    # rFF = ann.get_route_rFF(image)

    # ybound = [-20, 0]
    # ann.rFF_plot(ann.normalize(rFF, min=ybound[0], max=ybound[1]), ylim=[0, 1],
    #              ybound=ybound, title=f"{ann.model_name} rFF of view at ({coor[0]}, {coor[1]}) vs route memories",
    #              save_path="VIEW_ANALYSIS/BESTMATCH_VS_REALMATCH", save_data=save_data)

    # ann.rFF_plot(ann.normalize(rFF, min=min(rFF.values()), max=max(rFF.values())), ylim=[0, 1],
    #              ybound=[round(min(rFF.values())), round(max(rFF.values()))], title=f"{ann.model_name} rFF of view at ({coor[0]}, {coor[1]}) vs route memories",
    #              save_path="VIEW_ANALYSIS/BESTMATCH_VS_REALMATCH", save_data=save_data)