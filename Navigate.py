import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib.ticker as plticker
import cv2
from pyprobar import probar

class Navigate:

    def __init__(self, route, vis_deg, rot_deg, buffer):
        self.topdown_view = plt.imread("ant_world_image_databases/topdown_view.png")
        self.grid_path = "ant_world_image_databases/grid/"
        self.grid_data = pd.read_csv("ant_world_image_databases/grid/database_entries.csv", skipinitialspace = True)

        self.route_path = "ant_world_image_databases/routes/"+route+"/"
        self.route_data = pd.read_csv(self.route_path+"database_entries.csv", skipinitialspace = True)

        self.route = [[x / 10 for x in self.route_data['X [mm]'].tolist()], [y / 10 for y in self.route_data["Y [mm]"].tolist()]]
        self.start = [int(self.route_data['X [mm]'].iloc[0]), int(self.route_data["Y [mm]"].iloc[0])]
        self.goal = [int(self.route_data['X [mm]'].iloc[-1]), int(self.route_data["Y [mm]"].iloc[-1])]
        self.grid_bounds = [[int((math.floor(((min(self.route[0])-buffer)/10))*10)), int((math.floor(((min(self.route[1])-buffer)/10))*10))],
                            [int((math.ceil(((max(self.route[0])+buffer)/10))*10)), int((math.ceil(((max(self.route[1])+buffer)/10))*10))]]
        print(self.grid_bounds)
        print(self.topdown_view.shape)

        self.vis_deg = vis_deg
        self.rot_deg = rot_deg

    def database_analysis(self, spacing):
        delta_x = int(self.grid_bounds[1][0]-self.grid_bounds[0][0])
        delta_y = int(self.grid_bounds[1][1]-self.grid_bounds[0][1])

        print(int(delta_x/spacing))

        print(np.linspace(self.grid_bounds[0][0]/10, self.grid_bounds[1][0]/10, num=int(delta_x/spacing), endpoint=True, dtype=int))
        print(np.linspace(self.grid_bounds[0][1]/10, self.grid_bounds[1][1]/10, num=int(delta_y/spacing), endpoint=True, dtype=int))

        grid_view_familiarity = []
        for i in probar(np.linspace(self.grid_bounds[0][0]/10, self.grid_bounds[1][0]/10, num=int(delta_x/spacing), endpoint=True, dtype=int)):
            for j in np.linspace(self.grid_bounds[0][1]/10, self.grid_bounds[1][1]/10, num=int(delta_y/spacing), endpoint=True, dtype=int):
                filename = self.grid_data['Filename'].values[(self.grid_data['Grid X'] == i) & (self.grid_data['Grid Y'] == j)][0]
                grid_view = cv2.imread(self.grid_path + filename)
                grid_view = cv2.cvtColor(grid_view, cv2.COLOR_BGR2GRAY)
                grid_view_familiarity.append(self.most_familiar_bearing(grid_view))

        fig = plt.figure(figsize=((self.grid_bounds[1][0]-self.grid_bounds[0][0])/100, (self.grid_bounds[1][1]-self.grid_bounds[0][1])/100))
        #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        ax = fig.add_subplot()
        loc = plticker.MultipleLocator(base=spacing)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        ax.grid(which='major', axis='both', linestyle='-')

        img = ax.imshow(self.topdown_view)
        ax.set_xlim([self.grid_bounds[0][0], self.grid_bounds[1][0]])
        ax.set_ylim([self.grid_bounds[0][1], self.grid_bounds[1][1]])

        ax.plot(self.route[0], self.route[1], linewidth=2, color='r')

        X, Y = np.meshgrid(
            np.linspace(self.grid_bounds[0][0], self.grid_bounds[1][0], num=int(delta_x/spacing), endpoint=True, dtype=int),
            np.linspace(self.grid_bounds[0][1], self.grid_bounds[1][1], num=int(delta_y/spacing), endpoint=True, dtype=int))
        u = [math.cos(n) for n in grid_view_familiarity]
        v = [math.sin(n) for n in grid_view_familiarity]
        ax.quiver(X, Y, u, v, color='w', scale_units='xy', scale=(1/spacing)*2, width=0.01, headwidth=5)

        plt.xticks(rotation=90)
        plt.show()

    def most_familiar_bearing(self, curr_view):
        route_familiarity = []
        for filename in self.route_data['Filename'][:5]:
            route_view = cv2.imread(self.route_path + filename)
            route_view = cv2.cvtColor(route_view, cv2.COLOR_BGR2GRAY)

            view_familiarity = {}
            for i in np.linspace(0, self.vis_deg, num=self.rot_deg, endpoint=True, dtype=int):
                view_familiarity[i] = self.get_familiarity(curr_view, route_view, i)

            route_familiarity.append(view_familiarity)
        return min([min(dict, key=dict.get) for dict in route_familiarity])

    def get_familiarity(self, curr_view, route_view, i):
        rotated_view = np.roll(curr_view, int(curr_view.shape[1] * (i / 360)), axis=1)
        return np.square(np.subtract(route_view, rotated_view)).mean()

if __name__ == "__main__":
    nav = Navigate(route="ant1_route8", vis_deg=360, rot_deg=4, buffer=0)
    nav.database_analysis(30)