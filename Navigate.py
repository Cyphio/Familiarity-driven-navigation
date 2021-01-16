import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import cv2
from pyprobar import probar

class Navigate:

    def __init__(self, route, vis_deg, rot_deg, grid_buffer):
        self.topdown_view = plt.imread("ant_world_image_databases/topdown_view.png")
        self.grid_path = "ant_world_image_databases/grid/"
        self.grid_data = pd.read_csv("ant_world_image_databases/grid/database_entries.csv", skipinitialspace = True)

        self.route_path = "ant_world_image_databases/routes/"+route+"/"
        self.route_data = pd.read_csv(self.route_path+"database_entries.csv", skipinitialspace = True)

        self.route = [[x / 10 for x in self.route_data['X [mm]'].tolist()], [y / 10 for y in self.route_data["Y [mm]"].tolist()]]
        self.start = [int(self.route_data['X [mm]'].iloc[0]), int(self.route_data["Y [mm]"].iloc[0])]
        self.goal = [int(self.route_data['X [mm]'].iloc[-1]), int(self.route_data["Y [mm]"].iloc[-1])]
        self.grid_bounds = [[int((math.floor(((min(self.route[0]) - grid_buffer) / 10)) * 10)), int((math.floor(((min(self.route[1]) - grid_buffer) / 10)) * 10))],
                            [int((math.ceil(((max(self.route[0]) + grid_buffer) / 10)) * 10)), int((math.ceil(((max(self.route[1]) + grid_buffer) / 10)) * 10))]]

        self.vis_deg = vis_deg
        self.rot_deg = rot_deg

    def database_analysis(self, spacing):
        x_ticks = np.arange(self.grid_bounds[0][0], self.grid_bounds[1][0]+1, step=spacing, dtype=int)
        y_ticks = np.arange(self.grid_bounds[0][1], self.grid_bounds[1][1]+1, step=spacing, dtype=int)

        grid_view_familiarity = []
        for x in probar(x_ticks):
            for y in y_ticks:
                curr_view_path = self.grid_data['Filename'].values[(self.grid_data['Grid X'] == x/10) & (self.grid_data['Grid Y'] == y/10)][0]
                grid_view_familiarity.append(self.most_familiar_bearing(curr_view_path))

        fig = plt.figure(figsize=((self.grid_bounds[1][0]-self.grid_bounds[0][0])/100, (self.grid_bounds[1][1]-self.grid_bounds[0][1])/100))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        ax = fig.add_subplot()
        ax.xaxis.set_major_locator(plticker.FixedLocator(x_ticks))
        ax.yaxis.set_major_locator(plticker.FixedLocator(y_ticks))
        ax.grid(which='major', axis='both', linestyle=':')

        img = ax.imshow(self.topdown_view)
        ax.plot(self.route[0], self.route[1], linewidth=2, color='r')

        X, Y = np.meshgrid(x_ticks, y_ticks)
        u = [math.sin(math.radians(n)) for n in grid_view_familiarity]
        v = [math.cos(math.radians(n)) for n in grid_view_familiarity]
        ax.quiver(X, Y, u, v, color='w', scale_units='xy', scale=(1/spacing)*2, width=0.01, headwidth=5)

        ax.set_xlim([self.grid_bounds[0][0], self.grid_bounds[1][0]])
        ax.set_ylim([self.grid_bounds[0][1], self.grid_bounds[1][1]])

        plt.show()

    def most_familiar_bearing(self, curr_view_path):
        curr_view = cv2.imread(self.grid_path + curr_view_path)
        curr_view = cv2.cvtColor(curr_view, cv2.COLOR_BGR2GRAY)
        curr_view = cv2.resize(curr_view, (90, 17))
        route_familiarity = []
        for filename in self.route_data['Filename'][::100]:
            route_view = cv2.imread(self.route_path + filename)
            route_view = cv2.cvtColor(route_view, cv2.COLOR_BGR2GRAY)
            route_view = cv2.resize(route_view, (90, 17))
            view_familiarity = {}
            for i in np.arange(0, self.vis_deg, step=self.rot_deg, dtype=int):
                view_familiarity[i] = self.get_familiarity(curr_view, route_view, i)
            route_familiarity.append(view_familiarity)
        return max([max(x, key=x.get) for x in route_familiarity])

    # def most_familiar_bearing(self, curr_view):
    #     view_familiarity = {}
    #     for i in np.arange(0, self.vis_deg, step=self.rot_deg, dtype=int):
    #         route_familiarity = []
    #         for filename in self.route_data['Filename'][:5]:
    #             route_view = cv2.imread(self.route_path + filename)
    #             route_view = cv2.cvtColor(route_view, cv2.COLOR_BGR2GRAY)
    #             route_view = cv2.resize(route_view, (90, 17))
    #             route_familiarity.append(self.get_familiarity(curr_view, route_view, i))
    #         view_familiarity[i] = np.average(route_familiarity)
    #     return max(view_familiarity, key=view_familiarity.get)


    def get_familiarity(self, curr_view, route_view, i):
        rotated_view = np.roll(curr_view, int(curr_view.shape[1] * (i / 360)), axis=1)
        return -np.square(np.subtract(route_view, rotated_view)).mean()

if __name__ == "__main__":
    nav = Navigate(route="ant1_route8", vis_deg=360, rot_deg=4, grid_buffer=0)
    nav.database_analysis(30)