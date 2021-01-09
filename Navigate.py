import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

class Navigate:

    def __init__(self, route, vis_deg, rot_deg):
        self.topdown_view = plt.imread("ant_world_image_databases/topdown_view.png")

        self.grid_path = "ant_world_image_databases/grid/"
        self.grid_data = pd.read_csv("ant_world_image_databases/grid/database_entries.csv", skipinitialspace = True)

        self.route_path = "ant_world_image_databases/routes/"+route+"/"
        self.route_data = pd.read_csv(self.route_path+"database_entries.csv", skipinitialspace = True)

        self.current = [int(self.route_data['X [mm]'].iloc[0]), int(self.route_data["Y [mm]"].iloc[0]), int(self.route_data["Z [mm]"].iloc[0])]
        self.goal = [int(self.route_data['X [mm]'].iloc[-1]), int(self.route_data["Y [mm]"].iloc[-1]), int(self.route_data["Z [mm]"].iloc[-1])]

        self.vis_deg = vis_deg
        self.rot_deg = rot_deg

    def database_analysis(self, x, y):
        grid_view_familiarity = []

        for i in np.arange(0, 100, math.floor(100/x)):
            for j in np.arange(0, 100, math.floor(100/y)):
                filename = self.grid_data['Filename'].values[(self.grid_data['Grid X'] == i) & (self.grid_data['Grid Y'] == j)][0]
                grid_view = cv2.imread(self.grid_path + filename)
                grid_view = cv2.cvtColor(grid_view, cv2.COLOR_BGR2GRAY)
                grid_view_familiarity.append(self.most_familiar_bearing(grid_view))

        x_coor, y_coor = np.meshgrid(np.arange(0, 1000, math.floor(1000/x)), np.arange(0, 1000, math.floor(1000/y)))
        u = [math.cos(n) for n in grid_view_familiarity]
        v = [math.sin(n) for n in grid_view_familiarity]

        plt.imshow(self.topdown_view)
        plt.quiver(x_coor, y_coor, u, v)
        plt.show()

    def most_familiar_bearing(self, curr_view):
        route_familiarity = []
        for filename in self.route_data['Filename'][:5]:
            route_view = cv2.imread(self.route_path + filename)
            route_view = cv2.cvtColor(route_view, cv2.COLOR_BGR2GRAY)

            view_familiarity = {}
            for i in np.arange(0, self.vis_deg, self.rot_deg):
                view_familiarity[i] = self.get_familiarity(curr_view, route_view, i)

            route_familiarity.append(view_familiarity)
        return min([min(dict, key=dict.get) for dict in route_familiarity])
        #print(familiarity_dict)
        #plt.plot(familiarity_dict.keys(), familiarity_dict.values())
        #plt.show()

    def get_familiarity(self, curr_view, route_view, i):
        rotated_view = np.roll(curr_view, int(curr_view.shape[1] * (i / 360)), axis=1)
        return np.square(np.subtract(route_view, rotated_view)).mean()

if __name__ == "__main__":
    nav = Navigate("ant1_route2", 360, 4)
    nav.database_analysis(10, 10)