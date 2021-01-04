import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

class Navigate:

    def __init__(self, route, vis_deg, rot_deg):
        self.grid_path = "ant_world_image_databases/grid/"
        self.grid_data = pd.read_csv("ant_world_image_databases/grid/database_entries.csv", skipinitialspace = True)

        self.route_path = "ant_world_image_databases/routes/"+route+"/"
        self.route_data = pd.read_csv(self.route_path+"database_entries.csv", skipinitialspace = True)

        self.current = [int(self.route_data['X [mm]'].iloc[0]), int(self.route_data["Y [mm]"].iloc[0]), int(self.route_data["Z [mm]"].iloc[0])]
        self.goal = [int(self.route_data['X [mm]'].iloc[-1]), int(self.route_data["Y [mm]"].iloc[-1]), int(self.route_data["Z [mm]"].iloc[-1])]

        self.vis_deg = vis_deg
        self.rot_deg = rot_deg

    def database_analysis(self):
        # grid_familiarity = {}
        for filename in self.grid_data['Filename'][:50]:
            grid_view = cv2.imread(self.grid_path + filename)
            grid_view = cv2.cvtColor(grid_view, cv2.COLOR_BGR2GRAY)
            print(self.most_familiar_bearing(grid_view))

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
    nav.database_analysis()