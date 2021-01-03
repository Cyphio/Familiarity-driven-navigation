import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys

np.set_printoptions(threshold=sys.maxsize)

class Navigate:

    def __init__(self, route):
        self.grid_path = "ant_world_image_databases/grid/"
        self.grid_data = pd.read_csv("ant_world_image_databases/grid/database_entries.csv", skipinitialspace = True)

        self.route_path = "ant_world_image_databases/routes/"+route+"/"
        self.route_data = pd.read_csv(self.route_path+"database_entries.csv", skipinitialspace = True)

        self.current = [int(self.route_data['X [mm]'].iloc[0]), int(self.route_data["Y [mm]"].iloc[0]), int(self.route_data["Z [mm]"].iloc[0])]
        self.goal = [int(self.route_data['X [mm]'].iloc[-1]), int(self.route_data["Y [mm]"].iloc[-1]), int(self.route_data["Z [mm]"].iloc[-1])]

    def run(self, vis_deg, rot_deg):
        view = cv2.imread(self.grid_path + self.grid_data['Filename'][0])
        view = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)

        familiarity_dict = {}
        for i in np.arange(0, vis_deg, rot_deg):
            familiarity_dict[i] = self.get_familiarity(view, i)
        print(familiarity_dict)

        plt.plot(range(len(familiarity_dict)), familiarity_dict.values())
        plt.show()

    def get_familiarity(self, view, i):
        rotated_view = np.roll(view, int(view.shape[1] * ((i / 360) * 3)), axis=1)
        return np.sum((view.astype('float') - rotated_view.astype(float)) ** 2)

if __name__ == "__main__":
    nav = Navigate("ant1_route1")
    nav.run(360, 2)