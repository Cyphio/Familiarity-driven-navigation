import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

class Navigate:

    def __init__(self, route):
        self.grid_path = "ant_world_image_databases/grid/"
        self.grid_data = pd.read_csv("ant_world_image_databases/grid/database_entries.csv", skipinitialspace = True)

        self.route_path = "ant_world_image_databases/routes/"+route+"/"
        self.route_data = pd.read_csv(self.route_path+"database_entries.csv", skipinitialspace = True)

        self.current = [int(self.route_data['X [mm]'].iloc[0]), int(self.route_data["Y [mm]"].iloc[0]), int(self.route_data["Z [mm]"].iloc[0])]
        self.goal = [int(self.route_data['X [mm]'].iloc[-1]), int(self.route_data["Y [mm]"].iloc[-1]), int(self.route_data["Z [mm]"].iloc[-1])]

    def run(self, vis_deg, rot_deg):
        view = plt.imread(self.grid_path + self.grid_data['Filename'][0])

        familiarity_dict = {}
        for i in np.arange(0, vis_deg, rot_deg):
            rotated_view = np.roll(view, int(view.shape[1]*((i/360)*3)))
            ssd = np.sum((view - rotated_view) ** 2)
            familiarity_dict[i] = ssd
        print(familiarity_dict)

        plt.plot(range(len(familiarity_dict)), familiarity_dict.values())
        plt.show()

if __name__ == "__main__":
    nav = Navigate("ant1_route1")
    nav.run(360, 2)