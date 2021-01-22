import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import cv2
from pyprobar import probar
import datetime
import csv
from collections import defaultdict

class AnalysisToolkit:

    def __init__(self, route, vis_deg, rot_deg):
        self.topdown_view = plt.imread("ant_world_image_databases/topdown_view.png")
        self.grid_path = "ant_world_image_databases/grid/"
        self.grid_data = pd.read_csv("ant_world_image_databases/grid/database_entries.csv", skipinitialspace=True)

        self.route_name = route
        self.route_path = "ant_world_image_databases/routes/"+route+"/"
        self.route_data = pd.read_csv(self.route_path+"database_entries.csv", skipinitialspace=True)

        self.route = [[x / 10 for x in self.route_data['X [mm]'].tolist()], [y / 10 for y in self.route_data["Y [mm]"].tolist()]]
        self.start = [int(self.route_data['X [mm]'].iloc[0]/10), int(self.route_data["Y [mm]"].iloc[0]/10)]
        self.goal = [int(self.route_data['X [mm]'].iloc[-1]/10), int(self.route_data["Y [mm]"].iloc[-1]/10)]
        self.bounds = [[int((math.floor((min(self.route[0]) / 10)) * 10)), int((math.floor((min(self.route[1]) / 10)) * 10))],
                        [int((math.ceil((max(self.route[0]) / 10)) * 10)), int((math.ceil((max(self.route[1]) / 10)) * 10))]]

        self.vis_deg = vis_deg
        self.rot_deg = rot_deg

    def downsample(self, view):
        view = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        return cv2.resize(view, (90, 17))

    def database_analysis(self, spacing, bounds=None, save_data=False):
        if bounds is not None:
            self.bounds = bounds

        x_ticks = np.arange(self.bounds[0][0], self.bounds[1][0] + 1, step=spacing, dtype=int)
        y_ticks = np.arange(self.bounds[0][1], self.bounds[1][1] + 1, step=spacing, dtype=int)

        grid_view_familiarity = {}
        for y in probar(y_ticks):
            for x in x_ticks:
                curr_view_path = self.grid_data['Filename'].values[(self.grid_data['Grid X'] == x/10) & (self.grid_data['Grid Y'] == y/10)][0]
                curr_view = self.downsample(cv2.imread(self.grid_path + curr_view_path))
                grid_view_familiarity[str((x, y))] = self.evaluate(curr_view=curr_view)
        fig = plt.figure(figsize=(len(x_ticks), len(y_ticks)), dpi=spacing*10)
        ax = fig.add_subplot()

        ax.imshow(self.topdown_view)
        ax.plot(self.route[0], self.route[1], linewidth=2, color='gold')
        ax.add_patch(plt.Circle((self.start[0], self.start[1]), 5, color='green'))
        ax.add_patch(plt.Circle((self.goal[0], self.goal[1]), 5, color='red'))

        X, Y = np.meshgrid(x_ticks, y_ticks)
        u = [math.sin(math.radians(n)) for n in grid_view_familiarity.values()]
        v = [math.cos(math.radians(n)) for n in grid_view_familiarity.values()]
        ax.quiver(X, Y, u, v, color='w', scale_units='xy', scale=(1/spacing)*2, width=0.01, headwidth=5)

        ax.xaxis.set_major_locator(plticker.FixedLocator(x_ticks))
        ax.yaxis.set_major_locator(plticker.FixedLocator(y_ticks))
        ax.grid(which='major', axis='both', linestyle=':')
        ax.set_xlim([self.bounds[0][0], self.bounds[1][0]])
        ax.set_ylim([self.bounds[0][1], self.bounds[1][1]])
        ax.set_xticklabels(x_ticks, rotation=90, fontsize=20)
        ax.set_yticklabels(y_ticks, rotation=0, fontsize=20)

        if save_data:
            time = datetime.datetime.now()
            time = "%s-%s-%s_%s-%s-%s" % (time.day, time.month, time.year, time.hour, time.minute, time.second)
            filename = self.route_name + '_' + str(np.ptp(x_ticks)) + 'x' + str(np.ptp(y_ticks)) + '_' + str(spacing) + '_' + str(time)
            plt.savefig('DATABASE_ANALYSIS/' + self.model_name + '/' + filename + '.png')
            try:
                with open('DATABASE_ANALYSIS/' + self.model_name + '/' + filename + '.csv', 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=grid_view_familiarity.keys())
                    writer.writeheader()
                    writer.writerow(grid_view_familiarity)
            except IOError:
                print("I/O error")

        plt.show()

    # Rotational Image Difference Function
    def RIDF(self, curr_view, route_view, route_view_heading=0):
        RIDF = defaultdict(list)
        for i in np.arange(0, self.vis_deg, step=self.rot_deg, dtype=int):
            rotated_view = np.roll(curr_view, int(curr_view.shape[1] * (i / self.vis_deg)), axis=1)
            mse = np.sum((route_view.astype("float") - rotated_view.astype("float")) ** 2)
            mse /= float(route_view.shape[0] * route_view.shape[1])
            RIDF[(i + route_view_heading) % self.vis_deg].append(mse)
        return RIDF

    def RIDF_analysis(self, RIDF):
        plt.plot(*zip(*sorted(RIDF.items())))
        plt.xlabel("Angle")
        plt.ylabel("MSE")
        plt.title("RIDF")
        plt.show()