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

    def image_difference(self, minuend, subtrahend):
        return (minuend.astype("float") - subtrahend.astype("float")) ** 2

    def route_view_RIDF(self, curr_view):
        RIDF = defaultdict(list)
        for idx, filename in enumerate(self.route_data['Filename']):
            route_view = self.downsample(cv2.imread(self.route_path + filename))
            route_view_heading = int(self.rot_deg * round(float(self.route_data['Heading [degrees]'].iloc[idx])/self.rot_deg))
            [RIDF[k].append(v) for k, v in self.RIDF(curr_view, route_view, route_view_heading).items()]
        return RIDF

    def most_familiar_bearing(self, RIDF):
        familiarity_dict = {k: -np.amin(v) for k, v in RIDF.items()}
        return max(familiarity_dict, key=familiarity_dict.get)

    def matched_route_view(self, RIDF):
        min_RIDF_idx = {k: (np.amin(v), np.argmin(v)) for k, v in RIDF.items()}
        filename = self.route_data['Filename'].iloc[min(min_RIDF_idx.values())[1]]
        route_view_heading = min(min_RIDF_idx, key=min_RIDF_idx.get)
        matched_route_view = self.downsample(cv2.imread(self.route_path + filename))
        return matched_route_view, route_view_heading, filename

    def save_plot(self, plot, path="", filename=""):
        time = datetime.datetime.now()
        time = "%s-%s-%s_%s-%s-%s" % (time.day, time.month, time.year, time.hour, time.minute, time.second)
        plot.savefig(path + self.model_name + '/' + str(time) + '_' + filename + '.png')

    def save_dict_as_CSV(self, data, path="", filename=""):
        time = datetime.datetime.now()
        time = "%s-%s-%s_%s-%s-%s" % (time.day, time.month, time.year, time.hour, time.minute, time.second)
        try:
            with open(path + self.model_name + '/' + str(time) + '_' + filename + '.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data.keys())
                writer.writeheader()
                writer.writerow(data)
        except IOError:
            print("I/O error")

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
                RIDF = self.route_view_RIDF(curr_view)
                grid_view_familiarity[str((x, y))] = self.most_familiar_bearing(RIDF)
        fig = plt.figure(figsize=(len(x_ticks), len(y_ticks)), dpi=spacing*10)
        ax = fig.add_subplot()

        ax.imshow(self.topdown_view)
        ax.plot(self.route[0], self.route[1], linewidth=2, color='gold')
        ax.add_patch(plt.Circle((self.start[0], self.start[1]), 5, color='green'))
        ax.add_patch(plt.Circle((self.goal[0], self.goal[1]), 5, color='red'))

        X, Y = np.meshgrid(x_ticks, y_ticks)
        u = [math.cos(math.radians(n)) for n in grid_view_familiarity.values()]
        v = [math.sin(math.radians(n)) for n in grid_view_familiarity.values()]
        ax.quiver(X, Y, u, v, color='w', scale_units='xy', scale=(1/spacing)*2, width=0.01, headwidth=5)

        ax.xaxis.set_major_locator(plticker.FixedLocator(x_ticks))
        ax.yaxis.set_major_locator(plticker.FixedLocator(y_ticks))
        ax.grid(which='major', axis='both', linestyle=':')
        ax.set_xlim([self.bounds[0][0], self.bounds[1][0]])
        ax.set_ylim([self.bounds[0][1], self.bounds[1][1]])
        ax.set_xticklabels(x_ticks, rotation=90, fontsize=20)
        ax.set_yticklabels(y_ticks, rotation=0, fontsize=20)

        if save_data:
            filename = self.route_name + '_' + str(np.ptp(x_ticks)) + 'x' + str(np.ptp(y_ticks)) + '_' + str(spacing)
            self.save_plot(plt, "DATABASE_ANALYSIS/", filename)
            self.save_dict_as_CSV(grid_view_familiarity, "DATABASE_ANALYSIS/", filename)
        plt.show()

    def view_analysis(self, curr_view, curr_heading=0, save_data=False):
        matched_route_view, route_view_heading, filename = self.matched_route_view(self.route_view_RIDF(curr_view))
        rotated_view = np.roll(curr_view, int(curr_view.shape[1] * ((route_view_heading - curr_heading) / self.vis_deg)), axis=1)
        image_difference = self.image_difference(rotated_view, matched_route_view)
        RIDF = self.RIDF(curr_view, matched_route_view, route_view_heading)

        plt.figure()
        fig, ax = plt.subplots(3, 1)
        fig.tight_layout(pad=2.0, w_pad=0)

        ax[0].set_title("Current view, at rotation " + str(route_view_heading))
        ax[0].imshow(cv2.cvtColor(rotated_view.astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax[1].set_title("Best matched route view: " + filename)
        ax[1].imshow(cv2.cvtColor(matched_route_view.astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax[2].set_title("Image difference")
        ax[2].imshow(cv2.cvtColor(image_difference.astype(np.uint8), cv2.COLOR_BGR2RGB))
        if save_data:
            filename = "IMG_DIFF"
            self.save_plot(plt, "VIEW_ANALYSIS/", filename)
        plt.show()

        plt.plot(*zip(*sorted(RIDF.items())))
        plt.title("RIDF")
        plt.xlabel("Angle")
        plt.ylabel("MSE")
        if save_data:
            filename = "RIDF"
            self.save_plot(plt, "VIEW_ANALYSIS/", filename)
        plt.show()