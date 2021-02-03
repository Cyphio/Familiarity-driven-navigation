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
        self.route_name = route
        self.vis_deg = vis_deg
        self.rot_deg = rot_deg

        self.topdown_view = plt.imread("ant_world_image_databases/topdown_view.png")
        self.grid_path = "ant_world_image_databases/grid/"
        grid_data = pd.read_csv("ant_world_image_databases/grid/database_entries.csv", skipinitialspace=True)
        self.grid_X = [int(x/10) for x in np.array(grid_data['X [mm]'])]
        self.grid_Y = [int(y/10) for y in np.array(grid_data['Y [mm]'])]
        self.grid_filenames = {(int(grid_data['X [mm]'][idx]/10), int(grid_data['Y [mm]'][idx]/10)): filename
                               for idx, filename in enumerate(grid_data['Filename'])}

        self.route_path = "ant_world_image_databases/routes/"+route+"/"
        route_data = pd.read_csv(self.route_path+"database_entries.csv", skipinitialspace=True)
        self.route_filenames = np.array(route_data['Filename'])
        self.route_X = [int(x/10) for x in np.array(route_data['X [mm]'])]
        self.route_Y = [int(y/10) for y in np.array(route_data["Y [mm]"])]
        self.route_headings = np.array([int(rot_deg * round(float(heading)/rot_deg)) for heading in route_data['Heading [degrees]']])

        self.bounds = [[int((np.floor((min(self.route_X) / 10)) * 10)), int((np.floor((min(self.route_Y) / 10)) * 10))],
                        [int((np.ceil((max(self.route_X) / 10)) * 10)), int((np.ceil((max(self.route_Y) / 10)) * 10))]]

    def downsample(self, view):
        view = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        return cv2.resize(view, (90, 17))

    def image_difference(self, minuend, subtrahend):
        return (minuend.astype("float") - subtrahend.astype("float")) ** 2

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

        x_ticks = np.arange(self.bounds[0][0], self.bounds[1][0] + 1, spacing, dtype=int)
        y_ticks = np.arange(self.bounds[1][1], self.bounds[0][1] - 1, -spacing, dtype=int)
        cm = plt.get_cmap('YlOrRd')
        line_map = [cm(1. * i / (len(self.route_X) - 1)) for i in range(len(self.route_X) - 1)]
        quiver_map = []

        grid_view_familiarity = {}
        for y in probar(y_ticks):
            for x in x_ticks:
                curr_view = cv2.imread(self.grid_path + self.grid_filenames.get((x, y)))
                route_RIDF = self.route_RIDF(curr_view)

                familiar_heading = self.get_familiar_heading(route_RIDF)
                grid_view_familiarity[str((x, y))] = familiar_heading

                matched_route_view_idx = self.get_matched_route_view_idx(route_RIDF)
                quiver_map.append(line_map[matched_route_view_idx])
        print(grid_view_familiarity)
        fig = plt.figure(figsize=(len(x_ticks), len(y_ticks)), dpi=spacing*10)
        ax = fig.add_subplot()

        ax.imshow(self.topdown_view)

        ax.set_prop_cycle('color', line_map)
        [ax.plot(self.route_X[i:i + 2], self.route_Y[i:i + 2], linewidth=4) for i in range(len(line_map))]
        ax.add_patch(plt.Circle((self.route_X[0], self.route_Y[0]), 5, color='green'))
        ax.add_patch(plt.Circle((self.route_X[-1], self.route_Y[-1]), 5, color='red'))

        X, Y = np.meshgrid(x_ticks, y_ticks)
        u = [np.sin(np.deg2rad(n)) for n in grid_view_familiarity.values()]
        v = [np.cos(np.deg2rad(n)) for n in grid_view_familiarity.values()]
        ax.quiver(X, Y, u, v, color=quiver_map, scale_units='xy', scale=(1/spacing)*2, width=0.01, headwidth=5)

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

    def view_analysis(self, view, view_heading=0, save_data=False):
        route_RIDF = self.route_RIDF(view, view_heading)

        familiar_heading = self.get_familiar_heading(route_RIDF)

        matched_route_view_idx = self.get_matched_route_view_idx(route_RIDF)
        matched_route_view_filename = self.route_filenames[matched_route_view_idx]
        matched_route_view = cv2.imread(self.route_path + matched_route_view_filename)
        matched_route_view_downsampled = self.downsample(matched_route_view)

        rotated_view = np.roll(view, int(view.shape[1] * ((familiar_heading - view_heading) / self.vis_deg)), axis=1)
        rotated_view_downsampled = self.downsample(rotated_view)

        image_difference = self.image_difference(rotated_view_downsampled, matched_route_view_downsampled)
        view_RIDF = self.view_RIDF(rotated_view, matched_route_view, familiar_heading)

        plt.figure()
        fig, ax = plt.subplots(3, 1)
        fig.tight_layout(pad=2.0, w_pad=0)

        ax[0].set_title("View, at rotation " + str(familiar_heading))
        ax[0].imshow(cv2.cvtColor(rotated_view.astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax[1].set_title("Best matched route view: " + matched_route_view_filename)
        ax[1].imshow(cv2.cvtColor(matched_route_view.astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax[2].set_title("Image difference")
        ax[2].imshow(cv2.cvtColor(image_difference.astype(np.uint8), cv2.COLOR_BGR2RGB))
        if save_data:
            filename = "IMG_DIFF"
            self.save_plot(plt, "VIEW_ANALYSIS/", filename)
        plt.show()

        plt.plot(*zip(*sorted(view_RIDF.items())))
        plt.title("RIDF between view and best matched route view")
        plt.xlabel("Angle")
        plt.ylabel("MSE of pixel intensities")
        if save_data:
            filename = "RIDF"
            self.save_plot(plt, "VIEW_ANALYSIS/", filename)
        plt.show()

    def route_analysis(self, step):
        cm = plt.get_cmap('YlOrRd')
        line_map = [cm(1. * i / (len(self.route_filenames) - 1)) for i in range(len(self.route_filenames) - 1)]
        quiver_map = []

        route_view_familiarity = {}
        for idx, filename in enumerate(self.route_filenames[::step]):
            print(f"Current view under analysis: {filename}")
            curr_view = cv2.imread(self.route_path + filename)
            route_RIDF = self.route_RIDF(curr_view, self.route_headings[idx*step])

            familiar_heading = self.get_familiar_heading(route_RIDF)
            matched_route_view_idx = self.get_matched_route_view_idx(route_RIDF)

            route_view_familiarity[str((self.route_X[idx * step], self.route_Y[idx * step]))] = familiar_heading
            quiver_map.append(line_map[matched_route_view_idx])
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.imshow(self.topdown_view)
        ax.axis('equal')

        ax.set_prop_cycle('color', line_map)
        [ax.plot(self.route_X[i:i + 2], self.route_Y[i:i + 2], linewidth=1) for i in range(len(line_map))]
        ax.add_patch(plt.Circle((self.route_X[0], self.route_Y[0]), 5, color='green'))
        ax.add_patch(plt.Circle((self.route_X[-1], self.route_Y[-1]), 5, color='red'))

        X = [x for x in self.route_X[::step]]
        Y = [y for y in self.route_Y[::step]]
        u = [np.sin(np.deg2rad(n)) for n in route_view_familiarity.values()]
        v = [np.cos(np.deg2rad(n)) for n in route_view_familiarity.values()]

        ax.quiver(X, Y, u, v, color=quiver_map, scale_units='xy')

        ax.set_xlim([self.bounds[0][0], self.bounds[1][0]])
        ax.set_ylim([self.bounds[0][1], self.bounds[1][1]])

        plt.show()