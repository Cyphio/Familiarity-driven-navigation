# -*- coding: utf-8 -*-
from FunctionToolkit import FunctionToolkit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import cv2
from pyprobar import probar
import datetime
import csv
import itertools
import os
import seaborn as sns

class AnalysisToolkit(FunctionToolkit):
    def __init__(self, route, vis_deg, rot_deg):
        FunctionToolkit.__init__(self, route, vis_deg, rot_deg)

        time = datetime.datetime.now()
        self.time = "%s-%s-%s_%s-%s-%s" % (time.day, time.month, time.year, time.hour, time.minute, time.second)

    def save_plot(self, plot, path="", filename=""):
        if not os.path.isdir(path):
            os.makedirs(path)
        plot.savefig(f"{path}/{filename} {str(self.time)}.png")

    def save_dict_as_CSV(self, data, path="", filename=""):
        keys = data[0].keys()
        if not os.path.isdir(path):
            os.makedirs(path)
        try:
            with open(f"{path}/{str(self.time)}_{filename}.csv", 'w', newline='') as csvfile:
                dict_writer = csv.DictWriter(csvfile, keys)
                dict_writer.writeheader()
                dict_writer.writerows(data)
        except IOError:
            print("I/O error")

    def rFF_plot(self, rFF, title, ylim=None, ybound=None, save_path="VIEW_ANALYSIS", save_data=False):
        fig, ax = plt.subplots()
        ax.plot(*zip(*sorted(rFF.items())))
        plt.title(f"{title}\n"
                  f"Confidence: {round(self.get_signal_strength(rFF), 3)}, "
                  f"(Median) maximum: {round(max(rFF.values()), 3)} @ {self.get_most_familiar_heading(rFF)}°", wrap=True)
        plt.ylabel("Familiarity score")
        plt.xlabel("Angle")
        plt.xticks(np.arange(0, 361, 15), rotation=90)
        if ylim is not None:
            buffer = (ylim[1]-ylim[0])*0.05
            plt.ylim(ylim[0]-buffer, ylim[1]+buffer)
        if ybound is not None:
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.set_yticklabels([f"{ybound[0]}, 0.0", "0.2", "0.4", "0.6", "0.8", f"{ybound[1]}, 1.0"])
        plt.xlim(0, 360)
        plt.grid(which='major', axis='both', linestyle=':')
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        if save_data:
            self.save_plot(plt, save_path, title)
        plt.show()

    def gen_info_loss_data(self, coor, ybound, model_name, save_path, save_data=False):
        pos = f"({coor[0]}, {coor[1]})"
        for filename in os.listdir(f"VIEW_ANALYSIS/INFO_LOSS_TEST/{pos}/IMAGES"):
            x = os.path.splitext(filename)[0]
            print(x)
            image = cv2.imread(f"VIEW_ANALYSIS/INFO_LOSS_TEST/{pos}/IMAGES/{x}.png")
            self.rFF_plot(self.normalize(self.get_route_rFF(image), min=ybound[0], max=ybound[1]), ylim=[0, 1], ybound=ybound,
                          title=f"{model_name} rFF of view ({x}) at {pos} vs route memories",
                          save_path=save_path, save_data=save_data)

    def database_analysis(self, spacing, bounds=None, corridor=None, save_path="DATABASE_ANALYSIS", save_data=False):
        if bounds is None:
            bounds = self.bounds

        x_ticks = np.arange(bounds[0][0], bounds[1][0] + 1, spacing, dtype=int)
        y_ticks = np.arange(bounds[1][1], bounds[0][1] - 1, -spacing, dtype=int)

        if corridor is not None:
            quiver_coors = list(itertools.chain.from_iterable([list(zip(np.arange(int((np.floor((x-corridor) / 10) * 10)), int((np.floor((x+corridor) / 10) * 10))+1, spacing, dtype=int), itertools.repeat(y)))
                                                               for x, y in zip([self.route_X[min(range(len(self.route_Y)), key=lambda i: abs(self.route_Y[i]-y))] for y in y_ticks], y_ticks)]))
        else:
            quiver_coors = [(x, y) for x in x_ticks for y in y_ticks]

        cm = plt.get_cmap('YlOrRd')
        line_map = [cm(1. * i / len(self.route_X)) for i in range(len(self.route_X))]
        quiver_map = []

        data = []
        for x, y in probar(quiver_coors):
            view = cv2.imread(self.grid_path + self.grid_filenames.get((x, y)))

            rFF = self.get_route_rFF(view)
            familiar_heading = self.get_most_familiar_heading(rFF)

            matched_route_view_idx = self.get_matched_route_view_idx(view=view, view_heading=familiar_heading)
            quiver_map.append(line_map[matched_route_view_idx])

            data.append({"X_COOR": x, "Y_COOR": y, "HEADING": familiar_heading, "MATCHED_ROUTE_VIEW_IDX": matched_route_view_idx})

        if save_data:
            filename = f"{self.route_name}_{str(np.ptp(x_ticks))}x{str(np.ptp(y_ticks))}_{str(spacing)}"
            self.save_dict_as_CSV(data, save_path, filename)

    def show_database_analysis_plot(self, data_path, spacing, bounds=None, locationality=True,
                                    save_path="DATABASE_ANALYSIS", save_data=False):
        data = pd.read_csv(open(data_path))

        if bounds is None:
            bounds = self.bounds

        x_ticks = np.arange(bounds[0][0], bounds[1][0] + 1, spacing, dtype=int)
        y_ticks = np.arange(bounds[1][1], bounds[0][1] - 1, -spacing, dtype=int)

        cm = plt.get_cmap('YlOrRd')
        line_map = [cm(1. * i / len(self.route_X)) for i in range(len(self.route_X))]
        quiver_map = [line_map[idx] for idx in data['MATCHED_ROUTE_VIEW_IDX'].values]

        fig = plt.figure(figsize=(len(x_ticks), len(y_ticks)))
        ax = fig.add_subplot()

        ax.imshow(self.topdown_view)

        ax.add_patch(plt.Circle((self.route_X[0], self.route_Y[0]), 5, color='green'))
        ax.add_patch(plt.Circle((self.route_X[-1], self.route_Y[-1]), 5, color='red'))

        X, Y = data['X_COOR'].values, data['Y_COOR'].values
        u = [np.sin(np.deg2rad(heading)) for heading in data['HEADING'].values]
        v = [np.cos(np.deg2rad(heading)) for heading in data['HEADING'].values]
        if locationality:
            ax.set_prop_cycle('color', line_map)
            [ax.plot(self.route_X[i:i + 2], self.route_Y[i:i + 2], linewidth=4) for i in range(len(line_map))]
            ax.quiver(X, Y, u, v, color=quiver_map, scale_units='xy', scale=(1 / spacing) * 2, width=0.01, headwidth=5)
        else:
            ax.plot(self.route_X, self.route_Y, linewidth=4, color='r')
            ax.quiver(X, Y, u, v, color='w', scale_units='xy', scale=(1 / spacing) * 2, width=0.01, headwidth=5)

        ax.xaxis.set_major_locator(plticker.FixedLocator(x_ticks))
        ax.yaxis.set_major_locator(plticker.FixedLocator(y_ticks))
        ax.grid(which='major', axis='both', linestyle=':')
        ax.set_xlim([bounds[0][0], bounds[1][0]])
        ax.set_ylim([bounds[0][1], bounds[1][1]])
        ax.set_xticklabels(x_ticks, rotation=90, fontsize=20)
        ax.set_yticklabels(y_ticks, rotation=0, fontsize=20)

        if save_data:
            filename = f"{self.route_name}_{str(np.ptp(x_ticks))}x{str(np.ptp(y_ticks))}_{str(spacing)}"
            self.save_plot(plt, save_path, filename)
        plt.show()

    def error_boxplot(self, data_paths, index_titles=None, locationality=True, save_data=False):
        heading_errors = []
        indexes = []
        for idx, data_path in enumerate(data_paths):
            data = csv.DictReader(open(data_path))
            heading_errors.append([abs(self.get_ground_truth_heading(int(row['X_COOR']), int(row['Y_COOR'])) - int(row['HEADING'])) for row in data])
            if locationality:
                info_titles = f"AVG DIRECTIONAL ERROR: {round(self.avg_absolute_error(data_path), 2)}\n" \
                              f"DIRECTIONALLY CORRECT: {round(self.directionally_correct(data_path), 2)}%\n" \
                              f"LOCATIONALLY CORRECT: {round(self.locationally_correct(data_path), 2)}%"
            else:
                info_titles = f"AVG DIRECTIONAL ERROR: {round(self.avg_absolute_error(data_path), 2)}\n" \
                              f"DIRECTIONALLY CORRECT: {round(self.directionally_correct(data_path), 2)}%\n"
            if index_titles is not None:
                indexes.append(f"{index_titles[idx]}\n{info_titles}")
            else:
                indexes.append(f"{info_titles}")
        fig = plt.figure(dpi=750)
        ax = fig.add_subplot()
        df = pd.DataFrame(heading_errors, indexes)
        df.T.boxplot(vert=False, flierprops=dict(markerfacecolor='r', marker='s'))
        plt.title("Boxplot of directional errors in headings")
        plt.xlabel("Absolute heading error in degrees")
        plt.xticks(rotation=90)
        plt.yticks(fontsize=7)
        ax.xaxis.set_major_locator(plticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(plticker.AutoMinorLocator())
        plt.tight_layout()
        plt.xlim(-5)
        if save_data:
            filename = self.route_name + '_BOXPLOT'
            self.save_plot(plt, "DATABASE_ANALYSIS/", filename)
        plt.show()

    def view_analysis(self, view_1, view_2, view_1_heading=0, save_data=False):
        rFF = self.get_view_rFF(view_1, view_2, view_1_heading)
        familiar_heading = self.get_most_familiar_heading(rFF)

        rotated_view = self.rotate(view_1, familiar_heading-view_1_heading)

        rotated_view_downsampled = self.preprocess(rotated_view)
        view_2_downsampled = self.preprocess(view_2)

        image_difference = self.image_difference(rotated_view_downsampled, view_2_downsampled)

        plt.figure(dpi=750)
        fig, ax = plt.subplots(3, 1)
        fig.tight_layout(pad=2.0, w_pad=0)

        ax[0].set_title(f"Given view at heading: {familiar_heading}, rotated: {familiar_heading - view_1_heading}")
        ax[0].imshow(cv2.cvtColor(rotated_view_downsampled.astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax[1].set_title(f"view_1 at heading: {familiar_heading}, rotated: {familiar_heading - view_1_heading}")
        ax[1].imshow(cv2.cvtColor(view_2_downsampled.astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax[2].set_title("Image difference")
        ax[2].imshow(cv2.cvtColor(image_difference.astype(np.uint8), cv2.COLOR_BGR2RGB))
        if save_data:
            filename = "IMG_DIFF"
            self.save_plot(plt, "VIEW_ANALYSIS/", filename)
        plt.show()

        self.rFF_plot(rFF=rFF, title="rFF", ylim=None, save_data=save_data)

    def ground_truth_view_analysis(self, view_x, view_y, ybound, view_heading=0, save_data=False):
        view = cv2.imread(self.grid_path + self.grid_filenames.get((view_x, view_y)))

        ground_truth_view_coor = self.get_ground_truth_coor(view_x, view_y)
        ground_truth_view_idx = list(zip(self.route_X, self.route_Y)).index(ground_truth_view_coor)
        ground_truth_view_filename = self.route_filenames[ground_truth_view_idx]
        ground_truth_view_heading = self.route_headings[ground_truth_view_idx]
        ground_truth_view = cv2.imread(self.route_path + ground_truth_view_filename)

        rFF = self.normalize(self.get_view_rFF(view, ground_truth_view, view_heading),
                             min=ybound[0], max=ybound[1])
        familiar_heading = self.get_most_familiar_heading(rFF)

        rotated_view = self.rotate(view, familiar_heading-view_heading)

        rotated_view_downsampled = self.preprocess(rotated_view)
        ground_truth_view_downsampled = self.preprocess(ground_truth_view)

        image_difference = self.image_difference(rotated_view_downsampled, ground_truth_view_downsampled)

        plt.figure(dpi=750)
        fig, ax = plt.subplots(3, 1)
        fig.tight_layout(pad=2.0, w_pad=0)

        ax[0].set_title(f"View at: ({view_x}, {view_y}) at familiar heading vs ground truth view: {familiar_heading}°")
        ax[0].imshow(cv2.cvtColor(rotated_view_downsampled.astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax[1].set_title(f"Ground truth route view: {ground_truth_view_filename} at pose: "
                        f"({ground_truth_view_coor[0]}, {ground_truth_view_coor[1]}), {ground_truth_view_heading}°")
        ax[1].imshow(cv2.cvtColor(ground_truth_view_downsampled.astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax[2].set_title("Image difference")
        ax[2].imshow(cv2.cvtColor(image_difference.astype(np.uint8), cv2.COLOR_BGR2RGB))
        if save_data:
            filename = "REALMATCH_IMG_DIFF"
            self.save_plot(plt, "VIEW_ANALYSIS/", filename)
        plt.show()

        self.rFF_plot(rFF=rFF, title=f"PM rFF of view at ({view_x}, {view_y}) vs ground truth view",
                      ylim=[0, 1], ybound=ybound, save_data=save_data)

    def best_matched_view_analysis(self, view_x, view_y, ybound, view_heading=0, save_data=False):
        view = cv2.imread(self.grid_path + self.grid_filenames.get((view_x,view_y)))

        matched_route_view_idx = self.get_matched_route_view_idx(view)
        matched_route_view_filename = self.route_filenames[matched_route_view_idx]
        matched_route_view_heading = self.route_headings[matched_route_view_idx]
        matched_route_view = cv2.imread(self.route_path + matched_route_view_filename)

        route_rFF = self.normalize(self.get_route_rFF(view, view_heading),
                                  min=ybound[0], max=ybound[1])
        familiar_heading = self.get_most_familiar_heading(route_rFF)

        rotated_view = self.rotate(view, familiar_heading-view_heading)

        matched_route_view_downsampled = self.preprocess(matched_route_view)
        rotated_view_downsampled = self.preprocess(rotated_view)

        image_difference = self.image_difference(rotated_view_downsampled, matched_route_view_downsampled)

        plt.figure()
        fig, ax = plt.subplots(3, 1)
        fig.tight_layout(pad=2.0, w_pad=0)

        ax[0].set_title(f"View at: ({view_x}, {view_y}) at familiar heading vs route memories: {familiar_heading}°")
        ax[0].imshow(cv2.cvtColor(rotated_view_downsampled.astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax[1].set_title(f"Best matched route view: {matched_route_view_filename} at pose: "
                        f"({self.route_X[matched_route_view_idx]}, {self.route_Y[matched_route_view_idx]}), {matched_route_view_heading}°")
        ax[1].imshow(cv2.cvtColor(matched_route_view_downsampled.astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax[2].set_title("Image difference")
        ax[2].imshow(cv2.cvtColor(image_difference.astype(np.uint8), cv2.COLOR_BGR2RGB))
        if save_data:
            filename = "BESTMATCH_IMG_DIFF"
            self.save_plot(plt, "VIEW_ANALYSIS/", filename)
        plt.show()

        self.rFF_plot(rFF=route_rFF, title=f"PM rFF of view at ({view_x}, {view_y}) vs route memories", ylim=[0, 1], ybound=ybound, save_data=save_data)

        x_ticks = np.arange(self.bounds[0][0], self.bounds[1][0] + 1, 20, dtype=int)
        y_ticks = np.arange(self.bounds[1][1], self.bounds[0][1] - 1, -20, dtype=int)

        fig = plt.figure(figsize=(len(x_ticks), len(y_ticks)))
        ax = fig.add_subplot()

        ax.imshow(self.topdown_view)

        cm = plt.get_cmap('YlOrRd')
        line_map = [cm(1. * i / len(self.route_filenames)) for i in range(len(self.route_filenames))]
        ax.set_prop_cycle('color', line_map)
        [ax.plot(self.route_X[i:i + 2], self.route_Y[i:i + 2], linewidth=4) for i in range(len(line_map))]
        ax.add_patch(plt.Circle((self.route_X[0], self.route_Y[0]), 10, color='green'))
        ax.add_patch(plt.Circle((self.route_X[-1], self.route_Y[-1]), 10, color='red'))

        ax.quiver(view_x, view_y, np.sin(np.deg2rad(familiar_heading)), np.cos(np.deg2rad(familiar_heading)),
                  color=line_map[matched_route_view_idx], scale_units='xy', scale=0.1, width=0.01, headwidth=5)
        ground_truth_view_coor = self.get_ground_truth_coor(view_x, view_y)
        ax.add_patch(plt.Circle((view_x, view_y), 10, linewidth=3, color='cyan', fill=False))
        ax.plot(ground_truth_view_coor[0], ground_truth_view_coor[1], markersize=50, color='lime', marker='*')
        ax.plot(self.route_X[matched_route_view_idx], self.route_Y[matched_route_view_idx], markersize=50, color='deeppink', marker='*')

        ax.xaxis.set_major_locator(plticker.FixedLocator(x_ticks))
        ax.yaxis.set_major_locator(plticker.FixedLocator(y_ticks))
        ax.grid(which='major', axis='both', linestyle=':')
        ax.set_xlim([self.bounds[0][0], self.bounds[1][0]])
        ax.set_ylim([self.bounds[0][1], self.bounds[1][1]])
        ax.set_xticklabels(x_ticks, rotation=90, fontsize=20)
        ax.set_yticklabels(y_ticks, rotation=0, fontsize=20)

        if save_data:
            filename = "ENVIRONMENT"
            self.save_plot(plt, "VIEW_ANALYSIS/", filename)
        plt.show()

    def get_enviro_with_markers(self):
        cm = plt.get_cmap('YlOrRd')
        line_map = [cm(1. * i / len(self.route_X)) for i in range(len(self.route_X))]

        fig = plt.figure()
        ax = fig.add_subplot()

        ax.imshow(self.topdown_view)
        ax.axis('equal')

        ax.set_prop_cycle('color', line_map)
        [ax.plot(self.route_X[i:i + 2], self.route_Y[i:i + 2], linewidth=1) for i in range(len(line_map))]
        ax.add_patch(plt.Circle((self.route_X[0], self.route_Y[0]), 10, color='green'))
        ax.add_patch(plt.Circle((self.route_X[-1], self.route_Y[-1]), 10, color='red'))

        ax.plot(610, 600, markersize=10, color='yellow', marker='*')
        ax.plot(620, 600, markersize=10, color='pink', marker='*')
        ax.plot(700, 600, markersize=10, color='lime', marker='*')

        ax.set_xlim([self.bounds[0][0], self.bounds[1][0]])
        ax.set_ylim([self.bounds[0][1], self.bounds[1][1]])

        filename = "ZOOMEDOUT"
        self.save_plot(plt, "MISC/", filename)

        plt.show()

    def get_colour_map(self, pm_data_path, mlp_data_path, spacing, bounds=None, save_path="DATABASE_ANALYSIS", save_data=False):
        if bounds is None:
            bounds = self.bounds

        x_ticks = np.arange(bounds[0][0], bounds[1][0] + 1, spacing, dtype=int)
        y_ticks = np.arange(bounds[1][1], bounds[0][1] - 1, -spacing, dtype=int)

        pm_df = pd.read_csv(open(pm_data_path))
        pm_df["ABS_HEADING_ERROR"] = self.absolute_errors(pm_data_path)
        mlp_df = pd.read_csv(open(mlp_data_path))
        mlp_df["ABS_HEADING_ERROR"] = self.absolute_errors(mlp_data_path)

        diff_df = pd.merge(pm_df, mlp_df, how='inner', on=["X_COOR", "Y_COOR"])
        diff_df["HEADING_DIFF"] = diff_df["ABS_HEADING_ERROR_x"] - diff_df["ABS_HEADING_ERROR_y"]

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #     print(diff_df)

        fig = plt.figure(figsize=(len(x_ticks)+5, len(y_ticks)))
        ax = fig.add_subplot()

        ax.imshow(self.topdown_view)

        route_cm = plt.get_cmap('YlOrRd')
        line_map = [route_cm(1. * i / len(self.route_X)) for i in range(len(self.route_X))]
        ax.set_prop_cycle('color', line_map)
        [ax.plot(self.route_X[i:i + 2], self.route_Y[i:i + 2], linewidth=4) for i in range(len(line_map))]

        diff_cm = plt.cm.get_cmap('RdYlGn')
        shifted_diff_cm = self.shift_colour_map(diff_cm, midpoint=0.21)
        sc = ax.scatter(x=diff_df["X_COOR"], y=diff_df["Y_COOR"], c=diff_df["HEADING_DIFF"], s=700, cmap=shifted_diff_cm)
        cbar = plt.colorbar(sc)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(30)
        cbar.ax.get_yaxis().labelpad = 55
        cbar.ax.set_ylabel("MLP abs heading error subtracted from PM abs heading error. "
                           "Green indicates MLP outperforming PM", rotation=270, fontsize=50)

        ax.xaxis.set_major_locator(plticker.FixedLocator(x_ticks))
        ax.yaxis.set_major_locator(plticker.FixedLocator(y_ticks))
        ax.grid(which='major', axis='both', linestyle=':')
        ax.set_xlim([bounds[0][0], bounds[1][0]])
        ax.set_ylim([bounds[0][1], bounds[1][1]])
        ax.set_xticklabels(x_ticks, rotation=90, fontsize=20)
        ax.set_yticklabels(y_ticks, rotation=0, fontsize=20)

        plt.tight_layout()

        if save_data:
            filename = "COLOUR_PLOT"
            self.save_plot(plt, save_path, filename)
        plt.show()

if __name__ == "__main__":
    route_name = "ant1_route1"
    at = AnalysisToolkit(route=route_name, vis_deg=360, rot_deg=8)

    pm_path = "DATABASE_ANALYSIS/PERFECTMEMORY/ant1_route1/8_deg_px_res/16-3-2021_19-18-9_ant1_route1_140x740_20.csv"
    mlp_path = "DATABASE_ANALYSIS/MLP/ant1_route1/TRAINED_ON_90_DEGREES_DATA/22-4-2021_14-2-53_ant1_route1_140x740_20.csv"
    # mlp_path = "DATABASE_ANALYSIS/MLP/ant1_route1/TRAINED_ON_RAND_DATA/22-4-2021_14-23-11_ant1_route1_140x740_20.csv"

    at.get_colour_map(pm_path, mlp_path, spacing=20, save_data=False)

    # print(at.get_ground_truth_heading(560, 550))