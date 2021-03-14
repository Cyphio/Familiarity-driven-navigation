from AnalysisToolkit import AnalysisToolkit
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt

class PerfectMemory(AnalysisToolkit):

    def __init__(self, route, vis_deg, rot_deg):
        AnalysisToolkit.__init__(self, route, vis_deg, rot_deg)
        self.model_name = 'PERFECTMEMORY'

    # Rotational Image Difference Function for two views
    def get_view_rIDF(self, view_1, view_2, view_1_heading=0):
        view_1_downsampled = self.downsample(view_1)
        view_2_downsampled = self.downsample(view_2)
        view_rIDF = {}
        for i in np.arange(0, self.vis_deg, step=self.rot_deg, dtype=int):
            view_1_rotated = np.roll(view_1_downsampled, int(view_1_downsampled.shape[1] * (i / self.vis_deg)), axis=1)
            mse = np.sum(self.image_difference(view_1_rotated, view_2_downsampled)**2)
            mse /= float(view_1_downsampled.shape[0] * view_1_downsampled.shape[1])
            view_rIDF[(i + view_1_heading) % self.vis_deg] = mse
        return view_rIDF

    # Rotational Image Difference Function for a view over a route representation
    def get_route_rIDF(self, view, view_heading=0):
        route_rIDF = defaultdict(list)
        for idx, filename in enumerate(self.route_filenames):
            route_view = cv2.imread(self.route_path + filename)
            [route_rIDF[k].append(v) for k, v in self.get_view_rIDF(view, route_view, view_heading).items()]
        return route_rIDF

    # get the index of the best matching route view to a view given an rIDF for that view
    def get_matched_route_view_idx(self, route_rIDF):
        min_RIDF_idx = {k: (np.amin(v), np.argmin(v)) for k, v in route_rIDF.items()}
        return min(min_RIDF_idx.values())[1]

    # Calculates the signal strength of an rIDF
    def get_signal_strength(self, rIDF):
        return -(min(rIDF.values()) / np.array(list(rIDF.values())).mean())

    # Rotational Familiarity Function
    def get_rFF(self, route_rIDF):
        return {k: -np.amin(v) for k, v in route_rIDF.items()}

    # Get the most familiar heading given an rIDF for a view
    def get_most_familiar_heading(self, rFF):
        return max(rFF, key=rFF.get)


if __name__ == "__main__":
    pm = PerfectMemory(route="ant1_route1", vis_deg=360, rot_deg=2)

    # Database analysis
    # pm.database_analysis(spacing=30, save_data=True)
    # pm.database_analysis(spacing=10, bounds=[[490, 370], [550, 460]], save_data=True)
    # pm.database_analysis(spacing=20, corridor=30, save_data=True)
    # one_px_data_path = "DATABASE_ANALYSIS/PERFECTMEMORY/1_deg_px_res/10-3-2021_10-18-41_ant1_route1_140x740_20.csv"
    # two_px_data_path = "DATABASE_ANALYSIS/PERFECTMEMORY/2_deg_px_res/10-3-2021_0-14-54_ant1_route1_140x740_20.csv"
    # four_px_data_path = "DATABASE_ANALYSIS/PERFECTMEMORY/4_deg_px_res/9-3-2021_22-45-20_ant1_route1_140x740_20.csv"
    # eight_px_data_path = "DATABASE_ANALYSIS/PERFECTMEMORY/8_deg_px_res/9-3-2021_22-26-6_ant1_route1_140x740_20.csv"
    # sixteen_px_data_path = "DATABASE_ANALYSIS/PERFECTMEMORY/16_deg_px_res/9-3-2021_22-0-47_ant1_route1_140x740_20.csv"
    # pm.error_boxplot([one_px_data_path, two_px_data_path, four_px_data_path, eight_px_data_path, sixteen_px_data_path], save_data=False)
    # threshold = 20
    # print(f"AVG error: {pm.avg_error(data_path)}\n% correct: {pm.prcnt_correct(data_path, threshold)}")

    # Route view analysis
    # pm.route_analysis(step=100)

    # Off-route view analysis
    # filename = pm.grid_filenames.get((500, 500))
    # grid_view = cv2.imread(pm.grid_path + filename)
    # pm.view_analysis(view_1=grid_view, view_2=grid_view, save_data=False)

    # On-route view analysis
    # idx = 405
    # filename = pm.route_filenames[idx]
    # route_view = cv2.imread(pm.route_path + filename)
    # route_heading = pm.route_headings[idx]
    # pm.view_analysis(view_1=route_view, view_2=route_view, view_1_heading=route_heading, save_data=False)

    # Off-route best matched view analysis
    # pm.best_matched_view_analysis(view_x=610, view_y=810, save_data=True)

    # Off-route real match view analysis
    # pm.ground_truth_view_analysis(view_x=610, view_y=810, save_data=True)

    # 610.001, 600.01
    # yellow = cv2.imread(pm.route_path + pm.route_filenames[263])
    # yellow_heading = pm.route_headings[263]
    # rIDF = pm.get_view_rIDF(yellow, yellow, yellow_heading)
    # plt.plot(*zip(*sorted(rIDF.items())))
    # plt.title(f"rIDF between view at yellow star and itself\n"
    #           f"Confidence: {round(pm.get_signal_strength(rIDF), 2)}, Minimum: {round(min(rIDF.values()), 2)}")
    # plt.ylabel("MSE in pixel intensities")
    # plt.xlabel("Angle")
    # plt.xticks(np.arange(0, 361, 15), rotation=90)
    # plt.ylim(500, 1100)
    # plt.xlim(0, 360)
    # plt.grid(which='major', axis='both', linestyle=':')
    # plt.tight_layout()
    # filename = "YELLOW_RIDF"
    # pm.save_plot(plt, "MISC/", filename)
    # plt.show()
    #
    # pink = cv2.imread(pm.grid_path + pm.grid_filenames.get((620, 600)))
    # rIDF = pm.get_view_rIDF(pink, yellow)
    # plt.plot(*zip(*sorted(rIDF.items())))
    # plt.title(f"rIDF between view at pink star and view at yellow star\n"
    #           f"Confidence: {round(pm.get_signal_strength(rIDF), 2)}, Minimum: {round(min(rIDF.values()), 2)}")
    # plt.ylabel("MSE in pixel intensities")
    # plt.xlabel("Angle")
    # plt.xticks(np.arange(0, 361, 15), rotation=90)
    # plt.ylim(500, 1100)
    # plt.xlim(0, 360)
    # plt.grid(which='major', axis='both', linestyle=':')
    # plt.tight_layout()
    # filename = "PINK_RIDF"
    # pm.save_plot(plt, "MISC/", filename)
    # plt.show()
    #
    # green = cv2.imread(pm.grid_path + pm.grid_filenames.get((700, 600)))
    # rIDF = pm.get_view_rIDF(green, yellow)
    # plt.plot(*zip(*sorted(rIDF.items())))
    # plt.title(f"rIDF between view at green star and view at yellow star\n"
    #           f"Confidence: {round(pm.get_signal_strength(rIDF), 2)}, Minimum: {round(min(rIDF.values()), 2)}")
    # plt.ylabel("MSE in pixel intensities")
    # plt.xlabel("Angle")
    # plt.xticks(np.arange(0, 361, 15), rotation=90)
    # plt.ylim(500, 1100)
    # plt.xlim(0, 360)
    # plt.grid(which='major', axis='both', linestyle=':')
    # plt.tight_layout()
    # filename = "GREEN_RIDF"
    # pm.save_plot(plt, "MISC/", filename)
    # plt.show()
    #
    # on_route = cv2.imread(pm.route_path + pm.route_filenames[263])
    # data = {}
    # for x in np.arange(620, 900, step=10, dtype=int):
    #     off_route = cv2.imread(pm.grid_path + pm.grid_filenames.get((x, 600)))
    #     data[x-610] = pm.get_signal_strength(pm.get_view_rIDF(off_route, on_route))
    #
    # plt.plot(*zip(*sorted(data.items(), reverse=True)))
    # plt.title("Signal strength over distance")
    # plt.xlabel("X-axis displacement from yellow star (cm)")
    # plt.ylabel("Signal strength")
    # plt.xticks(np.arange(10, 281, 10), rotation=90)
    # plt.grid(which='major', axis='both', linestyle=':')
    # plt.tight_layout()
    #
    # filename = "GRAPH"
    # pm.save_plot(plt, "MISC/", filename)
    #
    # plt.show()