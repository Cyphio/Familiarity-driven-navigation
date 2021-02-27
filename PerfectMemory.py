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
            mse = np.sum(self.image_difference(view_1_rotated, view_2_downsampled))
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
    # pm.database_analysis(spacing=20, corridor=20, save_data=True)
    data_1_path = "DATABASE_ANALYSIS/PERFECTMEMORY/27-2-2021_18-6-49_ant1_route1_150x750_30.csv"
    data_2_path = "DATABASE_ANALYSIS/PERFECTMEMORY/27-2-2021_18-23-4_ant1_route1_140x740_20.csv"
    # print(pm.prcnt_correct(data_path=data_path, threshold=10))
    print(pm.error_boxplot(data_1_path, data_2_path))

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
    # pm.view_analysis(view_1=grid_view, view_2=route_view, view_2_heading=route_view_heading, save_data=False)

    # Off-route best matched view analysis
    # filename = pm.grid_filenames.get((500, 500))
    # grid_view = cv2.imread(pm.grid_path + filename)
    # pm.best_matched_view_analysis(view=grid_view)

    # route_rIDF = pm.get_route_rIDF(grid_view)
    # print(f"View best matches to route idx: {pm.get_matched_route_view_idx(route_rIDF)}\n")
    #
    # familiarity_dict = pm.get_view_familiarity(route_rIDF)
    # print(f"Signal strength: {pm.get_signal_strength(familiarity_dict)}\n")
    # print(f"Most familiar heading: {pm.get_most_familiar_heading(familiarity_dict)}\n")

    # pm.fig_generator()
    #
    # # 610.001, 600.01
    # yellow = cv2.imread(pm.route_path + pm.route_filenames[263])
    # yellow_heading = pm.route_headings[263]
    # rIDF = pm.get_view_rIDF(yellow, yellow, yellow_heading)
    # plt.plot(*zip(*sorted(rIDF.items())))
    # plt.title("rIDF between view at yellow star and itself\n"
    #           "Confidence: " + str(pm.get_signal_strength(rIDF)))
    # plt.xlabel("Angle")
    # plt.ylabel("MSE in pixel intensities")
    # plt.ylim([400, 1100])
    # filename = "YELLOW_RIDF"
    # pm.save_plot(plt, "MISC/", filename)
    # plt.show()
    #
    # pink = cv2.imread(pm.grid_path + pm.grid_filenames.get((620, 600)))
    # rIDF = pm.get_view_rIDF(pink, yellow)
    # plt.plot(*zip(*sorted(rIDF.items())))
    # plt.title("rIDF between view at pink star and view at yellow star\n"
    #           "Confidence: " + str(pm.get_signal_strength(rIDF)))
    # plt.xlabel("Angle")
    # plt.ylabel("MSE in pixel intensities")
    # plt.ylim([400, 1100])
    # filename = "PINK_RIDF"
    # pm.save_plot(plt, "MISC/", filename)
    # plt.show()
    #
    # green = cv2.imread(pm.grid_path + pm.grid_filenames.get((700, 600)))
    # rIDF = pm.get_view_rIDF(green, yellow)
    # plt.plot(*zip(*sorted(rIDF.items())))
    # plt.title("rIDF between view at green star and view at yellow star\n"
    #           "Confidence: " + str(pm.get_signal_strength(rIDF)))
    # plt.xlabel("Angle")
    # plt.ylabel("MSE in pixel intensities")
    # plt.ylim([400, 1100])
    # filename = "GREEN_RIDF"
    # pm.save_plot(plt, "MISC/", filename)
    # plt.show()

    # on_route = cv2.imread(pm.route_path + pm.route_filenames[263])
    # data = {}
    # # data[0] = pm.get_signal_strength(pm.get_view_rIDF(on_route, on_route))
    # for x in np.arange(620, 900, step=10, dtype=int):
    #     print(x)
    #     off_route = cv2.imread(pm.grid_path + pm.grid_filenames.get((x, 600)))
    #     data[x-610] = pm.get_signal_strength(pm.get_view_rIDF(off_route, on_route))
    #
    # plt.plot(*zip(*sorted(data.items(), reverse=True)))
    # plt.title("Signal strength over distance")
    # plt.xlabel("X-axis displacement from yellow star (cm)")
    # plt.ylabel("Signal strength")
    # plt.tight_layout()
    #
    # # filename = "GRAPH"
    # # pm.save_plot(plt, "MISC/", filename)
    #
    # plt.show()