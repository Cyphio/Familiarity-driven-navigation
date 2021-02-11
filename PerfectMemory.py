from AnalysisToolkit import AnalysisToolkit
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from collections import defaultdict

class PerfectMemory(AnalysisToolkit):

    def __init__(self, route, vis_deg, rot_deg):
        AnalysisToolkit.__init__(self, route, vis_deg, rot_deg)
        self.model_name = 'PERFECTMEMORY'

    # Rotational Image Difference Function for two views
    def rIDF(self, view_1, view_2, view_1_heading=0):
        view_1_downsampled = self.downsample(view_1)
        view_2_downsampled = self.downsample(view_2)
        view_RIDF = {}
        for i in np.arange(0, self.vis_deg, step=self.rot_deg, dtype=int):
            view_1_rotated = np.roll(view_1_downsampled, int(view_1_downsampled.shape[1] * (i / self.vis_deg)), axis=1)
            mse = np.sum(self.image_difference(view_1_rotated, view_2_downsampled))
            mse /= float(view_1_downsampled.shape[0] * view_1_downsampled.shape[1])
            view_RIDF[(i + view_1_heading) % self.vis_deg] = mse
        return view_RIDF

    # Rotational Image Difference Function for a view over a route representation
    def get_view_familiarity(self, view, view_heading=0):
        route_RIDF = defaultdict(list)
        for idx, filename in enumerate(self.route_filenames):
            route_view = cv2.imread(self.route_path + filename)
            [route_RIDF[k].append(v) for k, v in self.rIDF(view, route_view, view_heading).items()]
        return route_RIDF

    # Get the most familiar heading given an RIDF for a view
    def get_familiar_heading(self, RIDF):
        familiarity_dict = {k: -np.amin(v) for k, v in RIDF.items()}
        return max(familiarity_dict, key=familiarity_dict.get)

    # get the index of the best matching route view to a view given an RIDF for that view
    def get_matched_route_view_idx(self, RIDF):
        min_RIDF_idx = {k: (np.amin(v), np.argmin(v)) for k, v in RIDF.items()}
        return min(min_RIDF_idx.values())[1]

if __name__ == "__main__":
    pm = PerfectMemory(route="ant1_route1", vis_deg=360, rot_deg=2)

    # Database analysis
    # pm.database_analysis(spacing=10, bounds=[[490, 370], [550, 460]], save_data=True)
    # pm.database_analysis(spacing=30, save_data=True)

    # Route view analysis
    # pm.route_analysis(step=100)

    # Grid view analysis
    # filename = pm.grid_filenames.get((500, 500))
    # grid_view = cv2.imread(pm.grid_path + filename)
    # pm.view_analysis(view=grid_view, save_data=False)

    # On-route view analysis
    # idx = 405
    # filename = pm.route_filenames[idx]
    # route_view = cv2.imread(pm.route_path + filename)
    # route_view_heading = pm.route_headings[idx]
    # pm.view_analysis(grid_view, route_view, view_2_heading=route_view_heading, save_data=True)

    # fig = plt.figure()
    #
    # view = cv2.imread(pm.grid_path + pm.grid_filenames.get((500, 500)))
    # grid_view = cv2.imread(pm.route_path + pm.route_filenames[405])
    #
    # plt.imshow(cv2.cvtColor(view.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # filename = "VIEW_ROT_0"
    # pm.save_plot(plt, "MISC/", filename)
    # plt.show()
    #
    # view_rot_90 = np.roll(view, int(view.shape[1] * (90 / pm.vis_deg)), axis=1)
    # plt.imshow(cv2.cvtColor(view_rot_90.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # filename = "VIEW_ROT_90"
    # pm.save_plot(plt, "MISC/", filename)
    # plt.show()
    #
    # view_rot_180 = np.roll(view, int(view.shape[1] * (180 / pm.vis_deg)), axis=1)
    # plt.imshow(cv2.cvtColor(view_rot_180.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # filename = "VIEW_ROT_180"
    # pm.save_plot(plt, "MISC/", filename)
    # plt.show()
    #
    # view_rot_270 = np.roll(view, int(view.shape[1] * (270 / pm.vis_deg)), axis=1)
    # plt.imshow(cv2.cvtColor(view_rot_270.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # filename = "VIEW_ROT_270"
    # pm.save_plot(plt, "MISC/", filename)
    # plt.show()

    # view_RIDF = pm.view_RIDF(view, grid_view, 0)
    # plt.plot(*zip(*sorted(view_RIDF.items())))
    # plt.title("RIDF")
    # plt.xlabel("Angle")
    # plt.ylabel("MSE of pixel intensities")
    # filename = "VIEW_RIDF"
    # pm.save_plot(plt, "VIEW_ANALYSIS/", filename)
    # plt.show()