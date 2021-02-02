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

    def evaluate_familiarity(self, curr_view):
        curr_view_downsampled = self.downsample(curr_view)
        curr_view_RIDF = []
        for idx, filename in enumerate(self.route_filenames):
            route_view_downsampled = self.downsample(cv2.imread(self.route_path + filename))
            mse = np.sum(self.image_difference(curr_view_downsampled, route_view_downsampled))
            mse /= float(curr_view_downsampled.shape[0] * curr_view_downsampled.shape[1])
            curr_view_RIDF.append(mse)
        return route_view_RIDF

if __name__ == "__main__":
    pm = PerfectMemory(route="ant1_route5", vis_deg=360, rot_deg=4)

    # Database analysis
    # pm.database_analysis(spacing=10, bounds=[[600, 580], [630, 760]], save_data=True)
    pm.database_analysis(spacing=30, save_data=True)

    # Route view analysis
    # pm.route_analysis(step=100)

    # Grid view analysis
    # filename = pm.grid_filenames.get((480, 850))
    # grid_view = pm.downsample(cv2.imread(pm.grid_path + filename))
    # pm.view_analysis(curr_view=grid_view, save_data=False)

    # On-route view analysis
    # idx = 10
    # filename = pm.route_filenames[idx]
    # route_view = pm.downsample(cv2.imread(pm.route_path + filename))
    # route_view_heading = pm.route_headings[idx]
    # pm.view_analysis(curr_view=route_view, curr_view_heading=route_view_heading, save_data=False)