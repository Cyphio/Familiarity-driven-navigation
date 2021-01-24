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

    # Perfect Memory Rotational Image Difference Function
    def RIDF(self, curr_view, route_view, route_view_heading=0):
        RIDF = {}
        for i in np.arange(0, self.vis_deg, step=self.rot_deg, dtype=int):
            rotated_view = np.roll(curr_view, int(curr_view.shape[1] * (i / self.vis_deg)), axis=1)
            mse = np.sum((route_view.astype("float") - rotated_view.astype("float")) ** 2)
            mse /= float(route_view.shape[0] * route_view.shape[1])
            RIDF[(i + route_view_heading) % self.vis_deg] = mse
        return RIDF

if __name__ == "__main__":
    pm = PerfectMemory(route="ant1_route1", vis_deg=360, rot_deg=4)

    # Database analysis
    # pm.database_analysis(spacing=10, bounds=[[600, 800], [650, 850]], save_data=False)
    # pm.database_analysis(spacing=100, save_data=False)

    # View analysis
    curr_view = pm.downsample(cv2.imread(pm.grid_path + "image_+000000_+000000_+001800.png"))
    route_view = pm.downsample(cv2.imread(pm.route_path + "image_00082.png"))
    pm.view_analysis(curr_view)

