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

    def evaluate(self, curr_view):
        RIDF = defaultdict(list)
        for idx, filename in enumerate(self.route_data['Filename']):
            route_view = self.downsample(cv2.imread(self.route_path + filename))
            route_view_heading = int(self.rot_deg * round(float(self.route_data['Heading [degrees]'].iloc[idx])/self.rot_deg))
            RIDF = self.RIDF(curr_view, route_view, route_view_heading)
        RIDF = {k: -min(v) for k, v in RIDF.items()}
        return max(RIDF, key=RIDF.get)

if __name__ == "__main__":
    pm = PerfectMemory(route="ant1_route1", vis_deg=360, rot_deg=4)

    # curr_view = pm.downsample(cv2.imread(pm.grid_path + "image_+004900_+002200_+001800.png"))
    curr_view = pm.downsample(cv2.imread(pm.route_path + "image_00001.png"))
    route_view = pm.downsample(cv2.imread(pm.route_path + "image_00080.png"))

    RIDF = pm.RIDF(curr_view, route_view)
    pm.RIDF_analysis(RIDF)

    # pm.database_analysis(spacing=10, bounds=[[600, 800], [650, 850]], save_data=False)
    # pm.database_analysis(spacing=30, save_data=True)