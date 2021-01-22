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
        total_view_familiarity = defaultdict(list)
        for idx, filename in enumerate(self.route_data['Filename']):
            route_view = self.downsample(cv2.imread(self.route_path + filename))
            # route_view_heading = math.ceil(self.route_data['Heading [degrees]'].iloc[idx] / self.rot_deg) * self.rot_deg
            route_view_heading = int(self.rot_deg * round(float(self.route_data['Heading [degrees]'].iloc[idx])/self.rot_deg))
            for i in np.arange(0, self.vis_deg, step=self.rot_deg, dtype=int):
                rotated_view = np.roll(curr_view, int(curr_view.shape[1] * (i / self.vis_deg)), axis=1)
                mse = np.sum((route_view.astype("float") - rotated_view.astype("float")) ** 2)
                mse /= float(route_view.shape[0] * route_view.shape[1])
                total_view_familiarity[(i+route_view_heading) % self.vis_deg].append(mse)
        self.view_familiarity = {k: -min(v) for k, v in total_view_familiarity.items()}
        return max(self.view_familiarity, key=self.view_familiarity.get)

if __name__ == "__main__":
    pm = PerfectMemory(route="ant1_route1", vis_deg=360, rot_deg=4)

    # curr_view = pm.downsample(cv2.imread(pm.grid_path + "image_+004900_+002200_+001800.png"))
    # curr_view = pm.downsample(cv2.imread(pm.route_path + "image_00004.png"))
    # print(pm.evaluate(curr_view))

    # pm.database_analysis(spacing=10, bounds=[[600, 800], [650, 850]], save_data=False)
    pm.database_analysis(spacing=30, save_data=True)