from Parent import Parent
from collections import defaultdict
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

class PerfectMemory(Parent):

    pass

    def evaluate(self, curr_view):
        view_familiarity = defaultdict(list)
        for filename in self.route_data['Filename']:
            route_view = self.downsample(cv2.imread(self.route_path + filename))
            route_view_heading = math.ceil(self.route_data['Heading [degrees]'].iloc[0] / self.rot_deg) * self.rot_deg
            for i in np.arange(0, self.vis_deg, step=self.rot_deg, dtype=int):
                rotated_view = np.roll(curr_view, int(curr_view.shape[1] * (i / self.vis_deg)), axis=1)
                mse = np.sum((route_view.astype("float") - rotated_view.astype("float")) ** 2)
                mse /= float(route_view.shape[0] * route_view.shape[1])
                view_familiarity[(i+route_view_heading) % self.vis_deg].append(mse)
        view_familiarity = {k: -min(v) for k, v in view_familiarity.items()}
        return max(view_familiarity, key=view_familiarity.get)

if __name__ == "__main__":
    pm = PerfectMemory(route="ant1_route1", vis_deg=360, rot_deg=4)

    # # curr_view = pm.downsample(cv2.imread(pm.grid_path + "image_+004900_+002200_+001800.png"))
    # curr_view = pm.downsample(cv2.imread(pm.route_path + "image_00000.png"))
    # print(pm.evaluate(curr_view))

    # pm.database_analysis(spacing=10, bounds=[[600, 800], [650, 850]], save_data=False)
    pm.database_analysis(spacing=50, save_data=False)