import Navigate
from collections import defaultdict
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

class PerfectMemory(Navigate.Navigate):

    pass

    def evaluate(self, curr_view):
        view_familiarity = defaultdict(list)
        for filename in self.route_data['Filename']:
            route_view = self.downsample(cv2.imread(self.route_path + filename))
            route_view_heading = math.ceil(self.route_data['Heading [degrees]'].iloc[0] / self.rot_deg) * self.rot_deg
            for i in np.arange(0, self.vis_deg, step=self.rot_deg, dtype=int):
                rotated_view = np.roll(curr_view, int(curr_view.shape[1] * (i / self.vis_deg)), axis=1)
                # plt.imshow(rotated_view)
                # plt.show()
                # mse = sum(np.square(np.subtract(route_view, rotated_view)))
                mse = np.sum((route_view.astype("float") - rotated_view.astype("float")) ** 2)
                mse /= float(route_view.shape[0] * route_view.shape[1])
                view_familiarity[(i+route_view_heading) % self.vis_deg].append(mse)
        view_familiarity = {k: -min(v) for k, v in view_familiarity.items()}
        return max(view_familiarity, key=view_familiarity.get)