import Navigate
from collections import defaultdict
import numpy as np
import cv2

class PerfectMemory(Navigate.Navigate):

    pass

    def evaluate(self, curr_view):
        view_familiarity = defaultdict(list)
        for filename in self.route_data['Filename'][:5]:
            route_view = self.downsample(cv2.imread(self.route_path + filename))
            for i in np.arange(0, self.vis_deg, step=self.rot_deg, dtype=int):
                rotated_view = np.roll(curr_view, int(curr_view.shape[1] * (i / 360)), axis=1)
                mse = -np.square(np.subtract(route_view, rotated_view)).mean()
                view_familiarity[i].append(mse)
        view_familiarity = {k: np.sum(v) for k, v in view_familiarity.items()}
        return max(view_familiarity, key=view_familiarity.get)