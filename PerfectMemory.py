from AnalysisToolkit import AnalysisToolkit
import numpy as np
import cv2
from collections import defaultdict

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

    # Rotational Familiarity Function
    def get_rFF(self, route_rIDF):
        return {k: -np.amin(v) for k, v in route_rIDF.items()}

    # Calculates the signal strength of an rFF
    def get_signal_strength(self, rFF):
        return max(rFF.values()) / np.array(list(rFF.values())).mean()

    # Get the most familiar heading given an rFF for a view
    def get_most_familiar_heading(self, rFF):
        return max(rFF, key=rFF.get)

if __name__ == "__main__":
    pm = PerfectMemory(route="ant1_route1", vis_deg=360, rot_deg=2)

    # Database analysis
    # pm.database_analysis(spacing=10, bounds=[[490, 370], [550, 460]], save_data=True)
    # pm.database_analysis(spacing=30, save_data=False)

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
    # route_view_heading = pm.route_headings[idx]
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

    pm.fig_generator()

    # 610.001, 600.01
    green = cv2.imread(pm.route_path + pm.route_filenames[263])
    yellow = cv2.imread(pm.grid_path + pm.grid_filenames.get((580, 600)))
    pink = cv2.imread(pm.grid_path + pm.grid_filenames.get((530, 600)))
    blue = cv2.imread(pm.grid_path + pm.grid_filenames.get((460, 600)))
    red = cv2.imread(pm.grid_path + pm.grid_filenames.get((310, 600)))
    print(f"GREEN: {pm.get_signal_strength(pm.get_rFF(pm.get_route_rIDF(green)))}\n"
          f"YELLOW: {pm.get_signal_strength(pm.get_rFF(pm.get_route_rIDF(yellow)))}\n"
          f"PINK: {pm.get_signal_strength(pm.get_rFF(pm.get_route_rIDF(pink)))}\n"
          f"BLUE: {pm.get_signal_strength(pm.get_rFF(pm.get_route_rIDF(blue)))}\n"
          f"RED: {pm.get_signal_strength(pm.get_rFF(pm.get_route_rIDF(red)))}")