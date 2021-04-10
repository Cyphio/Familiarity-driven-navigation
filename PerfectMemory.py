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
        view_1_preprocessed = self.preprocess(view_1)
        view_2_preprocessed = self.preprocess(view_2)
        rIDF = {}
        for i in np.arange(0, self.vis_deg, step=self.rot_deg, dtype=int):
            view_1_rotated = self.rotate(view_1_preprocessed, i)
            mse = np.sum(self.image_difference(view_1_rotated, view_2_preprocessed)**2)
            mse /= float(view_1_preprocessed.shape[0] * view_1_preprocessed.shape[1])
            rIDF[(i + view_1_heading) % self.vis_deg] = mse
        return rIDF

    # Rotational Image Difference Function for a view over a route representation
    def get_route_rIDF(self, view, view_heading=0):
        route_rIDF = defaultdict(list)
        for idx, filename in enumerate(self.route_filenames):
            route_view = cv2.imread(self.route_path + filename)
            [route_rIDF[k].append(v) for k, v in self.get_view_rIDF(view, route_view, view_heading).items()]
        return route_rIDF

    # Rotational Familiarity Function of a view against a route stored in perfect memory
    def get_route_rFF(self, view, view_heading=0):
        return {k: -np.amin(v) for k, v in self.get_route_rIDF(view, view_heading).items()}

    # Rotational Familiarity Function of a view against another view
    def get_view_rFF(self, view_1, view_2, view_1_heading=0):
        return {k: -v for k, v in self.get_view_rIDF(view_1, view_2, view_1_heading).items()}

    # Get the most familiar heading given an rFF for a view
    def get_most_familiar_heading(self, rFF):
        return max(rFF, key=rFF.get)

    # Calculates the signal strength of an rFF
    def get_signal_strength(self, rFF):
        return max(rFF.values()) / np.array(list(rFF.values())).mean()

    # get the index of the best matching route view to a view
    def get_matched_route_view_idx(self, view, view_heading=0):
        # min_RIDF_idx = {k: (np.amin(v), np.argmin(v)) for k, v in self.get_route_rIDF(view, view_heading).items()}
        # return min(min_RIDF_idx.values())[1]
        view_preprocessed = self.preprocess(self.rotate(view, view_heading))
        x = {i: np.sum(self.image_difference(view_preprocessed, self.preprocess(cv2.imread(self.route_path+filename)))**2)
             for i, filename in enumerate(self.route_filenames)}
        return min(x, key=x.get)

if __name__ == "__main__":
    route_name = "ant1_route1"
    resolution = "8_deg_px_res"

    pm = PerfectMemory(route=route_name, vis_deg=360, rot_deg=8)

    # Database analysis
    # pm.database_analysis(spacing=20, save_data=True)
    # pm.database_analysis(spacing=10, bounds=[[490, 370], [550, 460]], save_data=True)
    # pm.database_analysis(spacing=20, corridor=30, save_path=f"DATABASE_ANALYSIS/PERFECTMEMORY/{route_name}/8_deg_px_res",
    #                      save_data=True)
    # pm.show_database_analysis_plot(data_path="DATABASE_ANALYSIS/PERFECTMEMORY/ant1_route1/16_deg_px_res/16-3-2021_18-58-18_ant1_route1_140x740_20.csv",
    #                                spacing=20, locationality=True,
    #                                save_path=f"DATABASE_ANALYSIS/PERFECTMEMORY/{route_name}/{resolution}", save_data=True)

    # one_px_data_path = "DATABASE_ANALYSIS/PERFECTMEMORY/1_deg_px_res/16-3-2021_21-1-3_ant1_route1_140x740_20.csv"
    # two_px_data_path = "DATABASE_ANALYSIS/PERFECTMEMORY/2_deg_px_res/16-3-2021_19-52-18_ant1_route1_140x740_20.csv"
    # four_px_data_path = "DATABASE_ANALYSIS/PERFECTMEMORY/4_deg_px_res/16-3-2021_17-36-29_ant1_route1_140x740_20.csv"
    # eight_px_data_path = "DATABASE_ANALYSIS/PERFECTMEMORY/8_deg_px_res/16-3-2021_19-18-9_ant1_route1_140x740_20.csv"
    # sixteen_px_data_path = "DATABASE_ANALYSIS/PERFECTMEMORY/16_deg_px_res/16-3-2021_18-58-18_ant1_route1_140x740_20.csv"
    # pm.error_boxplot([one_px_data_path, two_px_data_path, four_px_data_path, eight_px_data_path, sixteen_px_data_path],
    #                  ["1 degree resolution", "2 degree resolution", "4 degree resolution", "8 degree resolution", "16 degree resolution"],
    #                  save_data=True)
    # four_px_enviro_data_path = "DATABASE_ANALYSIS/PERFECTMEMORY/4_deg_px_res/15-3-2021_22-1-36_ant1_route1_140x740_20.csv"
    # pm.error_boxplot([four_px_data_path, four_px_enviro_data_path],
    #                  ["Within route corridor", "Across environment"],
    #                  save_data=True)
    # pm.error_boxplot(["DATABASE_ANALYSIS/PERFECTMEMORY/31-3-2021_18-23-11_ant1_route1_140x740_20.csv"])

    # Route view analysis
    # pm.route_analysis(step=100)

    # Off-route view analysis
    # filename = pm.grid_filenames.get((500, 500))
    # grid_view = cv2.imread(pm.grid_path + filename)
    # pm.view_analysis(view_1=grid_view, view_2=grid_view, save_data=False)

    # On-route view analysis
    # idx = 0
    # filename = pm.route_filenames[idx]
    # route_view = cv2.imread(pm.route_path + filename)
    # route_heading = pm.route_headings[idx]
    # pm.view_analysis(view_1=route_view, view_2=route_view, view_1_heading=route_heading, save_data=False)
    # rFF = pm.get_route_rFF(view=route_view, view_heading=route_heading)
    # pm.rFF_plot(rFF=rFF, ylim=None, save_data=False)
    # print(pm.get_most_familiar_heading(rFF))

    # Off-route best matched view analysis
    # pm.best_matched_view_analysis(view_x=610, view_y=810, save_data=True)

    # Off-route real match view analysis
    # pm.ground_truth_view_analysis(view_x=610, view_y=810, save_data=True)

    # view = cv2.imread(pm.grid_path + pm.grid_filenames[(510, 250)])
    # pm.rFF_plot(pm.get_route_rFF(view), title="PM rFF of view at (510, 250)")

    original = cv2.imread("VIEW_ANALYSIS/INFO_LOSS_TEST/(550, 560)/original.png")
    lost_left_tussock = cv2.imread("VIEW_ANALYSIS/INFO_LOSS_TEST/(550, 560)/lost_left_tussock.png")
    lost_middle_tussock = cv2.imread("VIEW_ANALYSIS/INFO_LOSS_TEST/(550, 560)/lost_middle_tussock.png")
    lost_right_tussock = cv2.imread("VIEW_ANALYSIS/INFO_LOSS_TEST/(550, 560)/lost_right_tussock.png")
    lost_sky_info = cv2.imread("VIEW_ANALYSIS/INFO_LOSS_TEST/(550, 560)/lost_sky_info.png")
    lost_ground_info = cv2.imread("VIEW_ANALYSIS/INFO_LOSS_TEST/(550, 560)/lost_ground_info.png")
    # plt.imshow(mlp.preprocess(lost_sky_info), cmap='gray')
    # plt.show()
    rFF = pm.rFF_plot(pm.get_route_rFF(lost_ground_info), ylim=[-1450, -350],
                       title="PM rFF of view at (550, 560) missing ground information", save_data=True)