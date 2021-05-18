from AnalysisToolkit import AnalysisToolkit
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import os

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
    # csv_file_path = "DATABASE_ANALYSIS/PERFECTMEMORY/ant1_route1/4_deg_px_res/16-3-2021_17-36-29_ant1_route1_140x740_20.csv"
    # pm.show_database_analysis_plot(data_path=csv_file_path,
    #                                spacing=20, locationality=True,
    #                                save_path=f"DATABASE_ANALYSIS/PERFECTMEMORY/{route_name}/{resolution}", save_data=False)

    # pm.scatter_confidence_against_error(spacing=20, corridor=30, save_data=True)


    # one_px_data_path = "DATABASE_ANALYSIS/PERFECTMEMORY/ant1_route1/1_deg_px_res/16-3-2021_21-1-3_ant1_route1_140x740_20.csv"
    # two_px_data_path = "DATABASE_ANALYSIS/PERFECTMEMORY/ant1_route1/2_deg_px_res/16-3-2021_19-52-18_ant1_route1_140x740_20.csv"
    # four_px_data_path = "DATABASE_ANALYSIS/PERFECTMEMORY/ant1_route1/4_deg_px_res/15-3-2021_22-1-36_ant1_route1_140x740_20.csv"
    # eight_px_data_path = "DATABASE_ANALYSIS/PERFECTMEMORY/ant1_route1/8_deg_px_res/PM-ant1_route1.csv"
    # sixteen_px_data_path = "DATABASE_ANALYSIS/PERFECTMEMORY/ant1_route1/16_deg_px_res/16-3-2021_18-58-18_ant1_route1_140x740_20.csv"
    # pm.error_boxplot([one_px_data_path, two_px_data_path, four_px_data_path, eight_px_data_path, sixteen_px_data_path],
    #                  ["1 degree resolution", "2 degree resolution", "4 degree resolution", "8 degree resolution", "16 degree resolution"],
    #                  save_data=True)
    # four_px_enviro_data_path = "DATABASE_ANALYSIS/PERFECTMEMORY/4_deg_px_res/15-3-2021_22-1-36_ant1_route1_140x740_20.csv"
    # pm.error_boxplot([four_px_data_path, four_px_enviro_data_path],
    #                  ["Within route corridor", "Across environment"],
    #                  save_data=True)
    # pm.error_boxplot(["DATABASE_ANALYSIS/PERFECTMEMORY/ant1_route1/4_deg_px_res/16-3-2021_17-36-29_ant1_route1_140x740_20.csv",
    #                   "DATABASE_ANALYSIS/PERFECTMEMORY/ant1_route1/4_deg_px_res/15-3-2021_22-1-36_ant1_route1_140x740_20.csv"],
    #                  ["Across environment", "Within route corridor"],
    #                  save_data=True)






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





    # save_data = True
    # coor = [510, 250]
    # save_path = f"VIEW_ANALYSIS/INFO_LOSS_TEST/({coor[0]}, {coor[1]})/PM"
    #
    # ybound = [-1250, -550]
    # pm.best_matched_view_analysis(view_x=coor[0], view_y=coor[1], save_data=save_data)
    # pm.ground_truth_view_analysis(view_x=coor[0], view_y=coor[1], save_data=save_data)

    # info_loss_ybound = [-700, -275]
    # pm.gen_info_loss_data(coor=coor, ybound=info_loss_ybound, model_name="PM", save_path=save_path, save_data=save_data)



    # on route rFF
    # idx = 0
    # view = cv2.imread(pm.route_path + pm.route_filenames[idx])
    # heading = pm.route_headings[0]
    # rff = pm.get_route_rFF(view, heading)
    # min_ = min(rff.values())
    # max_ = max(rff.values())
    # pm.rFF_plot(pm.get_route_rFF(view, heading), "Perfect Memory rFF of on route view")
    # pm.rFF_plot(pm.normalize(rff, min_, max_), ylim=[0, 1], ybound=[round(min_), round(max_)],
    #             title=f"PM rFF of on route view vs route memories", save_data=True)


    # Off route_rFF
    # coor = [630, 590]
    # view = cv2.imread(pm.grid_path + pm.grid_filenames[(coor[0], coor[1])])
    # rff = pm.get_route_rFF(view)
    # min_ = min(rff.values())
    # max_ = max(rff.values())
    # pm.rFF_plot(pm.normalize(rff, min_, max_), ylim=[0, 1], ybound=[round(min_), round(max_)],
    #             title=f"PM rFF of view at ({coor[0]}, {coor[1]}) vs route memories", save_data=True)

