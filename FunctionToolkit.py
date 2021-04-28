import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import cv2
from pyprobar import probar
import datetime
import csv
import itertools
import os

class FunctionToolkit:
    def __init__(self, route, vis_deg, rot_deg):
        self.route_name = route
        self.vis_deg = vis_deg
        self.rot_deg = rot_deg

        self.topdown_view = plt.imread("ant_world_image_databases/topdown_view.png")
        self.grid_path = "ant_world_image_databases/grid/"
        grid_data = pd.read_csv("ant_world_image_databases/grid/database_entries.csv", skipinitialspace=True)
        self.grid_X = [int(x/10) for x in np.array(grid_data['X [mm]'])]
        self.grid_Y = [int(y/10) for y in np.array(grid_data['Y [mm]'])]
        self.grid_filenames = {(int(grid_data['X [mm]'][idx]/10), int(grid_data['Y [mm]'][idx]/10)): filename
                               for idx, filename in enumerate(grid_data['Filename'])}

        self.route_path = "ant_world_image_databases/routes/"+route+"/"
        route_data = pd.read_csv(self.route_path+"database_entries.csv", skipinitialspace=True)
        self.route_filenames = np.array(route_data['Filename'])
        self.route_X = [int(x/10) for x in np.array(route_data['X [mm]'])]
        self.route_Y = [int(y/10) for y in np.array(route_data["Y [mm]"])]
        self.route_headings = np.array([int(rot_deg * round(float(heading)/rot_deg)) for heading in route_data['Heading [degrees]']])

        self.bounds = [[int((np.floor((min(self.route_X) / 10)) * 10)), int((np.floor((min(self.route_Y) / 10)) * 10))],
                        [int((np.ceil((max(self.route_X) / 10)) * 10)), int((np.ceil((max(self.route_Y) / 10)) * 10))]]

    def preprocess(self, view):
        view = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        return cv2.resize(view, (45, 8))

    def rotate(self, view, angle):
        return np.roll(view, int(view.shape[1] * (angle / self.vis_deg)), axis=1)

    def image_difference(self, minuend, subtrahend):
        return abs(minuend.astype("float") - subtrahend.astype("float"))

    def get_ground_truth_coor(self, x, y):
        return min(zip(self.route_X, self.route_Y), key=lambda route_coor: ((route_coor[0] - x) ** 2 + (route_coor[1] - y) ** 2))

    def get_ground_truth_heading(self, x, y):
        return self.route_headings[list(zip(self.route_X, self.route_Y)).index(self.get_ground_truth_coor(x, y))]

    def normalize(self, d, min, max=0):
        return {k: ((v - min) / (max - min)) for k, v in d.items()}

    def is_within_prcnt(self, a, b, prcnt):
        if b == 0:
            return a == 0
        return prcnt > abs(abs(b-a)/b)*100.0

    def absolute_errors(self, data_path):
        data = csv.DictReader(open(data_path))
        errors = []
        for row in data:
            real_heading = self.get_ground_truth_heading(int(row['X_COOR']), int(row['Y_COOR']))
            errors.append(abs(real_heading - int(row['HEADING'])))
        return errors

    def avg_absolute_error(self, data_path):
        return float(np.mean(self.absolute_errors(data_path)))

    def directionally_correct(self, data_path):
        threshold = 20
        data = csv.DictReader(open(data_path))
        correct_count, total_count = 0, 0
        for row in data:
            real_heading = self.get_ground_truth_heading(int(row['X_COOR']), int(row['Y_COOR']))
            correct_count += int((real_heading - threshold) % self.vis_deg <= int(row['HEADING']) <= (real_heading + threshold) % self.vis_deg)
            total_count += 1
        return (correct_count/total_count)*100

    def locationally_correct(self, data_path):
        threshold = 75
        data = csv.DictReader(open(data_path))
        correct_count, total_count = 0, 0
        for row in data:
            ground_truth_view_coor = self.get_ground_truth_coor(int(row['X_COOR']), int(row['Y_COOR']))
            matched_route_view_coor = list(zip(self.route_X, self.route_Y))[int(row['MATCHED_ROUTE_VIEW_IDX'])]
            correct_count += int((np.sqrt((np.square(ground_truth_view_coor[0] - matched_route_view_coor[0])) +
                                          (np.square(ground_truth_view_coor[1] - matched_route_view_coor[1])))) <= threshold)

            total_count += 1
        return (correct_count/total_count)*100