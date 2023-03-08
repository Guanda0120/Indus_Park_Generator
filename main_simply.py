# -*- coding: UTF-8 -*-
import time
from GeoAlgorithm.GeoProcess import GeoProcess
from PlanObject.PlanArea import PlanArea
from PlanObject.SubArea import SubArea
from PlanObject.VirtualProduct import VirtualProduct
from FileIO.readRhino import Read3dm
from FileIO.plotData import plot_polygon, plot_master
import matplotlib.pyplot as plt
from FileIO.exportRhino import ExportRhino
from CONFIG import Para
import numpy as np

# ============================ Read From CONFIG ==============================
file_dir = Para["file_dir"]
layer_name = Para["layer_name"]
max_area = Para["max_area"]
road_width = Para["road_width"]
redline_dist = Para["redline_dist"]
product_dict: dict = Para["product_dict"]
random_seeds = Para["random_seeds"]
sample_dist = Para["sample_dist"]
output_dir = Para["out_file_dir"]

# ============================ Read the Rhino File ===========================

# Read the 3dm file
model = Read3dm(file_dir)
rhino_model = model.read_Rhino()
model_dict, model_dictkey = model.layer_Dict()
pt_dict = model.transform_Point(model_dict, model_dictkey)
# Choose the planning area, default only one line in the list
plan_area_3d = pt_dict[layer_name][0]

# ============================ Split the area ==================================

# Init a PlanArea Object
plan_area = PlanArea(plan_area_3d)
# Split by max area and random seeds
sub_area_list = plan_area.long_side_split(max_area, random_seeds)
# Convert the sub area array to sub area instance
plan_area.to_subarea_obj(sub_area_list, road_width/2, redline_dist)
# Sampling each sub area
for sub_area in plan_area.subarea_list:
    sub_area.sample_area(dist=sample_dist, clear_outside=True)

# ============================ Init the virtual product ========================

# Create a virtual product list, contain virtual product
vp_list = [VirtualProduct(product_dict[key]) for key in product_dict.keys()]
# Sort as 1F area, from bigger to smaller
f1_area = np.array([vp.floor_area for vp in vp_list])
f1_area_idx = f1_area.argsort()[::-1]

# ============================ Insert Product ===================================

# Loop over all the product
st_time = time.time()

for vp in vp_list:
    plan_area.select_product(vp)

print("======================================================")
print(f"Time Duration: {time.time()-st_time}")
plan_area.init_building(0.9, 1.5)
plan_area.select_tree_point()

plot_building_line = []
for subarea in plan_area.subarea_list:
    plot_building_line.extend(subarea.satisfied_building)
for i in range(len(plot_building_line)):
    plot_building_line[i] = GeoProcess.closed_polygon(plot_building_line[i])

plot_road_line = [GeoProcess.closed_polygon(area.road_side_line) for area in plan_area.subarea_list]
plot_red_line = [GeoProcess.closed_polygon(area.construct_area) for area in plan_area.subarea_list]
plot_master(plot_red_line, plot_road_line, plot_building_line)

export = ExportRhino(plan_area, output_dir, 6)
export.export_rhino()
