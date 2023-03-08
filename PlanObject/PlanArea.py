import numpy as np
import time
import copy
import random
from GeoAlgorithm.GeoProcess import GeoProcess
from PlanObject.SubArea import SubArea
from PlanObject.VirtualProduct import VirtualProduct
from PlanObject.Building import Building
from FileIO.plotData import plot_area,scatter


class PlanArea:
    """
    This object is the all area of plan
    """

    def __init__(self, polyline: np.ndarray):
        """
        Pre processing the rhino read 3d array, flatten Z axis and get the area
        :param polyline: A (n,3) dimensional array
        """
        # polyline is original rhino read ndarray
        assert polyline.shape[1] == 3
        self.plan_area_3d = polyline
        # Flatten Z axis
        self.plan_area = GeoProcess.flatten_Z(polyline)
        # Compute the area of the plan area
        self.area = GeoProcess.area(self.plan_area)
        # Init it as None, Contain the SubArea object
        self.subarea_list = None
        # Init it as None, Contain the area of Subarea object
        self.subarea_m2_list = None
        # satisfied_building is the list contain Ndarray represent building polygon
        self.satisfied_building = []
        # satisfied_fire_bnd is the list contain Ndarray represent building half fair boundary polygon
        self.satisfied_fire_bnd = []
        # satisfied_area_idx is the list contain the area insert index
        self.satisfied_area_idx = []
        # Contain tree center point
        self.tree_pt = []

    def __str__(self):
        return f"Planning bound is {self.plan_area}, bound area is {self.area}"

    def long_side_split(self, max_area: float = 10000, random_seed: int = 42):
        """
        Random choose a point on longest segment on polygon, split the PlanArea,
        while all the area is less than max_area, exit the loop.
        :param max_area: The Max area of sub split area
        :param random_seed: Random choose on longest segment
        :return:
        """

        # Var use to while
        next_polygon = [self.plan_area]
        satisfied_polygon = []  # Satisfied polygon
        while_iter = 1  # iter
        st_t = time.time()

        while len(next_polygon) != 0:
            print(f"In the {while_iter}th while loop.")
            # All after clip
            all_polygon = []
            # Loop over the next_polygon clip it
            for polygon in next_polygon:
                # Choose the longest side
                pt_long, line_idx = GeoProcess().random_pt_longest_side(polygon, random_seeds=random_seed)
                # Clip it
                polygon_contain = GeoProcess().split_by_line(polygon, pt_long, line_idx)

                # Check area is satisfied or not
                for pl in polygon_contain:
                    # Not satisfied, need next clip
                    if GeoProcess.area(pl) > max_area:
                        all_polygon.append(pl)
                    # Satisfied, return
                    else:
                        satisfied_polygon.append(pl)

            next_polygon = copy.deepcopy(all_polygon)

            """
            # Follow is to plot data use
            print(f"satisfied_polygon: {satisfied_polygon}")
            print(f"next_polygon: {next_polygon}")
            plot_area(satisfied_polygon, next_polygon)
            """

            while_iter += 1
            print(f"summary: satisfied_polygon: {len(satisfied_polygon)}"
                  f" all_polygon: {len(all_polygon)}")
            print("======================================================")

        print(f'TIme Duration： {time.time() - st_t}')

        # Post processing the sub split space
        # Some situation may generate blank list, remove it; if it is nice polygon, close it
        for i in range(len(satisfied_polygon)):

            if satisfied_polygon[i].shape[0] != 0:
                satisfied_polygon[i] = GeoProcess.clear_redundant(GeoProcess.closed_polygon(satisfied_polygon[i]))
            else:
                satisfied_polygon.remove(satisfied_polygon[i])

        return satisfied_polygon

    def to_subarea_obj(self, satisfied_polygon: list, road_offset_dist: float, redline_dist: float):
        """
        Change the Ndarray satisfied_polygon to SubArea object,
        :param redline_dist:
        :param road_offset_dist:
        :param satisfied_polygon: List of Ndarray
        :return: Renew the subarea_list & subarea_m2_list of PlanArea
        """
        # Init SubArea instance, contain as list
        self.subarea_list = [SubArea(sub_area, road_offset_dist, redline_dist) for sub_area in satisfied_polygon]
        # Extract area info
        self.subarea_m2_list = [subarea.area for subarea in self.subarea_list]
        # Sort as area
        print(f"Area information is {self.subarea_m2_list}")
        idx = np.asarray(self.subarea_m2_list).argsort()[::-1]
        tmp_subarea = []
        tmp_m2 = []
        for id in idx:
            tmp_subarea.append(self.subarea_list[id])
            tmp_m2.append(self.subarea_m2_list[id])
        self.subarea_m2_list = tmp_m2
        self.subarea_list = tmp_subarea

    def select_product(self, virtual_product: VirtualProduct):
        """
        Insert the needed num of virtual product into the sub area
        :param virtual_product: The virtual product to insert
        :return: Renew the self.satisfied_building, self.satisfied_fair_bnd, self.subarea_idx,
                and remove inside polygon point in the subarea.flatten_sample_point
        """
        # Insert product to the
        needed_prod_num = virtual_product.product_num
        print(f"Needed num of product: {needed_prod_num}.")
        for sub_area in self.subarea_list:
            rest = sub_area.product_insert(virtual_product, needed_prod_num)
            needed_prod_num = rest
            print(f"Still rest {needed_prod_num} product to insert.")
            if needed_prod_num == 0:
                break

    def init_building(self, roof_thick: float, glass_offset: float):
        """
        Init building into building_container
        :param roof_thick: Sandwich bread thick
        :param glass_offset: Ham offset
        :return: Renew each subarea building container
        """
        for sub_area in self.subarea_list:
            if len(sub_area.satisfied_building) != 0:
                for idx in range(len(sub_area.satisfied_building)):
                    tmp_building = Building(sub_area.satisfied_building[idx], sub_area.satisfied_product[idx])
                    tmp_building.generate_horizon_facade(roof_thick, glass_offset)
                    sub_area.building_container.append(tmp_building)

    def select_tree_point(self):
        for sub_area in self.subarea_list:
            for pt in sub_area.flatten_sample_point:
                self.tree_pt.append(pt)
        random.shuffle(self.tree_pt)
        need_num = int(len(self.tree_pt)*0.15)
        self.tree_pt = self.tree_pt[:need_num]


if __name__ == "__main__":
    from FileIO.readRhino import Read3dm
    from FileIO.plotData import plot_polygon

    # Read the 3dm file
    path = "D:\\LiGD\\Indus_Park\\Test_File\\test_shenyang.3dm"
    model = Read3dm(path)
    rhino_model = model.read_Rhino()
    model_dict, model_dictkey = model.layer_Dict()
    pt_dict = model.transform_Point(model_dict, model_dictkey)
    # Choose the planning area
    plan_area_3d = pt_dict["Planning_Area"][0]

    plan_area = PlanArea(plan_area_3d)
    sub_area_list = plan_area.long_side_split()
    plot_polygon(sub_area_list)
    plan_area.to_subarea_obj(satisfied_polygon=sub_area_list, road_offset_dist=3.5, redline_dist=5)

    product_dict = [[3000, 3, 2, 0.6, 3], [1200, 4, 2, 0.6, 3]]
    prod_list = []
    temp_area = plan_area.subarea_list[-1]
    temp_area.sample_area(dist=3, clear_outside=True)
    for prod in product_dict:
        prod = VirtualProduct(prod)
        temp_area.product_insert(prod, 2)
