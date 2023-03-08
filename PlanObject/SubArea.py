import numpy as np
from GeoAlgorithm.GeoProcess import GeoProcess
from PlanObject.VirtualProduct import VirtualProduct
from FileIO.plotData import scatter

class SubArea:
    """
    This object is split area of it
    """

    def __init__(self, road_center_polyline: np.ndarray, half_road_dist: float, redline_dist: float):
        """
        Pre processing the rhino read 3d array, flatten Z axis and get the area
        :param road_center_polyline: A (n,3) dimensional array
        :param half_road_dist:
        :param redline_dist:
        """
        # polyline is original rhino read ndarray
        assert road_center_polyline.shape[1] == 2
        # Clear redundant pt in Ndarray list
        # Road center to road side
        self.half_road_dist = half_road_dist
        # Red line to road side
        self.redline_dist = redline_dist
        # Ndarray represent road center polygon
        self.road_center_line = GeoProcess.clear_redundant(road_center_polyline)
        # Ndarray represent road side polygon
        self.road_side_line = GeoProcess.offset(self.road_center_line, half_road_dist, 1)
        # Ndarray represent construct construct area polygon
        self.construct_area = GeoProcess.offset(self.road_side_line, redline_dist, 1)
        # Area is the m2 area of the current sub area
        self.area = GeoProcess.area(self.road_side_line)
        # Building container is the list to contain the building object
        self.building_container = []
        # Init sample point with None, or list of list of Ndarray represent 2d point, Have coordinate information
        self.sample_point = None
        # Init sample point with None, or list of Ndarray represent 2d point, no coordinate information
        self.flatten_sample_point = None
        # Minimal Bounding Box, with rotation
        self.bounding_box = None
        # Divided direction, self.uv_direction[0] for short direction, ~[1] for long direction
        self.uv_direction = None
        self.uv_dist = None
        # Contain satisfied
        self.satisfied_building = []
        self.satisfied_fire_bnd = []
        self.satisfied_product = []
        # Insert procedure variable
        self.tmp_satisfied_building = None
        self.tmp_satisfied_fire_bnd = None

    def __str__(self):
        return f"Planning bound is {self.road_side_line}, bound area is {self.area}"

    def sample_area(self, dist: float, clear_outside: bool):
        """
        Renew the area sample point list, with coordinate info version and without coordinate version
        :param dist: Point to point dist
        :param clear_outside: clear the out side polygon point or NOT
        :return: Renew object value
        """
        self.sample_point, self.uv_direction, self.uv_dist, self.bounding_box = GeoProcess().sample_polygon_area(
            self.construct_area, dist, clear_outside)

        # Flatten the sample point
        self.flatten_sample_point = []
        for sub_list_idx in range(len(self.sample_point)):
            self.sample_point[sub_list_idx] = np.array(self.sample_point[sub_list_idx])
            self.flatten_sample_point.extend(self.sample_point[sub_list_idx])
        self.flatten_sample_point = np.array(self.flatten_sample_point)

    def clear_pt(self, polygon_list):
        """
        Clear point if it is in the building polygon
        :param polygon_list: A list with multi polygon
        :return: Renew the flatten sample list
        """
        # CLear
        remove_idx = []
        for pt_idx in range(len(self.flatten_sample_point)):
            for polygon in polygon_list:
                if GeoProcess().point_in_polygon(self.flatten_sample_point[pt_idx], polygon):
                    remove_idx.append(pt_idx)
        self.flatten_sample_point = np.delete(self.flatten_sample_point, remove_idx, axis=0)

    def product_insert(self, test_product: VirtualProduct, needed_num: int):
        """
        Try to as much as test product
        :param test_product: A object contain virtual product to insert
        :param needed_num: The amount of this product
        :return: Renew the self.satisfied_product, self.satisfied_building, self.satisfied_fire_bnd
        """
        product_container = []
        product_fire_container = []
        # Loop over the point
        for pt in self.flatten_sample_point:
            # Construct the rectangle
            tmp_rec_1 = GeoProcess.rec_from_corner(pt, self.uv_direction[0], self.uv_direction[1],
                                                   test_product.long_side, test_product.short_side)
            tmp_rec_2 = GeoProcess.rec_from_corner(pt, self.uv_direction[0], self.uv_direction[1],
                                                   test_product.short_side, test_product.long_side)
            # Construct Fire rectangle
            tmp_fire_1 = GeoProcess.offset(tmp_rec_1, dist=test_product.half_fair_spacing, return_type=0)
            tmp_fire_2 = GeoProcess.offset(tmp_rec_2, dist=test_product.half_fair_spacing, return_type=0)

            # Check the polygon is inside the construct area
            if GeoProcess().pl_in_pl(self.construct_area, tmp_rec_1):
                product_container.append(tmp_rec_1)
                product_fire_container.append(tmp_fire_1)
            if GeoProcess().pl_in_pl(self.construct_area, tmp_rec_2):
                product_container.append(tmp_rec_2)
                product_fire_container.append(tmp_fire_2)

        # Renew the tmp_satisfied_building and tmp_satisfied_fair_bnd
        self.tmp_satisfied_building = np.array(product_container)
        self.tmp_satisfied_fire_bnd = np.array(product_fire_container)

        # Clear the polygon intersect with satisfied polygon
        first_remove = []
        for tmp_idx in range(len(self.tmp_satisfied_fire_bnd)):
            state = False
            for sat_pl in self.satisfied_fire_bnd:
                if GeoProcess().two_pl_intersec(sat_pl, self.tmp_satisfied_fire_bnd[tmp_idx]):
                    state = True
                    break
            if state:
                first_remove.append(tmp_idx)
        first_remove = np.array(first_remove).astype(int)

        self.tmp_satisfied_fire_bnd = np.delete(self.tmp_satisfied_fire_bnd, first_remove, axis=0)
        self.tmp_satisfied_building = np.delete(self.tmp_satisfied_building, first_remove, axis=0)

        cache_building = []
        cache_fire_bnd = []

        while self.tmp_satisfied_fire_bnd.shape[0] != 0:
            # Bug is here, need to loop over all
            cache_fire_bnd.append(self.tmp_satisfied_fire_bnd[0])
            cache_building.append(self.tmp_satisfied_building[0])
            remove_idx = []
            for bnd_idx in range(len(self.tmp_satisfied_fire_bnd)):
                if GeoProcess().two_pl_intersec(self.tmp_satisfied_fire_bnd[0], self.tmp_satisfied_fire_bnd[bnd_idx]):
                    remove_idx.append(bnd_idx)
            remove_idx = np.asarray(remove_idx)
            self.tmp_satisfied_fire_bnd = np.delete(self.tmp_satisfied_fire_bnd, remove_idx, axis=0)
            self.tmp_satisfied_building = np.delete(self.tmp_satisfied_building, remove_idx, axis=0)
            """
            # plot Use
            scatter(self.road_side_line, self.satisfied_building, self.flatten_sample_point)
            """
            self.clear_pt([cache_building[-1]])

        if len(cache_fire_bnd) <= needed_num:
            self.satisfied_building.extend(cache_building)
            self.satisfied_fire_bnd.extend(cache_fire_bnd)
            for i in range(len(cache_building)):
                self.satisfied_product.append(test_product)
            return needed_num - len(cache_fire_bnd)
        else:
            self.satisfied_building.extend(cache_building[:(needed_num + 1)])
            self.satisfied_fire_bnd.extend(cache_fire_bnd[:(needed_num + 1)])
            for i in range(len(cache_building[:(needed_num + 1)])):
                self.satisfied_product.append(test_product)
            return 0

    def insert_as_row(self, test_product: VirtualProduct, start_row: int, direction: str):
        """

        :param test_product:
        :param start_row:
        :param direction:
        :return:
        """
        assert direction in ["long", "short"]
        # Compute the needed row number
        if direction == "long":
            compute_row_num = int(np.ceil(test_product.short_side / self.uv_dist[1]))
            fire_row = int(np.ceil((test_product.short_side + 2 * test_product.half_fair_spacing) / self.uv_dist[1]))
        else:
            compute_row_num = int(np.ceil(test_product.long_side / self.uv_dist[1]))
            fire_row = int(np.ceil((test_product.long_side + 2 * test_product.half_fair_spacing) / self.uv_dist[1]))
        # Contain the satisfied polygon and fire bnd
        end_row = start_row + compute_row_num
        fire_row += start_row
        polygon_direction = []
        firebnd_direction = []

        if end_row <= len(self.sample_point):
            max_dist = []
            # Compute each row distance
            for pt_array in self.sample_point[start_row: end_row]:
                max_dist.append(np.linalg.norm(pt_array[0] - pt_array[-1]))

            # Select the min of distance
            max_dist = min(max_dist)
            # Compute the how much product can be insert
            if direction == "long":
                max_num = np.floor(max_dist - test_product.long_side) / (
                        test_product.long_side + 2 * test_product.half_fair_spacing) + 1
            else:
                max_num = np.floor(max_dist - test_product.short_side) / (
                        test_product.short_side + 2 * test_product.half_fair_spacing)
            max_num = int(max_num)

            if max_num > 0:
                # Insert product as one side
                for i in range(max_num):
                    tmp_start = self.sample_point[start_row][0]
                    # Check which direction to insert
                    if direction == "long":
                        start_direc = tmp_start + (test_product.long_side + 2 * test_product.half_fair_spacing) * i * \
                                      self.uv_direction[0]
                        polygon = GeoProcess.rec_from_corner(start_direc, self.uv_direction[0], self.uv_direction[1],
                                                             test_product.long_side, test_product.short_side)
                    else:
                        start_direc = tmp_start + (test_product.short_side + 2 * test_product.half_fair_spacing) * i * \
                                      self.uv_direction[0]
                        polygon = GeoProcess.rec_from_corner(start_direc, self.uv_direction[0], self.uv_direction[1],
                                                             test_product.short_side, test_product.long_side)

                    # Get the fire boundary
                    poly_fire_bnd = GeoProcess.offset(polygon, test_product.half_fair_spacing, 0)
                    inside_state = True  # Is the polygon inside the site
                    outside_fire_bnd = True  # Is the fire boundary have intersection with satisfied fire boundary

                    if not GeoProcess().two_pl_intersec(self.construct_area, polygon):
                        inside_state = False

                    for test_fire_bnd in self.satisfied_fire_bnd:
                        if GeoProcess().two_pl_intersec(test_fire_bnd, poly_fire_bnd):
                            outside_fire_bnd = False

                    if inside_state and outside_fire_bnd:
                        polygon_direction.append(polygon)
                        firebnd_direction.append(poly_fire_bnd)
        # Struct the var to return
        product_num = len(polygon_direction)

        return product_num, polygon_direction, firebnd_direction, [end_row, fire_row]

    def group_product_insert(self, test_product: VirtualProduct, needed_num: int):
        """
        Try to insert as much as test product, as row.
        :param test_product: A object contain virtual product to insert
        :param needed_num: The amount of this product
        :return: Renew the self.satisfied_product, self.satisfied_building, self.satisfied_fire_bnd
        """
        # Cache the sample point, while the length of tmp_sample_point is 0, Exit the function
        tmp_sample_point = self.sample_point
        start_row = 0
        for row_idx in range(len(self.sample_point)):
            start_row = row_idx
            if self.sample_point[row_idx].shape[0] != 0:
                break
        total_row = len(tmp_sample_point)
        print(f"total_row: {total_row}")
        while_iter = 0
        while total_row >= start_row or needed_num == 0:
            print(f"===================================================")
            print(f"While iter: {while_iter}")
            print(f"Satisfied polygon: {len(self.satisfied_building)}")
            long_direc_num, long_polygon, long_firebnd, long_end = self.insert_as_row(test_product, start_row, "long")
            short_direc_num, short_polygon, short_firebnd, short_end = self.insert_as_row(test_product, start_row,
                                                                                          "short")
            # Choose which direction to use
            if long_direc_num >= short_direc_num:
                tmp_product_num = long_direc_num
                tmp_polygon = long_polygon
                tmp_firebnd = long_firebnd
                tmp_end = long_end
            else:
                tmp_product_num = short_direc_num
                tmp_polygon = short_polygon
                tmp_firebnd = short_firebnd
                tmp_end = short_end
            print(f"End Condition: {tmp_end}")
            if len(tmp_polygon) != 0:
                # Add to satisfied_building list
                if tmp_product_num <= needed_num:
                    self.satisfied_building.extend(tmp_polygon)
                    self.satisfied_fire_bnd.extend(tmp_firebnd)
                    for i in range(len(tmp_polygon)):
                        self.satisfied_product.append(test_product)
                    needed_num = needed_num - tmp_product_num
                else:
                    self.satisfied_building.extend(tmp_polygon[:(needed_num + 1)])
                    self.satisfied_fire_bnd.extend(tmp_firebnd[:(needed_num + 1)])
                    for i in range(len(tmp_polygon[:(needed_num + 1)])):
                        self.satisfied_product.append(test_product)
                    needed_num = 0

                # Clear used sample point
                for pl_idx in range(len(tmp_polygon)):
                    test_ploygon = tmp_polygon[pl_idx]
                    for i in range(len(self.sample_point)):
                        remove_idx_list = []
                        for j in range(self.sample_point[i].shape[0]):
                            if GeoProcess().point_in_polygon(self.sample_point[i][j, :], test_ploygon):
                                remove_idx_list.append(j)
                        self.sample_point[i] = np.delete(self.sample_point[i], remove_idx_list, axis=0)
                start_row = tmp_end[1]
            else:
                start_row += 1
            while_iter += 1

        return needed_num


if __name__ == "__main__":
    # Read the 3dm file
    from FileIO.readRhino import Read3dm
    from FileIO.plotData import plot_polygon
    from PlanObject.PlanArea import PlanArea
    import CONFIG

    path = "D:\\LiGD\\Indus_Park\\Test_File\\test_polyline.3dm"
    model = Read3dm(path)
    rhino_model = model.read_Rhino()
    model_dict, model_dictkey = model.layer_Dict()
    pt_dict = model.transform_Point(model_dict, model_dictkey)
    # Choose the planning area
    plan_area_3d = pt_dict["Planning_Area"][0]

    plan_area = PlanArea(plan_area_3d)
    sub_area_list = plan_area.long_side_split()
    plan_1 = SubArea(sub_area_list[0], 3.5, 5)
    plan_1.sample_area(dist=6, clear_outside=True)
    print(f"uv_dist: {plan_1.uv_dist}")
    product = CONFIG.Para["product_dict"]["product_3"]
    product_1 = VirtualProduct(product)
    plan_1.group_product_insert(product_1, 22)

    # plan_1.product_insert(product_1, product_1.product_num)
    plot_line = []
    for b in plan_1.satisfied_building:
        plot_line.append(GeoProcess.closed_polygon(b))
    '''
    for pl in plan_1.satisfied_building:
        plot_line.append(GeoProcess.closed_polygon(pl))
    '''
    plot_line.append(GeoProcess.closed_polygon(plan_1.construct_area))
    plot_polygon(plot_line)
