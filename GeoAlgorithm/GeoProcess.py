import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


class GeoProcess:

    # Method Classification
    # ________________________________________________
    # ------------------------------------------------
    # | to_ndarray          | transform data type    |
    # | to_shapely          | transform data type    |
    # | flatten_Z           | transform data type    |
    # ------------------------------------------------
    # | area                | property               |
    # ------------------------------------------------
    # | closed_polygon      | compute geometry       |
    # | vertical_ray        | compute geometry       |
    # |

    @staticmethod
    def to_ndarray(polygon: Polygon) -> np.ndarray:
        """
        Transporm the shapely Polygon into ndarray
        :param polygon: A shapely Polygon
        :return: A ndarray
        """
        np_coord = np.zeros((len(polygon.exterior.coords), 2))

        for i in range(len(polygon.exterior.coords)):
            # Read the x,y coordinate
            np_coord[i, 0] = polygon.exterior.coords[i][0]
            np_coord[i, 1] = polygon.exterior.coords[i][1]

        return np_coord

    @staticmethod
    def to_shapely(np_coord: np.ndarray) -> Polygon:
        """
        Transporm the ndarray into shapely Polygon
        :param ndarray: a ndarray
        :return: A shapely Polygon
        """
        list_pt = []
        for i in range(np_coord.shape[0]):
            pt = (np_coord[i, 0], np_coord[i, 1])
            list_pt.append(pt)

        return Polygon(list_pt)

    @staticmethod
    def flatten_Z(np_coord: np.ndarray):
        """
        Given a (n,3) matrix
        :param np_coord:
        :return:
        """
        assert np_coord.shape[1] == 3
        return np_coord[:, 0:2]

    @staticmethod
    def clear_redundant(np_coord: np.ndarray):
        """
        For a numpy polygon, remove redundant vec
        :param np_coord: A Ndarray
        :return: A Ndarray, with unique item
        """
        # Check the index to remove
        remove_idx = []
        for i in range(np_coord.shape[0] - 1):
            if (np_coord[i] == np_coord[i + 1]).all():
                remove_idx.append(i + 1)
        # Remove it
        if remove_idx != []:
            np_coord = np.delete(np_coord, remove_idx, axis=0)

        return np_coord

    @staticmethod
    def area(polygon: np.ndarray):
        """
        Get the polygon area, Shoelace formula, https://en.wikipedia.org/wiki/Shoelace_formula
        :param polygon: np.ndarray
        :return: Area of polygon
        !! May have some tolerance +- 1E-10 !!
        """
        area = 0.5 * np.abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))
        return area

    @staticmethod
    def closed_polygon(polygon: np.ndarray):
        '''
        Cloesd an open polygon
        :param polygon: Polygon array
        :return: Closed polygon array
        '''

        if (polygon[0, :] != polygon[-1, :]).any():
            polygon = np.vstack((polygon, np.array([polygon[0]])))
        return polygon

    @staticmethod
    def vertical_ray(ray: np.ndarray):
        '''
        Return a vertical ray, counter clockwise
        :param ray: Ray to compute
        :return: Vertical Ray
        '''
        # Check the legal input
        assert ray.shape == (2,)

        vertical_ray = np.zeros((2))
        vertical_ray[0] = -ray[1]
        vertical_ray[1] = ray[0]
        return vertical_ray

    @staticmethod
    def offset(polygon: np.ndarray, dist: float, return_type: int):
        """
        Get offset polygon.
        https://stackoverflow.com/questions/54033808/how-to-offset-polygon-edges
        TODO there still be some bug here. ie...self intersection
        :param polygon: A Ndarray refer to a polygon
        :param dist: Offset distance
        :param return_type: Only int(0) and int(1) can be input here,
                            int(0): the longer polygon
                            int(1): the short polygon
        :return: A Ndarray refer to a polygon
        """
        # Check the input is legal or not
        assert polygon.shape[1] == 2
        assert return_type in [0, 1]

        # Closed the polygon if is not closed
        if (polygon[0, :] != polygon[-1, :]).any():
            polygon = np.vstack((polygon, np.array([polygon[0]])))

        # Compute the prev_vec & next_vec
        cur_pt = polygon[:-1, :]
        next_pt = polygon[1:, :]
        prev_pt = np.vstack((np.array([polygon[-2]]), cur_pt[:-1, :]))
        prev_vec = cur_pt - prev_pt
        next_vec = next_pt - cur_pt
        # Normalize prev_vec and next_vec
        prev_vec = prev_vec / np.tile(np.linalg.norm(prev_vec, axis=1), (2, 1)).T
        next_vec = next_vec / np.tile(np.linalg.norm(next_vec, axis=1), (2, 1)).T

        # Rotate Matrix
        ro_matrix = np.asarray([[0, -1], [1, 0]])
        # Two orthogonal vector
        prev_normal_d = np.dot(prev_vec, ro_matrix)
        next_normal_d = np.dot(next_vec, ro_matrix)

        # Move vector
        move_vec = prev_normal_d + next_normal_d
        proj_len = dist / np.sqrt(1 + np.sum(np.multiply(prev_normal_d, next_normal_d), axis=1))
        proj_len = np.tile(proj_len, (2, 1)).T

        # Get 2 polygon
        polygon_1 = cur_pt + proj_len * move_vec
        polygon_2 = cur_pt - proj_len * move_vec

        # Compare area
        pl_area_1 = 0.5 * np.abs(
            np.dot(polygon_1[:, 0], np.roll(polygon_1[:, 1], 1)) - np.dot(polygon_1[:, 1], np.roll(polygon_1[:, 0], 1)))
        pl_area_2 = 0.5 * np.abs(
            np.dot(polygon_2[:, 0], np.roll(polygon_2[:, 1], 1)) - np.dot(polygon_2[:, 1], np.roll(polygon_2[:, 0], 1)))

        # Check which to return
        if return_type == 0:
            # Return the bigger polygon
            if pl_area_1 > pl_area_2:
                return polygon_1
            else:
                return polygon_2
        else:
            # return the smaller polygon
            if pl_area_1 > pl_area_2:
                return polygon_2
            else:
                return polygon_1

    @staticmethod
    def line_2ray_inter(line: np.ndarray, start_pt: np.ndarray, ray_direc: np.ndarray):
        '''
        Find a intersection between line and ray, positive direction and negative direction
        Is use to split polygon by XL(CAD) line
        :param line: The line to check, finite line
        :param start_pt: The start point of a ray
        :param ray_direc: The direction of a ray
        :return: None if not on the line, inter_pt and coe if have intersection, and pt_line_state
        pt_line_state = [in_check_line_or_not, ray_parallel_or_not, start_pt_on_extend_line_or_not]
        '''
        # Assert the line has pass in a legal input
        assert line.shape == (2, 2)
        assert ray_direc.shape == start_pt.shape == (2,)

        line_start = line[0, :]
        line_ray = line[1, :] - line[0, :]

        # Construct a matrix
        # | line_ray_X  -ray_direc_X |
        # | line_ray_Y  -ray_direc_Y |
        matrix = np.column_stack((line_ray, -ray_direc))

        # para_ray is the direction of line_start and start_pt
        para_ray = start_pt - line_start
        # para_coe is to check whether in line or not, in line means on the extend part of line
        para_coe = para_ray / line_ray

        if np.linalg.det(matrix):
            # Invertable, have intersection
            # coe is the coefficient of two ray
            # coe[0] is the coef from line_start
            # coe[1] is the coef from start_pt
            int_coe = np.dot(np.linalg.inv(matrix), (start_pt - line_start))

            # Check the intersection point is on the line
            if 0 <= int_coe[0] <= 1:
                # Intersect pt is on the check line
                inter_pt = line_start + int_coe[0] * line_ray
                if int_coe[1] != 0:
                    # start_pt is not on the check line, ray and line is not parallel, have intersect point
                    # most normal state
                    pt_line_state = [False, False, False]
                else:
                    # start_pt is on the check line
                    pt_line_state = [True, False, False]
            else:
                # Intersect pt is NOT on the check line
                if int_coe[1] != 0:
                    # Intersect is NOT on the line, start is not on the line
                    inter_pt = None
                    pt_line_state = [False, False, False]
                else:
                    # Intersect is on the line, on the extend line, ray is not parallel
                    inter_pt = None
                    pt_line_state = [False, False, True]

        else:
            # Not Invertable, parallel but not on the line or extend part of line
            if 0 <= para_coe[0] <= 1:
                # Parallel and on the line
                inter_pt = None
                pt_line_state = [True, True, False]
            else:
                # Parallel but on the extend
                inter_pt = None
                pt_line_state = [False, True, True]

        return inter_pt, pt_line_state

    @staticmethod
    def sort_pt(pt_list: np.ndarray):
        '''

        :param pt_list:
        :return:
        '''
        # Check whether legal input
        assert pt_list.shape[1] == 2
        idx = pt_list[:, 1].argsort()
        pt_list = pt_list[idx]
        idx = idx[pt_list[:, 0].argsort(kind='mergesort')]
        pt_list = pt_list[idx]

        return pt_list, idx

    @staticmethod
    def rec_from_corner(coord: np.ndarray, long_direc: np.ndarray, short_direc: np.ndarray, long_dist: float,
                        short_dist: float):
        """
        Construct a rectangle from the corner of the rectangle
        :param coord: corner coordinate
        :param long_direc:
        :param short_direc:
        :param long_dist:
        :param short_dist:
        :return:
        """
        # Check the input is legal parameter
        assert coord.shape == long_direc.shape == short_direc.shape == (2,)

        # Construct a rectangle
        rec = np.zeros((4, 2))  # Only for rectangle building
        rec[0] = coord
        rec[1] = coord + long_dist * long_direc
        rec[2] = rec[1] + short_dist * short_direc
        rec[3] = rec[2] - long_dist * long_direc

        return rec

    @staticmethod
    def line_line_inter(line_1: np.ndarray, line_2: np.ndarray):
        """
        Check two line have intersection point or not. Hint: two line is finite line.
        :param line_1: Ndarray shape of (2,2) represent a line.
        :param line_2: Ndarray shape of (2,2) represent a line.
        :return: A (2, ) Ndarray if have intersection,
                 None if no intersection,
                 (2,2) Ndarray if two line in a line
        """
        # Check input is legal or not
        assert line_1.shape == line_2.shape == (2, 2)

        # inter_pt is the var to return
        inter_pt = None
        # Construct start pt and direction ray
        line_1_start = line_1[0]
        line_1_ray = line_1[1] - line_1[0]
        line_2_start = line_2[0]
        line_2_ray = line_2[1] - line_2[0]

        # Construct a matrix
        # | line_1_ray_X  -line_2_ray_X |
        # | line_1_ray_Y  -line_2_ray_Y |
        matrix = np.column_stack((line_1_ray, -line_2_ray))

        if np.linalg.det(matrix):
            # Invertable, have intersection, check intersection is in both two line or not
            # coe is the coefficient of two ray
            # coe[0] is the coef from line_start
            # coe[1] is the coef from start_pt
            int_coe = np.dot(np.linalg.inv(matrix), (line_2_start - line_1_start))
            if 0 <= int_coe[0] <= 1 and 0 <= int_coe[1] <= 1:
                inter_pt = line_1_start + int_coe[0] * line_1_ray

        else:
            # Compute The analitical of two line, namely " y = kx + b ",
            # if b is the same, check the if they have intersection
            k = np.array(np.diff(line_1, axis=0)[0, 0] / np.diff(line_1, axis=0)[0, 1],
                         np.diff(line_2, axis=0)[0, 0] / np.diff(line_2, axis=0)[0, 1])
            b = np.array([line_1[0, 1], line_2[1, 1]]) - k * np.array([line_1[0, 0], line_2[1, 0]])
            # Special case: k=0, vertical line; k=inf, horizon line
            # Check the slope, verify the intercept distance is a valid classify method.
            if k != np.inf:
                # The intercept distance is valid
                pass

            else:
                # The two line is vertical line, check the X axis
                pass

            if b[0] == b[1] and np.abs(k) != np.inf:
                # This case is for None vertical case, namely horizon and slash case
                # They are in line. But don't know if have intersection
                # Compare the Y axis, check line2 y axis value is in line_1 domain
                if np.min(line_1[:, 1]) <= line_2[0, 1] <= np.max(line_1[:, 1]) or np.min(line_1[:, 1]) <= line_2[
                    1, 1] <= np.max(line_1[:, 1]):
                    # Construct a Line
                    temp_line = np.vstack((line_1, line_2))
                    # Which dim sort start with
                    temp_line = temp_line[temp_line[:, 0].argsort()]
                    # The point in the middle is the intersection line
                    inter_pt = temp_line[1:3, :]
                    # Specific case, intersection is only a point
                    inter_pt = np.unique(inter_pt, axis=0)

            elif np.abs(k) == 0 and line_1[0, 0] == line_2[0, 0]:
                # The line is vertical line
                temp_line = np.vstack((line_1, line_2))
                temp_line = temp_line[temp_line[:, 1].argsort()]
                inter_pt = temp_line[1:3, :]
                inter_pt = np.unique(inter_pt, axis=0)

            elif np.abs(k) == np.inf and line_1[0, 1] == line_2[0, 1]:
                # The line is horizontal line
                temp_line = np.vstack((line_1, line_2))
                temp_line = temp_line[temp_line[:, 0].argsort()]
                inter_pt = temp_line[1:3, :]
                inter_pt = np.unique(inter_pt, axis=0)

        return inter_pt

    @staticmethod
    def explode_polyline(polygon: np.ndarray):
        '''
        Explode the polygon to single line
        :param polygon: a ndarray
        :return: line_array: shape of (n,2,2) array,
                 ray_direc: start from start pt, a ray direction, without normalize,
                 dist: distance of each line
        '''
        assert polygon.shape[1] == 2

        # Check whether it is a closed polygon, if not should add the first pt to the end
        # if polygon[0].all() != polygon[-1].all():
        #     polygon = np.append(polygon,polygon[0])

        if (polygon[0, :] != polygon[-1, :]).any():
            polygon = np.vstack((polygon, np.array([polygon[0]])))

        # Extract start point and end point
        start_pt = polygon[:-1, :]
        end_pt = polygon[1:, :]

        # Get ray direction and two pts distance
        ray_direc = end_pt - start_pt
        dist = np.linalg.norm(ray_direc, axis=1)

        # Struct the line with a (2,2) ndarray, if n lines construct a polygon, the ndarray should (n,2,2)
        line_array = []
        for i in range(ray_direc.shape[0]):
            tmp_line = np.stack((start_pt[i], end_pt[i]))
            line_array.append(tmp_line)

        line_array = np.array(line_array)

        return line_array, ray_direc, dist

    @staticmethod
    def bounding_box(points: np.ndarray) -> np.ndarray:
        '''
        Get the minimum area bounding box of the points, maybe some rotation, not only
        :param points: A ndarray with shape (n,2)
        :return: A ndarray with shape (4,2)
        '''

        # Get the convex points of the points list
        convex_pt = points[ConvexHull(points).vertices]

        # Calculate edge vec
        edges = convex_pt[1:] - convex_pt[:-1]

        # Calculate the unique angles, [-pi,pi]
        angles = np.arctan2(edges[:, 1], edges[:, 0])

        angles = np.abs(np.mod(angles, np.pi / 2))
        angles = np.unique(angles)

        rotations = np.vstack([
            np.cos(angles),
            np.cos(angles - np.pi / 2),
            np.cos(angles + np.pi / 2),
            np.cos(angles)]).T

        rotations = rotations.reshape((-1, 2, 2))

        rot_points = np.dot(rotations, convex_pt.T)

        # Find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        # Find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)

        # Return the best box
        x1 = max_x[best_idx]
        x2 = min_x[best_idx]
        y1 = max_y[best_idx]
        y2 = min_y[best_idx]
        r = rotations[best_idx]

        rval = np.zeros((4, 2))
        rval[0] = np.dot([x1, y2], r)
        rval[1] = np.dot([x2, y2], r)
        rval[2] = np.dot([x2, y1], r)
        rval[3] = np.dot([x1, y1], r)

        return rval

    @staticmethod
    def random_pt_inline(line: np.ndarray, random_seeds: int = 42, low_bnd: float = 0.3, upper_bnd: float = 0.7):
        '''
        Random choose a point in line, the scaler is between [low_bnd, upper_bnd], Normal distribution
        :param line: The line to be chose to random select a point
        :param random_seeds: Just random seed
        :param low_bnd: The scaler lower boundary
        :param upper_bnd: The scaler upper boundary
        :return: A point on line
        '''
        np.random.seed(random_seeds)
        a = np.random.rand(1)[0]
        # Scale the data into the boundary
        scaler = a - (a - (low_bnd + upper_bnd) / 2) * (upper_bnd - low_bnd)
        pt = line[0, :] + scaler * (line[1, :] - line[0, :])
        return pt

    def random_pt_longest_side(self, polygon: np.ndarray, random_seeds: int = 42, low_bnd: float = 0.3,
                               upper_bnd: float = 0.7):
        '''
        Random select a point on the longest side
        :param upper_bnd:
        :param low_bnd:
        :param random_seeds:
        :param polygon: A ndarray
        :return: The point array and index of point is on
        '''

        polygon = self.closed_polygon(polygon)
        line, _, line_dist = self.explode_polyline(polygon)
        idx = np.argmax(line_dist)
        pt = self.random_pt_inline(line[idx], random_seeds, low_bnd, upper_bnd)

        return pt, idx

    def point_in_polygon(self, pt: np.ndarray, polygon: np.ndarray, tolerance: float = 1e-6) -> bool:
        # https://en.wikipedia.org/wiki/Even%E2%80%93odd_rule
        # TODO add tolerance make it sense
        """
        Check the point is in the polygon or not
        :param pt: The Pt to test
        :param polygon: The Polygon to test
        :return: Bool, True for inside, False for outside
        """
        polygon = self.closed_polygon(polygon)
        num = polygon.shape[0]
        state: bool = False
        for i in range(num - 1):
            if (pt[0] == polygon[i, 0]) and (pt[1] == polygon[i, 1]):
                # pt is a corner
                state = True
                return state
            if (polygon[i, 1] > pt[1]) != (polygon[i + 1, 1] > pt[1]):
                # The two polygon pt is on the two side
                slope = (pt[0] - polygon[i, 0]) * (polygon[i + 1, 1] - polygon[i, 1]) - \
                        (polygon[i + 1, 0] - polygon[i, 0]) * (pt[1] - polygon[i][1])
                if slope == 0:
                    # pt is on boundary
                    state = True
                    return state
                if (slope < 0) != (polygon[i + 1, 1] < polygon[i, 1]):
                    state = not state

        return state

    def split_by_line(self, polygon: np.ndarray, split_pt: np.ndarray, seg_idx: int):
        """
        Split a Polygon start by a point and direction
        :param polygon: A (n,2) shape array, closed polygon
        :param split_pt: A (2,) shape array, refers a point
        :param seg_idx: the idx of segment that the point is on
        :return: a list of polygon array
        """

        # Assert legal input
        assert split_pt.shape == (2,)
        assert polygon.shape[1] == 2

        # Check the polygon is a closed polygon
        polygon = self.closed_polygon(polygon)
        line, line_direc, _ = self.explode_polyline(polygon)
        split_direc = self.vertical_ray(line_direc[seg_idx])

        # inter_pt is a intersection pt list, ex_line_idx is the index of line segment
        inter_pt = []
        ex_line_idx = []

        #  Loop over all the line segment, and get intersection
        for i in range(line.shape[0]):
            pt, state_list = self.line_2ray_inter(line[i], split_pt, split_direc)

            # Record pt and id if have a point
            if type(pt) == np.ndarray:
                inter_pt.append(pt)
                ex_line_idx.append(i)
        inter_pt = np.array(inter_pt)
        ex_line_idx = np.array(ex_line_idx)

        # clip_polygon is the polygon which is insert the pt
        clip_polygon = polygon
        clip_idx = ex_line_idx
        # Insert pt into Polygon
        for i in range(ex_line_idx.shape[0]):
            clip_polygon = np.insert(clip_polygon, clip_idx[i] + 1, inter_pt[i, :], axis=0)
            clip_idx += 1

        # Unique along axis 0, Prevent intersection is on the corner of Polygon
        # TODO Unique or not, That is a question
        # clip_polygon = np.unique(clip_polygon,axis=0)

        # Construct multiline, and check line in polygon or not
        inter_start_pt = inter_pt[:-1, :]
        inter_end_pt = inter_pt[1:, :]

        # inside_line_array Record the cross line, which is inside the Polygon
        inside_idx = []
        inside_line_array = []

        # Loop over check whether the split line is in polygon or not
        for i in range(inter_end_pt.shape[0]):
            tmp_line = np.stack((inter_start_pt[i], inter_end_pt[i]))
            tmp_mid_pt = inter_start_pt[i] + 0.5 * (inter_end_pt[i] - inter_start_pt[i])

            if self.point_in_polygon(tmp_mid_pt, polygon):
                # Record in POLYGON line and line idx
                inside_idx.append(i)
                inside_line_array.append(tmp_line)
        inside_line_array = np.array(inside_line_array)

        # Cliped polygon list
        rest_polygon_container = []
        tmp_polygon = clip_polygon

        # Start clip Polygon
        for i in range(inside_line_array.shape[0]):
            clip_start_pt = inside_line_array[i, 0, :]
            clip_end_pt = inside_line_array[i, 1, :]

            # Use a for loop, Not beautyful
            compared_vec = []
            for j in range(tmp_polygon.shape[0]):
                if (tmp_polygon[j] == clip_start_pt).all():
                    compared_vec.append(j)
                if (tmp_polygon[j] == clip_end_pt).all():
                    compared_vec.append(j)

            # Original flag generator Code
            # compared_vec = [np.argwhere(tmp_polygon == clip_start_pt)[0,0], np.argwhere(tmp_polygon == clip_end_pt)[0,0]]

            start_flag = np.min(compared_vec)
            end_flag = np.max(compared_vec)

            # Switch order of matrix
            rest_polygon_1 = tmp_polygon[start_flag:(end_flag + 1), :]
            rest_polygon_2 = np.vstack((tmp_polygon[end_flag:, :], tmp_polygon[:start_flag + 1, :]))

            # The Polygon still need clip
            if i < inside_line_array.shape[0] - 1:
                next_start = inside_line_array[i + 1, 0, :].tolist()
                next_end = inside_line_array[i + 1, 0, :].tolist()

                # Next clip is polygon_1
                if (next_start in rest_polygon_1.tolist()) and (next_end in rest_polygon_1.tolist()):
                    tmp_polygon = rest_polygon_1
                    rest_polygon_container.append(rest_polygon_2)

                # Next clip is polygon_2
                else:
                    tmp_polygon = rest_polygon_2
                    rest_polygon_container.append(rest_polygon_1)

            # The Polygon have no need to clip, return all
            else:
                rest_polygon_container.extend([rest_polygon_1, rest_polygon_2])

        return rest_polygon_container

    def sample_polygon_area(self, polygon: np.ndarray, divide_dist: float = 6, clear_not_inside: bool = True):
        # TODO bug is here
        """
        Divide a polygon by distance, split it in to point cloud
        :param polygon: Ndarray represent for a polygon
        :param divide_dist: Divide by length, not the real length of dist
        :param clear_not_inside: return list is clear the not in or fully return
        :return: coord: a list of list of 1d array;
                        u,v direction of the polygon(Normalized Direction),
                        uv_direc[0] refers to short direction
                        uv-direc[1] refers to long direction
                        bounding_box: the minimum bounding box of input polygon;
        """

        # Check whether input is legal
        assert polygon.shape[1] == 2
        # Get the minimum side bounding box of the polygon
        bounding_box = self.bounding_box(polygon)
        # Get 2 direction, distance of the bounding box
        direc_vec = np.array([bounding_box[1] - bounding_box[0], bounding_box[-1] - bounding_box[0]])
        print(f"direc_vec: {direc_vec}")
        direc_vec_dist = np.linalg.norm(direc_vec, axis=1)
        print(f"mesh_dist: {direc_vec_dist}")
        direc_vec_norm = direc_vec / np.tile(direc_vec_dist, (2, 1)).T
        print(f"direc_vec_norm: {direc_vec_norm}")
        # uv_direc = direc_vec_norm[direc_vec_dist.argsort()]
        uv_direc = direc_vec_norm
        # Get 2 direction mesh num and size
        mesh_num = np.round(direc_vec_dist / divide_dist)
        print(f"mesh_num:{mesh_num}")
        mesh_dist = direc_vec_dist / mesh_num
        uv_dist = mesh_dist
        mesh_point = []
        for i in range(int(mesh_num[1])):
            start_pt = bounding_box[0, :] + i * mesh_dist[1] * uv_direc[1]
            print(f"start_pt: {start_pt}")
            row_pt = []
            for j in range(int(mesh_num[0])):
                row_pt.append(start_pt + j * mesh_dist[0] * uv_direc[0])
            row_pt = np.array(row_pt)
            mesh_point.append(row_pt)
        coord = np.array(mesh_point)
        print(coord.shape)
        '''
        # Compute the first row point coordinate
        direc_1_sum = np.dot(np.tri(int(mesh_num[0]), int(mesh_num[0]), -1), np.ones((int(mesh_num[0]))) * mesh_dist[0])
        add_direc_1 = np.full((int(mesh_num[0]), 2), direc_vec_norm[0]) * np.tile(direc_1_sum, (2, 1)).T
        row_coord = add_direc_1 + bounding_box[0]

        # Compute the point on column coordinate
        direc_2_sum = np.dot(np.tri(int(mesh_num[1]), int(mesh_num[1]), -1), np.ones((int(mesh_num[1]))) * mesh_dist[1])
        add_direc_2 = np.full((int(mesh_num[1]), 2), direc_vec_norm[1]) * np.tile(direc_2_sum, (2, 1)).T

        # Add row and column
        row_coord_high = np.tile(row_coord, (int(mesh_num[1]), 1)).reshape((int(mesh_num[1]), int(mesh_num[0]), 2))
        add_direc_2 = np.tile(add_direc_2, (int(mesh_num[0]), 1)).reshape((int(mesh_num[1]), int(mesh_num[0]), 2))
        coord = row_coord_high + add_direc_2
        '''
        # If need to clear not in the polygon
        if clear_not_inside:
            row_list = []
            # Check every row
            for i in range(coord.shape[0]):
                col_list = []
                # Check every column
                for j in range(coord.shape[1]):
                    # Give bool
                    if self.point_in_polygon(coord[i, j, :], polygon):
                        col_list.append(coord[i, j, :])
                row_list.append(col_list)
            coord = row_list
        # Else no need to compute Bool with polygon
        else:
            coord = coord.tolist()

        return coord, uv_direc, uv_dist, bounding_box

    def pl_in_pl(self, subject_polygon: np.ndarray, object_polygon: np.ndarray) -> bool:
        """
        Check the object_polygon is inside subject_polygon or NOT
        :param subject_polygon: the larger polygon
        :param object_polygon: the smaller polygon
        :return: True for inside, False for outside
        """
        # Check the input is legal or not
        assert subject_polygon.shape[1] == 2
        assert object_polygon.shape[1] == 2

        # Check the polygon unique and closed
        subject_polygon = self.clear_redundant(subject_polygon)
        object_polygon = self.clear_redundant(object_polygon)
        subject_polygon = self.closed_polygon(subject_polygon)
        object_polygon = self.closed_polygon(object_polygon)

        # Explode the polygon
        subject_line, subject_direc, _ = self.explode_polyline(subject_polygon)
        object_line, object_direc, _ = self.explode_polyline(object_polygon)
        # Loop over every segment, check the intersection
        subj_idx = []
        inter_pt = []
        for i in range(subject_line.shape[0]):
            for j in range(object_line.shape[0]):
                # Compute the intersection
                interstate = self.line_line_inter(subject_line[i], object_line[j])
                if type(interstate) == np.ndarray:
                    # Have intersection
                    if len(interstate.shape) == 1:
                        # Two line doesn't parallel, but have inter
                        subj_idx.append(j)
                        inter_pt.append(interstate)
                    '''
                    else:
                        # Two line parallel, have inter, num is unknown
                        subj_idx.append(j)
                        inter_pt.append(interstate[0, :])
                        subj_idx.append(j)
                        inter_pt.append(interstate[1, :])
                    '''

        # Insert the intersection point
        inter_pt = np.array(inter_pt)
        subj_idx = np.array(subj_idx)
        # clip_polygon is the polygon which is insert the pt
        clip_polygon = object_polygon
        clip_idx = subj_idx

        # Insert pt into Polygon
        inter_pt = inter_pt[clip_idx.argsort()]
        clip_idx = np.sort(clip_idx)

        for i in range(len(subj_idx)):
            clip_polygon = np.insert(clip_polygon, clip_idx[i] + 1, inter_pt[i, :], axis=0)
            clip_idx += 1
        clip_polygon = self.clear_redundant(clip_polygon)
        # Clip polygon segment line middle point is in subject polygon

        # While False, return False
        # Close the clip polygon
        clip_polygon = self.closed_polygon(clip_polygon)
        clip_line, clip_line_direc, _ = self.explode_polyline(clip_polygon)
        # Get the mid point of each segment of polygon
        clip_mid_pt = clip_polygon[:-1, :] + 0.5 * clip_line_direc
        # Loop over the mid point
        for mid_pt in clip_mid_pt:
            if not self.point_in_polygon(mid_pt, subject_polygon):
                return False

        return True

    def two_pl_intersec(self, subject_polygon: np.ndarray, object_polygon: np.ndarray) -> bool:
        """
        Check the object_polygon is inside subject_polygon or NOT
        :param subject_polygon: the larger polygon
        :param object_polygon: the smaller polygon
        :return: True for inside, False for outside
        """
        # Check the input is legal or not
        assert subject_polygon.shape[1] == 2
        assert object_polygon.shape[1] == 2

        # Check the polygon unique and closed
        subject_polygon = self.clear_redundant(subject_polygon)
        object_polygon = self.clear_redundant(object_polygon)
        subject_polygon = self.closed_polygon(subject_polygon)
        object_polygon = self.closed_polygon(object_polygon)

        # Explode the polygon
        subject_line, subject_direc, _ = self.explode_polyline(subject_polygon)
        object_line, object_direc, _ = self.explode_polyline(object_polygon)
        # Loop over every segment, check the intersection
        subj_idx = []
        inter_pt = []
        for i in range(subject_line.shape[0]):
            for j in range(object_line.shape[0]):
                # Compute the intersection
                interstate = self.line_line_inter(subject_line[i], object_line[j])
                if type(interstate) == np.ndarray:
                    # Have intersection
                    return True

        for pt_1 in subject_polygon:
            if GeoProcess().point_in_polygon(pt_1, object_polygon):
                return True
        for pt_2 in object_polygon:
            if GeoProcess().point_in_polygon(pt_2, subject_polygon):
                return True

        return False


if __name__ == "__main__":
    from FileIO.plotData import plot_polygon
    import matplotlib.pyplot as plt
    from FileIO.readRhino import Read3dm

    '''
        path = "D:\\LiGD\\Indus_Park\\Test_File\\test_polyline.3dm"
        model = Read3dm(path)
        rhino_model = model.read_Rhino()
        model_dict, model_dictkey = model.layer_Dict()
        pt_dict = model.transform_Point(model_dict, model_dictkey)
        plan_area_3d = pt_dict["Planning_Area"][0]
        plan_area = GeoProcess.flatten_Z(plan_area_3d)
        print(plan_area)
        pt_long, line_idx = GeoProcess().random_pt_longest_side(plan_area, random_seeds=50)
        print(pt_long, line_idx)
        # Clip it
        polygon_contain = GeoProcess().split_by_line(plan_area, pt_long, line_idx)
        print(polygon_contain)
        for i in range(len(polygon_contain)):
            polygon_contain[i] = GeoProcess.closed_polygon(polygon_contain[i])
        test_pl = polygon_contain[-1]
    '''

    test_pl = np.array([[124.38041437, -6.16403083], [166.02889109, 31.86865627], [228.5420024, 73.78844845],
                        [174.25864686, 147.09974953], [68.9275611, 69.18657163]])

    a, b, c, d = GeoProcess().sample_polygon_area(test_pl, divide_dist=6, clear_not_inside=True)
    print(f"Minimum box: {d}")

    x = []
    y = []
    for i in a:
        for j in i:
            x.append(j[0])
            y.append(j[1])

    plt.scatter(x, y)

    plt.plot(GeoProcess.closed_polygon(test_pl)[:, 0], GeoProcess.closed_polygon(test_pl)[:, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    '''
        ======================================Line Line Intersection=================================================
    print(
        f"inter_pt: {GeoProcess.line_line_inter(np.array([[2, -1], [-1, -1]]), np.array([[0, -1], [-1, 0]]))}, Except: 1D Array")
    print("===========================================")
    print(
        f"inter_pt: {GeoProcess.line_line_inter(np.array([[0, 2], [0, -1]]), np.array([[0, 0], [1, 0]]))}, Except: 1D Array")
    print("===========================================")
    print(
        f"inter_pt: {GeoProcess.line_line_inter(np.array([[2, -1], [-1, -1]]), np.array([[0, 0], [1, 0]]))}, Except: None")
    print("===========================================")
    print(
        f"inter_pt: {GeoProcess.line_line_inter(np.array([[2, 0], [-1, 0]]), np.array([[0, 0], [1, 0]]))}, Except: (2,2) Array")
    print("===========================================")
    print(
        f"inter_pt: {GeoProcess.line_line_inter(np.array([[2, 0], [-1, 0]]), np.array([[4, 0], [2, 0]]))}, Except: (2,1) Array")
    print("===========================================")
    print(
        f"inter_pt: {GeoProcess.line_line_inter(np.array([[0, 4], [0, -2]]), np.array([[1, 2], [1, -1]]))}, Except: None")
    print("===========================================")
    print(
        f"inter_pt: {GeoProcess.line_line_inter(np.array([[0, 4], [0, -2]]), np.array([[0, 4], [0, 2]]))}, Except: (2,2) Array")
    print("===========================================")
    print(
        f"inter_pt: {GeoProcess.line_line_inter(np.array([[0, 4], [0, -2]]), np.array([[0, 4], [0, 5]]))}, Except: (2,1) Array")
    print("===========================================")
    print(
        f"inter_pt: {GeoProcess.line_line_inter(np.array([[4, 4], [-2, -2]]), np.array([[3, 3], [5, 5]]))}, Except: (2,2) Array")
    print("===========================================")
    print(f"inter_pt: {GeoProcess.line_line_inter(np.array([[1, 1], [0, 0]]), np.array([[0, 1], [1, 0]]))}")
    print("===========================================")
    print(f"inter_pt: {GeoProcess.line_line_inter(np.array([[1, 1], [1, 0]]), np.array([[2, 2], [2, 0]]))}")
    '''

    '''
    object_polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    subject_polygon = np.array([[-1, -1], [-1, 2], [2, 2], [2, -1]])

    object_polygon1 = np.array([[1, 0], [0, -1], [-1, 0], [0, 1]])
    subject_polygon1 = np.array([[-1, -1], [-1, 2], [2, 2], [2, -1]])

    object_polygon2 = np.array([[1, 1], [2, 2], [1, 3], [0, 2]])
    subject_polygon2 = np.array([[1, 0], [4, 4], [1, 7], [-3, 4]])

    object_polygon3 = np.array([[10, 10], [20, 20], [10, 30], [0, 20]])
    subject_polygon3 = np.array([[1, 0], [4, 4], [1, 7], [-3, 4]])

    print(f"In polygon? {GeoProcess().pl_in_pl(subject_polygon, object_polygon)}")
    print("=====================================================================")
    print(f"In polygon? {GeoProcess().pl_in_pl(subject_polygon1, object_polygon1)}")
    print("=====================================================================")
    print(f"In polygon? {GeoProcess().pl_in_pl(subject_polygon2, object_polygon2)}")
    print("=====================================================================")
    print(f"In polygon? {GeoProcess().pl_in_pl(subject_polygon3, object_polygon3)}")
    print("=====================================================================")
    # plot_polygon([GeoProcess.closed_polygon(object_polygon2),GeoProcess.closed_polygon(subject_polygon2)])

    # offset bug
    print(GeoProcess.offset(np.array([[-76.15498819, 170.40033424],
                                      [-149.73792199, 131.36868205],
                                      [-200.90591639, 227.83119219],
                                      [-127.32298259, 266.86284437],
                                      [-76.15498819, 170.40033424]]), 7, 1))
    '''
