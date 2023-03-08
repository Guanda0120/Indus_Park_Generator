import numpy as np
from PlanObject.VirtualProduct import VirtualProduct
from GeoAlgorithm.GeoProcess import GeoProcess
import shapely.geometry as shape_geo


class Building:
    '''

    '''

    def __init__(self, polygon: np.ndarray, virtual_product: VirtualProduct):
        """
        Init the building obj
        :param polygon: Ndarray represent polygon
        :param virtual_product:
        """
        self.polygon = polygon
        self.ground_area = virtual_product.floor_area
        self.total_area = virtual_product.total_area
        self.floor_num = virtual_product.floor_num
        self.floor_height = virtual_product.floor_height
        self.detail = None

    def generate_horizon_facade(self, roof_thick: float, glass_offset: float):

        """
        Generate the facade info list
        :return: return a dict with key polygon, z_coord, height
        """

        assert roof_thick <= 1.2
        assert glass_offset <= 3
        geo_dict = {
            "polygon": [],
            "Z_coord": [],
            "height": []
        }

        roof_polygon = self.polygon
        glass_polygon = GeoProcess.offset(roof_polygon, glass_offset, 1)
        glass_z_coord = np.dot(np.tri(self.floor_num, self.floor_num, k=-1),
                               np.ones(self.floor_num) * self.floor_height)
        glass_height = self.floor_height - roof_thick
        roof_z_coord = glass_z_coord+glass_height

        geo_dict["polygon"].append(glass_polygon)
        geo_dict["polygon"].append(roof_polygon)
        geo_dict["Z_coord"].append(glass_z_coord)
        geo_dict["Z_coord"].append(roof_z_coord)
        geo_dict["height"].append(glass_height)
        geo_dict["height"].append(roof_thick)
        self.detail = geo_dict


if __name__ == "__main__":
    from CONFIG import Para

    product_1 = Building(10000, 2000)
    product_1.locate_building(np.asarray([1000, 4000]), np.asarray([1, 0]), np.asarray([0, 1]))
    print(f"长边长度{product_1.long_len}")
    print(f"短边长度{product_1.short_len}")
    print(f"几何形体{product_1.building_polygon}")
