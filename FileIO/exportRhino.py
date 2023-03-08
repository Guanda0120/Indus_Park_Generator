import rhino3dm
import numpy as np
from PlanObject.PlanArea import PlanArea
from GeoAlgorithm.GeoProcess import GeoProcess


class ExportRhino:
    """
    Rhino export Module
    """

    def __init__(self, plan_area: PlanArea, output_dir, version: int):
        self.plan_area = plan_area
        self.output_dir = output_dir
        self.version = version
        self.file = rhino3dm.File3dm()

    @staticmethod
    def array2rhino_pl(polygon: np.ndarray, z_coord: float):
        """
        Convert the ndarray polygon to rhino polygon
        :param polygon: Ndarray represent polygon
        :param z_coord: Z Axis value
        :return: r3d.Curve
        """
        polygon = GeoProcess.closed_polygon(polygon)
        point_list = []
        for i in range(polygon.shape[0]):
            point_list.append(rhino3dm.Point3d(polygon[i, 0], polygon[i, 1], z_coord))
        polyline_rh = rhino3dm.Polyline(point_list)

        return rhino3dm.Curve.CreateControlPointCurve(polyline_rh, degree=1)

    @staticmethod
    def polyline2_brep(polyline_rhino: rhino3dm.Curve, height: float):
        """
        Extrude and convert to Rhino Brep
        :param polyline_rhino: r3d.Curve
        :param height: Extrude Height
        :return: Rhino Brep
        """
        return rhino3dm.Extrusion.Create(polyline_rhino, height, True).ToBrep(True)

    def pt_sphere(self, radius: float):
        sphere_list = []
        for pt in self.plan_area.tree_pt:
            center = rhino3dm.Point3d(pt[0], pt[1], radius)
            sphere_list.append(rhino3dm.Sphere(center, radius).ToBrep())
        return sphere_list

    def export_rhino(self):

        # Construct the layer info
        layer_name = ["Roof", "Glass", "Ground", "Grass"]
        layer_color = [(173, 216, 230, 255), (46, 139, 87, 255), (169, 169, 169, 255), (0, 100, 0, 255)]

        for idx in range(len(layer_name)):
            cur_layer = rhino3dm.Layer()
            cur_layer.Name = layer_name[idx]
            cur_layer.Visible = True
            cur_layer.Color = layer_color[idx]
            self.file.Layers.Add(cur_layer)

        # Get layer index
        layer_index = [each.Index for each in self.file.Layers]

        # Construct Building Object
        id_list = []
        layer_id = []
        for sub_area in self.plan_area.subarea_list:
            if len(sub_area.building_container) != 0:
                for tmp_building in sub_area.building_container:

                    polygon = tmp_building.detail["polygon"]
                    z_coord = tmp_building.detail["Z_coord"]
                    height = tmp_building.detail["height"]

                    for z_axis in z_coord[0]:
                        # Generate Glass
                        tmp_brep = self.polyline2_brep(self.array2rhino_pl(polygon[0], z_axis), height[0])
                        id = self.file.Objects.AddBrep(tmp_brep)
                        id_list.append(id)
                        layer_id.append(1)

                    for z_axis in z_coord[1]:
                        # Generate Glass
                        tmp_brep = self.polyline2_brep(self.array2rhino_pl(polygon[1], z_axis), height[1])
                        id = self.file.Objects.AddBrep(tmp_brep)
                        id_list.append(id)
                        layer_id.append(0)

            tmp_brep = self.polyline2_brep(self.array2rhino_pl(sub_area.road_side_line, -1), 1)
            id = self.file.Objects.AddBrep(tmp_brep)
            id_list.append(id)
            layer_id.append(2)

        # Add Tree
        tree_list = self.pt_sphere(radius=3)
        for tree_brep in tree_list:
            id = self.file.Objects.AddBrep(tree_brep)
            id_list.append(id)
            layer_id.append(3)

        # Assign layer info to object
        for id in range(len(id_list)):
            cur_object = self.file.Objects.FindId(str(id_list[id]))
            cur_object.Attributes.LayerIndex = layer_index[layer_id[id]]

        self.file.Write(self.output_dir, self.version)


if __name__ == "__main__":
    array = np.array([[1, 0], [1, 1], [0, 1]])
    a = ExportRhino.array2rhino_pl(array, 7)
    print(ExportRhino.polyline2_brep(a, 4))
    layer_color = [(173, 216, 230), (46, 139, 87), (169, 169, 169), (0, 100, 0)]
    print(layer_color[3])
