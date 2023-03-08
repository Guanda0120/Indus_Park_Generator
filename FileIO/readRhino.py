import rhino3dm as r3d
import numpy as np

class Read3dm():
    '''
    This module is designed to read rhino object to numpy。ndarray
    '''

    def __init__(self,path:str):
        self.path = path


    def read_Rhino(self):
        '''
        Read Rhino File
        :return: rhino Object
        '''
        model = r3d.File3dm.Read(self.path)
        return model

    def layer_Dict(self):
        '''
        Create a dictionary.
        The Keys is the layer name
        The value is a list of Rhino Object
        :return: the Dictionary and the Keys
        '''
        model = r3d.File3dm.Read(self.path)
        layer_obj = model.Layers

        # Creat a Dictionary match the layer name and object
        model_dict = {}
        for lay in layer_obj:
            model_dict[lay.Name] = []

        # Match the object to the dictionary
        model_objs = model.Objects
        for obj in model_objs:
            lay_idx = obj.Attributes.LayerIndex
            dict_key = layer_obj[lay_idx].Name
            model_dict[dict_key].append(obj)

        return model_dict,model_dict.keys()

    def transform_Point(self,model_dict:dict, model_keys:list):

        '''
        Turn Rhino Object [Line, Polyline, line like & polyline like curve]
        Into list of point of np.ndarray
        :param model_dict: func：layer_Dict >- dictionary
        :param model_keys: func: layer_Dict >- dictionary keys
        :return: A dictionary: key is the layer
                                object is a list of np.ndarray
        '''

        # Create a blank dictionary
        model_ptarray_dict = {}

        for lay in model_keys:
            model_ptarray_dict[lay] = []

        for key in model_keys:
            temp_obj = model_dict[key]

            for obj in temp_obj:

                # Check if the obj is Line obj
                # Append a ndarray to model_ptarray_dict[key]
                if obj.Geometry.ObjectType == r3d.Line:
                    line = obj.Geometry
                    pt_list = np.asarray([[line.From.X,line.From.Y,line.From.Z],[line.To.X,line.To.Y,line.To.Z]])
                    model_ptarray_dict[key].append(pt_list)

                # Check if the obj is Polyline obj
                # Append a ndarray to model_ptarray_dict[key]
                if obj.Geometry.ObjectType == r3d.Polyline:
                    polyline = obj.Geometry
                    order = polyline.SegmentCount
                    pt_list = []
                    for i in range(order):
                        pt = polyline.PointAt(i)
                        temp_pt = np.asarray([pt.X,pt.Y,pt.Z])
                        pt_list.append(temp_pt)
                    pt_list = np.asarray(pt_list)
                    model_ptarray_dict[key].append(pt_list)

                # Check if the obj is Curve obj
                # Append a ndarray to model_ptarray_dict[key]
                # Try curve could transform into polyline
                if (obj.Geometry.ObjectType == r3d.ObjectType.Curve) and (obj.Geometry.TryGetPolyline()!=False):
                    polyline_C = obj.Geometry.TryGetPolyline()
                    order = polyline_C.SegmentCount
                    pt_list = []
                    for i in range(order):
                        pt = polyline_C.PointAt(i)
                        temp_pt = np.asarray([pt.X, pt.Y, pt.Z])
                        pt_list.append(temp_pt)
                    pt_list = np.asarray(pt_list)
                    model_ptarray_dict[key].append(pt_list)

        return model_ptarray_dict


if __name__ == "__main__":
    path ="D:\\LiGD\\Indus_Park\\Test_File\\test_polyline.3dm"
    model = Read3dm(path)
    rhino_model = model.read_Rhino()
    model_dict, model_dictkey = model.layer_Dict()
    print(model.transform_Point(model_dict, model_dictkey))