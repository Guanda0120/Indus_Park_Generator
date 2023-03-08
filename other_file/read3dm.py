import rhino3dm as r3d
import numpy as np
import time
import matplotlib.pyplot as plt


class Axis:

    def __init__(self,col_cen_pt:np.ndarray,main_side:np.ndarray):
        self.col_cen_pt = col_cen_pt
        self.main_side = main_side

        assert self.col_cen_pt.shape[1]==self.main_side.shape[0]==2

        # sort col_cen_list
        if main_side[0]>=main_side[1]:
            self.col_cen_pt = self.col_cen_pt[self.col_cen_pt[:,0].argsort()]
            self.col_end2end = np.asarray([self.col_cen_pt[0,:],self.col_cen_pt[-1,:]])

        else:
            self.col_cen_pt = self.col_cen_pt[self.col_cen_pt[:, 1].argsort()]
            self.col_end2end = np.asarray([self.col_cen_pt[0, :], self.col_cen_pt[-1, :]])


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


class ColumnProcessing():

    def __init__(self, col_list:list, bnd_list:list):
        self.col_list = col_list
        self.bnd_list = bnd_list

    def dis2Point(pt1: np.ndarray, pt2: np.ndarray):
        return np.linalg.norm(pt1 - pt2)


    def main_Side(self, epsilon=0.05):

        '''
        Use self.bnd_list compute the main direction of the architecture
        :param epsilon: tolerance of some shift
        :return:main_vec: a ndarray shape of (n,2)
                side_vec_dict: a dictionary with key: 0 to n-1
        '''

        # Flatten the point, Eliminate Z axis
        main_side = self.bnd_list[0]
        main_side = main_side[:,:2]

        # Compute side vector
        main_side_1 = np.vstack((main_side[-1,:],main_side[:-1,:]))
        side_vec = main_side_1-main_side
        side_vec = np.divide(side_vec,np.tile(np.linalg.norm(side_vec,axis=1),(2,1)).T)  # Normalize vector

        # Angle of each vector cos, Triangle the Matrix
        cosine = np.abs(np.dot(side_vec,side_vec.T))
        cosine = np.tril(cosine,k=0)

        # Cosine Bool
        tolerance_cos = 1-epsilon
        cosine[cosine <= tolerance_cos] = 0
        cosine[cosine > tolerance_cos] = 1

        # Main vector index
        bool_cosine = np.sum(cosine,axis=1)
        main_vec_idx = np.where(bool_cosine==1)
        main_vec = side_vec[main_vec_idx,:][0]
        '''
        Unknown Bug Above!
        side_vec[main_vec_idx,:] has three axis
        Solu: add[0]
        '''

        # Classify all the side vector to main vector
        side_vec_dict = {}
        for i in range(main_vec.shape[0]):

            # Compute the angle between i^{th} main vec and all the vector
            angle = np.abs(np.dot(side_vec,main_vec[i]))
            same_idx = np.where(angle>tolerance_cos)
            side_vec_dict[i] = side_vec[same_idx,:][0]


        return main_vec,side_vec_dict


    def column_Porcess(self):

        col_ptlist = self.col_list
        # Read the main side of the building
        main_side_all,_ = self.main_Side(epsilon=0.05)

        # Get center point of each column
        cen_pt = []
        for temp_col in col_ptlist:
            temp_col = temp_col[:,:-1]
            temp_cen = np.sum(temp_col,axis=0)/temp_col.shape[0]
            cen_pt.append(temp_cen)
        cen_pt = np.asarray(cen_pt)

        # Mean Distance
        minus_cen = np.tile(cen_pt,(1,cen_pt.shape[0]))
        minus_cen = minus_cen.reshape(minus_cen.shape[0],minus_cen.shape[0],2)
        cen_vec = minus_cen-cen_pt
        cen_dis = np.linalg.norm(cen_vec,axis=2)

        # cos is a list contain every every main side cos angle
        cos = []
        main_dict = {}
        for i in range(main_side_all.shape[0]):

            # Var temp_main_side reshape the main side[i]{2 dimensional vec} to {num_pt,num_pt,2}dimensional tensor
            temp_main_side = np.tile(main_side_all[i], (cen_vec.shape[0],cen_vec.shape[1],1))
            # Compute cos and normalize and absolute
            temp_cos = np.sum(np.multiply(temp_main_side,cen_vec),axis=2)
            temp_cos = np.abs(np.divide(temp_cos,cen_dis))
            # Make Nan and Inf 0
            temp_cos[np.isnan(temp_cos)]=0
            temp_cos[np.isinf(temp_cos)] = 0
            cos.append(temp_cos)

            # Eliminate upper triangle matrix
            temp_cos = np.tril(temp_cos,-1)
            # Get 0-1 matrix
            cos_trig = temp_cos
            # TODO 1e-6
            cos_trig[cos_trig < 1-0.01] = 0
            cos_trig[cos_trig >= 1-0.02] = 1

            # Var idx is the list to loop, sum in column and delete in idx, forbide to loop
            idx = range(cen_dis.shape[0])
            idx = np.delete(idx,np.sum(cos_trig,axis=0)==0)


            # Var same_vec_pt_list contain the pt on same vec
            axis_vec_list = []

            # Make follow alg count the pt itself
            for j in idx:
                cos_trig[j, j] = 1

            while idx.shape[0] != 0:

                # Var same_idx is which pt is in line with the idx[0]
                same_idx = np.argwhere(cos_trig[:,idx[0]]==1)
                same_idx = np.reshape(same_idx,(same_idx.shape[0]))
                temp_pt_idx_list = list(same_idx)

                # Ideal situation will not append new, but it have some tolerance
                for list_num in same_idx:
                    sub_same_idx = np.argwhere(cos_trig[:,list_num]==1)
                    sub_same_idx = np.reshape(sub_same_idx, (sub_same_idx.shape[0]))
                    for j in sub_same_idx: temp_pt_idx_list.append(j)

                temp_pt_idx_list = list(np.unique(temp_pt_idx_list))

                temp_pt_list=[]
                for pt_idx in temp_pt_idx_list: temp_pt_list.append(cen_pt[pt_idx,:])
                temp_pt_list=np.asarray(temp_pt_list)
                temp_axis = Axis(temp_pt_list,main_side_all[i])

                axis_vec_list.append(temp_axis)

                for del_idx in temp_pt_idx_list:
                    idx = np.delete(idx,idx==del_idx)

            main_dict[str(i)] = axis_vec_list


        return cen_pt,main_dict


if __name__ == "__main__":

    # main below
    start_time = time.time()

    path = "D:\\LiGD\\ApartmentAlgorithim\\new.3dm"
    apartment_model = Read3dm(path)
    dict, dict_key = apartment_model.layer_Dict()
    print(f"main layer:{dict_key}")
    ptarray_dict = apartment_model.transform_Point(dict, dict_key)

    col_process = ColumnProcessing(ptarray_dict['XKOOL柱子'], ptarray_dict['XKOOL布局区域'])
    cen_pt,axis_dict = col_process.column_Porcess()
    print(axis_dict["0"][0].col_end2end)

    # axis.generate_Axis(main_side)
    end_time = time.time()

    # plot data
    fig = plt.figure(1, figsize=(30, 20), dpi=200)
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)

    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')

    for key in axis_dict.keys():
        for axis in axis_dict[key]:
            x= axis.col_end2end[:,0]
            y= axis.col_end2end[:,1]
            ax.plot(x, y, color="#9a1515", linewidth=1, ls='-.', solid_capstyle='projecting', zorder=20)

    plt.plot(cen_pt[:,0],cen_pt[:,1],'o',color='b')

    plt.show()
    plt.savefig("test.png")





    print(f"run time{end_time - start_time}")








