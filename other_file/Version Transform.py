import rhino3dm as r3d
import numpy as np

path_2 = "C:\\Users\\xkool1\\Downloads\\example_file_ae948f79-e57a-11ec-a7eb-00155d94159c (1).3dm"
path_3 = "C:\\Users\\xkool1\\Downloads\\example_file_ae948f79-e57a-11ec-a7eb-00155d94159c_6.3dm"
r3d.File3dm.Write(r3d.File3dm.Read(path_2),path_3,6)

