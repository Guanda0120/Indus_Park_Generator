import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from GeoAlgorithm.GeoProcess import GeoProcess
from FileIO.plotData import plot_polygon


def obj_func(x: np.ndarray):
    return np.sin(x) + np.cos(x)


def obj_func_1(x: np.ndarray):
    return np.abs(np.sin(x)) + np.abs(np.cos(x))

print()

'''
plot_polygon([GeoProcess.closed_polygon(polygon_1), GeoProcess.closed_polygon(polygon_2)])
x = np.arange(0, 5 * np.pi, 0.01)
y = obj_func(x)
y_1 = obj_func_1(x)
y_2 = y_1 ** 2
y_4 = y_2 ** 2
y_8 = y_4 ** 2
plt.plot(x, y)
plt.plot(x, y_1)
plt.plot(x, y_2)
plt.plot(x, y_4)
plt.plot(x, y_8)
plt.show()
'''