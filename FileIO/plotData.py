import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from GeoAlgorithm.GeoProcess import GeoProcess
from descartes.patch import PolygonPatch
import numpy as np


def plot_polygon(plg_list: list):
    fig = plt.figure(1, figsize=(16, 9), dpi=300)
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)

    # plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')

    for pl in plg_list:
        x = pl[:, 0]
        y = pl[:, 1]
        ax.plot(x, y, color="#8c00ff", linewidth=0.5, ls='-', solid_capstyle='projecting', zorder=20)

    plt.show()


def plot_master(red_line: list, road_line: list, building_line: list):
    fig = plt.figure(1, figsize=(16, 9), dpi=300)
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)

    # plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    for pl in red_line:
        x = pl[:, 0]
        y = pl[:, 1]
        ax.plot(x, y, color="#DC143C", linewidth=0.5, ls='-.', solid_capstyle='projecting', zorder=20)

    for pl in road_line:
        x = pl[:, 0]
        y = pl[:, 1]
        ax.plot(x, y, color="#4682B4", linewidth=0.5, ls='-', solid_capstyle='projecting', zorder=20)

    for pl in building_line:
        x = pl[:, 0]
        y = pl[:, 1]
        ax.plot(x, y, color="#4682B4", linewidth=1, ls='-', solid_capstyle='projecting', zorder=20)

    plt.show()


def plot_area(satisfied_area: list, uns_area: list):
    """
    Split the area
    :param satisfied_area:
    :param uns_area:
    :return:
    """
    fig = plt.figure(1, figsize=(30, 20), dpi=200)
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    if len(satisfied_area) != 0:
        for sat_area in satisfied_area:
            sat_area = GeoProcess.closed_polygon(sat_area)
            ax.fill(sat_area[:, 0], sat_area[:, 1], linewidth=1, ls='-', edgecolor="#8c00FF",
                    facecolor="#E6E6FA", zorder=1)

    if len(uns_area) != 0:
        for uns_area in uns_area:
            uns_area = GeoProcess.closed_polygon(uns_area)

            ax.fill(uns_area[:, 0], uns_area[:, 1], linewidth=1, ls='-', edgecolor="#DC143C",
                    facecolor="#DCDCDC", zorder=1)

    plt.show()


def scatter(land_pl, building_list, pt_list):
    fig = plt.figure(1, figsize=(30, 20), dpi=200)
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    land_pl = GeoProcess.closed_polygon(land_pl)
    ax.fill(land_pl[:, 0], land_pl[:, 1], linewidth=1, ls='-', edgecolor="#DC143C",
            facecolor="#DCDCDC", zorder=1)
    for building in building_list:
        building = GeoProcess.closed_polygon(building)

        ax.fill(building[:, 0], building[:, 1], linewidth=1, ls='-', edgecolor="#8c00FF",
                facecolor="#E6E6FA", zorder=1)
    plt.scatter(pt_list[:, 0], pt_list[:, 1], c="#8c00ff")
    plt.show()


if __name__ == "__main__":
    pass
