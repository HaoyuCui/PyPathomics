import pandas as pd
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union

# do not show warnings
import warnings
warnings.filterwarnings("ignore")


def get_triangle_area(p1, p2, p3):
    # calculate the area of a Delaunay triangle
    return 0.5 * np.abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def get_triangle_perimeter(p1, p2, p3):
    # calculate the perimeter of a triangle
    return np.linalg.norm(p1 - p2) + np.linalg.norm(p2 - p3) + np.linalg.norm(p3 - p1)


def get_triangle_angle(p1, p2, p3):
    # calculate the angles of a triangle
    a = np.linalg.norm(p1 - p2)
    b = np.linalg.norm(p2 - p3)
    c = np.linalg.norm(p3 - p1)
    return np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)), \
        np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)), \
        np.arccos((c ** 2 + a ** 2 - b ** 2) / (2 * c * a))


def get_triangle_feature(triangle):
    # calculate the area, perimeter and angles of a triangle
    p1, p2, p3 = triangle
    area = get_triangle_area(p1, p2, p3)
    perimeter = get_triangle_perimeter(p1, p2, p3)
    angle1, angle2, angle3 = get_triangle_angle(p1, p2, p3)
    return area, perimeter, angle1, angle2, angle3


def get_polygon_edges(triangles, coords):
    triangles_polygons = [Polygon([coords[index] for index in triangle]) for triangle in triangles]
    polygon = unary_union(triangles_polygons)
    return polygon


# def get_polygon_edges(triangles, coords):
#     edges = set()
#     for triangle in triangles:
#         for idx in range(3):
#             edge = (triangle[idx], triangle[(idx + 1) % 3])
#             if edge[0] > edge[1]:
#                 edge = (edge[1], edge[0])
#
#             if edge not in edges:
#                 edges.add(edge)
#             else:
#                 edges.remove(edge)
#
#     polygon_edges = []
#     for edge in edges:
#         p1 = tuple(coords[edge[0]])
#         p2 = tuple(coords[edge[1]])
#         polygon_edges.append(p1)
#         polygon_edges.append(p2)
#
#     return polygon_edges


def get_triangle_feature_df(triangles, coords, threshold=3000):
    # calculate the area, perimeter and angles of all triangles
    area_list = []
    perimeter_list = []
    angle_range_list = []
    valid_triangles = []  # 用于存储符合条件的三角形

    for triangle in tqdm(triangles):
        triangle_cor = coords[triangle]
        area, perimeter, angle1, angle2, angle3 = get_triangle_feature(triangle_cor)

        if perimeter <= threshold:
            valid_triangles.append(triangle)
            area_list.append(area)
            perimeter_list.append(perimeter)
            angle_range = max(angle1, angle2, angle3) - min(angle1, angle2, angle3)
            angle_range_list.append(angle_range)

    polygons = get_polygon_edges(valid_triangles, coords)
    # polygons = Polygon(polygons).buffer(1e-3)

    # fig, ax = plt.subplots()
    # for polygon in polygons:
    #     x, y = polygon.exterior.xy
    #     ax.plot(x, y)  # 绘制多边形的外轮廓
    #     ax.fill(x, y, alpha=0.5)
    #     # 绘制多边形内部的孔（如果有的话）
    #     for interior in polygon.interiors:
    #         x, y = interior.coords.xy
    #         ax.plot(x, y, 'r')
    #         ax.fill(x, y, 'r', alpha=0.5)
    #
    # plt.show()

    return pd.DataFrame({'Triangle_Area': area_list, 'Triangle_Perimeter': perimeter_list,
                         'Triangle_Angle_Range': angle_range_list}), polygons






