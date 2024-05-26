import logging

import yaml
import pandas as pd
from tqdm import tqdm
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

# do not show warnings
import warnings
warnings.filterwarnings("ignore")


# for config reading
def read_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {file_path}")
        raise
    except yaml.YAMLError as exc:
        logging.error(f"Error parsing the YAML file: {file_path}, "
                      f"make sure the feature-set & cell-types are <list> type")
        raise


def get_config():
    return read_yaml('../config.yaml')


# for triangle features
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


def get_triangle_feature_df(triangles, coords, threshold=3000):
    # calculate the area, perimeter and angles of all triangles
    area_list = []
    perimeter_list = []
    angle_range_list = []
    valid_triangles = []

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

    return pd.DataFrame({'Triangle_Area': area_list, 'Triangle_Perimeter': perimeter_list,
                         'Triangle_Angle_Range': angle_range_list}), polygons
