import logging
import os.path

import yaml
import pandas as pd
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def print_config(args, configs):
    logging.info("Arguments:")
    for arg in vars(args):
        print(f' - {arg}: {getattr(args, arg)}')
    logging.info("Configuration:")
    for key, value in configs.items():
        print(f' - {key}: {value}')


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
    if os.path.exists('config.yaml'):
        return read_yaml('config.yaml')
    elif os.path.exists('../config.yaml'):
        return read_yaml('../config.yaml')
    else:
        raise FileNotFoundError("Configuration file not found: config.yaml")


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

    return pd.DataFrame({'Triangle_Area': area_list, 'Triangle_Perimeter': perimeter_list,
                         'Triangle_Angle_Range': angle_range_list})
