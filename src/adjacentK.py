import os.path

import ripleyk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial import cKDTree

res = 0.2201  # um per pixel
radii_in_um = 32
radii = [radii_in_um // res]

sample_size = [-1, -1]

def getCoords(data:pd.DataFrame):
    centroid = data['Centroid']
    centroid = centroid.apply(lambda x: x[1:-1].split(','))

    xs = centroid.apply(lambda x: np.float32(x[0]))
    ys = centroid.apply(lambda x: np.float32(x[1]))
    return np.array(xs), np.array(ys)


def getAllCoords(root, slide_name):
    cell_coords = {}
    for cell_type in  ['I', 'S', 'T']:
        csv_file = os.path.join(root, f'{slide_name}_Feats_{cell_type}.csv')
        df = pd.read_csv(csv_file)
        xs, ys = getCoords(df)
        cell_coords[cell_type] = (xs, ys)
        sample_size[0], sample_size[1] = max(sample_size[0], max(xs)), max(sample_size[1], max(ys))
    return cell_coords


def getRelation(coords:dict, cell_types:list):
    assert sample_size[0] != -1, 'sample_size should be set'
    k = []
    if len(cell_types) == 2:
        for radius in radii:
            counts = 0
            score_vol = np.pi * radius**2
            bound_size = sample_size[0] * sample_size[1]
            alpha_x, alpha_y = coords[cell_types[0]][0], coords[cell_types[0]][1]
            beta_x, beta_y = coords[cell_types[1]][0], coords[cell_types[1]][1]
            tree = cKDTree(np.array([alpha_x, alpha_y]).T)
            for x, y in zip(beta_x, beta_y):
                # boundary_correct = False
                counts += len(tree.query_ball_point([x, y], radius, p=2))-1
            # CSR_Normalise
            # k_value = bound_size * counts / len(beta_x)**2 - score_vol
            # estimation
            k_value = counts / len(beta_x)
            k.append(k_value)
    else:
        raise ValueError('cell_types should be a list of 2')
    return k


if __name__ == '__main__':
    # allCoords = getAllCoords(r'C:\Users\Ed\Downloads\temp', '2023-31276')
    slide_name = 'TCAM'  # '2023-31276'
    allCoords = getAllCoords(r'C:\Users\Ed\Downloads\temp', slide_name)
    # I_S = getRelation(allCoords, ['I', 'S'])

    plt.figure(figsize=(24, 18))
    plt.scatter(allCoords['S'][0], allCoords['S'][1], c='blue', alpha=0.5, s=1)
    plt.scatter(allCoords['I'][0], allCoords['I'][1], c='green', alpha=0.5, s=1)
    plt.scatter(allCoords['T'][0], allCoords['T'][1], c='red', alpha=0.5, s=1)

    plt.rcParams['font.family'] = 'Times New Roman'

    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title('Scatter plot of Centroids')
    plt.gca().invert_yaxis()
    plt.savefig(f'{slide_name}_scatter.png')

    print(f'Cell \t Cell \t RipleyK')
    # print(f'I \t I \t {getRelation(allCoords, ["I"])}')
    # print(f'S \t S \t {getRelation(allCoords, ["S"])}')
    # print(f'T \t T \t {getRelation(allCoords, ["T"])}')
    #
    # print(f'I \t S \t {getRelation(allCoords, ["I", "S"])}')
    # print(f'I \t T \t {getRelation(allCoords, ["I", "T"])}')
    #
    # print(f'S \t I \t {getRelation(allCoords, ["S", "I"])}')
    # print(f'S \t T \t {getRelation(allCoords, ["S", "T"])}')
    #
    # print(f'T \t I \t {getRelation(allCoords, ["T", "I"])}')
    # print(f'T \t S \t {getRelation(allCoords, ["T", "S"])}')

    cell_types = ['I', 'S', 'T']
    matrix = np.zeros((len(cell_types), len(cell_types), len(radii)))

    for i, cell_type_a in enumerate(cell_types):
        for j, cell_type_b in enumerate(cell_types):
            matrix[i, j, :] = getRelation(allCoords, [cell_type_a, cell_type_b])
            print(f'{cell_type_a} \t {cell_type_b} \t {matrix[i, j, :]}')

    for idx, radius in enumerate(radii):
        plt.figure(figsize=(6.5, 6))
        sns.heatmap(matrix[:, :, idx], annot=True, fmt=".2f", xticklabels=cell_types, yticklabels=cell_types, cmap="seismic")
        plt.title(f"Distribution of expectations ({radii_in_um} μm)")
        plt.xlabel("Cell Type B (surroundings)")
        plt.ylabel("Cell Type A (targets)")
        plt.savefig(f'{slide_name}_heatmap_radius_{radius}.png')
        plt.close()





