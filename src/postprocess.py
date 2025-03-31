import logging
import os.path
from pathlib import Path

import pandas as pd
import numpy as np

from scipy.spatial import Delaunay, cKDTree

from src.utils import get_triangle_feature_df, get_coords


class FeatureExtractor:
    def __init__(self, slide, buffer_dir, feature_list, cell_types, statistic_types):
        self.slide = slide
        self.buffer_dir = buffer_dir
        self.feature_list = feature_list
        self.cell_types = cell_types
        self.statistic_types = statistic_types

        # RipleyK's parameters
        self.sample_size = [-1, -1]   # equal to the ROI's size
        self.radii_in_um = 32
        self.res = 0.2201  # um per pixel
        self.radii = [self.radii_in_um // self.res]

        assert len(self.statistic_types) > 0, 'In config.yaml: static_types should not be empty!'

    def read_csv_for_type(self, cell_type):
        buffer_file_path = f'{self.buffer_dir}/{self.slide}_Feats_{cell_type}.csv'
        if not os.path.exists(buffer_file_path):
            logging.error(f'File not found: {buffer_file_path}')
            raise FileNotFoundError
        return pd.read_csv(buffer_file_path)

    def compute_statistics(self, df, cell_type, remove_outliers):
        if remove_outliers:
            df = self.remove_outliers(df)
        stats = {}
        for col in df.columns:
            df[col] = df[col].astype(float)
            stats_dict = {}
            if 'basic' in self.statistic_types:
                stats_dict.update({
                    # basic statistics
                    f'{cell_type}_{col}_mean': df[col].mean(),
                    f'{cell_type}_{col}_std': df[col].std(),
                })
            if 'distribution' in self.statistic_types:
                stats_dict.update({
                    # distribution
                    f'{cell_type}_{col}_Q25': df[col].quantile(0.25),
                    f'{cell_type}_{col}_median': df[col].median(),
                    f'{cell_type}_{col}_Q75': df[col].quantile(0.75),
                    f'{cell_type}_{col}_IQR': df[col].quantile(0.75) - df[col].quantile(0.25),
                    f'{cell_type}_{col}_range': df[col].max() - df[col].min(),
                    f'{cell_type}_{col}_skew': df[col].skew(),
                    f'{cell_type}_{col}_kurt': df[col].kurt()
                })
            stats.update(stats_dict)
        return stats

    def remove_outliers(self, df, z_score_threshold=2):
        df = df.filter(regex=f'({"|".join(self.feature_list)})')
        z_scores = (df - df.mean()) / df.std()
        return df[(z_scores.abs() < z_score_threshold).all(axis=1)]

    def extract_features(self):
        features = {}
        cell_sum = 0
        cell_count = {}
        for cell_type in self.cell_types:
            df = self.read_csv_for_type(cell_type)
            cell_count[cell_type] = df.shape[0]
            cell_sum += cell_count[cell_type]
            features.update({
                f'{cell_type}_count': cell_count[cell_type]
            })
            features.update(self.compute_statistics(df, cell_type, remove_outliers=True))

        if len(self.cell_types) > 1:
            for cell_type in self.cell_types:
                features.update({
                    f'{cell_type}_ratio': cell_count[cell_type] / cell_sum,
                })
        return pd.DataFrame(features, index=[0])

    def extract_triangle_features(self):
        triangle_feature = {}
        for cell_type in self.cell_types:
            df = self.read_csv_for_type(cell_type)
            df['Centroid'] = df['Centroid'].apply(lambda x: x[1:-1].split(','))
            df['X'] = df['Centroid'].apply(lambda x: float(x[0]))
            df['Y'] = df['Centroid'].apply(lambda x: float(x[1]))
            coords = np.array(df[['X', 'Y']])
            triangles = Delaunay(df[['X', 'Y']]).simplices.tolist()
            # Delaunay triangles features
            triangle_feats = get_triangle_feature_df(triangles, coords)
            triangle_feature.update(self.compute_statistics(triangle_feats, cell_type, remove_outliers=False))

        return pd.DataFrame(triangle_feature, index=[0])


    def extract(self):
        if len(self.feature_list) == 0:
            raise ValueError('Feature list is empty! Check config file')
        elif len(self.cell_types) == 0:
            raise ValueError('Cell types list is empty! Check config file')

        triangle_feature = pd.DataFrame()

        if 'Triangle' in self.feature_list:
            triangle_feature =  self.extract_triangle_features()

        additional_features = triangle_feature

        return pd.concat([self.extract_features(), additional_features], axis=1)


# Core function
def postprocess_files(args, configs):
    process_queue = list(Path(args.seg).glob(f'*.json')) + list(Path(args.seg).glob(f'*.dat'))
    df_feats_list = []
    for i, slide in enumerate(process_queue):
        logging.info(f'Phase 2 Postprocessing \t {i + 1} / {len(process_queue)} \t {slide} ')
        slide = slide.stem
        extractor = FeatureExtractor(slide, args.buffer, feature_list=configs['feature-set'],
                                                 cell_types=configs['cell-types'],
                                                 statistic_types=configs['statistic-types'])
        slide_feats = extractor.extract()
        slide_feats['slide'] = slide
        df_feats_list.append(slide_feats)

    df_feats = pd.concat(df_feats_list, ignore_index=True)
    cols = ['slide'] + [col for col in df_feats.columns if col != 'slide']
    return df_feats[cols]


if __name__ == '__main__':
    from argparse import Namespace
    args, configs = Namespace(), {}
    args.seg = r'E:\hover-net-output\json'
    args.buffer = r'E:\hover-net-output\feature'
    configs['cell-types'] = ['I', 'S', 'T']
    configs['statistic-types'] = ['basic']
    configs['feature-set'] = ['Morph', 'Texture', 'Triangle']

    df_feats = postprocess_files(args, configs)

    df_feats.to_csv('ec-output.csv', index=False)