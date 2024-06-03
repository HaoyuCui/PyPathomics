import logging
import os.path

import pandas as pd
import numpy as np

from scipy.spatial import Delaunay

from src.utils import get_triangle_feature_df


class FeatureExtractor:
    def __init__(self, slide, buffer_dir, feature_list, cell_types, statistic_types):
        self.slide = slide
        self.buffer_dir = buffer_dir
        self.feature_list = feature_list
        self.cell_types = cell_types
        self.statistic_types = statistic_types
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

        if 'Triangle' not in self.feature_list:
            return self.extract_features()
        elif 'Triangle' in self.feature_list and len(self.cell_types) == 1:
            return self.extract_triangle_features()
        elif 'Triangle' in self.feature_list and len(self.cell_types) >= 2:
            return pd.concat([self.extract_features(), self.extract_triangle_features()], axis=1)
