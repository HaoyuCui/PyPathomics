import pandas as pd
import numpy as np

from scipy.spatial import Delaunay
from shapely.geometry import Polygon, MultiPolygon

from utils import get_triangle_feature_df


def intersect_ratio(p_a, p_b):
    intersection = p_a.intersection(p_b)
    if isinstance(intersection, Polygon):
        return intersection.area / p_a.area
    elif isinstance(intersection, MultiPolygon):
        return sum([poly.area for poly in intersection]) / p_a.area


class FeatureExtractor:
    def __init__(self, slide, sc_MTOP_dir, feature_list, cell_types):
        self.slide = slide
        self.sc_MTOP_dir = sc_MTOP_dir
        self.feature_list = feature_list
        self.cell_types = cell_types

    def read_csv_for_type(self, cell_type):
        return pd.read_csv(f'{self.sc_MTOP_dir}/{self.slide}_Feats_{cell_type}.csv')

    def compute_statistics(self, df, cell_type, remove_outliers):
        if remove_outliers:
            df = self.remove_outliers(df)
        stats = {}
        for col in df.columns:
            df[col] = df[col].astype(float)
            stats.update({
                # basic statistics
                f'{cell_type}_{col}_mean': df[col].mean(),
                f'{cell_type}_{col}_std': df[col].std(),
                # distribution
                f'{cell_type}_{col}_Q25': df[col].quantile(0.25),
                f'{cell_type}_{col}_median': df[col].median(),
                f'{cell_type}_{col}_Q75': df[col].quantile(0.75),
                f'{cell_type}_{col}_IQR': df[col].quantile(0.75) - df[col].quantile(0.25),
                f'{cell_type}_{col}_range': df[col].max() - df[col].min(),
                # coefficient of variation
                f'{cell_type}_{col}_CV': df[col].std() / df[col].mean(),
                # skewness
                f'{cell_type}_{col}_skew': df[col].skew(),
                # kurtosis
                f'{cell_type}_{col}_kurt': df[col].kurt()

            })
        return stats

    def remove_outliers(self, df, z_score_threshold=2):
        df = df.filter(regex=f'({"|".join(self.feature_list)})')
        z_scores = (df - df.mean()) / df.std()
        return df[(z_scores.abs() < z_score_threshold).all(axis=1)]

    def extract_features(self):
        features = {}
        cell_count = {}
        cell_sum = 0
        for cell_type in self.cell_types:
            df = self.read_csv_for_type(cell_type)
            cell_count[cell_type] = df.shape[0]
            cell_sum += cell_count[cell_type]
            features.update(self.compute_statistics(df, cell_type, remove_outliers=True))

        features.update({
            'I_Num': cell_count['I'],
            'S_Num': cell_count['S'],
            'T_Num': cell_count['T'],
            'I_Ratio': cell_count['I'] / cell_sum,
            'S_Ratio': cell_count['S'] / cell_sum,
            'T_Ratio': cell_count['T'] / cell_sum
        })

        return pd.DataFrame(features, index=[0])

    def extract_triangle_features(self):
        triangle_feature = {}
        polygon_area = {}
        for cell_type in self.cell_types:
            df = self.read_csv_for_type(cell_type)
            df['Centroid'] = df['Centroid'].apply(lambda x: x[1:-1].split(','))
            df['X'] = df['Centroid'].apply(lambda x: float(x[0]))
            df['Y'] = df['Centroid'].apply(lambda x: float(x[1]))
            coords = np.array(df[['X', 'Y']])
            triangles = Delaunay(df[['X', 'Y']]).simplices.tolist()
            # Delaunay triangles features
            triangle_feats, polygons = get_triangle_feature_df(triangles, coords)
            triangle_feature.update(self.compute_statistics(triangle_feats, cell_type, remove_outliers=False))
            # polygon features
            polygon_area[cell_type] = polygons

        p_I = polygon_area['I']
        p_S = polygon_area['S']
        p_T = polygon_area['T']

        # calculate the intersection over union of polygons, this process may take some time
        polygon_features = {
            'I_in_T': intersect_ratio(p_I, p_T),
            'I_in_S': intersect_ratio(p_I, p_S),
            'S_in_T': intersect_ratio(p_S, p_T),
            'S_in_I': intersect_ratio(p_S, p_I),
            'T_in_S': intersect_ratio(p_T, p_S),
            'T_in_I': intersect_ratio(p_T, p_I)
        }
        triangle_feature.update(polygon_features)

        return pd.DataFrame(triangle_feature, index=[0])

    def extract(self):
        if len(self.feature_list) == 0:
            raise ValueError('Feature list is empty!')
        elif len(self.cell_types) == 0:
            raise ValueError('Cell types list is empty!')

        if 'Triangle' not in self.feature_list:
            return self.extract_features()
        elif 'Triangle' in self.feature_list and len(self.cell_types) == 1:
            raise self.extract_triangle_features()
        elif 'Triangle' in self.feature_list and len(self.cell_types) >= 2:
            return pd.concat([self.extract_features(), self.extract_triangle_features()], axis=1)
