from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

import json
import joblib
from sklearn import set_config
set_config(enable_metadata_routing=True)

# imputers, classes, pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, FunctionTransformer, OrdinalEncoder, StandardScaler, MinMaxScaler, FunctionTransformer

from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler, RepeatedEditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN 
from imblearn.metrics import geometric_mean_score

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

KNNI = KNNImputer(n_neighbors=5, weights='uniform')
SP = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

# embedding
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from skorch import  NeuralNetClassifier
from skorch.helper import predefined_split, SliceDataset
from gensim.models import Word2Vec

# metrics
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold, cross_val_score, GroupShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, classification_report, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score,  make_scorer, average_precision_score, f1_score

# grouping
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# actual models
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.ensemble import RUSBoostClassifier, BalancedBaggingClassifier, BalancedRandomForestClassifier


embedding_cols = ['player', 'team']
one_hot = ['shot_technique', 'shot_type']
numerical = ['goal_distance', 'shot_angle', 'shot_zone_area']
binary = ['shot_deflected', 'shot_first_time', 'shot_open_goal']
body_part = ['shot_body_part']
locations = ['location_x', 'location_y']
timestamp = ['timestamp']
all_features = embedding_cols + one_hot + numerical + binary + body_part + locations + timestamp

engineered = [ 
    "interaction_dist_angle", 
    'effective_angle',
    "adjusted_shot_power",
    "shot_cone_area",
    "x_y_ratio",
    "distance_y_product",
    "quick_first_time",
    "open_goal_adjusted_angle",

    "location_r",
    "location_theta",
    "rel_x",
    "rel_y",
    "abs_distance_to_goal_center_y",
    "distance_squared",
    "inverse_distance",

    "time_seconds",
    "time_minutes",
    "time_remaining_half"
]

binary = binary + ['match_pressure', 'time_half']


class Word2VecEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, player_model, team_model):
        self.player_model = player_model
        self.team_model = team_model
        self.dim_player = player_model.vector_size
        self.dim_team = team_model.vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        player_embs = X['player'].astype(str).apply(lambda pid: self.player_model.wv[pid] if pid in self.player_model.wv else np.zeros(self.dim_player))
        team_embs = X['team'].astype(str).apply(lambda tid: self.team_model.wv[tid] if tid in self.team_model.wv else np.zeros(self.dim_team))

        player_embs = np.vstack(player_embs.values)
        team_embs = np.vstack(team_embs.values)

        return np.hstack([player_embs, team_embs])


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.max_opponents = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if not np.issubdtype(X["timestamp"].dtype, np.timedelta64):
            X["timestamp"] = pd.to_timedelta(X["timestamp"])

        X["interaction_dist_angle"] = X["goal_distance"] * X["shot_angle"]
        X['effective_angle'] = np.cos(X["shot_angle"]) / X['goal_distance']
        X["adjusted_shot_power"] = X["shot_first_time"] * X["goal_distance"]
        X["shot_cone_area"] = 0.5 * (X["goal_distance"] ** 2) * np.tan(X["shot_angle"])
        X["x_y_ratio"] = X["location_x"] / (X["location_y"] + 1e-6)
        X["distance_y_product"] = X["goal_distance"] * X["location_y"]
        X["quick_first_time"] = X["shot_first_time"] * X["location_x"]
        X["open_goal_adjusted_angle"] = X["shot_open_goal"] * X["shot_angle"]

        X["location_r"] = np.sqrt(X["location_x"]**2 + X["location_y"]**2)
        X["location_theta"] = np.arctan2(X["location_y"], X["location_x"])
        X["rel_x"] = X["location_x"] / 105
        X["rel_y"] = X["location_y"] / 68
        X["abs_distance_to_goal_center_y"] = np.abs(X["location_y"])
        X["distance_squared"] = X["location_x"]**2 + X["location_y"]**2
        X["inverse_distance"] = 1 / (X["goal_distance"] + 1e-6)

        X["time_seconds"] = X["timestamp"].dt.total_seconds()
        X["time_minutes"] = (X["time_seconds"] // 60).astype(int)
        X["time_half"] =( X["time_seconds"] // (45 * 60)).astype(int)  # 0 = 1st half, 1 = 2nd half
        X["time_remaining_half"] = (45 * 60) - (X["time_seconds"] % (45 * 60))
        X["match_pressure"] = (X["time_seconds"].apply(lambda t: 1 if t >= 40 * 60 else 0)).astype(int)
        
        return X



class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, player_model, team_model):
        self.player_model = player_model
        self.team_model = team_model
        self.feature_engineer = FeatureEngineeringTransformer()
        self.col_transformer = None  # Will define in fit

    def fit(self, X, y=None):
        X_fe = self.feature_engineer.fit_transform(X)

        self.col_transformer = ColumnTransformer([
            ("bodypart", OneHotEncoder(), body_part),
            ("one_hot", OneHotEncoder(handle_unknown="ignore"), one_hot),
            ("locations", "passthrough", locations),
            ("numerical", "passthrough", numerical),
            ("engineered", "passthrough", engineered),
            ("binary", "passthrough", binary),
            ("w2vec", Word2VecEmbedder(self.player_model, self.team_model), embedding_cols)
        ])

        self.col_transformer.fit(X_fe)
        return self

    def transform(self, X):
        if self.col_transformer is None:
            raise RuntimeError("PreprocessingTransformer must be fitted before calling transform.")
        X_fe = self.feature_engineer.transform(X)
        return self.col_transformer.transform(X_fe)
