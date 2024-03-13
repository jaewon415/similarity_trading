import numpy as np
from dtaidistance import dtw
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def calculate_absolute_corr_dtw(target_features, compare_features, user_distance, window_bool = False):
    distances = []
    correlations = []
    for dim in range(target_features.shape[1]):
        if window_bool:
            dim_distance = dtw.distance_fast(target_features[:, dim], compare_features[:, dim], window = int(user_distance * 0.2))
        else:
            dim_distance = dtw.distance_fast(target_features[:, dim], compare_features[:, dim])
        corr_coef = abs(np.corrcoef(target_features[:, dim], compare_features[:, dim])[0, 1])
        distances.append(dim_distance)
        correlations.append(corr_coef)
    return sum(correlations) / (sum(distances) + 1e-10)

def calculate_corr_dtw(target_features, compare_features, user_distance, window_bool = False):
    distances = []
    correlations = []
    for dim in range(target_features.shape[1]):
        if window_bool:
            dim_distance = dtw.distance_fast(target_features[:, dim], compare_features[:, dim], int(user_distance * 0.2))
        else:
            dim_distance = dtw.distance_fast(target_features[:, dim], compare_features[:, dim])
        corr_coef = np.corrcoef(target_features[:, dim], compare_features[:, dim])[0, 1]
        distances.append(dim_distance)
        correlations.append(corr_coef)
    return sum(correlations) / (sum(distances) + 1e-10)

def calculate_dtw(target_features, compare_features, user_distance, window_bool = False):
    distances = []
    for dim in range(target_features.shape[1]):
        if window_bool:
            dim_distance = dtw.distance_fast(target_features[:, dim], compare_features[:, dim], window = int(user_distance * 0.2))
        else:
            dim_distance = dtw.distance_fast(target_features[:, dim], compare_features[:, dim])
        distances.append(dim_distance)
    return sum(distances)

def calculate_pca_dtw(target_ts, compare_ts, user_distance, window_bool = False):
    scaler = StandardScaler()
    pca = PCA(n_components= 1)
    target_time_series = scaler.fit_transform(target_ts)
    compare_time_series = scaler.fit_transform(compare_ts)
    pca_features_x1, pca_features_x2 = pca.fit_transform(target_time_series), pca.fit_transform(compare_time_series)
    pca_features_x1 = pca_features_x1 - pca_features_x1[0]
    pca_features_x2 = pca_features_x2 - pca_features_x2[0]
    pca_features_x1, pca_features_x2 = pca_features_x1.flatten(), pca_features_x2.flatten()

    if window_bool:
        pca_dtw_distance = dtw.distance_fast(pca_features_x1, pca_features_x2, window = int(user_distance * 0.2))
    else:
        pca_dtw_distance = dtw.distance_fast(pca_features_x1, pca_features_x2)
    return pca_dtw_distance

def calculate_pca_similarity_factor(target_ts, compare_ts):
    scaler = StandardScaler()
    pca = PCA()
    target_time_series = scaler.fit_transform(target_ts)
    compare_time_series = scaler.fit_transform(compare_ts)
    pca_features_x1, pca_features_x2 = pca.fit_transform(target_time_series), pca.fit_transform(compare_time_series)

    num_columns = (pca_features_x1.shape[1], pca_features_x2.shape[1])

    similarity_factor = np.sum([
        (np.dot(pca_features_x1[:, idx1], pca_features_x2[:, idx2]) / (np.linalg.norm(pca_features_x1[:, idx1]) * np.linalg.norm(pca_features_x2[:, idx2]))) ** 2
        for idx1 in range(num_columns[0])
        for idx2 in range(num_columns[1])
    ])

    return similarity_factor