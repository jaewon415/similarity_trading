import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
import requests
import plotly.graph_objects as go
from datetime import datetime
from utils import *

pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(suppress=True, precision=6)

st.set_option('deprecation.showPyplotGlobalUse', False)
# st.set_page_config(layout="wide")

SECRET_KEY = ''

def calculate_date_distance(dt, date_str1, date_str2):
    """
    Calculate the absolute distance in days between two dates provided as strings
    @param date_str1: a string representing the start date in the format specified by date_format
    @param date_str2: a string representing the end date in the format specified by date_format
    @return: The absolute distance in days between the two dates
    """
    date_distance = len(dt.loc[date_str1: date_str2])
    return date_distance
    
def compute_distance(dt, users_target, users_compare, user_distance, algorithms):
    """
    Compute a distance matrix using Dynamic Time Warping between a target and a comparison time series
    @param dt: dataFrame containing time series data indexed by dates
    @param users_target: list containing two strings representing the start and end dates of the target time series
    @param users_compare: list containing two strings representing the start and end dates of the comparison time series
    @return matrix: a dictionary representing the distance matrix.
    """
    matrix = {}
    target_data = dt.loc[users_target[0]: users_target[1]]
    target_values = np.array(target_data.values)
    target_values -= target_values[0]

    compare_data = dt.loc[users_compare[0]: users_compare[1]]
    for i in range(len(compare_data) - user_distance + 1):
        sliced_data = compare_data.iloc[i: i + user_distance]
        compare_values = np.array(sliced_data)
        compare_values -= compare_values[0]
        if algorithms == 'DTW':
            score = calculate_dtw(target_values, compare_values, user_distance, window_bool = False)
        elif algorithms == 'DTWc':
            score = calculate_dtw(target_values, compare_values, user_distance, window_bool = True)
        elif algorithms == 'Corr/DTW':
            score = calculate_corr_dtw(target_values, compare_values, user_distance, window_bool = False)
        elif algorithms == 'Corr/DTWc':
            score = calculate_corr_dtw(target_values, compare_values, user_distance, window_bool = True)
        elif algorithms == 'aCorr/DTW':
            score = calculate_absolute_corr_dtw(target_values, compare_values, user_distance, window_bool = False)
        elif algorithms == 'aCorr/DTWc':
            score = calculate_absolute_corr_dtw(target_values, compare_values, user_distance, window_bool = True)
        elif algorithms == 'PCA w/ DTW':
            score = calculate_pca_dtw(target_values, compare_values, user_distance, window_bool = False)
        elif algorithms == 'PCA w/ DTWc':
            score = calculate_pca_dtw(target_values, compare_values, user_distance, window_bool = True)
        elif algorithms == 'PCA Similarity Factor':
            score = calculate_pca_similarity_factor(target_values, compare_values)
        matrix[sliced_data.index[0].strftime("%Y-%m-%d")] = score
    return matrix

def dates_score(dt, date_score_dict, user_distance):
    """
    Extract minimum dates, corresponding dates, and values from a similarity dictionary
    @param date_score_dict: a dictionary containing similarity values for different dates
    @return result: a list of tuples containing minimum dates, corresponding dates, and values
    """
    result = []
    for from_date in date_score_dict:
        score = date_score_dict[from_date]
        started_data = dt.loc[from_date:][:user_distance]
        from_date = datetime.strptime(from_date, "%Y-%m-%d")
        to_date = started_data.index[-1]
        result.append((from_date, to_date, score))
    return result

def have_overlap(range1, range2):
    """
    Check if two ranges overlap
    @param range1: a tuple representing the first range, consisting of start, end, and any additional data
    @param range2: a tuple representing the second range, consisting of start, end, and any additional data
    @return: True if the ranges overlap, False otherwise
    """
    start1, end1, _ = range1
    start2, end2, _ = range2
    return start1 <= end2 and start2 <= end1

def filter_overlaps(ranges):
    """
    Filter out overlapping ranges from a list of ranges
    @param ranges: a list of tuples representing ranges, each tuple consisting of start, end, and any additional data
    @return: a list of non-overlapping ranges
    """
    non_overlapping = [ranges[0]]
    for r1 in ranges[1:]:
        overlap = False
        for r2 in non_overlapping:
            if have_overlap(r1, r2):
                overlap = True
                break
        if not overlap:
            non_overlapping.append(r1)
    return non_overlapping

def normalize_df(df):
    """
    Normalize the values of a pandas DataFrame using z-score normalization.
    @param df: the DataFrame to be normalized
    @return df_normalized: the normalized DataFrame
    """
    df_normalized = (df - df.mean()) / df.std()
    return df_normalized

def data_select(selected_data):    
    """
    Fetch data from a GitHub repository and select specific columns.
    @param selected_data: a list of column names to select from the data
    @param token: GitHub authentication token for accessing private repositories
    @return data: a dataframe containing the selected columns
    """

    url = f'https://raw.githubusercontent.com/jaewon415/data/main/prototype.csv'
    headers = {'Authorization': f'token {SECRET_KEY}'}
    response = requests.get(url, headers=headers)
    csv_file = StringIO(response.text)
    data = pd.read_csv(csv_file, index_col= 0)[[selected_data]]
    data = data.dropna()
    return data

def generate_data(options):
    analysis_df = pd.DataFrame()
    for selected_data in options:
        data = data_select(selected_data)
        data.index = pd.to_datetime(data.index)
        data = data.sort_index(ascending=True)
        data.columns = [selected_data]
        analysis_df = pd.concat([analysis_df, data], axis = 1)
    analysis_df = analysis_df.loc[:,~analysis_df.columns.duplicated()]
    analysis_df.index = pd.to_datetime(analysis_df.index)
    analysis_df = analysis_df.dropna()
    return analysis_df

def create_figures(sample_data, target_dates, values_list, options, title_prefix, subtract=False, n_steps = 0, N = 3):
    figs = []
    WIDTH, HEIGHT = 700, 500
    for column in options:
        fig = go.Figure()
        if n_steps > 0:
            get_length = len(sample_data.loc[target_dates[0]: target_dates[1]])
            target_data = sample_data[target_dates[0]:][:get_length + n_steps]
            target_data.reset_index(drop=True, inplace=True)
            fig.add_vline(x=get_length, line_dash="dash", line_color="black", line_width=1.5)
        else:
            target_data = sample_data.loc[target_dates[0]: target_dates[1]]
            target_data.reset_index(drop=True, inplace=True)

        if subtract:
            target_trace = target_data[column] - target_data[column].iloc[0]
        else:
            target_trace = target_data[column]
        
        fig.add_trace(go.Scatter(x=target_data.index, y=target_trace, mode='lines', name=f'Target: {column.split("_")[-1]}'))
        
        for i, (start_date, end_date, _) in enumerate(values_list[:N], 1):
            if n_steps > 0:
                get_length = len(sample_data.loc[start_date:end_date]) 
                sliced_data = sample_data[start_date:][:get_length + n_steps]
                sliced_data.reset_index(drop=True, inplace=True)
                fig.add_vline(x=get_length, line_dash="dash", line_color="black", line_width=1.5)
            else:
                sliced_data = sample_data.loc[start_date:end_date]
                sliced_data.reset_index(drop=True, inplace=True)

            if subtract:
                sliced_trace = sliced_data[column] - sliced_data[column].iloc[0]
            else:
                sliced_trace = sliced_data[column]

            fig.add_trace(go.Scatter(x=sliced_data.index, y=sliced_trace, mode='lines', name=f'Graph {i}: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}'))

        fig.update_layout(
            title=f'{title_prefix} {column}',
            width=WIDTH,
            height=HEIGHT,
            xaxis=dict(showline=True, linewidth=1, linecolor='black', showgrid=True, gridwidth=1, gridcolor='gray'),
            yaxis=dict(showline=True, linewidth=1, linecolor='black', showgrid=True, gridwidth=1, gridcolor='gray'),
            legend=dict(
                orientation="h",
                yanchor="top",   # Anchor to the top
                y=-0.2,          # Push the legend below the graph
                xanchor="right",
                x=1
            )
        )
        fig.update_xaxes(showticklabels=False)
        figs.append(fig)
    return figs

def pca_variable_importance(sample_data, values_list, N = 3):
    return None
# def pca_variable_importance(sample_data, values_list, N=3):
#     results = []
#     for i, (start_date, end_date, _) in enumerate(values_list[:N], 1):
#         scaler, pca = StandardScaler(), PCA()
#         sliced_data = sample_data.loc[start_date:end_date]
#         sliced_data.reset_index(drop=True, inplace=True)
#         sliced_data = scaler.fit_transform(sliced_data)
#         pca.fit(sliced_data)
        
#         loadings = pca.components_
#         abs_sum_loadings = np.sum(np.abs(loadings), axis=0)
#         total_sum_loadings = np.sum(abs_sum_loadings)
#         impact_percentage = (abs_sum_loadings / total_sum_loadings) * 100
#         feature_impact_percentage = {sample_data.columns[i]: impact_percentage[i] for i in range(len(sample_data.columns))}
#         sorted_feature_impact_percentage = sorted(feature_impact_percentage.items(), key=lambda x: x[1], reverse=True)
#         sorted_feature_names = [item[0] for item in sorted_feature_impact_percentage]
#         sorted_impact_percentages = [item[1] for item in sorted_feature_impact_percentage]
        
#         loadings_df = pd.DataFrame(np.abs(loadings), columns=sample_data.columns)
#         loadings_df.index = [f"pca{i}_loadings" for i in range(1, loadings.shape[0] + 1)]
#         print(pca.explained_variance_ratio_)
#         df = pd.DataFrame(sorted_impact_percentages, index=sorted_feature_names, columns=[f'Impact Perc. of Graph {i}']).transpose()
#         results.append(df)
#         results.append(loadings_df)
#     return pd.concat(results, axis=0)
 
def main():
    st.sidebar.title('복수지표 유사기간 분석툴')
    options = st.sidebar.multiselect(
        'Time Series Data:',
        ('GT2 Govt', 'GT5 Govt', 'GT10 Govt', 'GT30 Govt', 'USYC2Y10 Index', 'USYC5Y30 Index', 'USYC1030 Index',
        'USGGBE10 Index', 'GTII10 Govt', 'GTDEM2Y Govt', 'GTDEM5Y Govt', 'GTDEM10Y Govt', 'DEYC2Y10 Index',
        'DEYC5Y30 Index', 'DEGGBE10 Index', 'GTDEMII10Y Govt', 'GTESP10Y Govt', 'GTITL10Y Govt', 'GTGBP10Y Govt',
        'GTCAD10Y Govt', 'GTAUD10Y Govt', 'GTJPY10Y Govt', 'CCSWNI5 BGN Curncy', 'IRSWNI5 BGN Curncy', 'ODF29 Comdty',
        'MPSW5E BGN Curncy', 'GVSK3YR Index', 'GVSK10YR Index', 'DXY Index', 'KRW Curncy', 'EUR Curncy', 'JPY Curncy',
        'CNH Curncy', 'BRL Curncy', 'INR Curncy', 'MXN Curncy', 'SPX Index', 'CCMP Index', 'DAX Index', 'BCOM Index',
        'XAU Curncy', 'USCRWTIC Index', 'TSFR3M Index', 'USGG3M Index', 'US0003M Index', 'UREPTA30 Index', 'LQD US Equity',
        'HYG US Equity', 'CDX IG CDSI GEN 5Y Corp', 'CDX HY CDSI GEN 5Y SPRD Corp', 'EMLC US Equity', 'EMB US Equity',
        'VIX Index', 'MOVE Index', '.VIXVXN Index', 'GFSIFLOW Index', 'JLGPUSPH Index', 'JLGPEUPH Index', 'MRIEM Index',
        'ACMTP10 Index', 'FWISUS55 Index', 'ILM3NAVG Index', 'ECRPUS 1Y Index', 'CESIUSD Index', 'CESIUSH Index', 'CESIUSS Index',
        'CESIEUR Index', 'CESIEUH Index', 'CESIEUS Index', 'CESIEM Index', 'CESIEMXP Index', 'CESIEMFW Index'
        )
    )
    
    if len(options) != 0:
        original_data = generate_data(options)
        sample_data = normalize_df(original_data)
        target_date = st.sidebar.date_input(
            "Target Date Range",
            (datetime(2023, 11, 1), datetime(2024, 1, 1)),
            format="YYYY/MM/DD"
        )
        target_date = [date.strftime("%Y-%m-%d") for date in target_date]
        user_target_distance = calculate_date_distance(sample_data, target_date[0], target_date[1])
        
        # max_year, max_month, max_day = sample_data.index[-1].year, sample_data.index[-1].month, sample_data.index[-1].day
        min_year, min_month, min_day = sample_data.index[0].year, sample_data.index[0].month, sample_data.index[0].day

        compare_date = st.sidebar.date_input(
            "Date Range for Analysis",
            (datetime(min_year, min_month, min_day), datetime(2023, 9, 30)),
            format="YYYY/MM/DD"
        )
        compare_date = [date.strftime("%Y-%m-%d") for date in compare_date]

        algorithms = st.sidebar.selectbox('Algorithms:', ('DTW', 'DTWc', 'Corr/DTW', 'Corr/DTWc', 'aCorr/DTW', 'aCorr/DTWc', 'PCA w/ DTW', 'PCA w/ DTWc','PCA Similarity Factor'))

        nsteps = st.sidebar.slider('N-Steps Ahead (in days)', min_value=0, max_value=100, value=0, step=10)

        if st.sidebar.button('Generate'):
            
            similarity_distance = compute_distance(sample_data, target_date, compare_date, user_target_distance, algorithms)
            dates_score_list = dates_score(sample_data, similarity_distance, user_target_distance)
            if algorithms == 'DTW' or algorithms == 'DTWc' or algorithms == 'PCA w/ DTW' or algorithms == 'PCA w/ DTWc':
                sorted_dates_score_list = sorted(dates_score_list, key=lambda x: x[2], reverse = False)
            else:
                sorted_dates_score_list = sorted(dates_score_list, key=lambda x: x[2], reverse = True)

            filtered_dates = filter_overlaps(sorted_dates_score_list)
            values_list = [(pd.to_datetime(start), pd.to_datetime(end), distance) for start, end, distance in filtered_dates]

            target_start_date = pd.to_datetime(target_date[0])
            target_end_date = pd.to_datetime(target_date[1])
            target_data = original_data.loc[target_start_date:target_end_date]
            target_data.reset_index(drop=True, inplace=True)

            original_figs = create_figures(original_data, [target_start_date, target_end_date], values_list, options, "(Original)", subtract=False, n_steps=nsteps, N = 3)
            aligned_figs = create_figures(original_data, [target_start_date, target_end_date], values_list, options, "(Aligned)", subtract=True, n_steps=nsteps, N = 3)

            num_columns = len(options)
            cols = st.columns(num_columns)

            for i, (original_fig, aligned_fig) in enumerate(zip(original_figs, aligned_figs)):
                with cols[i]:
                    st.plotly_chart(original_fig, use_container_width=True)
                    st.plotly_chart(aligned_fig, use_container_width=True)

if __name__ == "__main__":
    main()