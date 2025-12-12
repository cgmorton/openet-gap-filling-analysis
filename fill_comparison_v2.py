from datetime import datetime
import math
import os
import pprint
import random

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns
import sklearn.metrics
import sklearn.linear_model

from whittaker_eilers import WhittakerSmoother

# Include all MGRS grid zones except 12R and 16U since they are too small
MGRS_ZONES = [
    '10S', '10T', '10U', '11S', '11T', '11U', '12S', '12T', '12U',
    '13R', '13S', '13T', '13U', '14R', '14S', '14T', '14U', '15R', '15S', '15T', '15U',
    '16R', '16S', '16T', '17R', '17S', '17T', '18S', '18T', '19T'
    # '12R', '16U'
]
MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# Points were only built for NLCD 2024
NLCD_YEARS = [2024]


def main():
    print('Computing Gap Filling Comparison Statistics\n')

    # Read in the precomputed points and data CSV files as dataframes
    # points_df = read_points_csv_df(points_folder='points')
    data_df_dict = read_data_csv_df(data_folder='data')

    # CGM - The stats years are currently hardcoded in the compare function
    # Exclude 2016 from the statistics since there is not a full prior year to interpolate from
    # Including 2024 even though 2025 is not complete
    # stats_years = list(range(2017, 2025))

    # CGM - The min_mont_count parameter is currently set in the windows function
    # TODO: Add support for setting a minimum number of months in the year
    #   and minimum number of months in the growing season
    # min_month_count = 6
    # min_gs_month_count = 3

    # Building a single points dataframe and CSV from the MGRS grid zone points CSV files
    # overwrite_flag = False

    # Compute the maximum ETf per site
    # Assuming it is okay to make this for the full period of record
    print('Computing maximum ETf')
    etf_max_dict = {
        point_id: data_df_dict[point_id].agg(etf=('etf', 'max'))['etf'].to_dict()['etf']
        for point_id in data_df_dict.keys()
    }

    # Compute climos for each site
    # Only keep the climo value if there are at least "n" years of data
    print('Computing monthly climatologies')
    month_climo_count_min = 2
    month_climo_dict = {}
    for point_id in data_df_dict.keys():
        month_climo = data_df_dict[point_id].groupby(['month']).agg(
            etf=('etf', 'mean'),
            etf_median=('etf', 'median'),
            count=('etf', 'count'),
            et=('et', 'mean'),
            eto=('eto', 'mean'),
        )
        month_climo_count_mask = month_climo['count'] < month_climo_count_min
        month_climo.loc[month_climo_count_mask, ['etf', 'etf_median', 'et']] = np.nan
        month_climo_dict[point_id] = month_climo
        del month_climo, month_climo_count_mask


    # Process the target point combinations
    for name, point_mgrs_keep_list, point_nlcd_keep_list in [
        # # All NLCD classes by MGRS grid zone
        # ['mgrs10s', ['10S'], []],
        # ['mgrs10t', ['10T'], []],
        # ['mgrs11s', ['11S'], []],
        # ['mgrs11t', ['11T'], []],
        # ['mgrs12s', ['12S'], []],
        # ['mgrs12t', ['12T'], []],
        # ['mgrs13s', ['13S'], []],
        # ['mgrs13t', ['13T'], []],
        # ['mgrs14s', ['14S'], []],
        # ['mgrs14t', ['14T'], []],
        # ['mgrs15s', ['15S'], []],
        # ['mgrs15t', ['15T'], []],
        # ['mgrs16s', ['16S'], []],
        # ['mgrs16t', ['16T'], []],
        # ['mgrs17s', ['17S'], []],
        # ['mgrs17t', ['17T'], []],
        # ['mgrs18t', ['18T'], []],
        # ['mgrs19t', ['19T'], []],
        # # All MGRS grid zones by NLCD class
        # ['nlcd41', [], [41]],
        # ['nlcd42', [], [42]],
        # ['nlcd52', [], [52]],
        # ['nlcd71', [], [71]],
        # ['nlcd81', [], [81]],
        # ['nlcd82', [], [82]],
        # ['nlcd90', [], [90]],
        # ['nlcd95', [], [95]],
        # # All points
        ['all_points', [], []],
    ]:
        print(f'\n{name}')

        # Filter the points list to the target NLCD classes and MGRS grid zones
        point_id_list = list(data_df_dict.keys())
        if point_nlcd_keep_list:
            point_id_list = [p for p in point_id_list if int(p.split('_')[1][4:6]) in point_nlcd_keep_list]
        if point_mgrs_keep_list:
            point_id_list = [p for p in point_id_list if p.split('_')[0][0:3] in point_mgrs_keep_list]
        print(f'Points: {len(point_id_list)}')

        # Overwrite the summary stats file everytime this block is run
        output_txt = f'summary_stats_{name}.txt'
        output_f = open(output_txt, 'w')
        output_f.write('MGRS: '+ ', '.join(point_mgrs_keep_list) + '\n')
        output_f.write('NLCD: '+ ', '.join(map(str, point_nlcd_keep_list)) + '\n\n')
        output_f.close()

        randomly_drop_one_datapoint(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'Randomly drop one datapoint during the year',
            point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt,
        )
        randomly_drop_one_datapoint(
            [4, 5, 6, 7, 8, 9], 'Randomly drop one datapoint during the growing season (Apr-Sept)',
            point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt
        )
        randomly_drop_one_datapoint(
            [10, 11, 12, 1, 2, 3], 'Randomly drop one datapoint in the non-growing season (Oct-Mar)',
            point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt
        )
        randomly_drop_one_datapoint(
            [12, 1, 2], 'Randomly drop one datapoint during the winter (Dec-Feb)',
            point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt
        )

        randomly_drop_gap_datapoints(
            2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'Randomly drop a two month gap during the year',
            point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt
        )
        randomly_drop_gap_datapoints(
            2, [4, 5, 6, 7, 8, 9],
            'Randomly drop a two month gap during the growing season (Apr-Sept)',
            point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt
        )
        randomly_drop_gap_datapoints(
            2, [10, 11, 12, 1, 2, 3],
            'Randomly drop a two month gap during the the non-growing season (Oct-Mar)',
            point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt
        )

        randomly_drop_gap_datapoints(
            3, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'Randomly drop a three month gap during the year',
            point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt
        )
        randomly_drop_gap_datapoints(
            3, [4, 5, 6, 7, 8, 9],
            'Randomly drop a three month gap during the growing season (Apr-Sept)',
            point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt
        )
        randomly_drop_gap_datapoints(
            3, [10, 11, 12, 1, 2, 3],
            'Randomly drop a three month gap during the the non-growing season (Oct-Mar)',
            point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt
        )

        randomly_drop_gap_datapoints(
            4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'Randomly drop a four month gap during the year',
            point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt
        )
        randomly_drop_gap_datapoints(
            4, [4, 5, 6, 7, 8, 9],
            'Randomly drop a four month gap during the growing season (Apr-Sept)',
            point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt
        )
        randomly_drop_gap_datapoints(
            4, [10, 11, 12, 1, 2, 3],
            'Randomly drop a four month gap during the the non-growing season (Oct-Mar)',
            point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt
        )

        randomly_drop_gap_datapoints(
            6, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'Randomly drop a six month gap during the year',
            point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt
        )
        randomly_drop_gap_datapoints(
            6, [4, 5, 6, 7, 8, 9],
            'Randomly drop a six month gap during the growing season (Apr-Sept)',
            point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt
        )
        randomly_drop_gap_datapoints(
            6, [10, 11, 12, 1, 2, 3],
            'Randomly drop a six month gap during the the non-growing season (Oct-Mar)',
            point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt
        )


def randomly_drop_one_datapoint(
        months, title, point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt,
):
    output_list = []
    for point_id, year, window_df, year_month_mask in generate_window_dfs(
            point_id_list, data_df_dict, month_climo_dict, months=months
    ):
        tgt_mask = year_month_mask & window_df['etf'].notna()
        if not tgt_mask.any():
            print(f'{point_id} - {year} - no unmasked months, skipping')
            continue
        tgt_indices = window_df.loc[tgt_mask].sample(n=1).index
        output_list.extend(compute_filled_values(
            tgt_indices, point_id, year, window_df, month_climo_dict, etf_max_dict
        ))
    comparison_stats(pd.DataFrame(output_list), title=title, output_txt=output_txt)


def randomly_drop_gap_datapoints(
        dropped_months, months, title, point_id_list, data_df_dict, month_climo_dict, etf_max_dict, output_txt,
):
    output_list = []
    for point_id, year, window_df, year_month_mask in generate_window_dfs(
            point_id_list, data_df_dict, month_climo_dict, months=months
    ):
        # For the target year, identify all the gap windows with at least 1 month of data
        # This approach is assuming the gap will be 2 or months in the gap
        gap_mask = window_df['etf'].notna()
        for i in range(dropped_months - 1):
            gap_mask = gap_mask | window_df['etf'].notna().shift(-(i + 1))

        tgt_mask = year_month_mask & gap_mask
        if not tgt_mask.any():
            print(f'{point_id} - {year} - no unmasked months, skipping')
            continue

        tgt_indices = window_df.loc[tgt_mask].sample(n=1).index

        # Add an extra month index for each dropped month
        tgt_indices.freq = 'ms'
        for i in range(dropped_months - 1):
            tgt_indices = tgt_indices.append(pd.DatetimeIndex([tgt_indices[-1] + pd.DateOffset(months=1)]))

        values = compute_filled_values(
            tgt_indices, point_id, year, window_df, month_climo_dict, etf_max_dict
        )

        # Only keep values dictionaries that had data originally
        values = [v for v in values if not np.isnan(v['original'])]

        # Only keep 1 of the filled values from the window
        output_list.extend(random.sample(values, 1))

        del window_df, tgt_mask, tgt_indices, gap_mask

    comparison_stats(pd.DataFrame(output_list), title=title, output_txt=output_txt)


# def read_points_csv_df(points_folder):
#     # Read the separate points CSV files into a single dataframe
#     print('Reading mgrs point csv files')
#     points_df_list = [
#         pd.read_csv(os.path.join(points_folder, f'points_{mgrs_zone}_{nlcd_year}.csv'), index_col=None, header=0)
#         for nlcd_year in [2024]
#         for mgrs_zone in MGRS_ZONES
#         if os.path.isfile(os.path.join(points_folder, f'points_{mgrs_zone}_{nlcd_year}.csv'))
#     ]
#     points_df = pd.concat(points_df_list, axis=0, ignore_index=True)
#     print(f'Points: {len(points_df.index)}')
#
#     # The mgrs_zone value will eventually be added to the csv files
#     points_df['mgrs_zone'] = points_df['mgrs_tile'].str.slice(0, 3)
#
#     # Add a unique index to the points dataframe
#     points_df['index_group'] = points_df.groupby(['mgrs_tile', 'nlcd']).cumcount()
#     points_df['point_id'] = (
#             points_df["mgrs_tile"].str.upper() + '_' +
#             'nlcd' + points_df["nlcd"].astype(str).str.zfill(2) + '_' +
#             points_df["index_group"].astype(str).str.zfill(2)
#     )
#     del points_df['index_group']
#
#     # Round the lat and lon to 8 decimal places (probably should be 6)
#     points_df['latitude'] = round(points_df['latitude'], 8)
#     points_df['longitude'] = round(points_df['longitude'], 8)
#
#     return points_df


def read_data_csv_df(data_folder):
    # Read the CSV files into separate dataframes for each point
    print('Reading mgrs data csv files')
    data_df_dict = {}
    for mgrs_zone in MGRS_ZONES:
        # print(mgrs_zone)
        if not os.path.isfile(os.path.join(data_folder, f'data_{mgrs_zone}.csv')):
            continue

        mgrs_df = pd.read_csv(os.path.join(data_folder, f'data_{mgrs_zone}.csv'), index_col=None, header=0)

        # Set MGRS value to upper case
        # (at some point change this in all the data CSV files)
        mgrs_df['mgrs_tile'] = mgrs_df['mgrs_tile'].str.upper()
        mgrs_df['mgrs_zone'] = mgrs_df['mgrs_zone'].str.upper()

        # Compute the ET fraction
        mgrs_df['etf'] = mgrs_df['et'] / mgrs_df['eto']

        # Get the month for computing climos
        mgrs_df['date'] = pd.to_datetime(mgrs_df['date'])
        mgrs_df['year'] = mgrs_df['date'].dt.year
        mgrs_df['month'] = mgrs_df['date'].dt.month

        # Confirm that specific NLCD categories are not included
        # TODO: This probably isn't needed and switch to a check instead of masking
        for nlcd_skip in [11, 12, 21, 22, 23]:
            mgrs_df = mgrs_df[mgrs_df['nlcd'] != nlcd_skip]

        # Save dataframe for each point
        for point_id in mgrs_df['point_id'].unique():
            site_df = mgrs_df.loc[mgrs_df['point_id'] == point_id].copy()
            site_df.set_index('date', drop=True, inplace=True)
            site_df.sort_index(inplace=True)
            data_df_dict[point_id] = site_df
            del site_df
        del mgrs_df

    return data_df_dict


def generate_window_dfs(
        point_id_list,
        data_df_dict,
        month_climo_dict,
        months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        years=[2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        exclude_months_without_climo=True,
        min_month_count=3,
):
    """Generate Window Dataframes"""
    for i, point_id in enumerate(point_id_list):
        # Assume the data df dictionary exists in the global scope
        site_df = data_df_dict[point_id]

        for year in years:
            # Pull a three year window for each target year so that there are images to interpolate and fill from
            window_df = site_df[(site_df.index.year >= (year - 1)) & (site_df.index.year <= (year + 1))].copy()

            # If excluding months without climos, set them to NaN here
            # TODO: Test out adding the climo values to the window_df here
            #   instead of in the compute_filled_values() function below
            if exclude_months_without_climo:
                merge_df = pd.merge(window_df[['month']], month_climo_dict[point_id]['etf'], how="left", on="month")
                climo_nan_mask = merge_df['etf'].isna().values
                window_df.loc[climo_nan_mask, ['etf', 'et', 'count']] = np.nan

            year_mask = window_df.index.year == year
            year_month_mask = year_mask & window_df.index.month.isin(months)

            if window_df.loc[year_mask, 'etf'].count() < min_month_count:
                # Check if there are enough months in the target year
                # print(f'{point_id} - {i} - {year} - not enough unmasked months, skipping')
                continue
            elif window_df.loc[year_month_mask, 'etf'].isna().all():
                # Check that there are target months with data in the
                # print(f'{point_id} - {i} - {year} - no unmasked months in year/months, skipping')
                continue
            elif (window_df.loc[(window_df.index.year == year - 1), 'etf'].isna().all() or
                  window_df.loc[(window_df.index.year == year + 1), 'etf'].isna().all()):
                # Check if there is data in the prev/next year to interpolate from
                # print(f'{point_id} - {i} - {year} - no unmasked months in next/prev year, skipping')
                continue

            yield point_id, year, window_df, year_month_mask


def compute_filled_values(tgt_indices, point_id, year, window_df, month_climo_dict, etf_max_dict):
    """"""
    # Get a copy of the target value before clearing
    original_etf = window_df.loc[tgt_indices, 'etf'].values

    # Set the target row values to NaN
    window_df.loc[tgt_indices, ('etf', 'et', 'count')] = np.nan

    # Setup the Whittaker Smoothing for the full dataframe outside of the index loop
    # The smoothing function needs all nans filled with a value
    # The fill value is not important as long as the weight value is set to 0
    window_df['temp'] = window_df['etf'].copy()
    window_df.loc[np.isnan(window_df['temp']), 'temp'] = -1
    etf = window_df['temp'].values

    # TODO: Make sure weights are set to 0 for all temp==-1 rows
    #   This might be happening already with the .fillna(0) call but double check
    #   Right now the code is assuming count is NaN if etf is NaN

    # Default weights with 1 for data and 0 for missing values
    weight_a = window_df['count'].clip(lower=1, upper=1).fillna(0)
    if not any(weight_a):
        print(f'{point_id} - {year} - all weights 0, skipping')
        return []
    # CGM - I tested out building the smoother once and then updating lambda, but it didn't seem any faster
    whit_a_0p50 = WhittakerSmoother(lmbda=0.5, order=2, data_length=len(weight_a), weights=weight_a).smooth(etf)
    whit_a_0p20 = WhittakerSmoother(lmbda=0.2, order=2, data_length=len(weight_a), weights=weight_a).smooth(etf)
    whit_a_0p10 = WhittakerSmoother(lmbda=0.1, order=2, data_length=len(weight_a), weights=weight_a).smooth(etf)
    whit_a_0p05 = WhittakerSmoother(lmbda=0.05, order=2, data_length=len(weight_a), weights=weight_a).smooth(etf)
    whit_a_0p01 = WhittakerSmoother(lmbda=0.01, order=2, data_length=len(weight_a), weights=weight_a).smooth(etf)

    # CGM - I was testing out trying different weights but it didn't seem to change the values at all
    # # Compute weights based on the the scene count value
    # # Set count 0 images to a weight of 0
    # weight = window_df['count'].clip(lower=0, upper=1).fillna(0)

    # # Compute weights based on the the scene count value
    # # Set counts of 0 to a weight of 0.5 and all other to 1
    # weight = window_df['count'].add(1).clip(upper=2).divide(2).fillna(0)

    # # Compute weights based on the scene count value
    # # Set count weights as: 0 -> 0, 1 -> 0.5, 2+ -> 1
    # weight = window_df['count'].fillna(0).clip(upper=2).divide(2)

    # Process each target index separately
    values = []
    for i, (tgt_index, tgt_i) in enumerate(zip(tgt_indices, window_df.index.get_indexer(tgt_indices))):
        interp_value = window_df['etf'].interpolate(method='linear').loc[tgt_index]

        # Climos for all years
        climo_mean = month_climo_dict[point_id].loc[tgt_index.month, 'etf']
        climo_count = month_climo_dict[point_id].loc[tgt_index.month, 'count']
        climo_median = month_climo_dict[point_id].loc[tgt_index.month, 'etf_median']
        # # Climos with the target year excluded (not sure if this matters)
        # climo_mean = month_climo_dict[point_id][tgt_index.year].loc[tgt_index.month, 'etf']
        # climo_count = month_climo_dict[point_id][tgt_index.year].loc[tgt_index.month, 'count']
        # climo_median = month_climo_dict[point_id][tgt_index.year].loc[tgt_index.month, 'etf_median']

        # Compute various combinations of averaging the climo and interpolate values
        # Simple mean
        interp_clim_a = (climo_mean + interp_value) / 2
        # Simple mean with the median climo
        interp_clim_c = (climo_median + interp_value) / 2
        # Weight the climo based on the number of months in the climo?
        climo_months = 10
        interp_clim_b = (climo_mean * climo_count + interp_value * climo_months) / (climo_count + climo_months)

        # Conor's Approach
        # There is probably an easier way, but splitting the dataframe at the target index seemed to work pretty well
        window_prev_df = window_df.iloc[:tgt_i]
        window_next_df = window_df.iloc[tgt_i + 1:]
        prev_index = window_prev_df['etf'].last_valid_index()
        next_index = window_next_df['etf'].first_valid_index()
        prev_i = window_df.index.get_loc(prev_index)
        next_i = window_df.index.get_loc(next_index)
        w_prev = 0.5 * math.exp(1 - (tgt_i - prev_i))
        w_next = 0.5 * math.exp(1 - (next_i - tgt_i))
        value_prev = window_df['etf'].iloc[prev_i]
        value_next = window_df['etf'].iloc[next_i]
        climo_prev = month_climo_dict[point_id].loc[prev_index.month, 'etf']
        climo_next = month_climo_dict[point_id].loc[next_index.month, 'etf']
        conor = w_prev * (value_prev - climo_prev) + w_next * (value_next - climo_next) + climo_mean

        values.append({
            'index': tgt_index,
            'point_id': point_id,
            'mgrs': point_id.split('_')[0],
            'nlcd': int(point_id.split('_')[1][4:6]),
            'original': original_etf[i],
            # Filled values
            'interpolate': interp_value,
            'climo_mean': climo_mean,
            'climo_median': climo_median,
            'conor': conor,
            'interp_clim_a': interp_clim_a,
            'interp_clim_b': interp_clim_b,
            'interp_clim_c': interp_clim_c,
            'whit_a_0p50': min(max(whit_a_0p50[tgt_i], 0), etf_max_dict[point_id]),
            'whit_a_0p20': min(max(whit_a_0p20[tgt_i], 0), etf_max_dict[point_id]),
            'whit_a_0p10': min(max(whit_a_0p10[tgt_i], 0), etf_max_dict[point_id]),
            'whit_a_0p05': min(max(whit_a_0p05[tgt_i], 0), etf_max_dict[point_id]),
            'whit_a_0p01': min(max(whit_a_0p01[tgt_i], 0), etf_max_dict[point_id]),
        })

    return values


def comparison_stats(
        df, x_col='original', y_cols=[], title='', print_flag=True, write_flag=True, output_txt=None
):
    """"""
    output = [title]

    if not y_cols:
        y_cols = [
            'interpolate', 'climo_mean', 'climo_median',
            'conor',
            'interp_clim_a', 'interp_clim_b', 'interp_clim_c',
            'whit_a_0p50', 'whit_a_0p20', 'whit_a_0p10', 'whit_a_0p05', 'whit_a_0p01',
        ]

    # TODO: Build the format strings based on the number of parameters instead of hardcoding
    output.append('  {:>16s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}'.format(
        'method', 'rmse', 'mae', 'mbe', 'm', 'b', 'r2', 'n'
    ))
    for y_col in y_cols:
        # Remove any NaN rows before computing statistics
        stat_df = df[df[y_col].notna()]
        model = sklearn.linear_model.LinearRegression()
        model.fit(stat_df[[x_col]], stat_df[y_col])

        output.append('  {:>16s} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8d}'.format(
            y_col,
            sklearn.metrics.root_mean_squared_error(stat_df[x_col], stat_df[y_col]),
            sklearn.metrics.mean_absolute_error(stat_df[x_col], stat_df[y_col]),
            np.mean(stat_df[y_col] - stat_df[x_col]),
            # np.mean(stat_df[x_col] - stat_df[y_col]),
            # sklearn.metrics.r2_score(stat_df[x_col], stat_df[y_col]),
            model.coef_[0],
            model.intercept_,
            model.score(stat_df[[x_col]], stat_df[y_col]),
            # This count doesn't seem to change even when there are NaN values in the dataframe
            stat_df[y_col].count(),
        ))

    if print_flag:
        print('\n'.join(output))
    if write_flag:
        with open(output_txt, 'a') as output_f:
            output_f.write('\n'.join(output + ['\n']))


if __name__ == "__main__":
    main()
