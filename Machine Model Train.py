import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fftpack import rfft
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load


def create_meal_data(
        insulin_data_frame,
        cgm_data_frame,
        date_type
):
    insulin_df = insulin_data_frame.copy()
    insulin_df = insulin_df.set_index('date_time_stamp')
    timestamp_with_correct_difference = insulin_df.sort_values(by='date_time_stamp',
                                                               ascending=True).dropna().reset_index()
    timestamp_with_correct_difference['BWZ Carb Input (grams)'].replace(0.0, np.nan, inplace=True)
    timestamp_with_correct_difference = timestamp_with_correct_difference.dropna()
    timestamp_with_correct_difference = timestamp_with_correct_difference.reset_index().drop(columns='index')
    valid_timestamp_list = []
    for idx, i in enumerate(timestamp_with_correct_difference['date_time_stamp']):
        try:
            value = (timestamp_with_correct_difference['date_time_stamp'][idx + 1] - i).seconds / 60.0
            if value >= 120:
                valid_timestamp_list.append(i)
        except KeyError:
            break

    list1 = []
    if date_type == 1:
        for idx, i in enumerate(valid_timestamp_list):
            start = pd.to_datetime(i - timedelta(minutes=30))
            end = pd.to_datetime(i + timedelta(minutes=90))
            get_date = i.date().strftime('%-m/%-d/%Y')
            list1.append(
                cgm_data_frame.loc[cgm_data_frame['Date'] == get_date].set_index('date_time_stamp').between_time(
                    start_time=start.strftime('%-H:%-M:%-S'), end_time=end.strftime('%-H:%-M:%-S'))[
                    'Sensor Glucose (mg/dL)'].values.tolist())
        return pd.DataFrame(list1)
    else:
        for idx, i in enumerate(valid_timestamp_list):
            start = pd.to_datetime(i - timedelta(minutes=30))
            end = pd.to_datetime(i + timedelta(minutes=90))
            get_date = i.date().strftime('%Y-%m-%d')
            list1.append(cgm_data_frame.loc[pd.to_datetime(cgm_data_frame['Date']) ==
                                            pd.to_datetime(get_date)].set_index('date_time_stamp').between_time(
                start_time=start.strftime('%H:%M:%S'),
                end_time=end.strftime('%H:%M:%S')
            )['Sensor Glucose (mg/dL)'].values.tolist())
        return pd.DataFrame(list1)


def create_no_meal_data(
        insulin_data_frame,
        cgm_data_frame
):
    insulin_no_meal_df = insulin_data_frame.copy()
    test1_df = insulin_no_meal_df.sort_values(
        by='date_time_stamp',
        ascending=True
    ).replace(0.0, np.nan).dropna().copy()
    test1_df = test1_df.reset_index().drop(columns='index')
    valid_timestamp = []
    for idx, i in enumerate(test1_df['date_time_stamp']):
        try:
            value = (test1_df['date_time_stamp'][idx + 1] - i).seconds // 3600
            if value >= 4:
                valid_timestamp.append(i)
        except KeyError:
            break
    dataset = []
    for idx, i in enumerate(valid_timestamp):
        iteration_dataset = 1
        try:
            length_of_24_dataset = len(
                cgm_data_frame.loc[(cgm_data_frame['date_time_stamp'] >=
                                    valid_timestamp[idx] +
                                    pd.Timedelta(hours=2)) &
                                   (cgm_data_frame['date_time_stamp'] <
                                    valid_timestamp[idx + 1])]
            ) // 24
            while iteration_dataset <= length_of_24_dataset:
                if iteration_dataset == 1:
                    dataset.append(
                        cgm_data_frame.loc[(cgm_data_frame['date_time_stamp'] >=
                                            valid_timestamp[idx] +
                                            pd.Timedelta(hours=2)) &
                                           (cgm_data_frame['date_time_stamp'] <
                                            valid_timestamp[idx + 1])]
                        ['Sensor Glucose (mg/dL)']
                        [:iteration_dataset * 24].values.tolist())
                    iteration_dataset += 1
                else:
                    dataset.append(
                        cgm_data_frame.loc[(cgm_data_frame['date_time_stamp'] >=
                                            valid_timestamp[idx] +
                                            pd.Timedelta(hours=2)
                                            ) & (
                                cgm_data_frame['date_time_stamp'] < valid_timestamp[idx + 1]
                        )]['Sensor Glucose (mg/dL)']
                        [(iteration_dataset - 1) * 24:iteration_dataset * 24].values.tolist())
                    iteration_dataset += 1
        except IndexError:
            break
    return pd.DataFrame(dataset)


def create_meal_feature_matrix(meal_data):
    index = meal_data.isna().sum(axis=1).replace(0, np.nan).dropna().where(lambda x: x > 6).dropna().index
    meal_data_cleaned = meal_data.drop(meal_data.index[index]).reset_index().drop(columns='index')
    meal_data_cleaned = meal_data_cleaned.interpolate(method='linear', axis=1)
    index_to_drop_again = meal_data_cleaned.isna().sum(axis=1).replace(0, np.nan).dropna().index
    meal_data_cleaned = meal_data_cleaned.drop(meal_data.index[index_to_drop_again]).reset_index().drop(columns='index')
    meal_data_cleaned = meal_data_cleaned.dropna().reset_index().drop(columns='index')
    power_first_max = []
    index_first_max = []
    power_second_max = []
    index_second_max = []
    power_third_max = []
    for i in range(len(meal_data_cleaned)):
        array = abs(rfft(meal_data_cleaned.iloc[:, 0:30].iloc[i].values.tolist())).tolist()
        sorted_array = abs(rfft(meal_data_cleaned.iloc[:, 0:30].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        power_first_max.append(sorted_array[-2])
        power_second_max.append(sorted_array[-3])
        power_third_max.append(sorted_array[-4])
        index_first_max.append(array.index(sorted_array[-2]))
        index_second_max.append(array.index(sorted_array[-3]))
    meal_feature_matrix = pd.DataFrame()
    meal_feature_matrix['power_second_max'] = power_second_max
    meal_feature_matrix['power_third_max'] = power_third_max
    tm = meal_data_cleaned.iloc[:, 22:25].idxmin(axis=1)
    maximum = meal_data_cleaned.iloc[:, 5:19].idxmax(axis=1)
    list1 = []
    second_differential_data = []
    standard_deviation = []
    for i in range(len(meal_data_cleaned)):
        list1.append(np.diff(meal_data_cleaned.iloc[:, maximum[i]:tm[i]].iloc[i].tolist()).max())
        second_differential_data.append(
            np.diff(np.diff(meal_data_cleaned.iloc[:, maximum[i]:tm[i]].iloc[i].tolist())).max())
        standard_deviation.append(np.std(meal_data_cleaned.iloc[i]))
    meal_feature_matrix['2ndDifferential'] = second_differential_data
    meal_feature_matrix['standard_deviation'] = standard_deviation
    return meal_feature_matrix


def create_no_meal_feature_matrix(non_meal_data):
    index_to_remove_non_meal = non_meal_data.isna().sum(axis=1).replace(0, np.nan).dropna().where(
        lambda x: x > 5).dropna().index
    non_meal_data_cleaned = non_meal_data.drop(non_meal_data.index[index_to_remove_non_meal]).reset_index().drop(
        columns='index')
    non_meal_data_cleaned = non_meal_data_cleaned.interpolate(method='linear', axis=1)
    index_to_drop_again = non_meal_data_cleaned.isna().sum(axis=1).replace(0, np.nan).dropna().index
    non_meal_data_cleaned = non_meal_data_cleaned.drop(
        non_meal_data_cleaned.index[index_to_drop_again]).reset_index().drop(columns='index')
    non_meal_feature_matrix = pd.DataFrame()
    power_first_max = []
    index_first_max = []
    power_second_max = []
    index_second_max = []
    power_third_max = []
    for i in range(len(non_meal_data_cleaned)):
        array = abs(rfft(non_meal_data_cleaned.iloc[:, 0:24].iloc[i].values.tolist())).tolist()
        sorted_array = abs(rfft(non_meal_data_cleaned.iloc[:, 0:24].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        power_first_max.append(sorted_array[-2])
        power_second_max.append(sorted_array[-3])
        power_third_max.append(sorted_array[-4])
        index_first_max.append(array.index(sorted_array[-2]))
        index_second_max.append(array.index(sorted_array[-3]))
    non_meal_feature_matrix['power_second_max'] = power_second_max
    non_meal_feature_matrix['power_third_max'] = power_third_max
    first_differential_data = []
    second_differential_data = []
    standard_deviation = []
    for i in range(len(non_meal_data_cleaned)):
        first_differential_data.append(np.diff(non_meal_data_cleaned.iloc[:, 0:24].iloc[i].tolist()).max())
        second_differential_data.append(np.diff(np.diff(non_meal_data_cleaned.iloc[:, 0:24].iloc[i].tolist())).max())
        standard_deviation.append(np.std(non_meal_data_cleaned.iloc[i]))
    non_meal_feature_matrix['2ndDifferential'] = second_differential_data
    non_meal_feature_matrix['standard_deviation'] = standard_deviation
    return non_meal_feature_matrix


if __name__ == '__main__':
    insulin_data_frame = pd.read_csv(
        'InsulinData.csv',
        low_memory=False,
        usecols=['Date', 'Time', 'BWZ Carb Input (grams)']
    )

    cgm_data_frame = pd.read_csv(
        'CGMData.csv',
        low_memory=False,
        usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)']
    )

    insulin_data_frame['date_time_stamp'] = pd.to_datetime(
        insulin_data_frame['Date'] + ' ' + insulin_data_frame['Time']
    )
    cgm_data_frame['date_time_stamp'] = pd.to_datetime(
        cgm_data_frame['Date'] + ' ' + cgm_data_frame['Time']
    )

    insulin_data_frame_1 = pd.read_csv(
        'Insulin_patient2.csv',
        low_memory=False,
        usecols=['Date', 'Time', 'BWZ Carb Input (grams)']
    )

    cgm_data_frame_1 = pd.read_csv(
        'CGM_patient2.csv',
        low_memory=False,
        usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)']
    )

    insulin_data_frame_1['date_time_stamp'] = pd.to_datetime(
        insulin_data_frame_1['Date'] + ' ' + insulin_data_frame_1['Time']
    )
    cgm_data_frame_1['date_time_stamp'] = pd.to_datetime(
        cgm_data_frame_1['Date'] + ' ' + cgm_data_frame_1['Time']
    )

    meal_data = create_meal_data(
        insulin_data_frame,
        cgm_data_frame,
        1
    )
    meal_data1 = create_meal_data(
        insulin_data_frame_1,
        cgm_data_frame_1,
        2
    )
    meal_data = meal_data.iloc[:, 0:24]
    meal_data1 = meal_data1.iloc[:, 0:24]

    no_meal_data = create_no_meal_data(insulin_data_frame, cgm_data_frame)
    no_meal_data1 = create_no_meal_data(insulin_data_frame_1, cgm_data_frame_1)

    meal_feature_matrix = create_meal_feature_matrix(meal_data)
    meal_feature_matrix1 = create_meal_feature_matrix(meal_data1)
    meal_feature_matrix = pd.concat([meal_feature_matrix, meal_feature_matrix1]).reset_index().drop(columns='index')

    non_meal_feature_matrix = create_no_meal_feature_matrix(no_meal_data)
    non_meal_feature_matrix1 = create_no_meal_feature_matrix(no_meal_data1)
    non_meal_feature_matrix = pd.concat([non_meal_feature_matrix, non_meal_feature_matrix1]).reset_index().drop(
        columns='index'
    )

    meal_feature_matrix['label'] = 1
    non_meal_feature_matrix['label'] = 0
    total_data = pd.concat([meal_feature_matrix, non_meal_feature_matrix]).reset_index().drop(columns='index')
    dataset = shuffle(total_data, random_state=1).reset_index().drop(columns='index')
    principal_data = dataset.drop(columns='label')

    classifier = DecisionTreeClassifier(criterion='entropy')
    X, y = principal_data, dataset['label']
    classifier.fit(X, y)
    dump(classifier, 'model.pickle')
