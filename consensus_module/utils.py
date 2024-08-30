import pandas as pd

def get_formatted_dataset_and_indexes(dataset_name, id_column, target_column = 0):
    # importing dataset
    data = pd.read_csv(dataset_name)
    # getting dataset ids
    data_indexes = data[id_column].to_list()
    data = data.drop([target_column], axis=1) if target_column else data
    data = data.set_index(id_column)
    return data, data_indexes

def get_df_feature_names(df):
    return list(df.columns)

def get_df_row(dataset, row_number):
    return dataset.iloc[row_number]

def get_df_row_values(dataset, row_number):
    return dataset.values[row_number]

def get_round_percentage(weight):
    return round(weight*100, 1)