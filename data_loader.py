from enum import Enum
from typing import Optional

import pandas as pd

class Dataset(Enum):
    PRA = "pra"
    ADULT = "adult"
    STUDENT_PERFORMANCE = "student_performance"
    STUDENT_DROPOUT = "student_dropout"


def load_dataset(dataset_name, number_records:Optional[int]=None):
    if dataset_name == Dataset.PRA:
        data = load_pra()
    elif dataset_name == Dataset.ADULT:
        data = load_adult()
    elif dataset_name == Dataset.STUDENT_PERFORMANCE:
        data = load_student_performance()
    elif dataset_name == Dataset.STUDENT_DROPOUT:
        data = load_student_dropout()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    if number_records is not None and number_records < len(data['df']):
        data['df'] = data['df'].sample(n=number_records, random_state=42).reset_index(drop=True)
    
    return data


def load_pra():
    # df = pd.read_csv("data/pra_2023.csv")
    _df1 = pd.read_csv("data/pra_A.csv")
    _df2 = pd.read_csv("data/pra_B.csv")
    _shared_cols = ['sex', 'nationality', 'age', 'province', 'place_birth']
    _df2 = _df2.drop(columns=_shared_cols)
    # concatenate the two sources
    df = pd.concat([_df1, _df2], axis=1)
    df = df.drop(columns=["matricule"])  # drop IDs

    should_be_categorical_columns = ['nationality', 'place_birth', 'sex', 'province', 'household_duties', 'relation_to_activity1', 'relation_to_activity2', 'relationship', 'main_occupation', 'availability', 'search_work', 'search_reason', 'search_steps', 'search_method', 'main_activity', 'main_prof_situation' ,'main_sector' ,'contract_type']
    for col in should_be_categorical_columns:
        df[col] = df[col].astype(object)

    min_number_of_random_column_in_combinations = 2
    max_number_of_random_column_in_combinations = 8
    return {
        'df': df,
        'min_number_of_random_column_in_combinations': min_number_of_random_column_in_combinations,
        'max_number_of_random_column_in_combinations': max_number_of_random_column_in_combinations
        }


def load_adult():
    df = pd.read_csv("data/adult.csv")
    df = df.drop(columns=["fnlwgt"])

    # replace in all columns the '?' with None
    df = df.replace('?', None)

    return {
        'df': df,
        'min_number_of_random_column_in_combinations': 2,
        'max_number_of_random_column_in_combinations': 8
        }


def load_student_performance():
    df = pd.read_csv("data/student-mat.csv", delimiter=";")
    
    return {
        'df': df,
        'min_number_of_random_column_in_combinations': 2,
        'max_number_of_random_column_in_combinations': 15
        }

def load_student_dropout():
    df = pd.read_csv("data/students_dropout.csv", delimiter=";")

    should_be_categorical_columns = ['Marital status', 'Application mode', 'Application order', 'Course', 'Daytime/evening attendance', 'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation", "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "International", "Target" ]
    for col in should_be_categorical_columns:
        df[col] = df[col].astype(object)

    return {
        'df': df,
        'min_number_of_random_column_in_combinations': 2,
        'max_number_of_random_column_in_combinations': 10
        }