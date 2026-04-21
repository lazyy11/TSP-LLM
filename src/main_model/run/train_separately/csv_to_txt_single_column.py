import pandas as pd
import numpy as np
import os
import jsonlines

def generate_prompts(df, window_len, pre_len, output_path, k):
    with open(output_path, 'w') as file:
        for i in range(len(df) - window_len - pre_len):
            start_date = df.index[i].strftime('%m/%d/%Y %I:%M:%S %p')
            end_date = df.index[i + window_len - 1].strftime('%m/%d/%Y %I:%M:%S %p')
            true_values = ', '.join(
                f"{df.iloc[j, k]:.15g}" for j in range(i, i + window_len)
            )
            next_hours = ', '.join(
                df.index[i + window_len + j].strftime('%m/%d/%Y %I:%M:%S %p') for j in range(pre_len)
            )
            file.write(f"From {start_date} to {end_date}, the observation of {df.columns[k]} for the past is {true_values} on each hour. What is prediction on {next_hours}?\n")

def generate_labels(df, window_len, pre_len, output_path, k):
    with open(output_path, 'w') as file:
        for i in range(len(df) - window_len - pre_len):
            labels = ', '.join(
                f"{df.iloc[i + window_len + j, k]:.15g}" for j in range(pre_len)
            )
            file.write(f"The prediction is {labels}.\n")

def generate_no_prompt_prompts(df, window_len, pre_len, output_path, k):
    with open(output_path, 'w') as file:
        for i in range(len(df) - window_len - pre_len):
            true_values = ', '.join(
                f"{df.iloc[j, k]:.15g}" for j in range(i, i + window_len)
            )
            file.write(f"{true_values}.\n")

def generate_no_prompt_labels(df, window_len, pre_len, output_path, k):
    with open(output_path, 'w') as file:
        for i in range(len(df) - window_len - pre_len):
            labels = ', '.join(
                f"{df.iloc[i + window_len + j, k]:.15g}" for j in range(pre_len)
            )
            file.write(f"{labels}.\n")

def generate_json_data(df, window_len, pre_len, output_path, k, data_type):
    """
    Generate JSON format data for train and val datasets
    """
    items = []
    for i in range(len(df) - window_len - pre_len):
        start_date = df.index[i].strftime('%m/%d/%Y %I:%M:%S %p')
        end_date = df.index[i + window_len - 1].strftime('%m/%d/%Y %I:%M:%S %p')
        true_values = ', '.join(
            f"{df.iloc[j, k]:.15g}" for j in range(i, i + window_len)
        )
        next_hours = ', '.join(
            df.index[i + window_len + j].strftime('%m/%d/%Y %I:%M:%S %p') for j in range(pre_len)
        )
        
        text = f"From {start_date} to {end_date}, the observation of {df.columns[k]} for the past is {true_values} on each hour. What is prediction on {next_hours}?"
        
        labels = ', '.join(
            f"{df.iloc[i + window_len + j, k]:.15g}" for j in range(pre_len)
        )
        summary = f"The prediction is {labels}."
        
        items.append({"text": text, "summary": summary})
    
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(items)

def generate_no_prompt_json_data(df, window_len, pre_len, output_path, k, data_type):
    """
    Generate JSON format data for train and val datasets (no prompt version)
    """
    items = []
    for i in range(len(df) - window_len - pre_len):
        true_values = ', '.join(
            f"{df.iloc[j, k]:.15g}" for j in range(i, i + window_len)
        )
        
        text = f"{true_values}."
        
        labels = ', '.join(
            f"{df.iloc[i + window_len + j, k]:.15g}" for j in range(pre_len)
        )
        summary = f"{labels}."
        
        items.append({"text": text, "summary": summary})
    
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(items)

base_path = '/home/leizy/24sum/CFRL/data/'
files_name = ['TH']

# Define different prediction lengths
prediction_lengths = [1, 3, 5]

for f in files_name:
    input_path = base_path + f + '/final_combined_data.csv'
    df_daily = pd.read_csv(input_path, parse_dates=[0], index_col=0)

    total_rows = len(df_daily)
    train_size = int(total_rows * 0.7)
    val_size = int(total_rows * 0.1)
    test_size = total_rows - train_size - val_size

    df_train = df_daily.iloc[:train_size]
    df_val = df_daily.iloc[train_size:train_size + val_size]
    df_test = df_daily.iloc[train_size + val_size:]

    window_len = 24

    for pre_len in prediction_lengths:
        # Generate with_prompt data
        output_folder = f'/home/leizy/24sum/CFRL/data/train_separately_datasets/with_prompt/{pre_len}_step/{f}_S/'
    os.makedirs(output_folder, exist_ok=True)

        # generate train data as JSON
    for k in range(len(df_train.columns)):
        columns = df_train.columns
        path = output_folder + columns[k]
        os.makedirs(path, exist_ok=True)  # Create directory if it does not exist
            generate_json_data(df_train, window_len, pre_len, os.path.join(path, 'train.json'), k, 'train')

        # generate val data as JSON
    for k in range(len(df_val.columns)):
        columns = df_val.columns
        path = output_folder + columns[k]
        os.makedirs(path, exist_ok=True)  # Create directory if it does not exist
            generate_json_data(df_val, window_len, pre_len, os.path.join(path, 'val.json'), k, 'val')
        
        # generate test data (still as txt files)
    for k in range(len(df_test.columns)):
        columns = df_test.columns
        path = output_folder + columns[k]
        os.makedirs(path, exist_ok=True)  # Create directory if it does not exist
        generate_prompts(df_test, window_len, pre_len, os.path.join(path, 'test_x_prompt.txt'), k)

    for k in range(len(df_test.columns)):
        columns = df_test.columns
        path = output_folder + columns[k]
        os.makedirs(path, exist_ok=True)  # Create directory if it does not exist
        generate_labels(df_test, window_len, pre_len, os.path.join(path, 'test_y_prompt.txt'), k)

        # Generate no_prompt data
        output_folder_no_prompt = f'/home/leizy/24sum/CFRL/data/train_separately_datasets/no_prompt/{pre_len}_step/{f}_S/'
        os.makedirs(output_folder_no_prompt, exist_ok=True)

        # generate train data as JSON (no prompt)
        for k in range(len(df_train.columns)):
            columns = df_train.columns
            path = output_folder_no_prompt + columns[k]
            os.makedirs(path, exist_ok=True)  # Create directory if it does not exist
            generate_no_prompt_json_data(df_train, window_len, pre_len, os.path.join(path, 'train.json'), k, 'train')

        # generate val data as JSON (no prompt)
        for k in range(len(df_val.columns)):
            columns = df_val.columns
            path = output_folder_no_prompt + columns[k]
            os.makedirs(path, exist_ok=True)  # Create directory if it does not exist
            generate_no_prompt_json_data(df_val, window_len, pre_len, os.path.join(path, 'val.json'), k, 'val')
        
        # generate test data (no prompt, still as txt files)
        for k in range(len(df_test.columns)):
            columns = df_test.columns
            path = output_folder_no_prompt + columns[k]
            os.makedirs(path, exist_ok=True)  # Create directory if it does not exist
            generate_no_prompt_prompts(df_test, window_len, pre_len, os.path.join(path, 'test_x_prompt.txt'), k)

        for k in range(len(df_test.columns)):
            columns = df_test.columns
            path = output_folder_no_prompt + columns[k]
            os.makedirs(path, exist_ok=True)  # Create directory if it does not exist
            generate_no_prompt_labels(df_test, window_len, pre_len, os.path.join(path, 'test_y_prompt.txt'), k)

    print("done!")
