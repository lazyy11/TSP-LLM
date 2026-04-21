import os
import pandas as pd
import jsonlines


# ======== 常用配置：根据需要修改 ========
INPUT_CSV_PATH = '/home/leizy/24sum/CFRL/data/TH/final_combined_data.csv'
OUTPUT_JSON_DIR = '/home/leizy/24sum/CFRL/data/train_together_datasets/with_prompt/5_steps/jsonl_data'
WINDOW_LEN = 24
PRE_LEN = 5
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
WRITE_TEST_SPLIT = False  # 需要测试集时改为 True
# ========================================


def split_dataset(df, train_ratio, val_ratio):
    total_rows = len(df)
    train_size = int(total_rows * train_ratio)
    val_size = int(total_rows * val_ratio)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    return train_df, val_df, test_df


def build_pairs(df, window_len, pre_len):
    texts = []
    summaries = []
    num_rows = len(df)

    if num_rows <= window_len + pre_len:
        return texts, summaries

    num_clients = len(df.columns)
    for k in range(num_clients):
        col_name = df.columns[k]
        for i in range(num_rows - window_len - pre_len):
            start_ts = df.index[i].strftime('%m/%d/%Y %I:%M:%S %p')
            end_ts = df.index[i + window_len - 1].strftime('%m/%d/%Y %I:%M:%S %p')

            past_values = ', '.join(
                f"{df.iloc[j, k]:.15g}" for j in range(i, i + window_len)
            )
            future_timestamps = ', '.join(
                df.index[i + window_len + j].strftime('%m/%d/%Y %I:%M:%S %p')
                for j in range(pre_len)
            )
            future_values = ', '.join(
                f"{df.iloc[i + window_len + j, k]:.15g}" for j in range(pre_len)
            )

            prompt = (
                f"From {start_ts} to {end_ts}, the observation of {col_name} "
                f"for the past is {past_values} on each hour. "
                f"What is prediction on {future_timestamps}?"
            )
            answer = f"The prediction is {future_values}."

            texts.append(prompt)
            summaries.append(answer)

    return texts, summaries


def write_split(json_dir, split_name, texts, summaries):
    if not texts:
        print(f"{split_name} split 没有可写数据，跳过生成。")
        return

    os.makedirs(json_dir, exist_ok=True)
    output_path = os.path.join(json_dir, f"{split_name}.json")
    items = [{"text": texts[i], "summary": summaries[i]} for i in range(len(texts))]

    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(items)

    print(f"写入 {split_name} 数据 {len(items)} 条 -> {output_path}")


def main():
    if not os.path.exists(INPUT_CSV_PATH):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_CSV_PATH}")

    df = pd.read_csv(INPUT_CSV_PATH, parse_dates=[0], index_col=0)
    print(f"载入数据 {len(df)} 行，{len(df.columns)} 列")

    df_train, df_val, df_test = split_dataset(df, TRAIN_RATIO, VAL_RATIO)

    train_texts, train_summaries = build_pairs(df_train, WINDOW_LEN, PRE_LEN)
    val_texts, val_summaries = build_pairs(df_val, WINDOW_LEN, PRE_LEN)

    write_split(OUTPUT_JSON_DIR, 'train', train_texts, train_summaries)
    write_split(OUTPUT_JSON_DIR, 'val', val_texts, val_summaries)

    if WRITE_TEST_SPLIT:
        test_texts, test_summaries = build_pairs(df_test, WINDOW_LEN, PRE_LEN)
        write_split(OUTPUT_JSON_DIR, 'test', test_texts, test_summaries)

    print('完成！')


if __name__ == '__main__':
    main()


