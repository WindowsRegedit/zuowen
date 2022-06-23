import argparse
import os
import pickle

from tqdm import tqdm
from transformers import CpmTokenizer
from zuowen.utils import set_logger, absolute_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab_file",
        default=os.path.join(absolute_path, "vocab", "chinese_vocab.model"),
        type=str,
        required=False,
        help="词表路径")
    parser.add_argument(
        "--log_path",
        default=os.path.join(absolute_path, "log", "preprocess.log"),
        type=str,
        required=False,
        help="日志存放位置")
    parser.add_argument(
        "--data_path",
        default=os.path.join(absolute_path, "data", ""),
        type=str,
        required=False,
        help="数据集存放位置")
    parser.add_argument(
        "--save_path",
        default=os.path.join(absolute_path, "log", "train.pkl"),
        type=str,
        required=False,
        help="对训练数据集进行tokenize之后的数据存放位置")
    parser.add_argument(
        "--win_size",
        default=200,
        type=int,
        required=False,
        help="滑动窗口的大小，相当于每条数据的最大长度")
    parser.add_argument(
        "--step",
        default=200,
        type=int,
        required=False,
        help="滑动窗口的滑动步幅")
    args = parser.parse_args()

    logger = set_logger(args.log_path)

    tokenizer = CpmTokenizer(
        vocab_file=os.path.join(
            absolute_path,
            "vocab",
            "chinese_vocab.model"))
    eod_id = tokenizer.convert_tokens_to_ids("<eod>")
    sep_id = tokenizer.sep_token_id

    train_list = []
    logger.info("开始标记数据")
    for file in tqdm(os.listdir(args.data_path)):
        file = os.path.join(args.data_path, file)
        with open(file, "r", encoding="utf8") as reader:
            lines = reader.readlines()
            title = lines[1][3:].strip()
            lines = lines[7:]
            article = ""
            for line in lines:
                if line.strip() != "":
                    article += line
            title_ids = tokenizer.encode(title, add_special_tokens=False)
            article_ids = tokenizer.encode(article, add_special_tokens=False)
            token_ids = title_ids + [sep_id] + article_ids + [eod_id]

            win_size = args.win_size
            step = args.step
            start_index = 0
            end_index = win_size
            data = token_ids[start_index:end_index]
            train_list.append(data)
            start_index += step
            end_index += step
            while end_index + 50 < len(token_ids):
                data = token_ids[start_index:end_index]
                train_list.append(data)
                start_index += step
                end_index += step

    with open(args.save_path, "wb") as f:
        pickle.dump(train_list, f)


if __name__ == "__main__":
    main()
