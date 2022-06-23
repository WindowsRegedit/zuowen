import argparse

from transformers import GPT2LMHeadModel, CpmTokenizer
from zuowen.utils import *


def main():

    def generate_next_token(input_ids):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]
        next_token_logits = next_token_logits / args.temperature
        next_token_logits[unk_id] = -float("Inf")
        filtered_logits = top_k_top_p_filtering(
            next_token_logits, top_k=args.topk, top_p=args.topp)
        next_token_id = torch.multinomial(
            F.softmax(
                filtered_logits,
                dim=-1),
            num_samples=1)
        return next_token_id

    def generate(max_len):
        title_ids = tokenizer.encode(title, add_special_tokens=False)
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        input_ids = title_ids + [sep_id] + context_ids
        cur_len = len(input_ids)
        last_token_id = input_ids[-1]
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

        while True:
            next_token_id = generate_next_token(
                input_ids[:, -args.context_len:])
            input_ids = torch.cat(
                (input_ids, next_token_id.unsqueeze(0)), dim=1)
            cur_len += 1
            word = tokenizer.convert_ids_to_tokens(next_token_id.item())
            if cur_len >= max_len and last_token_id == 8 and next_token_id == 3:
                break
            if cur_len >= max_len and word in [
                    ".", "。", "！", "!", "?", "？", ",", "，"]:
                break
            if next_token_id == eod_id:
                break
        result = tokenizer.decode(input_ids.squeeze(0))
        return result

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="0",
        type=str,
        required=False,
        help="生成设备")
    parser.add_argument(
        "--temperature",
        default=1,
        type=float,
        required=False,
        help="生成温度")
    parser.add_argument(
        "--topk",
        default=0,
        type=int,
        required=False,
        help="最高几选一")
    parser.add_argument(
        "--topp",
        default=0.85,
        type=float,
        required=False,
        help="最高积累概率")
    parser.add_argument(
        "--repetition_penalty",
        default=1.0,
        type=float,
        required=False,
        help="重复惩罚参数")
    parser.add_argument(
        "--context_len",
        default=200,
        type=int,
        required=False,
        help="每一步生成时，参考的上文的长度")
    parser.add_argument(
        "--max_len",
        default=300,
        type=int,
        required=False,
        help="生成的最长长度")
    parser.add_argument(
        "--log_path",
        default=os.path.join(
            absolute_path,
            "log",
            "generate.log"),
        type=str,
        required=False,
        help="日志存放位置")
    parser.add_argument("--no_cuda", action="store_true", help="不使用GPU进行预测")
    parser.add_argument(
        "--model_path",
        type=str,
        default="WindowsRegedit/zuowen",
        help="模型存放位置")
    parser.add_argument("--title", type=str, default="家乡的四季", help="作文标题")
    parser.add_argument(
        "--context",
        type=str,
        default="家乡的四季,最美不过了",
        help="作文上文")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = "cuda" if args.cuda else "cpu"
    logger = set_logger(args.log_path)
    tokenizer = CpmTokenizer(
        vocab_file=os.path.join(
            absolute_path,
            "vocab",
            "chinese_vocab.model"))
    eod_id = tokenizer.convert_tokens_to_ids("<eod>")
    sep_id = tokenizer.sep_token_id
    unk_id = tokenizer.unk_token_id
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.eval()
    model = model.to(device)
    title = args.title
    context = args.context
    logger.info("标题：{}".format(title))
    logger.info("上文：{}".format(context))

    result = generate(args.max_len)
    result = result.split("<sep>")[1]
    logger.info("结果：{}\n".format(result))


if __name__ == "__main__":
    main()
