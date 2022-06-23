import argparse
import pickle
from datetime import datetime
from os.path import join

import torch.nn.utils.rnn as rnn_utils
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Config, CpmTokenizer
from zuowen.data_parallel import BalancedDataParallel
from zuowen.dataset import CPMDataset
from zuowen.utils import *


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="0,1,2,3,4,5,6,7,8,9",
        type=str,
        required=False,
        help="设置使用哪些显卡")
    parser.add_argument("--no_cuda", action="store_true", help="不使用GPU进行训练")
    parser.add_argument(
        "--vocab_path",
        default=os.path.join(
            absolute_path,
            "vocab",
            "chinese_vocab.model"),
        type=str,
        required=False,
        help="sp模型路径")
    parser.add_argument(
        "--model_config",
        default=os.path.join(
            absolute_path,
            "config",
            "cpm-small.json"),
        type=str,
        required=False,
        help="需要从头训练一个模型时，模型参数的配置文件")
    parser.add_argument(
        "--train_path",
        default=os.path.join(
            absolute_path,
            "data",
            "cpm-small.json"),
        type=str,
        required=False,
        help="经过预处理之后的数据存放路径")
    parser.add_argument(
        "--max_len",
        default=200,
        type=int,
        required=False,
        help="训练时，输入数据的最大长度")

    parser.add_argument(
        "--log_path",
        default=os.path.join(
            absolute_path,
            "log",
            "train.log"),
        type=str,
        required=False,
        help="训练日志存放位置")
    parser.add_argument(
        "--ignore_index",
        default=-100,
        type=int,
        required=False,
        help="对于ignore_index的label token不计算梯度")
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        required=False,
        help="训练的最大轮次")
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        required=False,
        help="训练的batch size")
    parser.add_argument(
        "--gpu0_bsz",
        default=6,
        type=int,
        required=False,
        help="0号卡的batch size")
    parser.add_argument(
        "--lr",
        default=1.5e-4,
        type=float,
        required=False,
        help="学习率")
    parser.add_argument(
        "--eps",
        default=1.0e-09,
        type=float,
        required=False,
        help="AdamW优化器的衰减率")
    parser.add_argument(
        "--log_step",
        default=1,
        type=int,
        required=False,
        help="多少步汇报一次loss")
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=6,
        type=int,
        required=False,
        help="梯度积累的步数")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        required=False)
    parser.add_argument(
        "--save_model_path",
        default=os.path.join(
            absolute_path,
            "model"),
        type=str,
        required=False,
        help="模型输出路径")
    parser.add_argument(
        "--pretrained_model",
        default="WindowsRegedit/zuowen",
        type=str,
        required=False,
        help="预训练的模型的路径")
    parser.add_argument(
        "--seed",
        type=int,
        default=os.urandom(16),
        help="设置随机种子")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="dataloader加载数据时使用的线程数量")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=4000,
        help="warm up步数")
    args = parser.parse_args()
    return args


def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(
        batch, batch_first=True, padding_value=5)
    labels = rnn_utils.pad_sequence(
        batch, batch_first=True, padding_value=-100)
    return input_ids, labels


def load_dataset(logger, args):
    logger.info("加载训练集")
    train_path = args.train_path

    with open(train_path, "rb") as f:
        train_list = pickle.load(f)
        train_dataset = CPMDataset(train_list, args.max_len)

    return train_dataset


def train_epoch(model, train_dataloader, optimizer, scheduler, logger,
                epoch, args):
    model.train()
    device = args.device
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()

    total_loss = 0
    epoch_correct_num = 0
    epoch_total_num = 0
    for batch_idx, (input_ids, labels) in tqdm(enumerate(train_dataloader)):
        try:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()
            batch_correct_num, batch_total_num = calculate_acc(
                logits, labels, ignore_index=ignore_index)
            epoch_correct_num += batch_correct_num
            epoch_total_num += batch_total_num
            batch_acc = batch_correct_num / batch_total_num

            total_loss += loss.item()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % args.log_step == 0:
                logger.info(
                    "batch {} of epoch {}, loss {}, batch_acc {}, lr {}".format(
                        batch_idx + 1,
                        epoch + 1,
                        loss.item() * args.gradient_accumulation_steps,
                        batch_acc,
                        scheduler.get_lr()))

            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("警告：内存不足")
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    logger.info(
        "epoch {}: loss {}, predict_acc {}".format(
            epoch + 1,
            epoch_mean_loss,
            epoch_mean_acc))

    logger.info("保存 epoch 为 {} 为模型".format(epoch + 1))
    model_path = join(args.save_model_path, "epoch{}".format(epoch + 1))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(model_path)
    logger.info("epoch {} 完成！".format(epoch + 1))
    epoch_finish_time = datetime.now()
    logger.info(
        "一次epoch所花时间： {}".format(
            epoch_finish_time -
            epoch_start_time))

    return epoch_mean_loss


def train(model, logger, train_dataset, args):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True)
    t_total = len(
        train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = transformers.AdamW(
        model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    logger.info("开始训练")

    train_losses = []
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        train_loss = train_epoch(
            model=model, train_dataloader=train_dataloader,
            optimizer=optimizer, scheduler=scheduler,
            logger=logger, epoch=epoch, args=args)
        train_losses.append(round(train_loss, 4))
        logger.info("训练 Loss 列表{}".format(train_losses))

    logger.info("训练完成")
    logger.info("总训练Loss:{}".format(train_losses))


def calculate_loss(logit, target, pad_idx, smoothing=True):
    if smoothing:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(2))
        target = target[..., 1:].contiguous().view(-1)

        eps = 0.1
        n_class = logit.size(-1)

        one_hot = torch.zeros_like(logit).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)

        non_pad_mask = target.ne(pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()
    else:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
        labels = target[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(logit, labels, ignore_index=pad_idx)
    return loss


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word


def main():
    args = set_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"
    args.cuda = not args.no_cuda

    logger = set_logger(args.log_path)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = "cuda" if args.cuda else "cpu"
    args.device = device
    logger.info("使用设备：{}".format(device))

    set_random_seed(args.seed, args.cuda)

    tokenizer = CpmTokenizer(
        vocab_file=os.path.join(
            absolute_path,
            "vocab",
            "chinese_vocab.model"))
    args.eod_id = tokenizer.convert_tokens_to_ids("<eod>")
    args.pad_id = tokenizer.pad_token_id

    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    if args.pretrained_model:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    else:
        model_config = GPT2Config.from_json_file(args.model_config)
        model = GPT2LMHeadModel(config=model_config)
    model = model.to(device)
    logger.info("模型配置:\n{}".format(model.config.to_json_string()))
    assert model.config.vocab_size == tokenizer.vocab_size

    if args.cuda and torch.cuda.device_count() > 1:
        model = BalancedDataParallel(args.gpu0_bsz, model, dim=0).cuda()
        logger.info("使用 GPU {} 以训练".format(args.device))
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info("模型参数数量 {}".format(num_parameters))
    logger.info("参数：{}".format(args))
    train_dataset = load_dataset(logger, args)
    train(model, logger, train_dataset, args)


if __name__ == "__main__":
    main()
