import pickle
import tempfile
from datetime import datetime
from os.path import join

import streamlit as st
import torch.nn.utils.rnn as rnn_utils
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, CpmTokenizer
from zuowen.data_parallel import BalancedDataParallel
from zuowen.dataset import CPMDataset
from zuowen.utils import *

st.sidebar.warning("想要获取与训练数据集，需要先预先将作文整合为pkl文件，请先使用前面一个工具进行转换！")

datas = {"loss": []}


def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(
        batch, batch_first=True, padding_value=5)
    labels = rnn_utils.pad_sequence(
        batch, batch_first=True, padding_value=-100)
    return input_ids, labels


def load_dataset():
    print("加载训练集")

    train_list = pickle.load(train_file)
    train_dataset = CPMDataset(train_list, max_len)

    return train_dataset


def train_epoch(model, train_dataloader, optimizer, scheduler,
                epoch):
    model.train()
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

            total_loss += loss.item()
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm)

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % log_step == 0:
                datas["loss"].append(loss.item() * gradient_accumulation_steps)
                if len(datas["loss"]) > 50:
                    datas["loss"].pop(0)
                train_chart.line_chart(datas)

            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("警告：内存不足")
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
            else:
                print(str(exception))
                raise exception

    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    print(
        "epoch {}: loss {}, predict_acc {}".format(
            epoch + 1,
            epoch_mean_loss,
            epoch_mean_acc))

    print("保存 epoch 为 {} 为模型".format(epoch + 1))
    model_path = join(save_model_path, "epoch{}".format(epoch + 1))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(model_path)
    print("epoch {} 完成！".format(epoch + 1))
    epoch_finish_time = datetime.now()
    print(
        "一次epoch所花时间： {}".format(
            epoch_finish_time -
            epoch_start_time))

    return epoch_mean_loss


def train(model, train_dataset):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True)
    t_total = len(
        train_dataloader) // gradient_accumulation_steps * epochs
    optimizer = transformers.AdamW(
        model.parameters(), lr=lr, eps=1.0e-09)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    print("开始训练")

    train_losses = []
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        train_loss = train_epoch(
            model=model, train_dataloader=train_dataloader,
            optimizer=optimizer, scheduler=scheduler,
            epoch=epoch)
        train_losses.append(round(train_loss, 4))
        print("训练 Loss 列表{}".format(train_losses))

    print("训练完成")
    print("总训练Loss:{}".format(train_losses))


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


st.title("吴凡的作文训练器")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"

train_file = st.file_uploader("提前转换成pkl格式的数据集", type=["pkl"])
warmup_steps = st.number_input("warm up步数", value=4000)
num_workers = st.number_input("dataloader加载数据时使用的线程数量", value=0)
epochs = st.number_input("训练的最大轮次", value=100)
batch_size = st.number_input("训练的batch size", value=16)
log_step = st.number_input("记录日志的步数", help="多少步汇报一次loss", value=1)
gpu0_bsz = st.number_input("GPU0负载", value=5, help="GPU0负载，仅在多GPU时有用")
max_len = st.number_input("最大长度", value=200, help="训练时，输入数据的最大长度")
gradient_accumulation_steps = st.number_input("梯度积累的步数", value=15)
ignore_index = st.slider(
    "忽略位置",
    value=-100,
    min_value=-500,
    max_value=500,
    step=50,
    help="对于忽略位置的label token不计算梯度")
max_grad_norm = st.slider(
    "Max Grad Norm",
    value=1.0,
    min_value=0.0,
    max_value=10.0)
lr = st.slider("学习率", value=1.5e-4, min_value=0.0, max_value=1.0, step=0.00001)
config_file = st.selectbox(
    "数据配置文件", [
        "cpm-small.json", "cpm-medium.json"], index=0)
cuda = not st.checkbox("禁用GPU", help="禁止使用GPU预测")

if st.button("开启训练"):
    cuda = torch.cuda.is_available() and cuda
    device = "cuda" if cuda else "cpu"
    print(device)
    set_random_seed(random.randint(1000, 10000), cuda)

    tokenizer = CpmTokenizer(
        vocab_file=os.path.join(
            absolute_path,
            "vocab",
            "chinese_vocab.model"))
    eod_id = tokenizer.convert_tokens_to_ids("<eod>")
    pad_id = tokenizer.pad_token_id
    save_model_path = tempfile.gettempdir()

    train_chart = st.line_chart(datas)

    model = GPT2LMHeadModel.from_pretrained("WindowsRegedit/zuowen")
    model = model.to(device)
    print("模型配置:\n{}".format(model.config.to_json_string()))
    assert model.config.vocab_size == tokenizer.vocab_size

    if cuda and torch.cuda.device_count() > 1:
        model = BalancedDataParallel(gpu0_bsz, model, dim=0).cuda()
        print("使用 GPU {} 以训练".format(device))
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print("模型参数数量 {}".format(num_parameters))
    train_dataset = load_dataset()
    train(model, train_dataset)
