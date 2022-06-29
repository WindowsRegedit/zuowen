import streamlit as st


st.title("作文生成器")
st.header("吴凡的作文生成器")
title = st.text_input("标题", placeholder="家乡的四季", help="作文标题")
context = st.text_input("作文上文", placeholder="家乡的四季,最美不过了", help="作文上文，可以理解为作文的开头（第一句话）")
max_len = st.slider(
    "作文长度",
    min_value=100,
    max_value=5000,
    step=50,
    value=200,
    help="作文长度")
st.subheader("高级")
temperature = st.slider(
    "输入温度",
    min_value=0.0,
    max_value=2.0,
    step=0.1,
    value=1.0)
context_len = st.slider(
    "参考上文长度",
    min_value=200,
    max_value=1000,
    step=50,
    help="每一步生成时，参考的上文的长度")
topk = st.slider("最高几选一", min_value=0, max_value=10)
topp = st.slider(
    "最高积累概率",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    value=0.85)


def generate_next_token(input_ids):
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    next_token_logits = logits[0, -1, :]
    next_token_logits = next_token_logits / temperature
    next_token_logits[unk_id] = -float("Inf")
    filtered_logits = top_k_top_p_filtering(
        next_token_logits, top_k=topk, top_p=topp)
    next_token_id = torch.multinomial(
        F.softmax(
            filtered_logits,
            dim=-1),
        num_samples=1)
    return next_token_id


@st.cache(show_spinner=False)
def generate(max_len):
    title_ids = tokenizer.encode(title, add_special_tokens=False)
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    input_ids = title_ids + [sep_id] + context_ids
    cur_len = len(input_ids)
    last_token_id = input_ids[-1]
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    while True:
        next_token_id = generate_next_token(input_ids[:, -context_len:])
        input_ids = torch.cat((input_ids, next_token_id.unsqueeze(0)), dim=1)
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


def gen_zuowen(title, context, max_len):
    print("标题：{}".format(title))
    print("上文：{}".format(context))

    result = generate(max_len)
    result = result.split("<sep>")[1]
    print("结果：{}\n".format(result))
    return result


if st.checkbox("禁用CUDA"):
    device = "cpu"
    st.warning("您禁用了CUDA(GPU).这会导致预测速度变慢！")
if st.button("开始自动写作"):
    with st.spinner("正在生成中，请稍等......"):
        from transformers import GPT2LMHeadModel, CpmTokenizer

        from zuowen.utils import *

        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"
        cuda = torch.cuda.is_available()
        device = "cuda" if cuda else "cpu"
        tokenizer = CpmTokenizer(
            vocab_file=os.path.join(
                absolute_path,
                "vocab",
                "chinese_vocab.model"))
        eod_id = tokenizer.convert_tokens_to_ids("<eod>")
        sep_id = tokenizer.sep_token_id
        unk_id = tokenizer.unk_token_id
        model = GPT2LMHeadModel.from_pretrained("WindowsRegedit/zuowen")
        model.eval()
        model = model.to(device)
        res = gen_zuowen(title, context, max_len)
    result = st.text_area(title, res, height=200)
