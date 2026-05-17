import os
import sys
import torch
import numpy as np
import cv2
import pickle
import gradio as gr
from torchvision import transforms
import argparse

# 确保能导入项目内模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.char_model import Model
from utils.class_emb_loader import load_class_emb, load_template

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# 预计算：类别嵌入 + 模板特征（只在启动时执行一次）
# ──────────────────────────────────────────────
def prepare_embeddings(args, net):
    """
    device-aware 版本的预计算函数。
    修正了原 prepare_emb.py 中 class_emb_module 返回值只有 1 个
    却被解包为 3 个的问题。
    """
    onehot_array, depth_array, part_array, part_embedding, depth_embedding = \
        load_class_emb(args)

    with torch.no_grad():
        all_class_embedding = net.class_emb_module(
            onehot_array.to(device),
            depth_array.to(device),
            part_array.to(device),
            part_embedding.to(device),
            depth_embedding.to(device),
        )
    all_class_embedding = all_class_embedding.cpu()

    # 分批计算模板特征，避免显存溢出
    char_templates = load_template(args)
    image_dummy = torch.zeros(
        1,
        char_templates.shape[1],
        char_templates.shape[2],
        char_templates.shape[3],
        device=device,
    )

    mini_batch = 512
    n = char_templates.shape[0]
    mini_bs = n // mini_batch
    if mini_bs == 0:
        chunks = [char_templates]
    else:
        chunks = [char_templates[i * mini_batch:(i + 1) * mini_batch]
                  for i in range(mini_bs)]
        if n % mini_batch != 0:
            chunks.append(char_templates[mini_bs * mini_batch:])

    feat_list = []
    with torch.no_grad():
        for chunk in chunks:
            _, template_feat = net.visual_module(image_dummy, chunk.to(device))
            feat_list.append(template_feat.cpu())

    template_feat = torch.cat(feat_list, dim=0)
    return all_class_embedding, template_feat


# ──────────────────────────────────────────────
# 模型 + 数据加载
# ──────────────────────────────────────────────
def load_everything(args):
    # 读取字符集
    with open(args.char_set, 'r', encoding='utf-8') as f:
        char_set = [line.strip()[0] for line in f.readlines()]

    # 构建并加载模型
    net = Model(args)
    if args.pretrained:
        print(f"加载模型权重：{args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location=device)
        net.load_state_dict(state_dict, strict=False)
    net = net.to(device)
    net.eval()

    # 预计算
    print(f"预计算类别嵌入和模板特征（共 {len(char_set)} 个类别）…")
    all_class_embedding, template_feat = prepare_embeddings(args, net)
    print("初始化完成！")
    return net, char_set, all_class_embedding, template_feat


# ──────────────────────────────────────────────
# 单张图片推理
# ──────────────────────────────────────────────
def predict_image(image, net, char_set, all_class_embedding, template_feat, top_k):
    if image is None:
        return {}

    # Gradio 传入 RGB numpy array，转为灰度
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            gray = image
    else:
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Otsu 二值化：自动确定阈值，得到黑白图
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 若背景为黑色（均值<128），则反色，确保白底黑字与训练分布一致
    if binary.mean() < 128:
        binary = cv2.bitwise_not(binary)

    # 转回 3 通道 BGR，与训练时的 cv2.imread 格式保持一致
    img_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # 缩放到 32×32，与训练预处理一致
    img_resized = cv2.resize(img_bgr, (32, 32))
    img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(device)  # (1,3,32,32)

    dummy_char_id = torch.zeros(1, 1, dtype=torch.long, device=device)

    with torch.no_grad():
        result = net(
            [img_tensor, template_feat.to(device), all_class_embedding.to(device)],
            dummy_char_id,
            sampling=False,
            training=False,
        )
    preds = result['pred']  # (1, num_classes)
    probs = torch.softmax(preds[0], dim=0)

    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(char_set)))
    return {char_set[idx.item()]: float(top_probs[i])
            for i, idx in enumerate(top_indices)}


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────
def build_args():
    parser = argparse.ArgumentParser(description="零样本汉字识别 Gradio 演示")

    # 必填：路径相关
    parser.add_argument('--pretrained',     type=str, required=True,  help='模型权重 .pth 路径')
    parser.add_argument('--char_set',       type=str, required=True,  help='字符集 .txt 路径')
    parser.add_argument('--emb_path',       type=str, required=True,  help='类别嵌入 .npz 路径')
    parser.add_argument('--template_path',  type=str, default='',     help='模板 .pkl 路径')

    # 模型结构（与训练时保持一致）
    parser.add_argument('--backbone',       type=str,   default='densenet36_dsbn')
    parser.add_argument('--emb_dim',        type=int,   default=266)
    parser.add_argument('--seen_classes',   type=int,   default=3755)
    parser.add_argument('--first_pooling',  action='store_true')
    parser.add_argument('--activation',     type=str,   default=None)
    parser.add_argument('--is_whole_model', action='store_true')
    parser.add_argument('--sampling',       action='store_true')
    parser.add_argument('--loadSize',       type=int,   default=32)
    parser.add_argument('--class_nums',     type=int,   default=3755)

    # 演示参数
    parser.add_argument('--top_k',  type=int, default=5,    help='显示 Top-K 候选结果')
    parser.add_argument('--port',   type=int, default=7860, help='Gradio 监听端口')
    parser.add_argument('--share',  action='store_true',    help='生成公网分享链接')

    # 以下参数 config.py 可能用到，保持兼容
    parser.add_argument('--trainBatchSize',  type=int,   default=128)
    parser.add_argument('--testBatchSize',   type=int,   default=64)
    parser.add_argument('--numOfWorkers',    type=int,   default=4)
    parser.add_argument('--trainRoot',       type=str,   default='')
    parser.add_argument('--valRoot',         type=str,   default='')
    parser.add_argument('--experiment',      type=str,   default='')
    parser.add_argument('--num_epochs',      type=int,   default=30)
    parser.add_argument('--resume_epoch',    type=int,   default=1)
    parser.add_argument('--eval_interval',   type=int,   default=5)
    parser.add_argument('--log_interval',    type=int,   default=100)
    parser.add_argument('--lr',              type=float, default=1.0)
    parser.add_argument('--log_file',        type=str,   default='')
    parser.add_argument('--vis_dir',         type=str,   default='')
    parser.add_argument('--is_vis',          action='store_true')
    parser.add_argument('--is_vis_tsne',     action='store_true')
    parser.add_argument('--modelsSavePath',  type=str,   default='')

    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()

    net, char_set, all_class_embedding, template_feat = load_everything(args)

    def predict_fn(image):
        return predict_image(
            image, net, char_set, all_class_embedding, template_feat, args.top_k
        )

    iface_kwargs = dict(
        fn=predict_fn,
        inputs=gr.Image(label="上传手写汉字图片"),
        outputs=gr.Label(num_top_classes=args.top_k, label=f"识别结果 Top-{args.top_k}"),
        title="零样本汉字识别系统",
        description=(
            f"基于零样本学习（深度嵌入网络 + Transformer 解码器）的手写汉字识别。\n"
            f"当前支持 {len(char_set)} 个汉字类别，上传图片后自动返回 Top-{args.top_k} 结果。"
        ),
    )
    try:
        interface = gr.Interface(**iface_kwargs, flagging_mode="never")
    except TypeError:
        interface = gr.Interface(**iface_kwargs, allow_flagging="never")

    print(f"启动 Gradio，访问 http://localhost:{args.port}")
    interface.launch(server_port=args.port, share=args.share)
