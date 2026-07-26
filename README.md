<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/readme/hero.svg">
  <img alt="零样本汉字识别 — 3755 字 · 90.17% Acc · Transformer Cross-Attention" src="assets/readme/hero.svg" width="100%">
</picture>

<br>

**3755 个汉字，零样本识别。** 不需要每个字的手写训练数据——通过 IDS 部件分解理解字形结构，结合印刷体模板与 Transformer 跨注意力匹配，从未见过的手写字也能识别。

---

## 是什么

一个基于**零样本学习（Zero-Shot Learning）**的手写汉字识别系统。核心思路是把"分类"转成"匹配"——用汉字的偏旁部首结构信息作为中间桥梁，让模型在没见过某个字手写样本的情况下也能认出它。

输入一张手写汉字图片 → 输出 Top-5 候选字及置信度。Gradio Web 界面开箱即用。

| | |
|---|---|
| **类别数** | 3755 个汉字（GB2312 一级字库） |
| **准确率** | 验证集 90.17% |
| **框架** | PyTorch · Gradio · OpenCV |
| **推理** | CPU 可跑，GPU 加速可选 |

---

## 为什么不一样

传统汉字识别需要每个字都有大量手写标注数据——3755 个字 × 每人书写风格差异，标注成本极高。

**零样本方案绕过了这个瓶颈：**

| 传统 OCR | 本方案 |
|---|---|
| 每个字需要数百张手写样本训练 | 只靠部件分解 + 印刷体模板，零手写样本 |
| 遇到新字必须重新标注训练 | 只要这个字有 IDS 部件记录就能识别 |
| 模型学到的是"这个字长什么样" | 模型学到的是"偏旁部首怎么组合成字" |

---

## 怎么做到的

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  手写图片     │───▶│ DenseNet + DSBN  │───▶│  视觉特征向量     │
│  (32×32)     │    │  双域 BN 主干    │    │                  │
└──────────────┘    └─────────────────┘    └────────┬─────────┘
                                                    │ cross-attention
┌──────────────┐    ┌─────────────────┐    ┌────────▼─────────┐
│  IDS 部件分解  │───▶│  ClassEmb 编码   │───▶│  ZSDecoder       │───▶ Top-K
│  明=日+月     │    │  onehot+depth    │    │  1层 4头 Transf. │
└──────────────┘    └─────────────────┘    └────────┬─────────┘
                                                    │
┌──────────────┐    ┌─────────────────┐             │
│ 宋体印刷模板   │───▶│ DenseNet + DSBN  │────────────┘
│ (Song font)  │    │  模板特征提取     │   concat
└──────────────┘    └─────────────────┘
```

### 四个关键组件

| 组件 | 作用 |
|---|---|
| **IDS 结构嵌入** (`class_embedding.py`) | 把每个汉字分解为偏旁部首（如 明→日+月），编码为语义向量。相同部首的字在嵌入空间中彼此接近 |
| **DenseNet + DSBN** (`densenetdsbn.py`) | 双域批归一化——手写域和印刷域各走各的 BN 统计量，两个空间对齐 |
| **印刷体模板** (`simsun_hwdb.pkl`) | 宋体字形经同一个 DSBN 主干提取特征，作为每个类别的"参考锚点" |
| **ZSDecoder** (`transformer.py`) | 将结构嵌入 + 模板特征拼接为查询向量，通过 cross-attention 与手写图像特征匹配 |

---

## 快速开始

### 安装

```bash
pip install torch torchvision gradio opencv-python numpy
```

> macOS 用户若 `opencv-python` 报错，换 `opencv-python-headless`。Apple Silicon 建议 conda 装 PyTorch 走 MPS 加速。

### 启动 Demo

```bash
python demo_char.py \
  --pretrained "checkpoint/exp_handwriting_1500_template_dense36dsbn_decloss/14_90.17.pth" \
  --char_set "data/hwdb/3755.txt" \
  --emb_path "data/hwdb/3755_parameter.npz" \
  --template_path "data/hwdb/simsun_hwdb.pkl" \
  --backbone densenet36_dsbn \
  --emb_dim 266 \
  --seen_classes 3755 \
  --top_k 5
```

浏览器访问 **http://localhost:7860**，上传手写汉字图片即可看到 Top-5 结果。

> 加 `--share` 生成公网分享链接。

---

## 参数说明

| 参数 | 作用 |
|---|---|
| `--pretrained` | 模型权重 `.pth` 路径 |
| `--char_set` | 汉字字表 `.txt`（每行一字） |
| `--emb_path` | 结构嵌入参数 `.npz` |
| `--template_path` | 宋体模板 `.pkl` |
| `--backbone` | 视觉主干（默认 `densenet36_dsbn`） |
| `--emb_dim` | 嵌入维度（需与权重一致，默认 266） |
| `--top_k` | 返回 Top-K 候选（默认 5） |
| `--port` | Gradio 端口（默认 7860） |

---

## 项目结构

```
├── demo_char.py                  # Gradio 启动入口
├── model/
│   ├── char_model.py             # 主模型（训练+推理）
│   ├── class_embedding.py        # IDS 结构嵌入编码
│   ├── transformer.py            # ZSDecoder 跨注意力解码
│   ├── densenetdsbn.py           # DenseNet + 双域 BN
│   ├── densenet.py               # DenseNet 基础实现
│   ├── transferLayer.py          # 嵌入维度映射
│   └── visual_backbone.py        # 主干网络选择器
├── utils/
│   ├── class_emb_loader.py       # 加载嵌入 + 模板
│   └── prepare_emb.py            # 预计算嵌入工具
├── data/hwdb/
│   ├── 3755.txt                  # 3755 汉字字表
│   ├── 3755_parameter.npz        # 结构嵌入参数
│   └── simsun_hwdb.pkl           # 宋体印刷模板特征
└── checkpoint/                   # 训练权重
```

---

## 许可

MIT
