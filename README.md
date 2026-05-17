# 零样本汉字识别 Demo

基于零样本学习（Zero-Shot Learning）的手写汉字识别系统演示。  
使用深度结构嵌入网络 + Transformer 解码器，支持 **3755** 个汉字类别的识别，无需对每个字都有训练样本。

---

## 项目结构

```
zero-shot-char-demo-export/
├── demo_char.py                                          # 启动入口
├── model/
│   ├── char_model.py                                     # 主模型
│   ├── class_embedding.py                                # 结构化类别嵌入
│   ├── transformer.py                                    # ZSDecoder 解码器
│   ├── densenetdsbn.py                                   # DenseNet + 双域BN 主干
│   ├── densenet.py                                       # DenseNet 基础实现
│   ├── transferLayer.py                                  # 嵌入维度映射层
│   └── visual_backbone.py                                # 视觉主干选择器
├── utils/
│   ├── class_emb_loader.py                               # 加载嵌入与模板
│   └── prepare_emb.py                                    # 预计算嵌入工具
├── data/
│   └── hwdb/
│       ├── 3755.txt                                      # 汉字字表（3755字）
│       ├── 3755_parameter.npz                            # 汉字结构嵌入参数
│       └── simsun_hwdb.pkl                               # 宋体印刷模板特征
└── checkpoint/
    └── exp_handwriting_1500_template_dense36dsbn_decloss/
        └── 14_90.17.pth                                  # 训练权重（验证集准确率 90.17%）
```

---

## 环境要求

- Python 3.7 及以上
- PyTorch（CPU 即可运行，有 NVIDIA GPU 速度更快）

---

## 安装依赖

```bash
pip install torch torchvision gradio opencv-python numpy
```

> **macOS 用户**：若 `opencv-python` 安装报错，请改用：
> ```bash
> pip install opencv-python-headless
> ```
>
> **Apple Silicon（M1/M2/M3）用户**：建议通过 conda 安装 PyTorch 以获得 MPS 加速支持：
> ```bash
> conda install pytorch torchvision -c pytorch
> ```

---

## 启动 Demo

### Windows（PowerShell）

```powershell
python demo_char.py `
  --pretrained "checkpoint/exp_handwriting_1500_template_dense36dsbn_decloss/14_90.17.pth" `
  --char_set "data/hwdb/3755.txt" `
  --emb_path "data/hwdb/3755_parameter.npz" `
  --template_path "data/hwdb/simsun_hwdb.pkl" `
  --backbone densenet36_dsbn `
  --emb_dim 266 `
  --seen_classes 3755 `
  --top_k 5
```

### macOS / Linux（bash/zsh）

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

启动后在浏览器访问 **http://localhost:7860**，上传手写汉字图片即可看到 Top-5 识别结果。

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pretrained` | 必填 | 模型权重 `.pth` 文件路径 |
| `--char_set` | 必填 | 字符集 `.txt` 文件路径（每行一个汉字） |
| `--emb_path` | 必填 | 汉字结构嵌入 `.npz` 文件路径 |
| `--template_path` | 必填 | 印刷体模板 `.pkl` 文件路径 |
| `--backbone` | `densenet36_dsbn` | 视觉主干网络类型 |
| `--emb_dim` | `266` | 结构嵌入维度（需与权重一致） |
| `--seen_classes` | `3755` | 字表中的类别总数 |
| `--top_k` | `5` | 返回 Top-K 候选结果数 |
| `--port` | `7860` | Gradio 服务端口 |
| `--share` | 不填 | 加上此参数可生成公网分享链接 |

---

## 工作原理简述

本系统采用**零样本学习**方法识别汉字，核心思路是将"分类"转化为"查询匹配"：

1. **结构嵌入**：每个汉字通过 IDS 部件分解（如"明" = "日" + "月"）生成语义向量，相同偏旁部首的字在嵌入空间中彼此接近。
2. **印刷体模板**：宋体字形图像经双域 BN（DSBN）主干提取特征，与手写特征处于同一空间。
3. **Transformer 解码**：将结构嵌入 + 模板特征拼接为每个类别的查询向量，通过 cross-attention 与手写图像特征匹配，得到分类分数。
4. **零样本泛化**：只要一个字有部件分解记录和字体模板，无需该字的任何手写训练样本即可识别。
