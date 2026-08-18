# Agentic Data Evolution

<p align="center">
  <a href="README.md">English</a> | <a href="README-zh.md">简体中文</a>
</p>

论文 "ADE: Agentic Data Evolution Framework for Human-Centered Objectives" 的官方实现。

## Overview

当目标不可执行且依赖上下文时，使大语言模型对齐人本目标十分困难。Agentic Data Evolution（ADE）将合成监督视为一系列持续演化的数据快照，并通过 Observation-Variation-Selection (OVS) 闭环流程改进这些快照。稳态准入机制以保守方式筛选更新，以实现跨轮次的持续改进。

![ADE 概览](figures/fig_main.png)

本仓库将数据构造、数据演化、训练流程和评测作为相互独立的阶段提供，用户可以接入自己的模型、服务端点和计算环境。

## Repository Layout

```text
.
├── src/
│   ├── construction/        # D(0) 和 DEV300 构造
│   ├── ovs/                 # 观察、变异、选择及其流程协调
│   └── evaluation/          # 预测生成和基准评测
├── data/                    # 评测示例和数据说明
└── scripts/                 # 构造、OVS 和 SFT 启动脚本
```

## Requirements

- Python 3.10 或更高版本。
- 根据所运行的阶段，需要一个或多个用于 D(0) 构造、OVS、预测生成或自动评测的 OpenAI-compatible 服务端点。
- `requirements.txt` 中列出的 D(0) 构造、OVS 和评测核心依赖。
- SFT 是可选流程，需要安装 `ms-swift`。我们的训练在一台配备 8 张 NVIDIA H100 GPU 的节点上完成。

创建环境并加载自己的服务端点配置：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，将所有占位符替换为你自己的配置。
set -a
source .env
set +a
```

环境模板分别包含 D(0) 构造、OVS、评测裁判模型以及待评测模型和 reference 模型预测生成所需的配置。`CONVERT_*` 变量在“Quick Start”部分按步骤设置。不要提交 `.env` 或真实凭据。

## Quick Start

请从仓库根目录运行以下命令。各阶段可以独立运行；完整流程通常依次执行 D(0) 构造、OVS、SFT 和评测。

### 1. 构造 `D(0)`

```bash
bash scripts/generate_d0.sh \
  --output-dir data/construction/raw \
  --model "$D0_MODEL" \
  --api-key "$D0_API_KEY" \
  --base-url "$D0_BASE_URL" \
  --prompt-language "$D0_PROMPT_LANGUAGE"
```

该命令会写入 `D0.jsonl`、`DEV300.jsonl` 和 `construction_manifest.json`。将 `D0.jsonl` 作为后续 SFT 数据的 OVS 输入，并将 `DEV300.jsonl` 留作内在评测集。

### 2. 使用 OVS 演化数据集

`run_ovs.sh` 会读取 `OVS_INPUT_DIR` 下的所有 JSONL 文件，并在配置的输出目录下写入演化后的数据快照。

```bash
mkdir -p data/ovs_input
cp data/construction/raw/D0.jsonl data/ovs_input/

export OVS_INPUT_DIR=data/ovs_input
export OVS_OUTPUT_DIR=/path/to/evolution
export OVS_LOG_FILE=/path/to/ovs.log
export OVS_ROUNDS=4
bash scripts/run_ovs.sh
```

请将 `DEV300.jsonl` 保持在 `OVS_INPUT_DIR` 之外；它用于留出评测。

最终轮次的数据快照写入 `$OVS_OUTPUT_DIR/round$OVS_ROUNDS/data/`。

### 3. 转换数据快照并运行 SFT

先将演化后的数据快照转换为 OpenAI chat-message JSONL 格式，再启动全参数 SFT：

```bash
export CONVERT_INPUT_DIR=/path/to/evolution/round4/data
export CONVERT_OUTPUT_DIR=/path/to/sft_data
python src/to_oai_msg.py

bash scripts/run_sft.sh \
  7b \
  /path/to/Qwen2.5-7B-Instruct \
  /path/to/sft_data/D0.jsonl \
  /path/to/checkpoint
```

对于 72B 模型启动器，将 `7b` 替换为 `72b`。这些脚本只负责训练 checkpoint，不负责提供模型服务或运行评测。

### 4. 生成并评测预测结果

通过 OpenAI-compatible API 暴露待比较的两个数据快照或模型，然后分别生成两份预测文件。两份文件必须包含相同的示例 ID 和 `answer` 字段。

为待评测模型生成预测结果：

```bash
python src/evaluation/llm_gen.py \
  --model "$SERVED_MODEL_NAME" \
  --api_key "$SERVED_MODEL_API_KEY" \
  --base_url "$SERVED_MODEL_BASE_URL" \
  --inp_file data/dev300/example.jsonl \
  --out_file results/predictions.jsonl
```

使用第二个数据快照或模型生成 reference predictions：

```bash
python src/evaluation/llm_gen.py \
  --model "$REFERENCE_MODEL_NAME" \
  --api_key "$REFERENCE_MODEL_API_KEY" \
  --base_url "$REFERENCE_MODEL_BASE_URL" \
  --inp_file data/dev300/example.jsonl \
  --out_file results/reference_predictions.jsonl
```

仓库不提供 `results/reference_predictions.jsonl`；运行自动裁判前，请先使用上述命令生成该文件。两份预测文件会根据 `id` 进行对齐：

```bash
python src/evaluation/eval_dev300.py \
  --inp_file results/predictions.jsonl \
  --ref_file results/reference_predictions.jsonl \
  --out_file results/eval.jsonl \
  --model "$JUDGE_MODEL" \
  --api_key "$JUDGE_API_KEY" \
  --base_url "$JUDGE_BASE_URL"
```

DEV300 和 EduBench 使用 LLM 裁判；MATH-500 和 ToxiCN 使用确定性评分。完整的外部数据说明、输入字段、基准许可证和各评测器的具体命令请参见 [data/README.md](data/README.md)。

## Citation

如果使用本仓库，请引用：

```bibtex
@misc{yu2026agentic,
  title     = {Agentic Data Evolution},
  author    = {Yu, Yang and Jiang, Yilin and Fei, Zexuan and Luo, Yiming and
               Song, Xingkai and Huang, Kaiyi and Zhou, Aimin and Lin, Xin and
               Tan, Fei},
  year      = {2026},
  url       = {https://github.com/ZeroLoss-Lab/Agentic-Data-Evolution}
}
```

使用相关数据时，也请引用 [data/README.md](data/README.md) 中列出的基准论文。

## License

原创源代码采用 MIT License 发布，完整文本请参见 [LICENSE](LICENSE)。示例数据和衍生基准文件可能受上游独立许可证约束；重新分发前请参阅 [data/README.md](data/README.md)。
