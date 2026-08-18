# Evaluation Data

This directory contains small input examples used to exercise the paper's automatic evaluation code. They are not the `D(0)` or `D(4)` training snapshots, full benchmark splits, or data for reporting results.

## DEV300

`dev300/example.jsonl` contains 10 examples from the held-out prompt format. Generate predictions for two snapshots or models, then provide those aligned prediction files to `src/evaluation/eval_dev300.py`.

### Construction

DEV300 is not built by a separate script. It is produced by the same `scripts/generate_d0.sh` run used to construct `D(0)` (see "Construct `D(0)`" in the top-level [README](../README.md)): `src/construction/generate_d0.py` writes both `D0.jsonl` and `DEV300.jsonl` from one invocation. DEV300 uses the same concept-to-topic-to-question pipeline as `D(0)`, uniformly sampled to 100 instances per objective (`--dev-size 300`, must be divisible by three), and rejects any candidate question whose BLEU-2 similarity to a `D(0)` question exceeds `--bleu2-threshold` (default `0.7`), so DEV300 does not overlap `D(0)`.

## EduBench

The official source is the paper [EduBench: A Comprehensive Benchmarking Dataset for Evaluating Large Language Models in Diverse Educational Scenarios](https://aclanthology.org/2026.acl-long.987/) (DOI: [10.18653/v1/2026.acl-long.987](https://doi.org/10.18653/v1/2026.acl-long.987)). The paper describes EduBench as a synthetic benchmark with nine educational scenarios and more than 4,000 educational contexts.

The ACL Anthology states that materials published in or after 2016 are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This applies to the ACL-hosted paper and its published materials; it does not establish a separate license for an EduBench dataset release. No official dataset download or independent dataset license is bundled with this repository. Obtain the full benchmark from an official author release and verify its terms before using or redistributing it.

`edubench/example.jsonl` is a synthetic, format-only fixture created for this repository. It is not an official EduBench release, a benchmark split, or data for reporting results. Generate predictions with `src/evaluation/llm_gen.py` using `model_question` and `model_answer` as the input and output tags, then score the generated file with `src/evaluation/eval_edubench.py` using the fixture as `--ref_file`.

```bash
python src/evaluation/llm_gen.py \
  --model "$SERVED_MODEL_NAME" \
  --api_key "$SERVED_MODEL_API_KEY" \
  --base_url "$SERVED_MODEL_BASE_URL" \
  --inp_file data/edubench/example.jsonl \
  --out_file results/edubench_predictions.jsonl \
  --inp_tag model_question \
  --out_tag model_answer

python src/evaluation/eval_edubench.py \
  --inp_file results/edubench_predictions.jsonl \
  --ref_file data/edubench/example.jsonl \
  --out_file results/edubench_eval.jsonl \
  --model "$JUDGE_MODEL" \
  --api_key "$JUDGE_API_KEY" \
  --base_url "$JUDGE_BASE_URL"
```

```bibtex
@inproceedings{xu-etal-2026-edubench,
    title = "{E}du{B}ench: A Comprehensive Benchmarking Dataset for Evaluating Large Language Models in Diverse Educational Scenarios",
    author = "Xu, Bin and Bai, Yu and Sun, Huashan and Lin, Yiguan and Liu, Siming and Liang, Xinyue and Li, Yaolin and Dong, Zhuangzhi and Zhang, Jingren and Deng, Yufan and Zou, Xinyu and Gao, Yang and Huang, Heyan",
    booktitle = "Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2026",
    url = "https://aclanthology.org/2026.acl-long.987/",
    doi = "10.18653/v1/2026.acl-long.987"
}
```

## MATH-500

The complete MATH-500 test set is not redistributed in this repository. Obtain the 500-instance `test.jsonl` from the [HuggingFaceH4/MATH-500 dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) and place it at `data/math500/test.jsonl`. The expected fields are `problem`, `answer`, `solution`, `subject`, and `level`. Use `src/evaluation/llm_gen.py --task math500` to generate a `response` field, then use `src/evaluation/eval_math500.py` for final-answer accuracy.

`data/math500/example.jsonl` contains five records copied from the upstream test file solely as a format smoke-test input. It is not a benchmark split and must not be used to report benchmark results.

```bibtex
@inproceedings{lightman2024iclr,
  author    = {Hunter Lightman and Vineet Kosaraju and Yuri Burda and
               Harrison Edwards and Bowen Baker and Teddy Lee and
               Jan Leike and John Schulman and Ilya Sutskever and Karl Cobbe},
  title     = {Let's Verify Step by Step},
  booktitle = {Proc. of ICLR},
  year      = {2024},
}
```

## ToxiCN

The complete ToxiCN test set is not redistributed. The upstream data are available from the [ToxiCN repository](https://github.com/DUT-lujunyu/ToxiCN) and are released under CC BY-NC-ND 4.0. The paper uses a 2,411-instance test subset. Convert the upstream JSON test split to JSONL and place it at `data/toxicn/test.jsonl`, adding a string `id`, `text` copied from `content`, and binary `label` copied from `toxic`. Use `src/evaluation/llm_gen.py --task toxicn` to generate a `response` field, then use `src/evaluation/eval_toxicn.py` for binary toxicity F1.

`data/toxicn/example.jsonl` contains five records derived from the upstream test input format solely as a format smoke-test input. It is not a benchmark split and must not be used to report benchmark results. These examples remain subject to the upstream CC BY-NC-ND 4.0 license.

```bibtex
@inproceedings{lu2023acl,
  author    = {Lu, Junyu and Xu, Bo and Zhang, Xiaokun and Min, Changrong and
               Yang, Liang and Lin, Hongfei},
  title     = {Facilitating Fine-grained Detection of {C}hinese Toxic Language:
               Hierarchical Taxonomy, Resources, and Benchmarks},
  booktitle = {Proc. of ACL},
  pages     = {16235--16250},
  year      = {2023},
}
```

Please cite the respective benchmark papers and comply with their licenses.

## Manifest

| File | Records | SHA-256 |
| --- | ---: | --- |
| `dev300/example.jsonl` | 10 | `ab29cbc4e18fffd164e8824c31e4be9817133d5bf8e94ed48a366051d8a062ca` |
| `edubench/example.jsonl` | 10 | `04be2490a026c0a047ca38934ddd4f7667ceb2001f462c0ace878fe5d5488dc5` |

The repository does not redistribute MATH-500 or ToxiCN, full human-evaluation records, judge-disagreement outputs, or generated training snapshots.
