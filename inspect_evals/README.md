# Observational Study

Evaluation harness for RLVR generalizability experiments, built on [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals).

## Supported Benchmarks

| Category | Benchmark |
|---|---|
| Medical | `pubmedqa`, `medqa` |
| Math | `aime2024`, `gsm8k`, `math500`, `amc23` |
| Reasoning | `tab_fact` |
| Legal / Finance | `legalbench`, `finben` |
| Code | `livecodebench`, `codeforces`, `humaneval`, `bigcodebench`, `mbpp`, `usaco` |
| Multilingual | `polyglot` |

## Structure

```
eval/
  src/inspect_evals/   # eval task implementations (subset of inspect_evals)
  eval.py              # sweep logic — accepts --models and --tasks flags
  eval.sh              # entry point — edit MODELS/TASKS at the top, then run
  pyproject.toml       # package config
  requirements.txt     # runtime dependencies
```

## Setup

```bash
pip install -e .
```

## Running Evals

Edit the `MODELS` and `TASKS` arrays at the top of `eval.sh`, then:

```bash
bash eval.sh
```

Or call `eval.py` directly:

```bash
python eval.py --models vllm/Qwen/Qwen2.5-Math-7B --tasks aime2024 math500 gsm8k
```

Results are written to `logs/<model>__<task>/`.
