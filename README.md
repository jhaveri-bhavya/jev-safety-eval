# jev-safety-eval

> ⚠️ **Content warning:** this project evaluates models on a dataset containing harmful,
> offensive and unsafe text.

## Goal
How well does **Jev** (TypeSafe's decision model) catch unsafe content? This repo is a
reproducible benchmark of Jev on user prompts, LLM responses and multi-turn dialogs,
reporting F1, AUPRC, calibration, cost and latency. Jev is compared against
**Gemma 4 31B** (an open-weights LLM judge) and cheap supervised baselines
(TF-IDF and MiniLM embeddings with logistic regression). It also tests a
confidence-gated Jev → Gemma cascade.

## Dataset
[NVIDIA Aegis AI Content Safety Dataset 1.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0)
(CC-BY-4.0): about 12k human-annotated texts covering 13 risk categories plus
Safe and Needs Caution. Only the 1,199-row test split is sent to the model APIs.

## Status
Work in progress: setup and API smoke tests are done (`scripts/smoke_*.py`). Results will follow.
