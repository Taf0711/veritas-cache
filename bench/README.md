# Trace: Quora Question Pairs

## Source

The trace comes from the Quora Question Pairs dataset.
The data is the GLUE QQP release.

The Hugging Face datasets-server endpoint for dataset quora returns a renamed-dataset error. The GLUE QQP zip archive contains the same Quora Question Pairs data.

## Contents

- 404276 labeled pairs (train and dev)
- 60397 equivalence classes with 2 or more members
- 8101 classes in the stream
- 20000 prompts in the stream
- stream seed 42

## Label noise

Quora duplicate labels are noisy.
A duplicate label does not always mean the same correct response.
The harness treats same class as same correct response.

## Reproduce

Run this command from the repository root.

```bash
python3 scripts/build_trace.py
```
