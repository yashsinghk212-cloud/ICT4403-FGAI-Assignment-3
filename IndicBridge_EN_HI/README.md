# IndicBridge: English-to-Hindi Machine Translation with Transformer, mT5, LLaMA and Gemma

Foundations of Generative AI (ICT4403) - Assignment 3: neural machine translation with encoder-decoder and decoder-only models.

## 1. Problem statement and objective

Build and compare four machine-translation systems for **English → Hindi** on the same parallel corpus and the same test set:

| Model | Architecture | How it is adapted here |
|---|---|---|
| Basic Transformer | encoder-decoder | trained from scratch |
| mT5-small | encoder-decoder | full fine-tuning |
| LLaMA (3.2, 1B) | decoder-only | LoRA fine-tuning |
| Gemma (3, 1B) | decoder-only | LoRA fine-tuning |

The objective is to (i) preprocess and split the corpus, (ii) evaluate the models with BLEU, chrF, training/fine-tuning time, inference time and parameter count and compare encoder-decoder with decoder-only designs, and (iii) examine the translations for responsible-AI concerns (bias, hallucination, cultural misrepresentation, sensitive content) and propose mitigations.

Hindi is written in Devanagari, is morphologically rich, marks grammatical gender on verbs and adjectives, and has three politeness levels of 'you' (तू / तुम / आप). It is also comparatively low-resource next to English-French, which makes it a demanding test of how much each architecture depends on pretraining.

## 2. Dataset

* **Name:** IITB English-Hindi Parallel Corpus
* **Source link:** https://huggingface.co/datasets/cfilt/iitb-english-hindi (original distribution: https://www.cfilt.iitb.ac.in/iitb_parallel/)
* **Subset used:** 60,000 training pairs, 1,000 validation pairs and 1,000 test pairs (sentences of 4-30 English words).
* **Preprocessing** (`prepare_data` in `mtkit.py`): Unicode NFC normalisation and whitespace cleaning; removal of empty pairs, pairs where source equals target, pairs outside the length limits, pairs whose word-length ratio exceeds 2.5, and duplicate source sentences; random shuffling with a fixed seed, then **disjoint** test / validation / training splits.
* Cleaned splits are written to `data/*.jsonl`; corpus statistics (pair counts, average and maximum lengths, vocabulary sizes, dropped-pair counts, sample pairs) to `results/dataset_stats.json`; the length histogram to `results/length_distribution.png`.

## 3. Models implemented

1. **Basic Transformer** - a 3+3-layer Transformer (d_model 256, 4 heads, feed-forward 1024) with an 8,000-token joint SentencePiece BPE vocabulary, shared (tied) embeddings, pre-LayerNorm, label smoothing 0.1, warm-up + inverse-square-root learning rate, early stopping on validation loss. Code in `mt_transformer.py`.
2. **mT5-small** (`google/mt5-small`, encoder-decoder) - fully fine-tuned with the prefix `translate English to Hindi:`.
3. **LLaMA 3.2 1B** (decoder-only) - **LoRA fine-tuning** (rank 16 on all attention and MLP projections) of the *base* checkpoints `meta-llama/Llama-3.2-1B` and `google/gemma-3-1b-pt` on 20,000 pairs for one epoch, with the loss computed on the Hindi target only.
4. **Gemma 3 1B** (decoder-only) - same adaptation recipe as LLaMA, so the two decoder-only models are compared under identical conditions.

All four models translate the **same test set** with **greedy decoding**, the same maximum output length and the same inference batch size (see `CONFIG` in `run_project.py`). Decoder-only models receive the prompt `Translate the following English sentences into Hindi.` followed by `English: <sentence>` and `Hindi:`.

## 4. Execution steps and dependencies

```bash
git clone https://github.com/<your-username>/en-hi-mt-transformer-mt5-llama-gemma.git
cd en-hi-mt-transformer-mt5-llama-gemma
pip install -r requirements.txt
huggingface-cli login            # LLaMA and Gemma are gated: accept their licences on huggingface.co first
python run_project.py --quick    # optional: tiny smoke test (a few minutes)
python run_project.py            # full run: data -> 4 models -> evaluation -> report
python run_project.py --stage report   # rebuild tables, plots and the results sections of this README
```

* Options: `--models transformer mt5 llama gemma` (any subset), `--stage data|train|report|all`, `--force` (re-run finished models).
* A GPU is strongly recommended (tested design target: a single 16-24 GB GPU such as a Colab T4/L4). Models that finish are cached in `results/metrics_<model>.json`, so an interrupted run can be resumed. On GPUs without bfloat16 support the 1B decoder-only models run in float32.
* Training sizes, epochs and model sizes can be reduced in `CONFIG` if memory or time is limited.

## 5. Results

All numbers below are produced by the code and inserted automatically by `python run_project.py --stage report`; BLEU uses SacreBLEU (`intl` tokenisation, signature stored in `results/metrics_*.json`) and chrF uses SacreBLEU defaults. Timing was measured on the hardware recorded in `results/environment.json`.

<!-- RESULTS_START -->
_Run `python run_project.py` to generate this section._
<!-- RESULTS_END -->

## 6. Model comparison and evaluation

**Encoder-decoder vs decoder-only.** Read the table above together with the following factors, which explain most differences between the four rows:

* **Model size.** The scratch Transformer is the smallest model and mT5-small has about 300 M parameters, most of them in its 250k-token embedding matrix; LLaMA-3.2-1B and Gemma-3-1B have roughly 1.0-1.2 B parameters. LoRA trains only a small fraction of the decoder-only weights, but inference still runs the full 1B-parameter network, so per-sentence latency and memory are higher than for the encoder-decoder models.
* **Pretraining.** The scratch Transformer sees only the Hindi training pairs. mT5 was pretrained on multilingual web text with a span-corruption objective (not translation), so it knows both languages but must learn the task during fine-tuning. LLaMA and Gemma were pretrained on far larger corpora with next-token prediction and have seen translation-like text, which is what makes few-shot prompting possible at all.
* **Tokenisation.** Hindi text costs many more tokens per word in LLaMA's 128k-token vocabulary than in Gemma's 262k-token vocabulary or in the small joint BPE vocabulary of the scratch model, so decoder-only inference tends to need longer generated sequences; this affects both speed and memory.
* **Adaptation method.** **LoRA fine-tuning** (rank 16 on all attention and MLP projections) of the *base* checkpoints `meta-llama/Llama-3.2-1B` and `google/gemma-3-1b-pt` on 20,000 pairs for one epoch, with the loss computed on the Hindi target only. The scratch Transformer and mT5 update all weights on the same pairs, so differences in the first two rows reflect pretraining, while differences between the decoder-only rows reflect the base model and tokenizer because the recipe is identical.
* **Architecture.** An encoder-decoder reads the whole source bidirectionally once and then decodes with cross-attention, which is efficient for a fixed-input/fixed-output task; a decoder-only model must process the prompt (and any in-context examples) causally and carries it in the KV cache while generating.

The automatic summary in Section 5 states which model leads on each metric; use it, not these general factors, as the evidence for any claim in your report.

## 7. Sampling-temperature analysis

Section 5 contains BLEU, chrF and the percentage of distinct outputs for each model at temperatures 0 (greedy), 0.3, 0.7 and 1.0 on the first 200 test sentences (`results/temperature_analysis.csv`, `results/temperature_analysis.png`). Lower temperatures concentrate probability on the most likely token and give stable, literal translations; higher temperatures flatten the distribution, increase variety and make omissions, wrong words and hallucinations more likely, so translation quality is expected to fall as temperature rises. Models with sharper output distributions (larger, or fine-tuned on the task) degrade more slowly than weaker models. Each temperature is a single sampled run with a fixed seed, so small differences are within sampling noise.

## 8. Responsible and ethical AI analysis

Evidence produced by the code: heuristic flags on all 1,000 test translations (`results/outputs/audit_<model>.csv`), the number of test sentences containing sensitive terms, and 15 hand-written **probe sentences** covering gender, politeness, culture, religion, social groups, health, violence, mental health, numbers/entities, idioms and ambiguity (`results/probe_translations.csv`).

**Concerns to examine in the outputs**

* **Bias.** English 'doctor', 'engineer' and 'CEO' carry no gender, but Hindi verbs and adjectives must be gendered, so a model is forced to guess; check whether the guess follows stereotypes (for example masculine for engineer/CEO and feminine for nurse/assistant). Also check the honorific level chosen for 'you' in the politeness probes and whether religious or caste-related words are rendered neutrally.
* **Hallucinated information.** Look at rows flagged `over_generation`, `number_mismatch`, `repetition_loop` and `entity_changed_or_dropped`: added content, changed figures, dropped names or looping text. Fluent but wrong output is the most dangerous failure because readers cannot see it without knowing the source. Small scratch-trained models tend to produce fluent-looking fragments and repetitions; prompted LLMs may add explanations or extra sentences.
* **Cultural misrepresentation.** Check festivals, clothing, food, honorifics and idioms in the probes: literal rendering of idioms, loss of cultural terms, or replacing an unfamiliar concept with a familiar but different one.
* **Inappropriate translation of sensitive content.** Check health, violence, religion, mental-health and group-identity sentences for softening, intensification, omission or insertion of judgement, and for unsafe handling of helpline or medical statements.

**Suggested measures**

1. Keep a human translator in the loop for medical, legal, safety, news and immigration content; never publish raw machine output for such material.
2. Show users a machine-translation notice and, where possible, a confidence or quality-estimation signal; allow easy reporting of errors.
3. Evaluate on gender-, culture- and domain-specific test sets (not only BLEU/chrF) and report results per category.
4. Mitigate gender bias with gender-balanced or counterfactual training data, explicit gender-neutral or both-gender outputs for ambiguous sources, and user-selectable gender/formality controls.
5. Detect hallucination with automatic checks (numbers, named entities, length ratio, repetition, back-translation or quality-estimation models) and route flagged outputs to review.
6. Curate training data: remove toxic, private or copyrighted material, and document the corpus (datasheet) and the models (model card) with known limitations.
7. Use constrained decoding or glossaries for names and terminology; keep temperature low (greedy/beam) for production translation.
8. Protect privacy: do not send confidential text to third-party services without consent; log and retain minimal data.
9. Monitor deployed systems and retrain or patch when bias or failure patterns are found.

**Observed examples (add after reading the output files):** list two or three concrete rows from `results/probe_translations.csv` or an audit file, say which model produced them, and which concern each illustrates.

## 9. Conclusion

<!-- CONCLUSION_START -->
_Generated after `python run_project.py` completes._
<!-- CONCLUSION_END -->

## 10. Repository layout

```
run_project.py     configuration + entry point
mtkit.py           data preparation, mT5 and LLaMA/Gemma runners, metrics, audit, report
mt_transformer.py  Transformer implementation, training and decoding
requirements.txt
data/              cleaned train / val / test splits (jsonl)
results/           metrics, plots, translations, audit files, probe translations
README.md
```
