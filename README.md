#  Multimodal Story Reasoning

**CNN-based Visual Encoder + Text Encoder + Temporal Fusion + Caption & Image-Feature Prediction**

This project implements a **lightweight multimodal deep learning system** for story-sequence reasoning:

* Input: a sequence of **images + captions**
* Model learns to:
  ✓ Generate the **next caption**
  ✓ Predict the **next image feature** (representation of last frame)

The architecture is designed for **efficient notebook training**, replacing heavy pretrained encoders with a notebook-friendly **lightweight CNN**.

---

##  Key Features

### ✔ Lightweight Visual Encoder (Custom CNN)

* Replaces heavy ResNet50/ViT
* Faster training, stable gradients, ideal for Colab/Jupyter
* 512-dim image embeddings

### ✔ Text Encoder (LSTM)

* Tokenized captions → 512-dim embedding

### ✔ Temporal Fusion

Per-frame fused representation:

```
TimeDistributed(VisualEncoder) + TimeDistributed(TextEncoder)
       → Dense → LSTM → Context Vector
```

### ✔ Two Decoders

1. **Text Decoder** — teacher forcing, generates next-frame caption
2. **Image-Feature Decoder** — predicts the CNN feature vector for final image

### ✔ Robust Training Loop (Notebook-Friendly)

Includes:

* Correct **token-level accuracy**
* Three forms of **sequence accuracy**:

  * `seq_strict` = strict token-id exact match
  * `seq_norm` = normalized text exact match (lowercase, punctuation removed)
  * `seq_fuzzy` = BLEU-based soft matching
* Handles **padding**, **empty caption repair**, and **stable loss**
* Step-level history saved to `results/loss_history.json`

### ✔ Evaluation Metrics

Automatically computes:

* BLEU
* ROUGE-L
* strict / normalized / fuzzy sequence accuracy
  All stored in `results/eval_metrics.json`.

---

##  Project Structure

```
src/
 ├── model.py        # CNN visual encoder + text encoder + multimodal model  
 ├── utils.py        # preprocessing, dataset loading, generators  
 ├── train.py        # full training pipeline  
notebooks/
 ├── notebook_cells/ # notebook-ready training + evaluation cells  
results/
 ├── loss_history.json
 ├── eval_metrics.json
config.yaml
requirements.txt
README.md
```

---

##  Installation

### 1. Clone repository

```bash
git clone <your_repo_url>
cd <your_repo>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Dataset

Default dataset:

```
daniel3303/StoryReasoning
```

Each sample contains:

* `frames` — list of image paths or pixel arrays
* `captions` — list of text descriptions
* All images + captions are truncated/padded to `seq_len`

---

## ⚙ Config File (`config.yaml`)

Example:

```yaml
dataset:
  hf_name: "daniel3303/StoryReasoning"
  seq_len: 3
  batch_size: 16
  image_size: 128
  max_caption_len: 32

model:
  image_feat_dim: 512
  text_embed_dim: 300
  text_hidden_dim: 512
  multimodal_dim: 512
  temporal_hidden_dim: 512
  text_decoder_hidden: 512
  vocab_size: 30522
  pad_token_id: 0
  bos_token_id: 101
  eos_token_id: 102

training:
  lr: 1e-4
  epochs: 5
  grad_clip: 1.0
  log_interval: 50
  save_dir: "results/checkpoints"
```

---

##  Training (Script Mode)

```bash
python src/train.py --config config.yaml
```

Checkpoints saved to:

```
results/checkpoints/ckpt_epochX.weights.h5
```

---

##  Notebook-Friendly Training (Recommended)

### Training cell features:

* Handles **padding repair**
* Tracks correct **token-level accuracy**
* Computes three types of **sequence accuracy**
* Stores per-step training logs
* Runs evaluation on random samples

### Evaluation cell includes:

* Greedy decoding
* BLEU / ROUGE-L
* Normalized text exact-match
* Fuzzy BLEU-based match
* Saves metrics to `results/eval_metrics.json`

### Plot loss + accuracy curves (from saved JSON)

Notebook includes:

* Loss curve (total/text/image)
* Token accuracy curve
* Strict / normalized / fuzzy seq accuracy curves

---

##  Metrics Explained

### Token Accuracy

Percentage of non-pad tokens predicted correctly.

### Strict Sequence Accuracy

A sequence counts as correct only if **every non-pad token** matches exactly.
→ Very harsh; usually near 0 early in training.

### Normalized Sequence Accuracy

Predictions and targets normalized by:

* Lowercasing
* Removing punctuation
* Collapsing whitespace

More realistic for story captions.





## 🛠 Extending the Model

You can easily extend:

* Swap CNN → MobileNetV2 / EfficientNet
* Add attention over frames
* Add transformer caption decoder
* Add CLIP loss for image-text alignment

---

##  Acknowledgements

Dataset by: **daniel3303/StoryReasoning**
Text tokenization by HuggingFace Transformers
Image models built with TensorFlow/Keras


