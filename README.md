# Clothes-Changing Person Re-Identification: Model Comparison & Evaluation

A compact, reproducible PyTorch pipeline for comparing Re-ID backbones on **clothes-changing** person re-identification, and for checking whether a model's **confidence** can tell its correct matches from its wrong ones.

The project fine-tunes several backbones on **PRCC** (clothes-changing) and **Market-1501** (no clothes change, used as a control), evaluates them under same-clothes and clothes-changing protocols, and analyses how trustworthy each model's confidence is (AUROC and abstention / risk–coverage curves).

> Everything runs in a single Google Colab notebook (T4 GPU is enough). Runs and results are saved to Google Drive so a session timeout never loses work.

---

## Highlights

- **Multiple backbones, one training recipe:** OSNet-x1.0, OSNet-AIN, ResNet-50, DenseNet-121 (ViT-Small is also registered).
- **Two evaluation settings on PRCC:**
  - `sc` (same-clothes): the true match wears the same outfit as the query.
  - `cc` (clothes-changing): the true match wears a different outfit.
- **Market-1501 control:** the standard setting with no clothes labels.
- **Confidence / uncertainty analysis:** AUROC of two confidence scores and abstain curves (answer only the most confident queries).
- **Honest evaluation:** no checkpoint selection on the test set. The **last epoch** is saved and evaluated.
- **Experimental extras:** colour-augmentation variant, outfit-aware PK sampler, and a DIFFER-style CLIP ViT-B/16 model (see [Extra experiments](#extra-experiments)).

---

## Project structure

The notebook writes these files into `cc-reid-uncertainty/` using `%%writefile`:

```
cc-reid-uncertainty/
├── reid/
│   ├── datasets.py       # PRCC + Market-1501 parsers
│   ├── metrics.py        # CMC / mAP, sc/cc protocols, per-query records
│   ├── loader.py         # transforms, dataset, PK batch sampler (+ outfit-aware)
│   ├── model.py          # backbones (Torchreid CNNs + timm ViT) and losses
│   └── differ_model.py   # DIFFER-style CLIP ViT-B/16 model
├── train.py              # fine-tune one backbone on one dataset
├── evaluate.py           # evaluate a trained run, save metrics + per-query records
├── check_dataset.py      # sanity-check dataset counts (run this first)
├── confidence.py         # AUROC and risk-coverage analysis
├── report.py             # tables and figures
├── analysis.py           # cross-model summary table + AUROC table
├── train_differ.py       # DIFFER-style training
├── make_captions.py      # pseudo text descriptions with a small open VLM
├── make_text_feats.py    # captions -> frozen CLIP text embeddings
├── runs/                 # checkpoints and training logs (symlinked to Drive)
└── results/              # json / npz metrics and figures (symlinked to Drive)
```

---

## Datasets

| Dataset | Purpose | Train | Query | Gallery |
|---|---|---|---|---|
| **PRCC** | clothes-changing | 17,896 imgs / 150 IDs | 7,416 imgs / 71 IDs | 3,384 imgs / 71 IDs |
| **Market-1501** | control (no clothes change) | 12,936 imgs / 751 IDs | 3,368 imgs / 750 IDs | 15,913 imgs / 751 IDs |

**PRCC protocol:** gallery = camera A, query = camera B (same clothes) + camera C (different clothes). Evaluation is **single-shot** (one random gallery image per identity) and repeated for 10 trials, then averaged.

Download the datasets yourself from their official sources and set the paths in the notebook. The notebook expects:

```
/content/data/prcc_raw/prcc/rgb/{train,val,test}/...
/content/data/Market-1501-v15.09.15/{bounding_box_train,bounding_box_test,query}/
```

---

## Method

**Training** (identical for every backbone, for a fair comparison)

| Setting | Value |
|---|---|
| Loss | Label-smoothed cross-entropy (ε = 0.1) + batch-hard triplet (margin 0.3) |
| Optimizer | Adam, lr 3.5e-4 (1e-4 for ViT), weight decay 5e-4 |
| Schedule | 5-epoch linear warm-up, then cosine decay to 0 |
| Epochs | 30 |
| Batching | PK sampler: P = 16 identities × K = 4 images |
| Input | 256 × 128, random flip / pad-crop / random erasing |
| Precision | Mixed precision (AMP) on GPU |
| Weights | ImageNet-pretrained, last epoch saved |

**Evaluation**

- Features are L2-normalised and averaged with a horizontally flipped copy (flip test-time augmentation).
- Similarity is cosine similarity.
- Standard Re-ID filtering removes same-identity, same-camera gallery items.
- Metrics: Rank-1, Rank-5, Rank-10, mAP.

**Confidence scores** (for each query)

1. `top1_sim`: similarity of the best gallery match.
2. `margin`: top-1 similarity minus the best similarity of a *different* identity.

Each score is evaluated with **AUROC** (does confidence separate right from wrong? 0.5 = useless, 1.0 = perfect) and with **risk–coverage curves** (rank-1 accuracy when answering only the most confident fraction of queries, compared against a perfect-abstention oracle).

---

## Results

Rank-1 / Rank-5 / mAP in %. Mean ± std over seeds where more than one seed was run (`seeds` column).

| Train set | Setting | Model / variant | Rank-1 | Rank-5 | mAP | Seeds |
|---|---|---|---|---|---|---|
| Market-1501 | general | OSNet-x1.0 | 93.3 | 97.5 | 82.3 | 1 |
| Market-1501 | general | OSNet-AIN | 91.8 | 97.2 | 80.4 | 1 |
| Market-1501 | general | DenseNet-121 | 90.7 | 96.6 | 78.0 | 1 |
| Market-1501 | general | ResNet-50 | 90.5 | 96.7 | 79.9 | 1 |
| PRCC | cc | OSNet-x1.0 | 38.9 ± 2.2 | 58.0 | 48.5 ± 2.0 | 3 |
| PRCC | cc | OSNet-AIN | 33.4 | 58.4 | 45.8 | 1 |
| PRCC | cc | OSNet-x1.0 + colour aug | 30.5 | 58.0 | 43.9 | 1 |
| PRCC | cc | ResNet-50 | 23.6 | 44.8 | 34.2 | 1 |
| PRCC | cc | DenseNet-121 | 22.1 | 40.8 | 32.3 | 1 |
| PRCC | sc | OSNet-AIN | 98.4 | 99.9 | 99.1 | 1 |
| PRCC | sc | OSNet-x1.0 | 98.4 ± 0.4 | 99.8 | 99.0 ± 0.2 | 3 |
| PRCC | sc | OSNet-x1.0 + colour aug | 97.9 | 99.7 | 98.7 | 1 |
| PRCC | sc | ResNet-50 | 92.6 | 98.6 | 95.3 | 1 |
| PRCC | sc | DenseNet-121 | 91.1 | 97.6 | 94.0 | 1 |

### Confidence quality (AUROC)

| Train set | Setting | Variant | AUROC (margin) | AUROC (top-1) |
|---|---|---|---|---|
| Market-1501 | general | OSNet-x1.0 | 0.914 | 0.750 |
| PRCC | sc | OSNet-x1.0 | 0.984 | 0.972 |
| PRCC | cc | OSNet-x1.0 | 0.775 | 0.706 |
| PRCC | cc | OSNet-AIN | 0.698 | 0.656 |
| PRCC | cc | ResNet-50 | 0.643 | 0.676 |
| PRCC | cc | DenseNet-121 | 0.709 | 0.731 |

### Key takeaways

- **Clothes change is the real challenge.** On PRCC, OSNet-x1.0 reaches about 98% Rank-1 when clothes stay the same, but only about 39% when they change.
- **OSNet clearly beats ResNet-50 and DenseNet-121** in the clothes-changing setting (≈39% vs ≈22–24% Rank-1).
- **Simple colour augmentation did not help** (30.5% vs 38.9% Rank-1 in the cc setting, single seed).
- **Confidence is much less reliable under clothes change.** The `margin` score separates right from wrong well in the same-clothes setting (AUROC ≈ 0.98) but drops to ≈ 0.64–0.78 in the clothes-changing setting, so abstaining on low-confidence queries helps, but far from perfectly.
- The `margin` score is generally a better confidence signal than raw `top1_sim` for the best-performing models.

> Most variants were run with a single seed, so differences of a few points should be read as indicative rather than conclusive.

---

## How to run

1. **Open the notebook in Google Colab** and select a **GPU runtime** (T4 is fine).
2. Run the setup cells: mount Google Drive, clone [`deep-person-reid`](https://github.com/KaiyangZhou/deep-person-reid) (Torchreid), and install `yacs gdown h5py tensorboard timm`.
3. Run the `%%writefile` cells to create the `reid/` package and the scripts.
4. Download and unpack PRCC and Market-1501 and check them first:

   ```bash
   python check_dataset.py --dataset prcc   --root /content/data/prcc_raw/prcc
   python check_dataset.py --dataset market --root /content/data/Market-1501-v15.09.15
   ```

5. **Train and evaluate** (example: OSNet-x1.0 on PRCC):

   ```bash
   python train.py    --dataset prcc --root $PRCC --seed 0 --epochs 30 --workers 2
   python evaluate.py --run runs/prcc_base_s0 --root $PRCC --workers 2
   ```

   Other options: `--arch {osnet_x1_0, osnet_ain_x1_0, resnet50, densenet121, vit_small}`, `--color_aug`, `--cc_sampler`, `--seed N`.

6. **Analyse and report:**

   ```bash
   python confidence.py     # AUROC + risk-coverage
   python report.py         # tables + figures
   ```

Figures are written to `results/figures/` (`fig1_sc_vs_cc.png`, `fig2_abstain.png`, `fig3_auroc.png`).

---

## Extra experiments

The notebook also contains code for:

- **Outfit-aware sampler** (`--cc_sampler`): builds batches so that each identity appears in more than one outfit.
- **DIFFER-style model** (`reid/differ_model.py`, `train_differ.py`): CLIP ViT-B/16 image encoder with a BNNeck classifier and non-biometric detaching (gradient-reversal) heads. Pseudo text descriptions are generated with a small open VLM (`make_captions.py`) and encoded with a frozen CLIP text encoder (`make_text_feats.py`), as a lightweight replacement for the large captioning models of the original paper so that it fits on a T4.

Results for these extra experiments are not included in the tables above.

---

## Requirements

- Python 3.10+ (developed on Colab)
- PyTorch + torchvision, with CUDA recommended
- [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)
- `timm`, `yacs`, `gdown`, `h5py`, `tensorboard`
- `numpy`, `scipy`, `pandas`, `matplotlib`, `Pillow`

---

## Acknowledgements

- [Torchreid / deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) by Kaiyang Zhou for the OSNet, ResNet and DenseNet implementations.
- The **PRCC** and **Market-1501** dataset authors. Please follow each dataset's own license and citation requirements.

---

## Author

**Nimra Abdulhaq**
Email: [nimraabdulhaqq@gmail.com](mailto:nimraabdulhaqq@gmail.com)
