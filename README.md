# TLDChoiceNet: Quantitatively Choosing a Transfer Learning Dataset

**Predict how well a transfer-learning dataset will work — before spending the compute to fine-tune on it.**

[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](#citation)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-ff6f00.svg)](https://www.tensorflow.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Jing Ning, James D. Braza — Stanford University · [CS 330: Deep Multi-Task and Meta-Learning](https://cs330.stanford.edu/)

> Joint work. Originally developed at
> [jamesbraza/cs330-project](https://github.com/jamesbraza/cs330-project); this repository
> preserves the full commit history.

---

## The problem

You have a small dataset for a target task. You have candidate pre-trained datasets A, B,
and C — similar example counts, similar class counts — but compute to fine-tune on only
**one** of them. Which do you pick?

Conventional wisdom says take the biggest, most diverse source. But no quantitative method
existed to make that call. TLDChoiceNet predicts the post-fine-tuning test accuracy for each
candidate, so the choice becomes a measurement instead of a guess.

## Results

| Model / metric | Result |
| --- | --- |
| TLDChoiceNet v1 — test MSE | 0.154 |
| TLDChoiceNet v2 — test MSE | **0.031** (5× lower) |
| Distribution distance (DD) vs. accuracy — *R²* | 0.894 |
| Average class correlation (ACC) vs. accuracy — *R²* | **0.974** |

Two results worth separating:

**The learned predictor.** v2 cuts test MSE 5× over v1 by embedding both inputs *per class*
rather than averaging across all of them, and by learning the post-embedding reduction rather
than fixing it. It has ~1.15M trainable parameters against v1's ~314K.

**The unsupervised metrics — no training required at all.** Average class correlation reaches
an *R²* of 0.974 against fine-tune accuracy. You can compute it from a pre-trained ResNet50 v2
forward pass and pick your transfer dataset without training anything.

## Approach

### The Transfer Learning Dataset (TLDS)

The core obstacle is that there is no dataset of *"transfer learning dataset → resulting
accuracy"* pairs to learn from. So we built one. Each entry is a 3-tuple of
`(transfer learning dataset, transfer-learned model, resulting test accuracy)`, spanning four
deliberately chosen regimes:

| Regime | Source | Subsets |
| --- | --- | --- |
| Similar | Plant diseases | 3 × 10 classes |
| Dissimilar | Bird species | 3 × 10 classes |
| Random | CIFAR-100 and ImageNet | 40 × 10 classes |
| No transfer learning | Random initialisation (control) | 1 |

Each invocation of the generation script produces 57 data points; varying the seed yields
more. Fine-tuning throughout is on a 22-class plant-leaves dataset.

### TLDChoiceNet

Both versions take an embedded fine-tuning dataset and an embedded transfer-learning
dataset/model, and regress to a single number — predicted test accuracy.

- **v1** embeds the fine-tuning dataset via PCA to 256 dimensions and the transfer-learned
  model via its flattened last Conv2D, applies a LoRA-similar learned reduction, and
  concatenates.
- **v2** fixes v1's central flaw: both embeddings ignored class specifics. v2 embeds
  **per class**, keeps the 10 highest-activation classes, and *adds* the two reduced
  matrices instead of concatenating them.

### Two unsupervised metrics

**Distribution distance (DD)** — the Euclidean distance between two datasets in
(mean, |skew|, |kurtosis|) space over normalised pixel values:

$$\mathrm{DD}(i,j)=\sqrt{(\mu_i-\mu_j)^2+(s_i-s_j)^2+(\kappa_i-\kappa_j)^2}$$

**Average class correlation (ACC)** — mean pairwise correlation between per-class embeddings
from an ImageNet-pretrained headless ResNet50 v2:

$$\mathrm{ACC}(i,j)=\frac{1}{nm}\sum_{k=1}^{n}\sum_{l=1}^{m}\mathrm{Cor}_{kl}$$

ACC separates similar from dissimilar transfer datasets far more sharply (0.501 vs. 0.276 —
45% lower for dissimilar) and explains fine-tune accuracy better than DD does.

## Key findings

**Pre-trained weights push dissimilar classes apart in latent space.** Computing ACC from raw
normalised pixels gives 0.507 for similar classes and 0.438 for dissimilar — a gap of just
0.06. Running the same computation through ImageNet-pretrained ResNet50 v2 weights widens
that gap to 0.22 (0.49 vs. 0.27). The pre-trained weights are actively increasing class
separation, which is a concrete answer to *what* transfer learning transfers.

**A dataset's low-level pixel statistics explain much of the transfer effect.** DD, computed
from nothing but mean, skew, and kurtosis of pixel values, reaches an *R²* of 0.894.

**Similar transfer datasets move weights less during fine-tuning.** L2 distance between
pre-trained and fine-tuned weights stays smaller across epochs for similar transfer datasets,
corroborated by centered kernel alignment (CKA).

### Honest limitations

- The reported MSE figures come from a test subset that **shares its fine-tuning dataset**
  with the training subset. On a genuinely unseen fine-tuning dataset, v1 degrades badly —
  MSE 0.46, with accuracy predictions off by as much as 80%. Generalising across fine-tuning
  datasets needs either a v3 architecture or a TLDS spanning multiple fine-tuning datasets.
- Test accuracy spans only ~15% from random initialisation to the best transfer-learned
  model, which is a narrow band in which to resolve differences.
- Dissimilar transfer datasets scored about the same as random ones, so image diversity alone
  may matter less than expected here.
- Everything is image classification on one fine-tuning task. Generalisation to other
  domains is untested.

## Datasets

| Dataset | Role |
| --- | --- |
| [Plant leaves][4] (22 classes) | Fine-tuning target |
| [Plant diseases][2] / [`plant_village`][5] | Similar transfer source |
| [Birds 450 species][3] | Dissimilar transfer source |
| CIFAR-100, ImageNet | Random transfer sources |

Download them all with the Kaggle API:

```bash
kaggle datasets download -p data/plant-diseases --unzip vipoooool/new-plant-diseases-dataset
kaggle datasets download -p data/plant-leaves --unzip csafrit2/plant-leaves-for-image-classification
kaggle datasets download -p data/bird-species --unzip gpiosenka/100-bird-species
```

## Getting started

Developed with Python 3.10.

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

The pipeline runs in three stages — build the TLDS, fine-tune, then train ChoiceNet on the
result:

```bash
bash training/1.run_tl_training.sh    # train TransferModel on each TL dataset
bash training/2.run_fine_tune.sh      # fine-tune onto plant leaves, record accuracy
bash training/3.run_choicenet.sh      # train TLDChoiceNet on the resulting TLDS
```

> The shell scripts carry absolute paths from the original GPU machine
> (`/data1/cs330/project/...`). Point them at your own directories before running.

Monitor training:

```bash
tensorboard --logdir training      # then open http://localhost:6006/
```

### Code quality tooling

```bash
python -m pip install -r requirements-qa.txt
pre-commit install
```

## Repository layout

```
data/         Dataset loading, preprocessing, and TLDS source configs
models/       TransferModel CNN and ChoiceNet v1/v2 architectures
training/     TLDS creation, fine-tuning, and ChoiceNet training entry points
embedding/    ResNet50 v2 dataset embedding and weight-matrix preprocessing
experiments/  Metric analysis — CKA, KS tests, correlation matrices, weight distance
```

## Citation

```bibtex
@article{ning2026tldchoicenet,
  title   = {TLDChoiceNet: Quantitatively Choosing a Transfer Learning Dataset},
  author  = {Ning, Jing and Braza, James D.},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Acknowledgements

We thank Chelsea Finn and Daniel Zeng for helpful discussions and feedback, including ideas on
ImageNet embedding and measuring weight distance across training.

[2]: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
[3]: https://www.kaggle.com/datasets/gpiosenka/100-bird-species
[4]: https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification
[5]: https://www.tensorflow.org/datasets/catalog/plant_village
