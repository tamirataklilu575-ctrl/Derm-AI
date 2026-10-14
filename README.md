<h1 align="center">Derm AI</h1>
<p align="center"><b>Melanoma detection from skin lesion images: CNN vs. RNN vs. Vision Transformer</b></p>
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="Task" src="https://img.shields.io/badge/Task-Binary%20Image%20Classification-4f46e5">
  <img alt="Status" src="https://img.shields.io/badge/Status-Baseline%20complete-2ea44f">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
</p>

> **Medical disclaimer.** Derm AI is an educational and experimental project. It is **not** a clinically validated diagnostic tool and must not be used to diagnose melanoma, rule out cancer, or replace professional medical assessment.

---

## Overview

Melanoma is the most dangerous form of skin cancer, and catching it early greatly improves outcomes. Yet early melanomas are hard to distinguish from benign lesions by eye, and access to dermatologists is limited in many regions.

**Derm AI** is a supervised binary image classification project in medical computer vision. It trains and compares three deep learning architectures on the same data, split and evaluation protocol, to study which approach is the most reliable candidate for supporting early screening:

| Model | Architecture | Idea |
|-------|--------------|------|
| **CNN** | EfficientNetB0 (ImageNet pretrained, frozen) + dense head | Learns local spatial features through convolutions |
| **RNN** | EfficientNetB0 feature maps → Bidirectional LSTM | Reads the feature-map rows as a sequence |
| **Vision Transformer** | `vit-keras` backbone (frozen) + trainable head | Models relationships between image patches with self-attention |

## Key Findings

This is a **baseline experiment** (5,000 images, 3 epochs per model). The main results:

- The **Vision Transformer** achieved the best **AUC-ROC (0.690)** and was the only model with any useful separation between the two classes.
- The **CNN reached 98.1% accuracy but detected 0 of 13 melanomas.** It predicted "benign" for nearly every image, which is a textbook example of why accuracy is misleading on imbalanced medical data.
- The **RNN** over-predicted melanoma (flagging every melanoma but also nearly every benign image) and scored below chance (AUC 0.316), so it did not learn a useful signal in this short run.
- Even the best model is **not clinically useful yet** (sensitivity 23%). The value of this project is the complete, reproducible pipeline and the evaluation lessons, which are ready to scale to a full training run.

## Results

Evaluated on a held-out test set of **702 images (689 benign, 13 melanoma)**. Models are sorted by AUC-ROC.

| Model | AUC-ROC | Sensitivity (Recall) | Specificity | Precision | F1 | Accuracy |
|-------|:-------:|:--------------------:|:-----------:|:---------:|:--:|:--------:|
| **Vision Transformer** | **0.690** | 0.231 (3 / 13) | 0.845 | 0.027 | 0.049 | 0.833 |
| CNN (EfficientNetB0) | 0.604 | 0.000 (0 / 13) | 1.000 | 0.000 | 0.000 | 0.981 |
| RNN (CNN + BiLSTM) | 0.316 | 1.000 (13 / 13) | 0.000 | 0.019 | 0.036 | 0.019 |

**How to read this table.** For screening, *sensitivity* (not missing a melanoma) and *AUC-ROC* (ranking quality independent of threshold) matter far more than accuracy. With only 13 melanoma cases in the test set, every figure above has a wide margin of error, so treat these numbers as a pipeline check rather than a final ranking of the architectures.

## Experimental Setup

| Item | Setting |
|------|---------|
| Dataset | SIIM-ISIC 2020 Melanoma Classification (256×256 JPEG mirror) |
| Labels | 0 = benign, 1 = melanoma |
| Images used | 5,000 (stratified sample) |
| Split | 70 / 15 / 15 train / validation / test; by patient when `patient_id` is available, otherwise stratified |
| Imbalance handling | Melanoma class oversampled in the **training set only** + class weights; validation and test keep their natural distribution |
| Input size / batch | 224×224 / 32 |
| Augmentation | Horizontal & vertical flips, rotation, zoom, contrast |
| Optimizer / loss | Adam / binary cross-entropy |
| Regularization | Dropout 0.35, early stopping and learning-rate reduction on validation AUC |
| Epochs | 3 per model (quick-test setting) |
| Reproducibility | Fixed random seed (42) |

## Pipeline

1. **Data:** download ISIC 2020 through `kagglehub` and match images with `train.csv`
2. **EDA:** class distribution and sample images
3. **Split:** train / validation / test with leakage prevention
4. **Balance:** oversample the minority class in the training set
5. **Input pipeline:** `tf.data` loading, resizing and augmentation shared by all models
6. **Training:** three models under identical conditions
7. **Evaluation:** accuracy, precision, sensitivity, specificity, F1, AUC-ROC, confusion matrices and ROC curves
8. **Inference:** save the best model and predict on a single image

## Limitations

- Short training (3 epochs) on a 5,000-image subset; results will change with the full dataset and longer training.
- Only 13 melanoma cases in the test set, so metrics are statistically unstable.
- Backbones were frozen; fine-tuning was not explored.
- A single decision threshold (0.5) was used; screening would require threshold tuning for high sensitivity.
- No external validation on other populations, devices or clinical settings.

## Roadmap

- [ ] Train on the full dataset (`MAX_IMAGES = None`) for more epochs
- [ ] Fine-tune the pretrained backbones
- [ ] Tune the decision threshold for high sensitivity
- [ ] Add precision-recall curves and confidence intervals
- [ ] Add explainability (e.g. Grad-CAM) to show what the model focuses on
- [ ] Cross-validation and external validation

## Getting Started

```bash
pip install tensorflow pandas numpy scikit-learn pillow matplotlib seaborn kagglehub vit-keras
jupyter notebook Skin_Lesion_ISIC_CNN_RNN_Transformer_FIXED.ipynb
```

Run the cells from top to bottom. The dataset downloads automatically (a Kaggle account may be required for `kagglehub`). To run the full experiment, change `MAX_IMAGES` to `None` and increase `EPOCHS` in the configuration cell.

## Repository Structure

```
Derm-AI/
├── Skin_Lesion_ISIC_CNN_RNN_Transformer_FIXED.ipynb   # Full pipeline: data, models, evaluation
├── README.md                                          # Project documentation
└── LICENSE                                            # MIT License
```

## What I Learned

- Building an end-to-end medical image classification pipeline
- Handling severe class imbalance without leaking information into validation and test data
- Comparing CNN, recurrent and transformer architectures fairly under identical conditions
- Choosing clinically meaningful metrics, and why accuracy alone can hide a model that misses every positive case
- Documenting limitations honestly for AI-assisted medical applications

## Technologies

Python · TensorFlow / Keras · scikit-learn · pandas · NumPy · Matplotlib · Seaborn · EfficientNetB0 · Bidirectional LSTM · Vision Transformer (`vit-keras`) · Jupyter Notebook

## Dataset and Acknowledgements

- SIIM-ISIC 2020 Melanoma Classification challenge, used through the public 256×256 JPEG mirror on Kaggle (`cdeotte/jpeg-melanoma-256x256`).
- Rotemberg, V. et al. *A patient-centric dataset of images and metadata for identifying melanomas using clinical context.* Scientific Data 8, 34 (2021).
- The ISIC data is distributed under its own license (CC BY-NC); please review it before any reuse. The images are not included in this repository.

## License

The code in this repository is released under the [MIT License](LICENSE). The ISIC dataset is covered by its own license (see above).

## Author

**Aklilu Tamirat**
Machine Learning Student · Ethiopian Artificial Intelligence Institute (EAII)
GitHub: [@tamirataklilu575-ctrl](https://github.com/tamirataklilu575-ctrl)
