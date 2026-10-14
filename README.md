# Derm-AI: Melanoma Detection from Skin Lesion Images

A supervised **binary image classification** project in medical computer vision. It aims to identify melanoma early and accurately from skin lesion images, which is hard to do by eye and especially difficult where dermatologists are scarce. The project could support doctors with early screening, and by comparing **CNN, RNN and Vision Transformer** models it explores which approach is most reliable for medical diagnosis support.

> **Educational use only.** This is not a medical diagnostic tool.

## Dataset

- **SIIM-ISIC 2020 Melanoma Classification** (256x256 JPEG mirror on Kaggle: `cdeotte/jpeg-melanoma-256x256`), downloaded automatically with `kagglehub`.
- Labels: `0` = benign, `1` = melanoma.
- The dataset is heavily imbalanced (melanoma is the minority class).
- The raw images are **not** stored in this repository.

## Models compared

| Model | Description |
|-------|-------------|
| **CNN** | EfficientNetB0 (ImageNet pretrained, frozen) + dense classification head |
| **RNN** | EfficientNetB0 feature maps treated as a sequence, then a Bidirectional LSTM |
| **Transformer** | Vision Transformer backbone (`vit-keras`, frozen) + small trainable head |

## Pipeline

1. Download the ISIC data and match images with `train.csv`
2. Exploratory data analysis (class distribution, sample images)
3. 70 / 15 / 15 train / validation / test split (patient-level when `patient_id` is available, otherwise stratified)
4. Oversample the melanoma class in the **training set only**; validation and test keep their natural distribution
5. Data augmentation (flips, rotation, zoom, contrast) and a shared `tf.data` pipeline
6. Train each model with class weights, early stopping and learning-rate reduction, monitoring validation AUC
7. Evaluate with accuracy, precision, sensitivity (recall), specificity, F1, AUC-ROC, confusion matrices and ROC curves
8. Save the best model and run single-image prediction

## Results (quick-test baseline)

These results come from a **quick-test run**: 5,000 images, 3 epochs per model, 702 test images (only 13 melanoma cases).

| Model | Accuracy | Precision | Sensitivity (Recall) | Specificity | F1 | AUC-ROC |
|-------|----------|-----------|----------------------|-------------|-----|---------|
| CNN (EfficientNetB0) | 0.981 | 0.000 | 0.000 | 1.000 | 0.000 | 0.604 |
| RNN (CNN + BiLSTM) | 0.019 | 0.019 | 1.000 | 0.000 | 0.036 | 0.316 |
| Transformer (ViT) | 0.833 | 0.027 | 0.231 | 0.845 | 0.049 | **0.690** |

**Best model by AUC-ROC: Vision Transformer (0.69).**

### How to read these results

- The CNN's 98% accuracy is misleading: it predicted *benign* for almost every image, so it caught **none** of the 13 melanomas. With such an imbalanced test set, accuracy alone is not a reliable metric.
- The RNN did the opposite and predicted melanoma for almost every image. Its AUC is below 0.5, so it did not learn a useful signal in this short run.
- The Vision Transformer was the only model with some separation between classes, but its sensitivity and precision are still low.
- With only 3 epochs, 5,000 images and 13 melanoma cases in the test set, these numbers are a pipeline check, not a final conclusion.

## Next steps

- Set `MAX_IMAGES = None` to train on the full dataset
- Train for more epochs and fine-tune the pretrained backbones
- Tune the decision threshold to prioritize sensitivity, which matters most for screening
- Report precision-recall curves and confidence intervals

## How to run

1. Install requirements: `pip install tensorflow pandas numpy scikit-learn pillow matplotlib seaborn kagglehub vit-keras`
2. Open `Skin_Lesion_ISIC_CNN_RNN_Transformer_FIXED.ipynb`
3. Run all cells top to bottom. The dataset downloads automatically (a Kaggle account may be required for `kagglehub`).

## Tech stack

Python, TensorFlow / Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, EfficientNetB0, Bidirectional LSTM, Vision Transformer (`vit-keras`)

## Disclaimer

This project is for research and education only and must not be used for medical diagnosis or treatment decisions.
