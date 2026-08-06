# Knee Osteoarthritis Severity Grading using Multimodal Deep Learning

## Live Demo - https://knee-oa-severity-grading.streamlit.app/

## Overview

This project presents a multimodal deep learning framework for automated Knee Osteoarthritis (OA) severity grading using knee X-ray images and clinical metadata. The system combines image-based deep learning models with structured clinical and radiographic information to improve OA severity prediction.

The project explores:
- Image-only models
- Metadata-only models
- Multimodal models
- Multi-class classification
- Ordinal classification

The study demonstrates that multimodal ordinal models outperform image-only and standard multi-class approaches.

---

## Problem Statement

Knee Osteoarthritis severity is traditionally assessed manually using radiographic images. This process is:
- Time-consuming
- Subjective
- Observer-dependent

The goal of this project is to improve OA severity grading using multimodal deep learning approaches that combine X-ray images with clinical metadata.

---

## Dataset

The dataset includes:
- Knee X-ray images
- Clinical metadata
- Radiographic metadata
- Kellgren–Lawrence (KL) grades (0–4)

### Metadata Features
- Age
- BMI
- Gender
- Knee pain scores
- Joint space narrowing
- Osteophyte indicators
- Sclerosis indicators

---

## Model Architectures

### Image-Only Models
- ResNet34
- Swin Transformer (Swin-Tiny)
- Multi-class and Ordinal variants

### Metadata-Only Model
- Fully connected neural network
- Ordinal classification

### Multimodal Models
- Naive multimodal model
- ResNet + Metadata
- Swin + Metadata
- Multi-class and Ordinal variants

---

## Ordinal Classification

Ordinal classification is used to model the progressive nature of OA severity.

Instead of predicting independent classes directly, the model predicts thresholds:
- KL > 0
- KL > 1
- KL > 2
- KL > 3

This improves prediction consistency and reduces large misclassification errors.

---

## Evaluation Metrics

The models are evaluated using:
- Macro F1-score
- Balanced Accuracy
- AUC Score
- Confusion Matrix

---

## Results

| Model | Pipeline | F1 Score | Balanced Accuracy | AUC |
|------|------|------|------|------|
| ResNet | Image-only | 0.6408 | 0.6355 | 0.8631 |
| Swin | Image-only | 0.6901 | 0.7020 | 0.9059 |
| Swin Ordinal | Image-only | 0.7015 | 0.7011 | 0.9615 |
| ResNet Ordinal | Image-only | 0.6633 | 0.6500 | 0.9461 |
| Meta | Metadata | 0.5610 | 0.6557 | 0.9606 |
| Naive | Multimodal | 0.7637 | 0.7644 | 0.9278 |
| ResNet + Meta | Multimodal | 0.8175 | 0.8115 | 0.9724 |
| Swin + Meta | Multimodal | 0.8258 | 0.8237 | 0.9354 |
| **Swin + Meta Ordinal** | **Multimodal** | **0.8312** | **0.8269** | **0.9702** |

### Key Findings
- Multimodal models outperform image-only models
- Ordinal classification improves performance over multi-class classification
- KL1 is the hardest class to predict
- KL0, KL1, and KL2 exhibit strong overlap

---

## Streamlit Interface

A Streamlit-based interface is developed to:
- Upload knee X-ray images
- Predict KL grade
- Display confidence score and risk level
- Generate AI-based health guidance
- Support multimodal prediction using linked metadata

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Streamlit
- Groq API

---

## Project Structure

```textKnee-OA-Severity-Grading/
│
├── app/
│   ├── app.py
│   ├── samples/
│   │  ├── grade0.png
│   │  ├── grade1.png
│   │  ├── grade2.png
│   │  ├── grade3.png
│   │  └── grade4.png
│   │
│   ├── models/
│   │   ├── swin_ord.pth
│   │   ├── swin_meta_ord.pth
│   │
│   ├── artifacts/
│   │   ├── scaler.pkl
│   │   ├── metadata.csv
│   │
│   ├── utils/
│   │   ├── predict.py
│   │   ├── risk.py
|   |   ├── advice.py
|   |   ├── scaler.py
│
├── training/
│   ├── img_resnet.py
│   ├── img_swin.py
│   ├── meta.py
│   ├── img_meta_naive.py
│   ├── resnet_meta_multi.py
│   ├── swin_meta_multi.py
│   ├── resnet_ord.py
│   ├── swin_ord.py
│   ├── swin_meta_ord.py
│
├── images/
│   ├── confusion_matrix/
│   ├── roc_curves/
│
├── report/
│   ├── final_report.pdf
│   ├── final_presentation.pptx
│
├── README.md
├── requirements.txt
├── .gitignore
└── .gitattributes

```
## Future Improvements

Future enhancements include:

* Adding metadata such as previous injury and hereditary factors
* Improving KL1 prediction
* Exploring advanced multimodal fusion methods
* Expanding dataset diversity
* Improving explainability and interpretability

---

## Author

**Jinalkumari Vaniya**
Data Science Major
Hofstra University

Advisor: Corey Elowsky

---

## License

This project is for educational and research purposes.
