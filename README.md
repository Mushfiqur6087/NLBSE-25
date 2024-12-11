# NLBSE-25: Enhancing Multi-Label Code Comment Classification

Welcome to the repository for **"Enhancing Multi-Label Code Comment Classification with Data Augmentation and Transformer-Based Architectures"**, submitted as a solution for the NLBSE'25 Code Comment Classification Tool Competition. This repository hosts all relevant scripts, datasets, and model configurations used in the project.

---

## Abstract

Code comment classification is vital for software comprehension and maintenance. This repository demonstrates a multi-step solution that achieves a **6.7% accuracy improvement** over baseline models by combining **synthetic dataset generation** and **fine-tuned transformer-based models**.

Key Points:
- Translation-retranslation for **linguistic diversity** in data augmentation.
- Transformer architectures (**BERT, RoBERTa, CodeBERT, XLNet**) for multi-label classification.
- Tailored frameworks for Java, Python, and Pharo databases.

---

## Repository Structure

```plaintext
.
├── Dataset Generation/     # Scripts for data augmentation (translation-retranslation pipelines).
├── Datasets/               # Original, augmented, and filtered datasets.
├── HyperParameter tuning Python/ # Optuna-based hyperparameter optimization scripts.
├── Model-Saving/           # Fine-tuned transformer models and checkpoints.
├── roBERTa-large-hyperparameter-java-pharo/ # Scripts for RoBERTa tuning on specific languages.
└── final-score.ipynb       # Results, evaluation metrics, and plots.
