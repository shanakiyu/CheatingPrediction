Predicting Cheating in Film Plots
Authors: Yacine Touileb & Nina Vivier Barte
Date: November 2024

Overview
As part of a broader study on romantic love in cinema, this project develops a binary classification pipeline to predict whether a film's plot involves a cheating or jealousy situation, based solely on its textual description.
The data comes from Letterboxd, a social film platform, where movies were manually annotated by the authors. The target variable txtCheat takes the value 1 if the plot involves cheating/jealousy, and 0 otherwise.

Repository Structure
├── Touileb_VivierBarte_FeatureEngineering_02.ipynb   # Feature extraction pipeline
├── Touileb_VivierBarte_Analysis_02.ipynb             # Modelling and evaluation
├── Letterboxd_annotations - Letterboxd_annotations.csv  # Raw annotated dataset
├── baby_bagofwords.csv                               # Specific bag of words features
├── bagofwords.csv                                    # General bag of words features
└── word_embeddings.csv                               # spaCy word embedding features

Data
The dataset (Letterboxd_annotations) contains film entries with the following key fields:
ColumnDescriptionnameFilm titledescriptionPlot summary (used as input)txtCheatTarget label — 1 = cheating/jealousy, 0 = nonetxtMarr, txtDivorce, etc.Other romantic situation annotations

⚠️ The dataset is imbalanced: films without cheating are significantly more common than films with cheating, which motivated the use of recall as the primary evaluation metric.


Pipeline
1. Feature Engineering (FeatureEngineering_02.ipynb)
Three feature representations are extracted from film descriptions:
Specific Bag of Words (baby_bagofwords.csv)
Counts occurrences of 18 hand-picked semantically relevant words:
cheat, betray, affair, unfaithful, infidelity, adultery, guilt, suspicion, etc.
General Bag of Words (bagofwords.csv)
Counts all words (excluding stopwords) that appear at least 100 times across the corpus, yielding a high-dimensional vocabulary-based representation.
Word Embeddings (word_embeddings.csv)
Averages the 300-dimensional spaCy (en_core_web_md) word vectors across all tokens in each description, producing a dense semantic representation per film.

2. Modelling (Analysis_02.ipynb)
Each feature set is standardized with StandardScaler and split into train/test sets (80/20). The following models are trained and evaluated:
SectionFeature SetModel8.1.1Specific BoWLogistic Regression8.1.2General BoWLogistic Regression8.1.3Word EmbeddingsLogistic Regression8.2.1Specific BoWNeural Network (MLP)8.2.2General BoWNeural Network (MLP)8.2.3Word EmbeddingsNeural Network (MLP)8.3Word EmbeddingsOptimized MLP (deeper architecture)8.4.1Word EmbeddingsLogistic Regression (class-balanced)8.4.2Word EmbeddingsL1-regularized Logistic Regression (class-balanced)
Best performing model: L1-regularized balanced Logistic Regression on word embeddings
→ Train recall: 0.682 | Test recall: 0.50

Key Findings

Simple bag-of-words models suffer from severe bias due to insufficient features.
Word embeddings significantly improve recall by capturing semantic context.
Class imbalance is a critical challenge: standard models default to predicting "no cheating." Using class_weight='balanced' was essential to get meaningful recall on the minority class.
Even the best model remains limited — a class-aware neural network could be a promising next step given more resources.


Requirements
bashpip install pandas numpy scikit-learn spacy transformers torch matplotlib seaborn
python -m spacy download en_core_web_md
The notebooks were originally developed in Google Colab (GPU: T4). File upload cells use google.colab.files — replace these with standard pd.read_csv() calls if running locally.

Limitations & Future Work

The dataset is manually annotated, which may introduce subjectivity.
Class imbalance limits model performance on the positive class.
Future directions include: BERT sentence embeddings, data augmentation, and class-balanced neural networks.
