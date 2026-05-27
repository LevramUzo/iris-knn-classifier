# 🌸 Iris Flower Classification — KNN Supervised Learning

A complete supervised machine learning pipeline built as **Project 2 of the DecodeLabs AI Engineering Internship (Batch 2026)**. This project marks the transition from rule-based deterministic AI to statistical pattern recognition — teaching a machine to classify flower species from measurements it has never seen before.

---

## 💡 The Core Concept

In Project 1, I wrote explicit rules — if the user says "hello", respond with "Hi there."

In Project 2, I provide labelled examples and let the machine derive the logic itself.

> *"We do not write the rules. We provide history, and the machine derives the logic."* — DecodeLabs

This is the fundamental shift from **heuristic AI** to **supervised learning** — and it's the same principle behind every production ML system in use today.

---

## 🏗️ Pipeline Architecture — IPO Model

```
INPUT                →    PROCESS              →    OUTPUT
Iris Dataset              Train/Test Split           Confusion Matrix
Feature Scaling           KNN Algorithm (K=5)        F1 Score
                          Model Training             Classification Report
```

---

## 📊 The Dataset — Iris Benchmark

| Property | Value |
|----------|-------|
| Samples | 150 (balanced) |
| Classes | 3 (Setosa, Versicolor, Virginica) |
| Features | 4 (Sepal Length, Sepal Width, Petal Length, Petal Width) |
| Source | sklearn.datasets |

---

## 🧠 The Algorithm — K-Nearest Neighbors

KNN operates on the **Proximity Principle** — similar things exist in close proximity.

For each new flower:
1. Calculate distance to all training flowers
2. Find the 5 nearest neighbours (K=5)
3. Majority vote determines the predicted species

**Why K=5?** The Elbow Method shows K=1 overfits (noise) and K=100 underfits (generic). K=5 sits at the optimal point — the elbow of the error curve.

**Why StandardScaler first?** Without scaling, features with larger values (e.g. sepal length ~5cm) unfairly dominate distance calculations over smaller features (e.g. petal width ~0.2cm). StandardScaler normalises all features to mean=0, variance=1.

---

## ✨ Features

- ✅ Automatic dataset loading via scikit-learn
- ✅ Feature scaling with StandardScaler
- ✅ Stratified 80/20 train-test split with shuffle
- ✅ KNN model training and prediction
- ✅ Full classification report (precision, recall, F1 per class)
- ✅ Weighted F1 Score
- ✅ Confusion Matrix visualisation saved as PNG

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Weighted F1 Score | **1.0000** |
| Test Accuracy | **100%** (30/30 correct) |
| False Positives | 0 |
| False Negatives | 0 |

The model achieved perfect classification on the test set — demonstrating that KNN with proper feature scaling is highly effective on the Iris benchmark.

---

## 🚀 How to Run

**Requirements:** Python 3.6+

```bash
git clone https://github.com/LevramUzo/iris-knn-classifier.git
cd iris-knn-classifier
pip install scikit-learn matplotlib seaborn
python classifier.py
```

---

## 🖥️ Sample Output

```
Dataset shape: (150, 4)
Classes: ['setosa' 'versicolor' 'virginica']
Training samples: 120
Testing samples: 30
Model trained successfully!

=== Classification Report ===
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00        9
   virginica       1.00      1.00      1.00        11
    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

Weighted F1 Score: 1.0000
```

---

## 🛠️ Tech Used

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| scikit-learn | Dataset, model, metrics |
| StandardScaler | Feature normalisation |
| KNeighborsClassifier | Classification algorithm |
| matplotlib + seaborn | Confusion matrix visualisation |

---

## 🧪 Key Concepts Demonstrated

**Why F1 Score over Accuracy?**
Accuracy is a mirage on imbalanced datasets. A model predicting "healthy" for everyone achieves 99% accuracy on a disease affecting 1% of people — but catches zero sick patients. F1 Score balances Precision and Recall, giving a truer picture of model performance.

**TP, FP, FN, TN explained:**
- **TP (True Positive)** — Predicted Setosa, actually Setosa ✅
- **FP (False Positive)** — Predicted Setosa, actually another species ❌ False Alarm
- **FN (False Negative)** — Actual Setosa predicted as another species ❌ Missed Detection
- **TN (True Negative)** — Predicted NOT Setosa, actually NOT Setosa ✅

---

## 🔗 Project Progression

```
Project 1: Rule-Based Chatbot    → Deterministic logic, guardrail architecture
Project 2: KNN Classifier        → Supervised learning, pattern recognition  ← YOU ARE HERE
Project 3: Deep Learning & CNNs  → Neural networks, computer vision (coming)
```

---

## 🔗 Related Projects

- **Rule-Based AI Chatbot** — Deterministic guardrail layer architecture
- **AI Training Data Logger** — RLHF annotation pipeline simulation
- **Computer Vision Object Detection** — RF-DETR Nano, 88.5% precision

---

## 👨‍💻 Author

**Marvellous Opara**
BSc Industrial Physics | AI Engineering Intern @ DecodeLabs
[GitHub](https://github.com/LevramUzo) • oparamarvellous456@gmail.com
