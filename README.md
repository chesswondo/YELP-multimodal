# Multimodal Yelp Classification: From Logistic Regression to Late Fusion

This project implements a progressive approach to multi-class classification using the **Yelp Image Dataset**. The goal is to classify images into 5 categories (`food`, `drink`, `inside`, `outside`, `menu`) using varying levels of model complexity: ranging from a simple text-only baseline to a multimodal ensemble combining Computer Vision (CNN) and NLP (LoRA Fine-tuned Transformers).


## 📊 Project Results

We conducted an ablation study to compare different architectures. The final ensemble achieved the best performance by correcting errors where visual context was ambiguous but textual context was strong.

| Stage | Approach | Accuracy | Description |
| :--- | :--- | :--- | :--- |
| **I** | **Text Only (Baseline)** | **74.9%** | TF-IDF + Logistic Regression on captions. |
| II | Text Only (Deep) | 77.1% | Fine-tuned DistilBERT. |
| **III** | **Vision Only** | **95.0%** | EfficientNet-B0 (Fine-tuned). |
| **IV** | **Multimodal Fusion** | **95.6%** | EfficientNet + LoRA-BERT + Meta-features + XGBoost. |

Key Insight: While the Vision model is dominant (95%), the fusion approach reduced the remaining error rate by ~12%, correctly classifying difficult cases like "Menu" (visually ambiguous) or distinguishing "Inside" from "Food".


## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **CV:** PyTorch, Timm (EfficientNet), Albumentations
* **NLP:** Hugging Face Transformers (DistilBERT), PEFT (LoRA for efficient fine-tuning)
* **Tabular/Fusion:** Scikit-learn, XGBoost
* **Hardware:** Single GPU RTX 3060 (12GB VRAM)


## 📂 Dataset Structure
<pre>
project_root/  
├── data/  
│   ├── train.json          # Original JSONL metadata  
│   ├── test.json           # Original JSONL metadata  
│   ├── train/              # Images (e.g., train/photo1.jpg)  
│   ├── test/               # Images (e.g., test/photo2.jpg)  
│   └── ...                 # Generated CSVs and models will appear here  
├── notebooks/
│   ├── stage1.ipynb           # Text-only LogReg + fine-tuned DistilBERT
│   ├── stage2.ipynb           # Image-only EfficientNet-B0
│   └── stage3.ipynb           # Different multimodal fusions
├── README.md
└── requirements.txt  
</pre>

## 🧠 Approach Details

**1. Stage 1 (Text)**: Proved that captions contain significant signal but suffer from ambiguity (e.g., "Delicious!" applies to both Food and Drink).

**2. Stage 2 (Vision)**: Solved the majority of cases. Used heavy augmentations (Albumentations) and timm library.

**3. Stage 3 (Fusion)**: Adopted a Late Fusion strategy.

- **Visual Stream**: Probabilities from EfficientNet.
- **Text Stream**: Logits from DistilBERT (LoRA).
- **Meta Stream**: Handcrafted features (caption length, word count).
- **Aggregator**: XGBoost trained on the concatenated feature vector.


## 📈 Error Analysis
The ensemble the most successfully corrects CNN errors in ambience scenarios such as:

- ***Food/Inside***: CNN sees a plate on a table -> predicts *Food*; Text says "Cozy atmosphere" -> Ensemble corrects to *Inside*.
- ***Inside/Outside***: CNN sees chairs and tables -> predicts *Inside*; Text says "Bar in the outdoor area" -> Ensamble corrects to *Outside*.