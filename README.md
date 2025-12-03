Fall 2025 | DS 5690 – Gen AI Models in Theory & Practice | Vanderbilt University

<img width="780" height="390" alt="image" src="https://github.com/user-attachments/assets/5929f872-8165-453d-8ec5-07d0b512588b" />

# FOMC Statement & S&P 500 Time-Series Analysis
Transformer-based sentiment + macro/technical features to study how FOMC communication relates to S&P 500 returns


## 🔎 Quick Info
| Item                              | Value / Outcome                                          | Notes                                                                 |
|----------------------------------|----------------------------------------------------------|-----------------------------------------------------------------------|
| Sample                           | 2020–2024 FOMC meetings (8 per year)                    | Matched to monthly S&P 500 returns around each statement             |
| Target                           | Next-period S&P 500 excess return (`ret_1m_next`)       | Monthly horizon chosen to smooth noisy day-of announcement moves     |
| Best classical model             | ≈ 0.20 in-sample R²                                     | Macro + technical features only (no sentiment)                       |
| Incremental value of sentiment   | Small, statistically weak incremental lift              | FOMC tone adds limited predictive power on top of return history     |
| Trading backtest                 | Low, unstable risk-adjusted performance                 | Simple long/flat strategies on model signals underperform buy-and-hold |
| Key takeaway                     | Communication matters, but is hard to monetize          | Strong contemporaneous relationships, weak out-of-sample prediction  |

## 📂 Repository Structure
```text
FOMC-statement-S-P-500-time-series-analysis/
│
├── data/                       # Raw and processed data
│   ├── sp500_monthly.csv
│   ├── merged_train_2020_2024.csv
│   ├── merged_test_2025.csv
│
├── notebooks/                  # Ordered workflow for full reproduction
│   ├── 01_data_preparation.ipynb
│   ├── 02_transformer_sentiment.ipynb
│   ├── 03_merge_sentiment_market.ipynb
│   ├── 04_train_deep_models.ipynb
    ├── 05_backtesting.ipynb
│
├── outputs/                    # Predictions, plots, and saved weights
│   ├── binary_2025_predictions.csv
│   ├── monthly_aggregated_predictions.csv
│   ├── *_best.pth
│   ├── *_cm.png
│
```
---

## Table of Contents

1. Problem Statement & Overview  
2. Methodology  
3. Implementation & Demo  
4. Assessment & Evaluation  
5. Model & Data Cards  
6. Critical Analysis
7. Next Steps and Future Work  
8. Documentation & Resource Links  


## 1. Problem Statement & Overview

<img width="1211" height="694" alt="image" src="https://github.com/user-attachments/assets/7455a2b3-20ff-4ab3-b644-34a47bfd0513" />

Modern equity markets react strongly to Federal Reserve communication, especially around FOMC meetings. Traders and journalists routinely describe statements as “hawkish” or “dovish”, but it is less clear whether that **language**, as captured by modern NLP models, contains *predictive* information about future equity returns.

**Project overview.**  
This project builds an end-to-end pipeline that connects **transformer models** for text with classical time-series tools for markets:

- We run each paragraph of every FOMC statement through a **transformer-based sentiment model** (FinBERT/LLM-style), extracting `positive`, `negative`, and `neutral` probabilities as **numerical features**.
- These transformer-derived features are merged with S&P 500 monthly return “regimes” (five buckets from strongly negative to strongly positive) aligned to the *next* FOMC meeting month.
- On top of the transformer features, we train several deep models (MLP, 1D-CNN, attention-augmented MLP, ResNet-style MLP, and an autoencoder classifier) to predict whether the next regime will be **up vs. down/flat**.
- In parallel, we build a more traditional **time-series feature set** (past returns, volatility, momentum, simple macro controls) and run linear models / regressions to benchmark the incremental value of the transformer sentiment.

**Research question.**  
> After controlling for basic return history and technical features, do transformer-based sentiment scores for FOMC statements add meaningful explanatory or predictive power for next-period S&P 500 returns?

**High-level findings.**

- FOMC sentiment from transformers is **correlated with contemporaneous market moves**, and deep models trained on these features achieve reasonable paragraph- and month-level accuracy on a small 2025 test set.
- However, once we include standard return and volatility features, the **incremental lift from sentiment is small and unstable**, and simple long/flat strategies based on the models **do not beat buy-and-hold** over the available sample.
- In this setup, FOMC communication appears more useful for **explaining** how markets react around meetings than for building a robust, out-of-sample trading signal.


## 2. Methodology

Our pipeline integrates transformer-based NLP with deep learning and simple return-based evaluation. The workflow follows four stages:

---

### 2.1 Data Collection & Event Alignment

**FOMC Statements.**  
We collected the text of every FOMC policy statement (2020–2025). Each statement was split into paragraphs for fine-grained sentiment extraction.
https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

**Market Alignment.**  
Daily S&P 500 prices were converted into **monthly returns**.  
For each statement, we assign a binary label for the *following month*:

- **1** → next-month S&P 500 return > 0  
- **0** → next-month return ≤ 0  

This aligns the project with a forward-looking prediction task tied directly to FOMC communication.

---

### 2.2 Transformer-Based Sentiment Extraction

Each paragraph is passed through a **transformer sentiment model** (FinBERT/LLM), producing:

- `positive_score`  
- `negative_score`  
- `neutral_score`  
- `paragraph_num`  

These **transformer-derived features** serve as the *only* inputs to our deep neural networks, forming the core representation of FOMC tone.

---

### 2.3 Deep Learning Models on Transformer Features

We trained five neural architectures on the transformer sentiment vectors:

1. **MLP** – baseline dense network with BatchNorm + Dropout  
2. **CNN1D** – temporal-style convolution over the sentiment feature vector  
3. **AttentionMLP** – applies learned attention over the sentiment embedding  
4. **ResNetMLP** – ResNet-style skip connections for stable training  
5. **AutoencoderClassifier** – compresses tone into a low-dimensional latent code and classifies from it

Training details:
- Train data: **2020–2024** (paragraph-level)  
- Test data: **2025**  
- Loss: class-weighted cross-entropy  
- Optimizer: Adam  
- Early stopping on validation accuracy  

---

### 2.4 Paragraph → Month Aggregation

Each model predicts a binary direction per paragraph (UP or DOWN/FLAT).  
We convert them to signed votes:

- 1 → +1  
- 0 → −1  

For each FOMC meeting month:

$$
\text{monthly score} = \sum_{\text{paragraphs}} \text{signed prediction}
$$

This yields five interpretable model-based sentiment signals:

- `MLP_score`  
- `CNN1D_score`  
- `AttentionMLP_score`  
- `ResNetMLP_score`  
- `AutoencoderClassifier_score`

These monthly scores serve as the **inputs to the backtesting and evaluation** phase.

---

### 2.5 Baseline Statistical Modeling (for comparison)

To contextualize the transformer-based pipeline, we built simple regressions using standard return-history features (1m/3m/6m returns, volatility).  
These served only as **benchmarks** to test whether FOMC sentiment added incremental value.

---

### 2.6 Model Architecture Deep Dive  
To understand how transformer-derived sentiment flows into our predictive system, we examine the two best-performing architectures: **CNN1D** and the **AutoencoderClassifier**. Both models operate on the 4-dimensional transformer sentiment vector as defined in section 2.2

---

## **A. CNN1D — Best Overall Performance (0.80 Accuracy)**  
**Why a 1D convolution?**  
Even though our feature vector is only length 4, a CNN can still learn *local interactions* between sentiment components — e.g., patterns like “high positive + low negative” or “neutral swings across paragraphs.” Treating the sentiment vector as a tiny 1-D sequence allows convolution to act as a **feature interaction extractor**.

**Architecture (from code):**
- Conv1d(1 → 64 channels, kernel size 3, padding=1)  
- BatchNorm + ReLU  
- Conv1d(64 → 32 channels, kernel size 3, padding=1)  
- BatchNorm + ReLU  
- AdaptiveMaxPool1d(1) → compress entire paragraph into a single summary unit  
- Fully Connected: 32 → 32 → 2 classes  
- Dropout = 0.3

**Total parameters:** ~12k  
(lightweight, low risk of overfitting)

**Why it worked well:**  
- CNN filters learn *interactions* between sentiment classes that MLPs cannot easily separate.  
- Adaptive max pooling provides a stable “summary” representation even across variable magnitudes.  
- Strong regularization (BatchNorm, Dropout) stabilizes small-sample learning.

**Interpretation:**  
The CNN likely learned patterns such as:  
- “Strong positive minus moderate negative = UP signal”  
- “High neutral scores paired with low positives = ambiguous/down”  
- “Paragraph early vs late (paragraph_num) interactions”

This architecture extracted more nuanced sentiment patterns than simple linear models.

---

## **B. AutoencoderClassifier — Latent Tone Representation (0.78 Accuracy)**  
**Why an autoencoder?**  
We hypothesized that sentiment tone may lie on a **low-dimensional manifold** — e.g., a single latent “hawkish–dovish” axis.  
The autoencoder maps the 4 sentiment features into an 8-dimensional latent space, encouraging the model to learn a structured ‘tone’ representation rather than relying on raw scores.

**Architecture (from code):**

**Encoder:**
- 4 → 32 → 16 → 8 (latent vector)

**Decoder:**
- 8 → 16 → 32 → 4 (reconstruction)

**Classifier:**
- 8 → 16 → 2

**Why it worked well:**
- The reconstruction objective forces the network to capture core sentiment structure in the latent space, rather than memorizing individual paragraphs. 
- Regularization from the reconstruction objective prevents overfitting.  
- The latent space clusters UP vs DOWN/FLAT months more cleanly.

**Interpretation:**  
The autoencoder appears to learn:
- A smooth “hawkish → dovish” dimension capturing the overall tone  
- Paragraph position effects (early vs late sentiment)  
- The magnitude of positivity/negativity rather than raw probabilities  

This model provided the **most stable** month-level prediction scores.

---

## **C. Why Transformers Matter in These Architectures**  
Both of these deep models operate exclusively on **transformer-extracted sentiment**.  
This means:

\[
\text{Transformer → Sentiment Embedding → Deep Model → Monthly Score}
\]

The transformer model provides:
- A **learned semantic representation** of FOMC tone  
- Probabilistic estimates grounded in financial text pretraining  
- Structure that downstream models rely on to generalize

Without transformer sentiment, MLP/CNN/Autoencoder models would only see raw text counts or bag-of-words features — far less expressive.

---

## **D. Small-Data Strategy (Why These Models Didn’t Overfit)**  
Even with small training size (380 paragraphs), both best-performing models avoided overfitting due to:

- Dropout (0.3–0.4)  
- BatchNorm  
- Early stopping  
- Class weighting  
- Low-parameter architectures  
- Sentiment features already regularized by the transformer model  

The transformer effectively acts as a “feature extractor,” reducing the burden on downstream networks.

---

## **E. Summary: What the Models Learned**
- CNN1D learns **interactions** between sentiment channels (positive/negative/neutral × paragraph index).  
- AutoencoderClassifier learns a **latent tone representation**, smoothing out noise in raw sentiment.  
- Both models agree ~80% of the time with actual monthly S&P direction after aggregation.  
- Transformer features were **essential** — no deep model had any chance of learning from raw text alone in such a small dataset.

## 3. Implementation & Demo

This project is fully reproducible and organized around a clean, ordered notebook pipeline.  
Each notebook corresponds to a major stage of the workflow, moving from raw text → transformer sentiment → deep models → monthly predictions.

---

### 3.1 Notebook Workflow

#### **`01_data_preparation.ipynb`**
- Loads daily SP500 data and aggregates to monthly returns  
- Constructs the forward-looking binary label  
  - **1** → next-month return > 0  
  - **0** → next-month return ≤ 0  
- Performs initial EDA on market behavior around FOMC meetings  

---

#### **`02_transformer_sentiment.ipynb`**
- Loads raw FOMC statements (2020–2025)  
- Splits each statement into paragraphs  
- Sends each paragraph through a **transformer-based sentiment model** (FinBERT / LLM)  
- Produces transformer-derived features:
  - `positive_score`  
  - `negative_score`  
  - `neutral_score`  
  - `paragraph_num`  
- Saves paragraph-level sentiment CSVs  

Transformers provide the **core semantic representation** for all downstream models.

---

#### **`03_merge_sentiment_market.ipynb`**
- Aligns each statement’s date to the **next month’s SP500 return**  
- Merges sentiment scores with market labels  
- Creates:
  - `merged_train_2020_2024.csv` (paragraph-level training data)  
  - `merged_test_2025.csv` (held-out evaluation set)  
- Handles cases where 2025 sentiment lacks market features by keeping the transformer-only representation  

---

#### **`04_train_deep_models.ipynb`**
- Loads merged training and test datasets  
- Trains five architectures on transformer sentiment features:
  - **MLP**  
  - **CNN1D**  
  - **AttentionMLP**  
  - **ResNetMLP**  
  - **AutoencoderClassifier**  
- Uses:
  - class-weighted loss  
  - validation-based early stopping  
  - Adam optimizer + LR scheduling  
- Saves:
  - best model weights (`*_best.pth`)  
  - paragraph-level predictions (`binary_2025_predictions.csv`)  
  - confusion matrices for each model
  - 
#### **`05_backtesting.ipynb`**
- Loads `monthly_aggregated_predictions.csv` and 2025 SP500 returns  
- For each model’s monthly score, builds a simple **long/flat strategy**:
  - Go long SP500 when model score > 0  
  - Stay in cash when model score ≤ 0  
- Computes:
  - directional hit rate (how often sign matches reality)  
  - cumulative strategy return vs buy-and-hold  
  - basic risk/return metrics and plots (e.g., equity curves)  
- Summarizes which transformer-based model produced the most stable and interpretable signal
---

### 3.2 Full Pipeline Diagram
```
Raw FOMC Text
↓ (paragraph split)
Transformer Sentiment Model
↓ (positive / negative / neutral / paragraph index)
Transformer Feature Matrix (4D per paragraph)
↓
Deep Learning Models (MLP / CNN / Attention / ResNet / Autoencoder)
↓
Paragraph Predictions (0/1)
↓ (convert 0→–1, 1→+1)
Monthly Aggregation
↓
Final Monthly Signal Scores (one per model)
↓
Backtesting & Evaluation
```

### 3.3 Core Algorithm (Pseudocode)

```text
for each FOMC statement:
    # Split text into paragraphs
    paragraphs = split_into_paragraphs(statement)

    # Transformer sentiment extraction
    features = []
    for p in paragraphs:
        sentiment = transformer_model(p)   # FinBERT / LLM
        features.append([
            sentiment.positive,
            sentiment.negative,
            sentiment.neutral,
            paragraph_index
        ])

    # Deep model prediction on transformer features
    signed_preds = []
    for f in features:
        pred = deep_model(f)               # outputs 0 or 1
        signed = +1 if pred == 1 else -1
        signed_preds.append(signed)

    # Aggregate paragraph predictions into monthly score
    monthly_score = sum(signed_preds)

return monthly_score for each FOMC meeting month
```

## Usage Example

Below is a minimal example showing how to reproduce the model predictions for a new FOMC statement using our released artifacts.  
This mirrors the paragraph → transformer sentiment → deep model → monthly score workflow used in the project.

```python
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from Model_Training_DL import MLP, CNN1D, AttentionMLP, ResNetMLP, AutoencoderClassifier

# ---------------------------------------------------------
# 1. Load transformer sentiment model (FinBERT-style)
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_sentiment_scores(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = sentiment_model(**inputs)
    probs = outputs.logits.softmax(dim=1).detach().numpy()[0]
    return {
        "positive_score": float(probs[0]),
        "negative_score": float(probs[1]),
        "neutral_score":  float(probs[2])
    }

# ---------------------------------------------------------
# 2. Split a new FOMC statement into paragraphs
# ---------------------------------------------------------
statement = """
The Federal Reserve is committed to achieving maximum employment and 2 percent inflation...
(etc)
"""
paragraphs = [p for p in statement.split("\n") if p.strip()]

# Extract transformer sentiment + paragraph index
rows = []
for i, para in enumerate(paragraphs):
    scores = get_sentiment_scores(para)
    scores["paragraph_num"] = i
    rows.append(scores)

df = pd.DataFrame(rows)

# ---------------------------------------------------------
# 3. Load a trained deep model (CNN1D in this example)
# ---------------------------------------------------------
input_dim = 4     # [positive, negative, neutral, paragraph_num]
model = CNN1D(input_dim=input_dim, n_classes=2)
model.load_state_dict(torch.load("outputs/binary_CNN1D_best.pth"))
model.eval()

# Predict paragraph-level direction (0 = down/flat, 1 = up)
with torch.no_grad():
    X = torch.tensor(df.values).float()
    logits = model(X)
    preds = logits.argmax(dim=1).numpy()

# ---------------------------------------------------------
# 4. Convert to signed votes and aggregate
# ---------------------------------------------------------
signed = [+1 if p == 1 else -1 for p in preds]
monthly_score = sum(signed)

print("Paragraph predictions:", preds)
print("Monthly score:", monthly_score)
```

## 4. Assessment & Evaluation

We evaluate the system along two axes:

1. **Predictive accuracy** (paragraph-level classification, 2025 holdout)  
2. **Portfolio performance** (simple long/flat strategies based on monthly model scores)

---

### 4.1 Paragraph-Level Model Performance (2025 Test Set)

All models were trained on 380 paragraphs (2020–2024) and evaluated on 50 paragraphs from 2025.  
Each model ingests **only transformer-derived sentiment features**, testing whether they capture forward-looking tone.

| Model                   | Val Acc | 2025 Test Acc | Notes |
|-------------------------|---------|----------------|-------|
| **CNN1D**               | 0.658   | **0.800**      | Best directional classifier |
| **AutoencoderClassifier** | 0.658 | **0.780**      | Strong latent representation |
| **ResNetMLP**           | 0.645   | 0.720          | Stable, lightweight |
| **MLP**                 | 0.645   | 0.700          | Solid but less expressive |
| **AttentionMLP**        | 0.553   | 0.560          | Highly variable |
| **Baseline (majority class)** | —     | 0.780          | Highlights class imbalance |

**Key takeaway:**  
Transformers give the models enough structure to achieve **70–80% accuracy** on paragraph-level sentiment classification, but performance skews heavily toward the *majority class (1 = up)*.

---

### 4.2 Monthly Aggregation Results

Aggregating paragraph predictions into monthly scores produces strong, consistent signals for the best models:

- **CNN1D** and **AutoencoderClassifier** generate the most stable, positive month-level signals  
- **AttentionMLP** exhibits inconsistent behavior (negative scores for positive months)  
- All models correctly identify that 2025 was broadly “UP,” except the January decline  

This supports the idea that **transformer sentiment is contemporaneously informative**, even when predictive power is limited.

---

### 4.3 Backtesting: Model Strategies vs SP500 Buy-and-Hold
<img width="1389" height="690" alt="image" src="https://github.com/user-attachments/assets/4be425cc-5510-47aa-894b-9e8721d088f5" />
### Model Backtest Portfolio Values (2025)

| Period   | MLP_score | CNN1D_score | AttentionMLP_score | ResNetMLP_score | AutoencoderClassifier_score |
|----------|-----------|-------------|---------------------|------------------|------------------------------|
| 2025-01  | 99.64     | 98.93       | 100.00              | 99.64            | 98.58                       |
| 2025-02  | 93.91     | 93.24       | 98.56               | 93.91            | 92.90                       |
| 2025-05  | 96.24     | 97.86       | 98.56               | 96.24            | 97.51                       |
| 2025-06  | 97.28     | 99.98       | 99.10               | 97.28            | 99.62                       |
| 2025-07  | 97.75     | 101.89      | 99.57               | 98.21            | 101.52                      |

We evaluate a simple long/flat strategy:

- **Long SP500** when monthly score > 0  
- **Flat (0 exposure)** when ≤ 0  

The chart below compares cumulative portfolio value (Jan–Jul 2025):

**Observations:**

- **CNN1D_score** and **AutoencoderClassifier_score** slightly outperform buy-and-hold for the 2025 sample  
- **AttentionMLP_score** is nearly flat and underperforms the benchmark  
- All models show **very similar curve shapes** → they are largely responding to the same underlying 2025 market trend  
- No model generates a meaningfully different trajectory from SP500  

**Conclusion:**  
While transformers helped produce coherent sentiment signals, **these signals did not translate into superior risk-adjusted performance**.  
The models mostly echo the market’s prevailing direction rather than providing predictive edge.

---
## 5. Model & Data Cards

This section documents the datasets and deep learning models used in this project following a simplified “model card / data card” standard. These cards clarify scope, intended use, limitations, and evaluation context.

---

# **5.1 Data Cards**

### **Dataset: FOMC Transformer-Sentiment Dataset (2020–2025)**  
**Description:**  
Paragraph-level dataset containing transformer-derived sentiment scores for every FOMC statement from January 2020 through July 2025.

**Source Documents:**  
- Federal Open Market Committee policy statements  
- Official release dates from the Federal Reserve  
- Preprocessed into paragraph units

**Features (Transformer-Derived):**
- `positive_score`  
- `negative_score`  
- `neutral_score`  
- `paragraph_num` (position within statement)

**Labels:**
- `target_binary`:  
  - 1 → next-month S&P 500 return > 0  
  - 0 → next-month return ≤ 0  
- `target_5class`: unused in final modeling but included for completeness (−2 to +2 regimes)

**Train/Test Split:**
- **Train:** 2020–2024 (380 paragraphs, ~40 independent FOMC events)  
- **Test:** 2025 (50 paragraphs, 5 independent events)

**Intended Use:**  
- Research on whether FOMC tone contains predictive information  
- Benchmarking transformer sentiment features

**Limitations:**  
- Only ~8 events per year → small dataset  
- Paragraphs share the same label → samples are *not* independent  
- Market returns are noisy targets  
- Cannot support large neural networks or sequence models reliably

---
## 6. Critical Analysis

Despite building a complete end-to-end pipeline—from transformer sentiment extraction to deep model training to portfolio backtesting—the results must be interpreted with caution. The core limitation is **not the modeling**, but the **scale and structure of the available data**.

---

### 6.1 Scarcity of FOMC Events (Only ~8 per Year)

FOMC statements are inherently sparse:

- ~8 statements per year  
- ~40 statements across the 2020–2024 training window  
- 5 statements in the 2025 evaluation window  

Even though we split each statement into paragraphs, these paragraphs all share the same **monthly label**, which causes:

- **Inflated sample size** — 380 paragraphs is not 380 independent observations  
- **Overfitting risk** — models may memorize paragraph patterns instead of true economic signals  
- **Tiny test set** — only 5 independent months of out-of-sample evaluation

This is a structural limitation of the FOMC event cycle.

---

### 6.2 Transformer Sentiment Is Stable, but the Market Label Is Noisy

The transformer provides clean, consistent sentiment embeddings.  
The *target* (next-month S&P direction) is extremely noisy:

- Monthly returns often move for reasons unrelated to policy tone  
- Macro shocks, earnings cycles, and geopolitical events dominate single-month behavior  
- The market during 2020–2024 was predominantly upward-trending

This creates a **signal-to-noise mismatch**:

- Inputs → semantic, structured transformer features  
- Outputs → highly noisy market labels  

A simple baseline (“always predict UP”) achieved **0.78 accuracy**, matching or beating several deep models.  
This reflects the noisiness of the label, not a failure of the transformer.

---

### 6.3 Backtesting Fragility

The backtest covers **five months** of 2025. This is too small to measure:

- Drawdowns  
- Regime shifts  
- Sharpe / Sortino  
- Stability across market conditions  

On such a tiny horizon, any apparent outperformance (or underperformance) is unreliable and likely accidental.

---

### 6.4 Transformer Features Are Useful — But Insufficient Alone

CNN1D and the AutoencoderClassifier produced consistent and interpretable month-level signals:

- ~80% directional agreement after aggregation  
- Meaningful paragraph-level voting behavior  
- Smooth month-to-month score changes  

However, **S&P 500 one-month-ahead returns are not reliably predictable from FOMC text alone**.  
Transformers extract meaningful tone — the market label simply does not consistently react one month later in a learnable way.

---

## 7. Next Steps and Future Work

This project establishes a strong foundation for NLP × macro-finance research. The next phase should increase sample size and refine the prediction target.

---

### 7.1 Expand Beyond FOMC Statements

To grow from ~40 documents to hundreds or thousands:

- FOMC minutes  
- Press conferences  
- Fed speeches and testimonies  
- Beige Book  
- Fed staff economic projections  
- Policy debate transcripts and interviews  

These sources occur far more frequently and influence expectations.

---

### 7.2 Use Better Predictive Targets

Next-month S&P 500 returns are too noisy. Better supervised targets include:

- Intraday reaction around statement release  
- 1–3 day post-FOMC drift  
- Volatility regimes instead of returns  
- Treasury yield reactions  
- Fed Funds futures “surprise” (text vs. market-implied expectations)

These align much more closely with academic literature on monetary policy surprises.

---

### 7.3 Improve Text Representations

Instead of only paragraph-level sentiment:

- Use full-statement embeddings (e.g., Instructor-XL, FinE5)  
- Use pooled token embeddings instead of fixed sentiment scores  
- Extract topics (inflation, employment, liquidity, risk)  
- Model paragraph sequences with LSTMs / transformers  

This would allow the model to capture structure beyond simple positive/negative tone.

---

### 7.4 Combine Sentiment With Macro Covariates

A realistic forecasting setup would incorporate:

- CPI, unemployment, industrial production  
- Yield curve shape  
- Fed Funds futures  
- Corporate credit spreads  
- Equity volatility indices (VIX, VVIX)

Transformer sentiment would then act as one feature among many.

---

### 7.5 Extend Backtesting Across Historical Regimes

If older FOMC communications are added (1994–present), the model could be evaluated across:

- Dot-com bubble  
- 2008 financial crisis  
- 2013 taper tantrum  
- COVID liquidity crisis  
- 2022–2023 hiking cycle  

Only then can we test robustness and real-world applicability.

---

### Summary

The transformer pipeline works and produces meaningful tone embeddings.  
The limiting factor is the **tiny dataset** and the **noisy, hard-to-predict target**.

Future work should focus on:

1. Increasing sample size  
2. Improving the market target  
3. Using richer text embeddings  
4. Combining sentiment with macroeconomic covariates  

This will determine whether FOMC communication contains **predictive** information or is primarily useful for **explaining** market reactions.

## 8. Documentation & Resource Links

This section provides direct links to external tools, models, and datasets used throughout the project, along with references to the course materials that informed our methodology. All links are lightweight and intended for reproducibility.

---

### 🔗 **Data Sources**

**Federal Reserve FOMC Statements (2020–2025)**  
Official archive of policy statements  
https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm  

**S&P 500 Price Data (via Yahoo Finance)**  
Used for daily → monthly return aggregation  
https://finance.yahoo.com/quote/%5EGSPC/history  

---

### 🤖 **Transformer & NLP Resources**

**FinBERT (Financial Sentiment Transformer)**  
The model used to score positive/negative/neutral sentiment  
https://huggingface.co/yiyanghkust/finbert-tone  

**HuggingFace Transformers Library**  
Core API used for model loading and inference  
https://github.com/huggingface/transformers  

**Course Material – “Transformers: A Quick Introduction”**  
Foundational reading on attention mechanisms and transformer architecture  
(Provided in DS 5690 course resources)

---

### 📊 **Modeling & Time-Series References**

**PyTorch**  
Deep learning framework used for all architectures  
https://pytorch.org/  

**scikit-learn**  
Used for scaling, class weighting, and evaluation metrics  
https://scikit-learn.org/stable/  

**pandas**  
Data manipulation and merging of sentiment + market features  
https://pandas.pydata.org/  

---

### 📁 **Project Repository**

**Full Source Code (GitHub)**  
https://github.com/jbanov/FOMC-statement-S-P-500-time-series-analysis  

This includes:
- All notebooks (`01–05`)  
- Deep model training script (`Model_Training_DL.py`)  
- Generated datasets  
- Confusion matrices & saved weights  
- Monthly backtesting results  

---

### 🧪 **Reproduction Guide**

To fully reproduce the experiment:

1. Clone the repository  
2. Install packages → `pip install -r requirements.txt`  
3. Run the notebooks in numeric order (`01` → `05`)  
4. Refer to `Model_Training_DL.py` for the full deep learning training loop  
5. Output files will be written to the `outputs/` folder  

---

### 📚 **Academic Context & Readings**

**DS 5690 – Gen AI Models in Theory & Practice**  
Vanderbilt University  
- Core topics used in this project:  
  - Transformer architectures  
  - Prompt-based NLP  
  - Embedding-based classification  
  - Attention mechanisms  
  - Small-data deep learning strategies  

**Relevant Papers & References**  
- *Vaswani et al. (2017)* — “Attention is All You Need”  
- *Araci (2019)* — “FinBERT: Financial Sentiment Analysis”  

---

### 📝 **License & Usage Notes**

This repository is intended **exclusively for academic use** in DS 5690 at Vanderbilt University.  
It is *not* intended for financial forecasting or real-world trading.



