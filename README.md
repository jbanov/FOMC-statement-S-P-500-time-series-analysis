# FOMC Statement & S&P 500 Time-Series Analysis  
Analysis of Federal Reserve communication and its influence on equity markets.

## 📌 Project Overview
This project examines whether the sentiment of Federal Reserve FOMC statements helps explain or predict movements in the S&P 500 index. Using a combination of transformer-based sentiment analysis, time-series feature engineering, and classical statistical tools, this repository builds a structured pipeline to analyze the relationship between central bank language and market behavior.

---

## 📂 Repository Structure

```
FOMC-statement-S-P-500-time-series-analysis/
│
├── data/                     # Raw and processed data
│   ├── sp500_features.csv
│   ├── fomc_statements.csv
│
├── notebooks/                # Jupyter notebooks
│   ├── 01_sp500_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_sentiment_merge.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Methods & Techniques

### **Data Acquisition**
- S&P 500 data pulled using `yfinance`
- FOMC statements provided as CSV (date + text)
- Monthly and quarterly resampling for macro alignment

### **Feature Engineering**
Includes key time-series signals often used in asset management:
- 1-month / 3-month / 6-month returns  
- Rolling volatility  
- Rolling momentum indicators  
- Lagged returns for lead–lag modeling  
- Event-aligned windows around FOMC announcements  

### **Sentiment Modeling**
- Transformer-based embeddings or sentiment scoring of FOMC text  
- Experimenting with:
  - FinBERT
  - LLM-based API sentiment extraction
  - Custom prompt-based scoring

### **Statistical Analysis**
- Correlation analysis  
- Lead–lag tests  
- Linear and multivariate regressions  
- Visualization of sentiment vs. market performance  

---

## 🎯 Goals
1. Determine whether FOMC statement sentiment has predictive power for next-period S&P 500 returns.  
2. Identify which engineered features best capture market reactions to policy communication.  
3. Build an interpretable analysis pipeline combining NLP and time-series methods.  
4. Produce plots, regressions, and findings suitable for final class presentation.

---

## 🚀 Getting Started

### **Clone the repo**
```bash
git clone https://github.com/jbanov/FOMC-statement-S-P-500-time-series-analysis.git
cd FOMC-statement-S-P-500-time-series-analysis
```

### **Install dependencies**
```bash
pip install -r requirements.txt
```

### **Launch Jupyter**
```bash
jupyter lab
```

---

## 🧩 Next Steps / Roadmap
- [ ] Finish SP500 data collection  
- [ ] Build initial EDA (Notebook 01)  
- [ ] Add time-series features (Notebook 02)  
- [ ] Create sentiment scoring pipeline  
- [ ] Merge sentiment + SP500 features  
- [ ] Run regressions and produce visualizations  
- [ ] Finalize project results for presentation  

---


## 📄 License
This project is for academic use within Vanderbilt University coursework.
