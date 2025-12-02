import pandas as pd
import numpy as np

# Load paragraph-level sentiment data
sentiment = pd.read_csv('../data/fed_sentiment_results.csv')

# Load market data
market = pd.read_csv('../data/sp500_features.csv')

# Parse dates for sentiment
def parse_date(s):
    s = s.split('_')[0]
    months = {'Jan': 1, 'Feb': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
              'July': 7, 'Aug': 8, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    for m, num in months.items():
        if s.startswith(m):
            year = int(s.replace(m, ''))
            return pd.Timestamp(year=year, month=num, day=1)
    return None

sentiment['date'] = sentiment['statement_date'].apply(parse_date)
sentiment['year_month'] = sentiment['date'].dt.to_period('M')

# Convert market DATE to datetime and filter to 2020+
market['DATE'] = pd.to_datetime(market['DATE'])
market = market[market['DATE'] >= '2020-01-01'].copy()
market['year_month'] = market['DATE'].dt.to_period('M')

# Merge on year-month
merged = pd.merge(
    sentiment,
    market,
    on='year_month',
    how='inner',
    suffixes=('', '_market')
)

# Sort by date and paragraph
merged = merged.sort_values(['DATE', 'statement_date', 'paragraph_num']).reset_index(drop=True)

# For each unique statement date, get the NEXT month's market regime
unique_stmts = merged[['statement_date', 'DATE', 'year_month']].drop_duplicates().sort_values('DATE').reset_index(drop=True)

# Create mapping
stmt_to_next_regime = {}

for i in range(len(unique_stmts) - 1):
    current_stmt = unique_stmts.iloc[i]
    next_stmt = unique_stmts.iloc[i + 1]
    
    next_month_data = market[market['DATE'] == next_stmt['DATE']]
    
    if len(next_month_data) > 0:
        regime = next_month_data.iloc[0]
        stmt_to_next_regime[current_stmt['statement_date']] = {
            'next_ret_high_negative': regime['ret_high_negative'],
            'next_ret_negative': regime['ret_negative'],
            'next_ret_flat': regime['ret_flat'],
            'next_ret_positive': regime['ret_positive'],
            'next_ret_high_positive': regime['ret_high_positive']
        }

# Add target columns
merged['next_ret_high_negative'] = merged['statement_date'].map(lambda x: stmt_to_next_regime.get(x, {}).get('next_ret_high_negative', np.nan))
merged['next_ret_negative'] = merged['statement_date'].map(lambda x: stmt_to_next_regime.get(x, {}).get('next_ret_negative', np.nan))
merged['next_ret_flat'] = merged['statement_date'].map(lambda x: stmt_to_next_regime.get(x, {}).get('next_ret_flat', np.nan))
merged['next_ret_positive'] = merged['statement_date'].map(lambda x: stmt_to_next_regime.get(x, {}).get('next_ret_positive', np.nan))
merged['next_ret_high_positive'] = merged['statement_date'].map(lambda x: stmt_to_next_regime.get(x, {}).get('next_ret_high_positive', np.nan))

# Create target
conditions = [
    merged['next_ret_high_negative'] == 1,
    merged['next_ret_negative'] == 1,
    merged['next_ret_flat'] == 1,
    merged['next_ret_positive'] == 1,
    merged['next_ret_high_positive'] == 1
]
choices = [0, 1, 2, 3, 4]
merged['target'] = np.select(conditions, choices, default=np.nan)

# Remove rows without target
merged = merged[merged['target'].notna()].copy()
merged['target'] = merged['target'].astype(int)

# Check available columns and build feature list
available_cols = merged.columns.tolist()
print("Available columns:", available_cols[:20], "...")

# Features - only use what exists
feature_cols = [
    'positive_score', 'negative_score', 'neutral_score',
    'paragraph_num',
    'ret_1m', 'vol_3m', 'vol_6m', 'ret_1m_z', 'vol_3m_z', 'vol_6m_z',
    'ret_high_negative', 'ret_negative', 'ret_flat', 'ret_positive', 'ret_high_positive'
]

# Only keep features that exist
feature_cols = [col for col in feature_cols if col in merged.columns]

print("\nParagraph-Level Dataset:")
print(f"Total rows: {len(merged)}")
print(f"Unique statements: {merged['statement_date'].nunique()}")
print(f"Date range: {merged['DATE'].min()} to {merged['DATE'].max()}")
print(f"Features: {len(feature_cols)}")

# Check missing
print("\nMissing values in features:")
missing = merged[feature_cols].isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values!")

# Train/test split
train = merged[merged['DATE'] < '2024-01-01'].copy()
test = merged[merged['DATE'] >= '2024-01-01'].copy()

print(f"\nTrain: {len(train)} paragraphs from {train['statement_date'].nunique()} statements (2020-2023)")
print(f"Test: {len(test)} paragraphs from {test['statement_date'].nunique()} statements (2024+)")

# Class distribution
print("\nTarget distribution:")
print("\nTrain:")
for i in range(5):
    count = (train['target'] == i).sum()
    print(f"  Class {i}: {count} ({100*count/len(train):.1f}%)")

print("\nTest:")
for i in range(5):
    count = (test['target'] == i).sum()
    print(f"  Class {i}: {count} ({100*count/len(test):.1f}%)")

# Save
merged.to_csv('../data/merged_paragraph_level.csv', index=False)
print("\nSaved: ../data/merged_paragraph_level.csv")
print(f"Feature columns: {feature_cols}")

# Show sample
print("\nSample rows:")
print(merged[['statement_date', 'paragraph_num', 'positive_score', 'negative_score', 
              'neutral_score', 'ret_1m', 'target']].head(10))