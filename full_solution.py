import pandas as pd
import re
import os

print("1. Fast Loading Test Datasets...")
test_s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", usecols=['entity_id', 'business_name'])
test_s2 = pd.read_csv("dataset/test/test_source2.tsv", sep="\t", usecols=['entity_id', 'business_name'])
test_s3 = pd.read_csv("dataset/test/test_source3.tsv", sep="\t", usecols=['entity_id', 'business_name'])

os.makedirs("output", exist_ok=True)

print("2. Fast Vectorized Cleaning...")
def clean_series(series):
    return series.fillna('').astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

test_s1['clean'] = clean_series(test_s1['business_name'])
test_s2['clean'] = clean_series(test_s2['business_name'])
test_s3['clean'] = clean_series(test_s3['business_name'])

print("3. Combining Target Sources (Source 2 + Source 3)...")
s23 = pd.concat([test_s2, test_s3], ignore_index=True)
s23['prefix'] = s23['clean'].str[:3]

# Create blocking map using vectorized groupby
print("4. Grouping & Indexing Prefix Blocks...")
blocking_dict = s23[s23['prefix'] != ''].groupby('prefix')['entity_id'].apply(list).to_dict()

# Exact match dictionary for high-precision matching
exact_match_dict = s23[s23['clean'] != ''].groupby('clean')['entity_id'].apply(list).to_dict()

print("5. Processing 1.7M Matches Fast...")
s1_clean = test_s1['clean'].values
s1_ids = test_s1['entity_id'].values

matching_results = []
candidate_results = []

for s1_id, name in zip(s1_ids, s1_clean):
    prefix = name[:3] if len(name) >= 3 else ''
    
    # Candidates from prefix block
    cands = blocking_dict.get(prefix, [])[:10]
    
    # Matches from exact name match (subset of candidates to satisfy validation)
    exacts = exact_match_dict.get(name, [])
    matches = [m for m in exacts if m in cands][:5]
    
    candidate_results.append(','.join(cands))
    matching_results.append(','.join(matches))

print("6. Saving Validation-Compliant Files...")
df_match = pd.DataFrame({'source1_entity_id': s1_ids, 'matched_entity_ids': matching_results})
df_cand = pd.DataFrame({'source1_entity_id': s1_ids, 'candidate_entity_ids': candidate_results})

df_match.to_csv("output/matching_results.tsv", sep="\t", index=False)
df_cand.to_csv("output/candidate_pairs.tsv", sep="\t", index=False)

print("SUCCESS! All 1.7M records processed in seconds!")