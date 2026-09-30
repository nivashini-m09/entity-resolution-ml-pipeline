import pandas as pd
import re
import os

print("1. Loading Test Data for Precision Matching...")
test_s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t")
test_s2 = pd.read_csv("dataset/test/test_source2.tsv", sep="\t")
test_s3 = pd.read_csv("dataset/test/test_source3.tsv", sep="\t")

os.makedirs("output", exist_ok=True)

print("2. Advanced Preprocessing (Removing Corporate Suffixes)...")
def clean_text(series):
    # Remove corporate noise words like Inc, LLC, Ltd to improve matching accuracy
    cleaned = series.fillna('').astype(str).str.lower()
    cleaned = cleaned.str.replace(r'\b(inc|llc|corp|co|ltd|limited|corporation)\b', '', regex=True)
    return cleaned.str.replace(r'[^a-z0-9]', '', regex=True)

test_s1['clean_name'] = clean_text(test_s1['business_name'])
test_s2['clean_name'] = clean_text(test_s2['business_name'])
test_s3['clean_name'] = clean_text(test_s3['business_name'])

print("3. Indexing & Grouping Targets...")
s23 = pd.concat([test_s2, test_s3], ignore_index=True)
s23['prefix'] = s23['clean_name'].str[:3]

blocking_dict = s23[s23['prefix'] != ''].groupby('prefix')['entity_id'].apply(list).to_dict()
exact_match_dict = s23[s23['clean_name'] != ''].groupby('clean_name')['entity_id'].apply(list).to_dict()

print("4. Processing High-Precision Matches...")
s1_clean = test_s1['clean_name'].values
s1_ids = test_s1['entity_id'].values

matching_results = []
candidate_results = []

for s1_id, name in zip(s1_ids, s1_clean):
    prefix = name[:3] if len(name) >= 3 else ''
    
    # Candidate selection
    cands = blocking_dict.get(prefix, [])[:10]
    
    # Exact cleaned name matches strictly within candidates
    exacts = exact_match_dict.get(name, [])
    matches = [m for m in exacts if m in cands][:5]
    
    candidate_results.append(','.join(cands))
    matching_results.append(','.join(matches))

print("5. Writing Final Results...")
df_match = pd.DataFrame({'source1_entity_id': s1_ids, 'matched_entity_ids': matching_results})
df_cand = pd.DataFrame({'source1_entity_id': s1_ids, 'candidate_entity_ids': candidate_results})

df_match.to_csv("output/matching_results.tsv", sep="\t", index=False)
df_cand.to_csv("output/candidate_pairs.tsv", sep="\t", index=False)

print("\nSUCCESS! High-precision solution execution complete!")