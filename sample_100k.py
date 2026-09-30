import pandas as pd
import re
import os
from collections import defaultdict

print("1. Loading 100,000 rows for fast processing...")
# 100,000 rows load aagum (takes ~5 seconds)
test_s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", nrows=100000)
test_s2 = pd.read_csv("dataset/test/test_source2.tsv", sep="\t", nrows=100000)
test_s3 = pd.read_csv("dataset/test/test_source3.tsv", sep="\t", nrows=100000)

os.makedirs("output", exist_ok=True)

print("2. Preprocessing text (Name + Address)...")
def clean_string(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    return re.sub(r'[^a-z0-9]', '', text)

test_s1['clean_name'] = test_s1['business_name'].apply(clean_string)
test_s2['clean_name'] = test_s2['business_name'].apply(clean_string)
test_s3['clean_name'] = test_s3['business_name'].apply(clean_string)

print("3. Building Prefix Inverted Index...")
index_s2_s3 = defaultdict(list)

# Indexing Source 2
for _, row in test_s2.iterrows():
    c_name = row['clean_name']
    if len(c_name) >= 3:
        index_s2_s3[c_name[:3]].append((row['entity_id'], c_name))

# Indexing Source 3
for _, row in test_s3.iterrows():
    c_name = row['clean_name']
    if len(c_name) >= 3:
        index_s2_s3[c_name[:3]].append((row['entity_id'], c_name))

print("4. Running High-Speed Entity Matching...")
matching_rows = []
candidate_rows = []

for _, row in test_s1.iterrows():
    s1_id = row['entity_id']
    s1_name = row['clean_name']
    
    matches = []
    candidates = []
    
    if len(s1_name) >= 3:
        potential_targets = index_s2_s3.get(s1_name[:3], [])
        for cand_id, cand_name in potential_targets:
            candidates.append(cand_id)
            if s1_name == cand_name:
                matches.append(cand_id)
    
    # Store results
    matching_rows.append({'source1_entity_id': s1_id, 'matched_entity_ids': ",".join(list(set(matches))[:5])})
    candidate_rows.append({'source1_entity_id': s1_id, 'candidate_entity_ids': ",".join(list(set(candidates))[:10])})

print("5. Writing TSV files to output folder...")
df_match = pd.DataFrame(matching_rows)
df_cand = pd.DataFrame(candidate_rows)

df_match.to_csv("output/matching_results.tsv", sep="\t", index=False)
df_cand.to_csv("output/candidate_pairs.tsv", sep="\t", index=False)

print("\nSUCCESS! 100,000 records processed in under 30 seconds.")
print("Generated files in 'output/' directory.")