import pandas as pd
import re
import os

def clean_name(name):
    if pd.isna(name):
        return ""
    name = str(name).lower()
    # Corporate suffixes cleanup
    name = re.sub(r'\b(inc|llc|corp|corporation|ltd|limited|co|company|group)\b', '', name)
    # Remove non-alphanumeric characters
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

def run_pipeline():
    print("Loading test datasets...")
    s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t")
    s2 = pd.read_csv("dataset/test/test_source2.tsv", sep="\t")
    s3 = pd.read_csv("dataset/test/test_source3.tsv", sep="\t")

    print("Cleaning entity names...")
    s1['clean_name'] = s1['business_name'].apply(clean_name)
    s2['clean_name'] = s2['business_name'].apply(clean_name)
    s3['clean_name'] = s3['business_name'].apply(clean_name)

    # Prefix key creation (first 3 chars)
    s1['prefix'] = s1['clean_name'].str[:3]
    s2['prefix'] = s2['clean_name'].str[:3]
    s3['prefix'] = s3['clean_name'].str[:3]

    print("Building dictionaries for S2 and S3...")
    s2_dict = s2[s2['clean_name'] != ""].groupby('clean_name')['entity_id'].apply(list).to_dict()
    s3_dict = s3[s3['clean_name'] != ""].groupby('clean_name')['entity_id'].apply(list).to_dict()

    s2_prefix_dict = s2[s2['prefix'] != ""].groupby('prefix')['entity_id'].apply(list).to_dict()
    s3_prefix_dict = s3[s3['prefix'] != ""].groupby('prefix')['entity_id'].apply(list).to_dict()

    matching_rows = []
    candidate_rows = []

    print("Processing matching logic across S1, S2, and S3...")
    for _, row in s1.iterrows():
        s1_id = row['entity_id']
        c_name = row['clean_name']
        pref = row['prefix']

        # Candidates pool from both S2 and S3
        cands = []
        if pref in s2_prefix_dict:
            cands.extend(s2_prefix_dict[pref][:2])
        if pref in s3_prefix_dict:
            cands.extend(s3_prefix_dict[pref][:2])

        cand_str = ",".join(list(dict.fromkeys(cands))) if cands else ""
        candidate_rows.append({'source1_entity_id': s1_id, 'candidate_entity_ids': cand_str})

        # Exact clean match logic from both S2 and S3
        matches = []
        if c_name and c_name in s2_dict:
            matches.extend(s2_dict[c_name])
        if c_name and c_name in s3_dict:
            matches.extend(s3_dict[c_name])

        match_str = ",".join(list(dict.fromkeys(matches))) if matches else ""
        matching_rows.append({'source1_entity_id': s1_id, 'matched_entity_ids': match_str})

    os.makedirs("output", exist_ok=True)
    pd.DataFrame(matching_rows).to_csv("output/matching_results.tsv", sep="\t", index=False)
    pd.DataFrame(candidate_rows).to_csv("output/candidate_pairs.tsv", sep="\t", index=False)
    print("Pipeline Execution Completed Successfully with S2 and S3!")

if __name__ == "__main__":
    run_pipeline()