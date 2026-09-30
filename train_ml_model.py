import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\b(inc|llc|corp|corporation|ltd|limited|co|company|group)\b', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def train_and_run_ml_pipeline():
    print("🚀 Loading 1,000 Sample Records for ML Training...")
    
    # 1000 sample rows loading
    s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", nrows=1000)
    s2 = pd.read_csv("dataset/test/test_source2.tsv", sep="\t", nrows=1000)
    s3 = pd.read_csv("dataset/test/test_source3.tsv", sep="\t", nrows=1000)

    print("🧹 Cleaning & Preprocessing Text...")
    s1['clean_name'] = s1['business_name'].apply(clean_text)
    s2['clean_name'] = s2['business_name'].apply(clean_text)
    s3['clean_name'] = s3['business_name'].apply(clean_text)

    # All unique business names corpus
    all_names = pd.concat([s1['clean_name'], s2['clean_name'], s3['clean_name']]).unique()

    print("🧠 Training ML Model (TF-IDF Character N-Gram Vectorizer)...")
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    vectorizer.fit(all_names) # Model training step

    print("🔢 Converting Business Names into Numerical Vector Spaces...")
    s1_vecs = vectorizer.transform(s1['clean_name'])
    s2_vecs = vectorizer.transform(s2['clean_name'])
    s3_vecs = vectorizer.transform(s3['clean_name'])

    print("🎯 Predicting Similarity Matrix using Cosine Similarity...")
    sim_s1_s2 = cosine_similarity(s1_vecs, s2_vecs)
    sim_s1_s3 = cosine_similarity(s1_vecs, s3_vecs)

    matching_rows = []
    candidate_rows = []

    SIMILARITY_THRESHOLD = 0.85 # High precision threshold for F0.5 optimization

    for i in range(len(s1)):
        s1_id = s1.iloc[i]['entity_id']

        # Candidates (Top-2 S2 + Top-2 S3)
        s2_top_idx = np.argsort(sim_s1_s2[i])[-2:]
        s3_top_idx = np.argsort(sim_s1_s3[i])[-2:]

        cands = [s2.iloc[idx]['entity_id'] for idx in s2_top_idx if sim_s1_s2[i][idx] > 0.3]
        cands.extend([s3.iloc[idx]['entity_id'] for idx in s3_top_idx if sim_s1_s3[i][idx] > 0.3])

        cand_str = ",".join(cands) if cands else ""
        candidate_rows.append({'source1_entity_id': s1_id, 'candidate_entity_ids': cand_str})

        # High Precision Exact Matches
        matches = []
        for idx in range(len(s2)):
            if sim_s1_s2[i][idx] >= SIMILARITY_THRESHOLD:
                matches.append(s2.iloc[idx]['entity_id'])

        for idx in range(len(s3)):
            if sim_s1_s3[i][idx] >= SIMILARITY_THRESHOLD:
                matches.append(s3.iloc[idx]['entity_id'])

        match_str = ",".join(list(set(matches))) if matches else ""
        matching_rows.append({'source1_entity_id': s1_id, 'matched_entity_ids': match_str})

    print("✅ ML Model Training & Inference Completed Successfully!")
    print(f"📊 Model Matrix Shape: {s1_vecs.shape} (Trained Vocabulary Size)")

if __name__ == "__main__":
    train_and_run_ml_pipeline()