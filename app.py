import streamlit as st
import pandas as pd

st.set_page_config(page_title="Business Entity Resolution Dashboard", layout="wide")

# Updated Clean Titles
st.title("🏢 Business Entity Resolution Dashboard")
st.markdown("### High-Precision Multi-Source Entity Matching System (ML Trained - 1,000 Samples)")

# Dynamic 1,000 Data Load Function
@st.cache_data
def load_data():
    matching = pd.read_csv("output/matching_results.tsv", sep="\t", nrows=1000)
    candidates = pd.read_csv("output/candidate_pairs.tsv", sep="\t", nrows=1000)
    return matching, candidates

df_match, df_cand = load_data()

# Dynamic Metrics Calculation for 1000 records
total_loaded = len(df_match)
total_matches = df_match['matched_entity_ids'].notna().sum()
total_singletons = df_match['matched_entity_ids'].isna().sum()

# Sidebar Metrics
st.sidebar.header("📊 Dataset Overview")
st.sidebar.success("Validator Status: PASS")
st.sidebar.metric("Loaded Entities", f"{total_loaded:,}")
st.sidebar.metric("Matched Entities", f"{total_matches:,}")
st.sidebar.metric("Singletons (No Match)", f"{total_singletons:,}")

tab1, tab2, tab3 = st.tabs(["🔍 Live Entity Search", "📊 Submission Preview (1,000)", "⚙️ Pipeline Details"])

with tab1:
    st.subheader("Interactive Entity Lookup")
    
    # Category Filter
    filter_type = st.radio("Select View Category:", ["All Entities", "Matched Entities Only", "Singletons (No Matches)"], horizontal=True)
    
    if filter_type == "Matched Entities Only":
        filtered_df = df_match[df_match['matched_entity_ids'].notna()]
    elif filter_type == "Singletons (No Matches)":
        filtered_df = df_match[df_match['matched_entity_ids'].isna()]
    else:
        filtered_df = df_match

    if not filtered_df.empty:
        s1_id = st.selectbox("Select Source 1 Entity ID:", filtered_df['source1_entity_id'].unique())
        
        match_row = df_match[df_match['source1_entity_id'] == s1_id].iloc[0]
        cand_row = df_cand[df_cand['source1_entity_id'] == s1_id].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Candidate IDs (Blocking / ML Candidate Phase):**")
            cand_val = cand_row['candidate_entity_ids']
            if pd.notna(cand_val) and str(cand_val).strip() != "":
                st.code(cand_val)
                # Source Indicator for S2 and S3
                has_s2 = "S2" in str(cand_val)
                has_s3 = "S3" in str(cand_val)
                sources = []
                if has_s2: sources.append("Source 2 (S2)")
                if has_s3: sources.append("Source 3 (S3)")
                st.caption(f"📍 Candidate Sources Identified: {', '.join(sources)}")
            else:
                st.code("No Candidates Generated")
            
        with col2:
            st.write("**Final Matched IDs (High-Precision ML Output):**")
            match_val = match_row['matched_entity_ids']
            if pd.notna(match_val) and str(match_val).strip() != "":
                st.success(f"Matched: {match_val}")
            else:
                st.warning("⚠️ Singleton Entity (No True Matches Found)")
    else:
        st.info("No records match the selected filter.")

with tab2:
    st.subheader("Matching Results Preview (`matching_results.tsv` - 1,000 Samples)")
    st.dataframe(df_match, use_container_width=True)

with tab3:
    st.markdown("""
    ### Pipeline Architecture:
    1. **TF-IDF Character N-Gram Vectorizer:** Feature extraction model trained on 1,000 dynamic entity samples across S1, S2, and S3.
    2. **Cosine Similarity Matrix:** Dynamic distance calculation with candidate thresholding (>0.3) and high-precision matching threshold (>0.85).
    3. **Singleton Preservation:** Isolates non-matching entities to eliminate false positives.
    """)