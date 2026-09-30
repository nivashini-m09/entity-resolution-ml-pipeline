# Entity Resolution Methodology Document

## 1. Candidate Generation / Blocking Strategy
Used prefix-based blocking (first 3 characters of normalized business names) across Source 2 and Source 3 datasets.

## 2. Preprocessing & Normalization
Cleaned corporate suffixes (e.g., Inc, LLC, Corp, Ltd) and non-alphanumeric characters to improve matching recall and precision.

## 3. High-Precision Matching
Extracted top candidate matches per Source 1 entity and applied exact normalized name matching, maximizing {0.5}$ score while guaranteeing full validation compliance.
