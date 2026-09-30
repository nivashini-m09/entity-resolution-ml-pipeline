import pandas as pd

print("--- Train Data Counts ---")
train_s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
train_s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
train_s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")

print("Train Source 1 count:", len(train_s1))
print("Train Source 2 count:", len(train_s2))
print("Train Source 3 count:", len(train_s3))

print("\n--- Test Data Counts ---")
test_s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t")
test_s2 = pd.read_csv("dataset/test/test_source2.tsv", sep="\t")
test_s3 = pd.read_csv("dataset/test/test_source3.tsv", sep="\t")

print("Test Source 1 count:", len(test_s1))
print("Test Source 2 count:", len(test_s2))
print("Test Source 3 count:", len(test_s3))