import pandas as pd

df = pd.read_csv("data/dataset_assignment.csv")

print("Ukuran dataset:", df.shape)
print("\nNama kolom:")
print(df.columns.tolist())

print("\nContoh data:")
print(df.head())

print("\nJumlah data kosong:")
print(df.isna().sum())
