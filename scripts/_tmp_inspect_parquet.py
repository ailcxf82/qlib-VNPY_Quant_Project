import pandas as pd

p = "git_ignore_folder/combined_factors_df.parquet"
df = pd.read_parquet(p)
print("path:", p)
print("shape:", df.shape)
print("index names:", df.index.names)
print("index levels:", df.index.nlevels)
print("columns:", list(df.columns))
print("dtypes:", {k: str(v) for k, v in df.dtypes.to_dict().items()})
idx = df.index
print(
    "date range:",
    idx.get_level_values(0).min(),
    "~",
    idx.get_level_values(0).max(),
)
print("instruments count:", idx.get_level_values(1).nunique())
print("head:")
print(df.head(3))
print("non_null:")
print(df.notna().sum().to_dict())
