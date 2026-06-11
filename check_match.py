import os
import pandas as pd
from src.data import sources as S

sys_path = os.path.join(S.DEEP_SEE, "system_out", "forecasts", "han_S01001.parquet")
if os.path.exists(sys_path):
    df = pd.read_parquet(sys_path)
    print("Columns:", df.columns)
    print("Unique targets:", df["target"].unique())
    print("Sample rows where target=do:")
    print(df[df["target"] == "do"].head(10))
