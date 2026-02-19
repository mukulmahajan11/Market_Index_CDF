import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("artifacts/mood_index_output.csv")
asset = "ASSET_000"

d = df[df["asset_id"] == asset]

plt.figure(figsize=(14,6))
plt.plot(pd.to_datetime(d["date"]), d["mood_ewma"])
plt.title(f"{asset} Market Mood Index")
plt.ylabel("Mood (0-100)")
plt.xticks(rotation=45)
plt.show()
