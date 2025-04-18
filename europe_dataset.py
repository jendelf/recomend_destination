import pandas as pd


df = pd.read_csv("attractions.csv")  

# границы европы
min_lat, max_lat = 35, 72  
min_lon, max_lon = -25, 45  


df_europe = df[
    (df["LATITUDE"] >= min_lat) & (df["LATITUDE"] <= max_lat) &
    (df["LONGITUDE"] >= min_lon) & (df["LONGITUDE"] <= max_lon)
]


df_europe.to_csv("europe_only.csv", index=False)

print(f"Фильтрация завершена! Осталось {len(df_europe)} записей.")
