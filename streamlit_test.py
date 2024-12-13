import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import geopandas as gpd
# import osmnx as ox
# import contextily as cx
import data_preprocessing as dp
import folium
import streamlit as st
from streamlit.components.v1 import html


years = range(2020, 2024)
data_list = []

# Schleife über jedes Jahr
for year in years:
    df = pd.read_csv(f"MVG_Rad_Fahrten_{year}.csv", sep=";", decimal=",", parse_dates=["STARTTIME       ", "ENDTIME         "])
    
    # Entfernen von Leerzeichen in den Columns
    column_names = [dp.remove_space(column) for column in df.columns]
    df.columns = column_names

    # Hinzufügen des eingelesenen DataFrames zur Liste
    data_list.append(df)

# alle Daten zu einem einzigen DataFrame kombinieren 
df = pd.concat(data_list, ignore_index=True)

# Entfernen von Leerzeichen bei Stationsnamen
df["RENTAL_STATION_NAME"] = df["RENTAL_STATION_NAME"].apply(dp.remove_space)
df["RETURN_STATION_NAME"] = df["RETURN_STATION_NAME"].apply(dp.remove_space)

# Löschen von "Row"
df = df.drop("Row", axis=1)

# Formatierung der Koordinaten + Entfernung ungültiger Daten
df = dp.handle_coordinates(df)

# Formatierung von is_station
df = dp.handle_is_station(df)

# Auffüllen fehlender Werte anhand des Vorhandenseins oder Fehlens von "station_name"-Werten
df = dp.fill_is_station_values(df)


# Hinzufügen einer Spalte für die Dauer
df["DURATION"] = df["ENDTIME"] - df["STARTTIME"]

## Code aus Jupyter Notebooks, wird für Streamlit nicht benötigt
# # Darstellung auf einer Karte

# geo_start = gpd.points_from_xy(x=january_1_2023['STARTLON'], crs="EPSG:4326", y=january_1_2023['STARTLAT'])
# gdf_start = gpd.GeoDataFrame(january_1_2023, geometry=geo_start)

# geo_end = gpd.points_from_xy(x=january_1_2023['ENDLON'], y=january_1_2023["ENDLAT"], crs="EPSG:4326")
# gdf_end = gpd.GeoDataFrame(january_1_2023, geometry=geo_end)
# gdf_start = gdf_start.to_crs(epsg=3857)
# gdf_end = gdf_end.to_crs(epsg=3857)

# # Plotten des GeoDataFrames
# fig, ax = plt.subplots(figsize=(10, 10))

# # Plotten der Punkte
# gdf_start.plot(ax=ax, marker='x', color='purple', markersize=5, alpha=0.7)
# gdf_end.plot(ax=ax, marker='x', color='green', markersize=5, alpha=0.7)
# # Füge die Basemap von contextily hinzu
# cx.add_basemap(ax, crs=gdf_start.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)


# Testframe für Streamlit: 1. Januar 2023
january_1_2023 = df[(df["STARTTIME"].dt.year == 2023) & (df["STARTTIME"].dt.month == 1) & (df["STARTTIME"].dt.day == 1)]

january_1_2023_folium = january_1_2023.dropna()

# Geopandas-geometry, wird nicht benötigt
# geo_start = gpd.points_from_xy(x=january_1_2023['STARTLON'], crs="EPSG:4326", y=january_1_2023['STARTLAT'])
# gdf_start_jan_1_2023 = gpd.GeoDataFrame(january_1_2023, geometry=geo_start)

# Initialisiere die Karte mit Zentrum bei Mittelwerten von Breiten- und Längengrad
map_center = [january_1_2023_folium['STARTLAT'].mean(), january_1_2023_folium['STARTLON'].mean()]
my_map = folium.Map(location=map_center, zoom_start=13)

# Hinzufügen von Punkten für Ausleihort
for index, row in january_1_2023_folium.iterrows():
    folium.CircleMarker(location=[row["STARTLAT"], row["STARTLON"]],
                        radius=1.3,
                        color="purple",
                        fill=True,
                        fill_color="purple",
                        fill_opacity=0.5
                        ).add_to(my_map)

#Hinzufügen von Punkten für Rückgabeort    
for index, row in january_1_2023_folium.iterrows():
    folium.CircleMarker(location=[row["ENDLAT"], row["ENDLON"]],
                        radius=1.3,
                        color="green",
                        fill=True,
                        fill_color="green",
                        fill_opacity=0.5
                        ).add_to(my_map)
    

# Lade die GeoJSON-Datei der Stadtviertel
city_districts = gpd.read_file("neighbourhoods.geojson")

# Füge die Stadtviertel als GeoJSON auf die Karte hinzu
folium.GeoJson(
    city_districts,
    name="Stadtviertel",
    style_function=lambda feature: {
        'fillColor': 'lightblue',  # Füllfarbe der Stadtviertel
        'color': 'blue',  # Randfarbe
        'weight': 2,  # Randstärke
        'opacity': 0.6,  # Rand-Opazität
        'fillOpacity': 0.2  # Füll-Opazität
    }
).add_to(my_map)


map_html = my_map._repr_html_()

html(map_html, width=1000, height=800)