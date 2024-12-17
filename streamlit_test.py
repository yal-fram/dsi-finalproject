import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import geopandas as gpd
# import osmnx as ox
# import contextily as cx
import data_preprocessing as dp
import folium
import streamlit as st
# from streamlit.components.v1 import html
from streamlit_folium import st_folium

import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go

@st.cache_data
def read_files(start_year=2020, end_year=2024):
    years = range(start_year, end_year)
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
    return df

@st.cache_data
def format_files(df):
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

    # Entfernen ungültiger Daten
    df = dp.remove_invalid_datetime(df)

    # Hinzufügen des Stadtviertels
    df = dp.add_city_district(df)

    # Hinzufügen, ob Punkte in Stadtbereich ("city area")
    df = dp.add_city_status(df)

    return df

# Removing data with NULL values

df_all_data = format_files(read_files())
df = df_all_data.dropna()


day_input = st.date_input("Wähle ein Datum:",
                      value=pd.to_datetime("2023-01-01"),
                      min_value=pd.to_datetime("2020-01-01"),
                      max_value=pd.to_datetime("2023-12-31"))

chosen_day = df[(df["STARTTIME"].dt.year == day_input.year) & (df["STARTTIME"].dt.month == day_input.month) & (df["STARTTIME"].dt.day == day_input.day)].dropna().copy()

# Initialisiere die Karte mit Zentrum bei Mittelwerten von Breiten- und Längengrad
map_center = [48.137154, 11.576124] # Munich city centre
my_map = folium.Map(location=map_center, zoom_start=12)


if st.checkbox("Startpunkte"):
# Hinzufügen von Punkten für Ausleihort
    for index, row in chosen_day.iterrows():
        start_coordinates = [row["STARTLAT"], row["STARTLON"]]
        end_coordinates = [row["ENDLAT"], row["ENDLON"]]
        folium.Circle(location=start_coordinates,
                            radius=20,
                            color="purple",
                            fill=True,
                            fill_color="purple",
                            fill_opacity=0.5
                            ).add_to(my_map)

if st.checkbox("Endpunkte"):
#Hinzufügen von Punkten für Rückgabeort
    for index, row in chosen_day.iterrows():
        start_coordinates = [row["STARTLAT"], row["STARTLON"]]
        end_coordinates = [row["ENDLAT"], row["ENDLON"]]   
        folium.Circle(location=end_coordinates,
                            radius=20,
                            color="green",
                            fill=True,
                            fill_color="green",
                            fill_opacity=0.5
                            ).add_to(my_map)

if st.checkbox("Strecke (Luftlinie)"):
# Hinzufügen einer Linie zwischen Start- und Rückgabeort
    for index, row in chosen_day.iterrows():
        start_coordinates = [row["STARTLAT"], row["STARTLON"]]
        end_coordinates = [row["ENDLAT"], row["ENDLON"]]
        folium.PolyLine(locations=[start_coordinates, end_coordinates],
                        color="grey",
                        weight=2.5,
                        opacity=0.3).add_to(my_map)
    

# Lade die GeoJSON-Datei der Stadtviertel
city_districts = gpd.read_file("neighbourhoods.geojson")


if st.checkbox("Stadtviertel"):
# Füge die Stadtviertel als GeoJSON auf die Karte hinzu
    folium.GeoJson(
        city_districts,
        name="Stadtviertel",
        style_function=lambda feature: {
            'fillColor': 'lightblue',  # Füllfarbe der Stadtviertel
            'color': 'blue',
            'weight': 2,
            'opacity': 0.3,
            'fillOpacity': 0.2
        }
    ).add_to(my_map)


# Lade die GeoJSON-Datei der City Area
city_area = gpd.read_file("city_area.geojson")

if st.checkbox("Stadtbereich", help="Bereich, in dem Fahrräder auch abseits von Stationen zurückgegeben werden können"):
# Füge die City Area auf die Karte hinzu
    folium.GeoJson(
        city_area,
        name="Stadtbereich",
        style_function=lambda feature: {
            'fillColor': 'lightgreen',  # Füllfarbe der City Area
            'color': 'green',
            'weight': 2,
            'opacity': 0.6,
            'fillOpacity': 0.25
        }
    ).add_to(my_map)


# map_html = my_map._repr_html_()

# Uncomment zum Anzeigen der Karte:
# html(map_html, width=1000, height=800)

st_folium(my_map, width=700)

# Zeitreihenanalyse mit Plotly und Prophet
# Extrahieren von Date / Hour
df['DATE'] = df['STARTTIME'].dt.date  # Datum extrahieren
df['HOUR'] = df['STARTTIME'].dt.hour  # Stunde extrahieren

# Anzahl der Fahrten pro Tag
daily_counts = df.groupby('DATE').size().reset_index(name='DAILY_COUNTS')

# Linienplot der täglichen Anzahl der Fahrten
fig_daily = px.line(
    daily_counts,
    x='DATE',
    y='DAILY_COUNTS',
    title='Tägliche Anzahl der Fahrten (2020-2023)',
    labels={'DAILY_COUNTS': 'Fahrten', 'DATE': 'Datum'},
    template="plotly_white"
)

# Titel in die Mitte setzen
fig_daily.update_layout(
    title={
        'text': "Tägliche Anzahl der Fahrten (2020-2023)",
        'y': 0.9,  # Y-Position des Titels
        'x': 0.5,  # X-Position des Titels
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=24)
    }
)

# Prophet-Modell initialisieren
prophet_data = daily_counts.rename(columns={'DATE': 'ds', 'DAILY_COUNTS': 'y'})
model = Prophet(growth='linear', seasonality_mode='additive', interval_width=0.90)
model.fit(prophet_data)

# Zukunftsdaten erstellen und Vorhersage
future = model.make_future_dataframe(periods=365, freq='D', include_history=True)
forecast = model.predict(future)

# Prophet-Visualisierung
fig_forecast = plot_plotly(model, forecast)
fig_components = plot_components_plotly(model, forecast)

# Streamlit-Anzeige
st.sidebar.header("Zeitreihendanalyse:")
if st.sidebar.button("Tägliche Anzahl der Fahrten"):
    st.plotly_chart(fig_daily, use_container_width=True)

if st.sidebar.button("Prophet-Vorhersage"):
    st.plotly_chart(fig_forecast, use_container_width=True)
    st.plotly_chart(fig_components, use_container_width=True)
