import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import geopandas as gpd
import data_preprocessing as dp
import folium
import streamlit as st
# from streamlit.components.v1 import html
from streamlit_folium import st_folium
from datetime import time, timedelta, datetime, date

import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go

start_year = 2020
end_year = 2023

@st.cache_data
# Einlesen der Dateien
def read_files(start_year=2020, end_year=2023):
    years = range(start_year, end_year + 1)
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
df_all_data = format_files(read_files(start_year=start_year, end_year=end_year))
df = df_all_data.dropna()


# Lade die GeoJSON-Datei der Stadtviertel
city_districts = gpd.read_file("neighbourhoods.geojson")

# Lade die GeoJSON-Datei der City Area
city_area = gpd.read_file("city_area.geojson")

# Session State für die Karte initialisieren
if "show_map" not in st.session_state:
    st.session_state.show_map = False

chosen_depth = st.selectbox("Welchen Zeitraum möchtest du betrachten?", ("Jahre", "Monate", "Tage"), index=None, placeholder="Zeitraum auswählen")

if chosen_depth == "Monate":

    # Session State initialisieren
    if "map_config" not in st.session_state:
        st.session_state.map_config = {
            "show_startpoints": False,
            "show_endpoints": False,
            "show_lines": False,
            "show_city_districts": False,
            "show_city_area": False
        }
    if "chosen_time_data" not in st.session_state:
        st.session_state.chosen_time_data = None

    year_input = st.multiselect("Wähle die Jahre aus:",                            
                            list(range(start_year, end_year + 1)))
    
    month_input = st.multiselect("Wähle die Monate aus:",
                                 list(range(1, 13)))
    st.write(year_input)
    st.write(month_input)
    st.write("Mehr passiert hier gerade noch nicht!")

if chosen_depth == "Tage":

    # Session State initialisieren
    if "map_config" not in st.session_state:
        st.session_state.map_config = {
            "show_startpoints": False,
            "show_endpoints": False,
            "show_lines": False,
            "show_city_districts": False,
            "show_city_area": False
        }
    if "chosen_time_data" not in st.session_state:
        st.session_state.chosen_time_data = None

    # Auswahl des Startdatums
    day_input_start = st.date_input("Wähle ein Startdatum:",
                        value=date(2023,12,31),
                        min_value=date(2020, 1, 1),
                        max_value=date(2023, 12, 31))

    # Auswahl des Enddatums
    day_input_end = st.date_input("Wähle ein Enddatum:",
                        value=date(2023,12,31),
                        min_value=date(2020, 1, 1),
                        max_value=date(2023, 12, 31))
    
    # Prüfen der Daten auf Gültigkeit
    if day_input_end < day_input_start:
        st.write("Das Enddatum muss nach dem Startdatum liegen.")
        valid = False
    elif day_input_end - day_input_start >= timedelta(days=7):
        st.write("Es können nicht mehr als 7 Tage ausgewählt werden.")
        valid = False
    else:
        valid = True

    # Auswahl des Zeitraums
    daytime_input = st.slider("Wähle die Tageszeit:",
                            min_value=time(0),
                            max_value=time(23, 59, 59),
                            value=(time(0), time(23, 59, 59)),
                            step=timedelta(hours=1))

    # speichert gewünschten Zeitraum in einer Variablen
    # chosen_time = (datetime.combine(day_input_start, daytime_input[0]), datetime.combine(day_input_end, daytime_input[1]))
    if valid:


        st.write("Was soll auf der Karte angezeigt werden?")
        show_startpoints = st.checkbox("Startpunkte", value=st.session_state.map_config["show_startpoints"])
        show_endpoints = st.checkbox("Endpunkte", value=st.session_state.map_config["show_endpoints"])
        show_lines = st.checkbox("Strecke (Luftlinie)", value=st.session_state.map_config["show_lines"])
        show_city_districts = st.checkbox("Stadtviertel", value=st.session_state.map_config["show_city_districts"])
        show_city_area = st.checkbox("Stadtbereich", value=st.session_state.map_config["show_city_area"],
                                     help="Bereich, in dem Fahrräder auch abseits von Stationen zurückgegeben werden können")


        if st.button("Hier klicken für Auswertung und Aktualisierung der Karte"):
            # Speichern des DataFrames im Session State
            st.session_state.chosen_time_data = df[(((df["STARTTIME"].dt.date >= day_input_start) & (df["STARTTIME"].dt.date <= day_input_end))\
                            | ((df["ENDTIME"].dt.date >= day_input_start) & (df["ENDTIME"].dt.date <= day_input_end)))\
                            & (((df["STARTTIME"].dt.time >= daytime_input[0]) & (df["STARTTIME"].dt.time <= daytime_input[1]))
                            | ((df["ENDTIME"].dt.time >= daytime_input[0]) & (df["ENDTIME"].dt.time <= daytime_input[1])))].dropna().copy()
            
            # Speichern der Checkbox-Werte im Session State
            st.session_state.map_config["show_startpoints"] = show_startpoints
            st.session_state.map_config["show_endpoints"] = show_endpoints
            st.session_state.map_config["show_lines"] = show_lines
            st.session_state.map_config["show_city_districts"] = show_city_districts
            st.session_state.map_config["show_city_area"] = show_city_area
            
            st.session_state.show_map = True
        
        # Zeige die Karte nur, wenn "show_map" True ist
        if st.session_state.show_map and st.session_state.chosen_time_data is not None:

            # Initialisieren der Karte
            map_center = [48.137154, 11.576124] # Munich city centre
            munich_map = folium.Map(location=map_center, zoom_start=12)
            
            for index, row in st.session_state.chosen_time_data.iterrows():
                start_coordinates = [row["STARTLAT"], row["STARTLON"]]
                end_coordinates = [row["ENDLAT"], row["ENDLON"]]

                # Hinzufügen der Startpunkte
                if st.session_state.map_config["show_startpoints"]:
                    folium.Circle(location=start_coordinates,
                                        radius=20,
                                        color="purple",
                                        fill=True,
                                        fill_color="purple",
                                        fill_opacity=0.5
                                        ).add_to(munich_map)
                
                # Hinzufügen der Endpunkte
                if st.session_state.map_config["show_endpoints"]: 
                        folium.Circle(location=end_coordinates,
                                            radius=20,
                                            color="green",
                                            fill=True,
                                            fill_color="green",
                                            fill_opacity=0.5
                                            ).add_to(munich_map)
                
                # Hinzufügen einer Linie zwischen Start- und Rückgabeort
                if st.session_state.map_config["show_lines"]:
                        folium.PolyLine(locations=[start_coordinates, end_coordinates],
                                        color="grey",
                                        weight=2.5,
                                        opacity=0.3).add_to(munich_map)
                
            # Füge die Stadtviertel als GeoJSON auf der Karte hinzu
            if st.session_state.map_config["show_city_districts"]:
                folium.GeoJson(
                    city_districts,
                    name="Stadtviertel",
                    style_function=lambda feature: {
                        "fillColor": "lightblue",  # Füllfarbe der Stadtviertel
                        "color": "blue",
                        "weight": 2,
                        "opacity": 0.3,
                        "fillOpacity": 0.2
                    }
                ).add_to(munich_map)

                # Füge den Stadtbereich als GeoJSON auf der Karte hinzu
            if st.session_state.map_config["show_city_area"]:
                folium.GeoJson(
                    city_area,
                    name="Stadtbereich",
                    style_function=lambda feature: {
                        "fillColor": "lightgreen",  # Füllfarbe des Stadtbereichs
                        "color": "green",
                        "weight": 2,
                        "opacity": 0.6,
                        "fillOpacity": 0.25
                    }
                ).add_to(munich_map)
            
            # Anzeigen der Karte
            st_folium(munich_map, width=700)
            st.write(st.session_state.chosen_time_data)


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
