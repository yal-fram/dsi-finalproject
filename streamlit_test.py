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
    """Reading csv files and concatenating them in a Pandas DataFrame

    Args:
        start_year (int, optional): year to start with. Defaults to 2020.
        end_year (int, optional): year to end with. Defaults to 2023.

    Returns:
        pd.DataFrame: Pandas DataFrame
    """
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
    """Formatting and Cleaning Pandas DataFrame. Combines several functions defined and executes them consecutively.

    Args:
        df (pd.DataFrame): Pandas DataFrame

    Returns:
        pd.DataFrame: formatted and cleaned DataFrame
    """
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

    # Removing data with NULL values
    df = df.dropna()

    # Hinzufügen der Distanz
    # dauert ein paar Minuten, uncomment nur, wenn benötigt!
    # df = dp.calculate_distance(df)

    # Hinzufügen des Stadtviertels
    df = dp.add_city_district(df)

    # Hinzufügen, ob Punkte in Stadtbereich ("city area")
    df = dp.add_city_status(df)

    return df

# Load Dataframe
df = format_files(read_files(start_year=start_year, end_year=end_year))

@st.cache_data
def load_geojson(geojson):
    geojson_file = gpd.read_file(geojson)
    return geojson_file

# Lade die GeoJSON-Datei der Stadtviertel
city_districts = load_geojson("neighbourhoods.geojson")

# Lade die GeoJSON-Datei der City Area
city_area = load_geojson("city_area.geojson")

# Session State für die Karte initialisieren
if "show_map" not in st.session_state:
    st.session_state.show_map = False

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

# Streamlit-Titel
st.header(f":blue[MVG-Mieträder] :bike: :blue[in München {start_year} - {end_year}]")

# Session State initialisieren
if "geo_days" not in st.session_state:
    st.session_state.geo_days = False
if "geo_months" not in st.session_state:
    st.session_state.geo_months = False
if "geo_years" not in st.session_state:
    st.session_state.geo_days = False

# Streamlit-Seitenleiste
st.sidebar.header(""); st.sidebar.header(""); st.sidebar.header("")
st.sidebar.header("Zeitreihenanalyse")


if st.sidebar.button("Tägliche Anzahl der Fahrten"):
    st.plotly_chart(fig_daily, use_container_width=True)


if st.sidebar.button("Prophet-Vorhersage"):
    st.plotly_chart(fig_forecast, use_container_width=True)
    st.plotly_chart(fig_components, use_container_width=True)



st.sidebar.header("Geografische Auswertung")
if st.sidebar.button("nach Jahren"):
    # Session States aktualisieren
    st.session_state.geo_years = True
    st.session_state.geo_months = False
    st.session_state.geo_days = False

    st.write("Hier passiert noch gar nichts.")


if st.sidebar.button("nach Monaten"):
    # Session States aktualisieren
    st.session_state.geo_years = False
    st.session_state.geo_months = True
    st.session_state.geo_days = False

    # Session States für Monate initialisieren
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


if st.sidebar.button("nach Tagen"):
    # Session States aktualisieren
    st.session_state.geo_years = False
    st.session_state.geo_months = False
    st.session_state.geo_days = True

    # Session States für Tage initialisieren
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

if st.session_state.geo_days:
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
            st.write(f"Anzahl Fahrten im gewählten Zeitraum: {st.session_state.chosen_time_data.shape[0]}")
            avg_length = st.session_state.chosen_time_data["DURATION"].mean().seconds // 60
            if avg_length > 60:
                avg_length_str = f"{avg_length // 60} Stunden, {avg_length%60} Minuten"
            else:
                avg_length_str = f"{avg_length} Minuten"
            st.write(f"Durchschnittliche Fahrtenlänge im gewählten Zeitraum: {avg_length_str}")
            # st.write(f"Durchschnittliche Entfernung (Luftlinie) im gewählten Zeitraum: {st.session_state.chosen_time_data["DISTANCE"].mean():.1f} Kilometer")
            st.write(f"Beliebtestes Startviertel: {st.session_state.chosen_time_data["CITY_DISTRICT_START"].mode()[0]}")
            st.write(f"Beliebtestes Zielviertel: {st.session_state.chosen_time_data["CITY_DISTRICT_END"].mode()[0]}")
            # Ausleihen an Stationen im Stadtgebiet:
            rental_station_city_number = st.session_state.chosen_time_data[((st.session_state.chosen_time_data["RENTAL_IS_STATION"] == 1)\
                                             & (st.session_state.chosen_time_data["RENTAL_IS_CITY"] == 1))].shape[0]
            # Ausleihen an Stationen außerhalb des Stadtgebiets:
            rental_station_not_city_number = st.session_state.chosen_time_data[((st.session_state.chosen_time_data["RENTAL_IS_STATION"] == 1)\
                                             & (st.session_state.chosen_time_data["RENTAL_IS_CITY"] == 0))].shape[0]
            # Rückgaben an Stationen im Stadtgebiet:
            return_station_city_number = st.session_state.chosen_time_data[((st.session_state.chosen_time_data["RETURN_IS_STATION"] == 1)\
                                             & (st.session_state.chosen_time_data["RETURN_IS_CITY"] == 1))].shape[0]
            # Rückgaben an Stationen außerhalb des Stadtgebiets:
            return_station_not_city_number = st.session_state.chosen_time_data[((st.session_state.chosen_time_data["RENTAL_IS_STATION"] == 1)\
                                             & (st.session_state.chosen_time_data["RETURN_IS_CITY"] == 0))].shape[0]
            st.write(f"Ausleihen an Stationen inner- / außerhalb des Stadtgebiets: {rental_station_city_number} / {rental_station_not_city_number}")
            st.write(f"Rückgaben an Stationen inner- / außerhalb des Stadtgebiets: {return_station_city_number} / {return_station_not_city_number}")