import pandas as pd
import numpy as np
import geopandas as gpd
import data_preprocessing as dp
import folium
import streamlit as st
# from streamlit.components.v1 import html
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from datetime import time, timedelta, datetime, date
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go

start_year = 2020
end_year = 2023


# Einlesen der Dateien
@st.cache_data
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


# Formatieren der Dateien
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
    # dauert lange, uncomment nur, wenn erwünscht und ausreichend Rechenleistung verfügbar!
    # wird später zur Angabe der mittleren Distanz verwendet
    # diese Zeilen müssten dann entsprechend auch ent-kommentiert werden: suche nach "Entfernung" und "DISTANCE"
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


# Zeitreihenanalyse mit Plotly und Prophet

# Initialisieren von session_state für Zeitreihenanalyse
if "time" not in st.session_state:
    st.session_state.time = False
if "fig_daily" not in st.session_state:
    st.session_state.fig_daily = False
if "fig_forecast" not in st.session_state:
    st.session_state.fig_forecast = False
if "fig_components" not in st.session_state:
    st.session_state.fig_components = False

# Session States für geographische Auswertung initialisieren
if "geo_days" not in st.session_state:
    st.session_state.geo_days = False
if "geo_months" not in st.session_state:
    st.session_state.geo_months = False
# if "geo_years" not in st.session_state:
#     st.session_state.years = False
# Session State für die Karte initialisieren
if "show_map" not in st.session_state:
    st.session_state.show_map = False

# Session States für Monate initialisieren
if "map_config_months" not in st.session_state:
    st.session_state.map_config_months = {
        "show_stations": False,
        "show_heatmap": False,
        "show_city_districts": False,
        "show_city_area": False
    }

# Session States für Tage initialisieren
if "map_config_days" not in st.session_state:
    st.session_state.map_config_days = {
        "show_startpoints": False,
        "show_endpoints": False,
        "show_lines": False,
        "show_city_districts": False,
        "show_city_area": False}

def reset_views():
    # st.session_state.geo_years = False
    st.session_state.geo_months = False
    st.session_state.geo_days = False
    st.session_state.show_map = False
    st.session_state.map_config_months["show_stations"] = False
    st.session_state.map_config_months["show_heatmap"] = False
    st.session_state.map_config_months["show_city_districts"] = False
    st.session_state.map_config_months["show_city_area"] = False
    st.session_state.map_config_days["show_startpoints"] = False
    st.session_state.map_config_days["show_endpoints"] = False
    st.session_state.map_config_days["show_lines"] = False
    st.session_state.map_config_days["show_city_districts"] = False
    st.session_state.map_config_days["show_city_area"] = False
# Streamlit-Titel
st.header(f":blue[MVG-Mieträder] :bike: :blue[in München {start_year} - {end_year}]")

# Streamlit-Seitenleiste
st.sidebar.header(""); st.sidebar.header("")
st.sidebar.header("Zeitreihenanalyse")

if st.sidebar.button("Zeitreihenanalyse starten", type="primary"):
    # Extrahieren von Date / Hour
    st.session_state.time = df.copy()
    st.session_state.time['DATE'] = st.session_state.time['STARTTIME'].dt.date  # Datum extrahieren
    st.session_state.time['HOUR'] = st.session_state.time['STARTTIME'].dt.hour  # Stunde extrahieren

    # Anzahl der Fahrten pro Tag
    daily_counts = st.session_state.time.groupby('DATE').size().reset_index(name='DAILY_COUNTS')

    # Linienplot der täglichen Anzahl der Fahrten
    st.session_state.fig_daily = px.line(
        daily_counts,
        x='DATE',
        y='DAILY_COUNTS',
        title='Tägliche Anzahl der Fahrten (2020-2023)',
        labels={'DAILY_COUNTS': 'Fahrten', 'DATE': 'Datum'},
        template="plotly_white"
    )

    # Titel in die Mitte setzen
    st.session_state.fig_daily.update_layout(
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
    st.session_state.fig_forecast = plot_plotly(model, forecast)
    st.session_state.fig_components = plot_components_plotly(model, forecast)




if st.sidebar.button("Tägliche Anzahl der Fahrten"):
    reset_views()
    st.plotly_chart(st.session_state.fig_daily, use_container_width=True)






if st.sidebar.button("Prophet-Vorhersage"):
    reset_views()
    st.plotly_chart(st.session_state.fig_forecast, use_container_width=True)
    st.plotly_chart(st.session_state.fig_components, use_container_width=True)






st.sidebar.header("Geografische Auswertung")

# if st.sidebar.button("nach Jahren"):
#     # Session States aktualisieren
#     reset_views()
#     st.write("Hier passiert noch gar nichts.")



if st.sidebar.button("Überblick nach Monaten"):
    # Session States aktualisieren
    reset_views()
    st.session_state.geo_months = True


    if "chosen_months" not in st.session_state:
        st.session_state.chosen_months = None

if st.sidebar.button("Detailansicht nach Tagen"):
    # Session States aktualisieren
    reset_views()
    st.session_state.geo_days = True

    if "chosen_days" not in st.session_state:
        st.session_state.chosen_days = None

st.sidebar.header(""); st.sidebar.header(""); st.sidebar.header(""); st.sidebar.header("");

if st.sidebar.button("Let it snow!", type="primary"):
    reset_views()
    st.markdown(
    """
    <h1 style="text-align: center; color:rgb(0, 104, 201);"><br><br>
        Vielen Dank für die Aufmerksamkeit!
    </h1>
    """, 
    unsafe_allow_html=True
    )
    st.snow()
# tagsüber (Tagmodus): rgb(0, 104, 201), abends (Nachtmodus): rgb(96, 180, 255)


# Wenn Monate ausgewertet werden sollen
if st.session_state.geo_months:
    # Auswahl der Jahre und Monate
    year_input = st.multiselect("Wähle die Jahre aus:",                            
                            list(range(start_year, end_year + 1)),
                            placeholder="Jahre auswählen")
    month_input = st.multiselect("Wähle die Monate aus:",
                            list(range(1, 13)),
                            placeholder="Monate auswählen")

    st.write("Was soll auf der Karte angezeigt werden?")
    # Auswahl der gewünschten Parameter
    show_stations = st.checkbox("Stationen", value=st.session_state.map_config_months["show_stations"])
    show_heatmap = st.checkbox("Heatmap", value=st.session_state.map_config_months["show_heatmap"])
    show_city_districts = st.checkbox("Stadtviertel", value=st.session_state.map_config_months["show_city_districts"], key="districts_months")
    show_city_area = st.checkbox("Stadtbereich", value=st.session_state.map_config_months["show_city_area"],
                                help="Bereich, in dem Fahrräder auch abseits von Stationen zurückgegeben werden können", key="area_months")


    if st.button("Hier klicken für Auswertung und Aktualisierung der Karte", key="map_months"):
        if not year_input or not month_input:
            st.write("Bitte wähle Jahre und Monate aus.")
            valid_month = False
        else:
            valid_month = True

        if valid_month:
            # Speichern des DataFrames im Session State
            df_list = []
            for year in year_input:
                df_temp = df[((df["STARTTIME"].dt.year == year) | (df["ENDTIME"].dt.year == year))].copy()
                df_list.append(df_temp)
            df_years_temp = pd.concat(df_list)
            df_list = []
            for month in month_input:
                df_temp = df_years_temp[((df_years_temp["STARTTIME"].dt.month == month) | (df_years_temp["ENDTIME"].dt.month == month))].copy()
                df_list.append(df_temp)
            st.session_state.chosen_months = pd.concat(df_list)
            
            # Speichern der Checkbox-Werte im Session State
            st.session_state.map_config_months["show_stations"] = show_stations
            st.session_state.map_config_months["show_heatmap"] = show_heatmap
            st.session_state.map_config_months["show_city_districts"] = show_city_districts
            st.session_state.map_config_months["show_city_area"] = show_city_area
            
            st.session_state.show_map = True

    # Zeige die Karte nur, wenn "show_map" True ist
    if st.session_state.show_map and st.session_state.chosen_months is not None:

        # Initialisieren der Karte
        map_center = [48.137154, 11.576124] # Munich city centre
        munich_map = folium.Map(location=map_center, zoom_start=11)

        # benutzte Stationen ermitteln
        stations = dp.get_station_data(st.session_state.chosen_months)

        # Nutzungshäufigkeit der Stationen ermitteln
        frequency_start = st.session_state.chosen_months["RENTAL_STATION_NAME"].value_counts()
        frequency_end = st.session_state.chosen_months["RETURN_STATION_NAME"].value_counts()

        # Heatmap
        if st.session_state.map_config_months["show_heatmap"]:
            # heat-Daten für Stationen:
            # (geht schneller, umfasst aber nur Stationen)
            # heat_data = dp.get_heatmap_data(st.session_state.chosen_months, stations)

            # heat-Daten für alle Punkte:
            # (dauert länger, ist aber spannender, da auch mit freien Rückgaben)
            heat_data_start = list(zip(st.session_state.chosen_months["STARTLAT"], st.session_state.chosen_months["STARTLON"]))
            heat_data_end = list(zip(st.session_state.chosen_months["STARTLAT"], st.session_state.chosen_months["STARTLON"]))
            heat_data = heat_data_start + heat_data_end

            heat_map = HeatMap(heat_data, min_opacity=0.2, radius=25, blur=18)
            heat_map.add_to(munich_map)

        # Hinzufügen der Stationen
        if st.session_state.map_config_months["show_stations"]:
            # Prüfen, ob Ausleih- und Rückgabewerte für jede Station vorhanden, ansonsten 0 einsetzen (kein Wert vorhanden => Wert = 0)
            for station, coordinates in stations.items():
                if station not in frequency_start.index.values:
                    frequency_start[station] = 0
                if station not in frequency_end.index.values:
                    frequency_end[station] = 0

                # Hinzufügen zur Karte    
                folium.Marker(location=coordinates,
                              icon=folium.Icon(color="darkblue",
                                     icon="bicycle",
                                     prefix="fa"),
                               tooltip=f"{station}: insgesamt {frequency_start[station] + frequency_end[station]}\
                                <br>(Ausleihe: {frequency_start[station]}, Rückgabe: {frequency_end[station]})"
                                    ).add_to(munich_map)
            
        # Füge die Stadtviertel als GeoJSON auf der Karte hinzu
        if st.session_state.map_config_months["show_city_districts"]:
            folium.GeoJson(
                city_districts,
                name="Stadtviertel",
                style_function=lambda feature: {
                    "fillColor": "lightblue",  # Füllfarbe der Stadtviertel
                    "color": "blue",
                    "weight": 3,
                    "opacity": 0.3,
                    "fillOpacity": 0.2
                }
            ).add_to(munich_map)

            # Füge den Stadtbereich als GeoJSON auf der Karte hinzu
        if st.session_state.map_config_months["show_city_area"]:
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

        # Weitere Infos
        # Berechnungen
        # Durchschnittliche Dauer
        avg_length = st.session_state.chosen_months["DURATION"].median().seconds // 60
        if avg_length > 60:
            avg_length_str = f"{avg_length // 60} Stunden, {avg_length%60} Minuten"
        else:
            avg_length_str = f"{avg_length} Minuten"

        # Ausleihen an Stationen im Stadtgebiet:
        rental_station_city_number = st.session_state.chosen_months[((st.session_state.chosen_months["RENTAL_IS_STATION"] == 1)\
                                            & (st.session_state.chosen_months["RENTAL_IS_CITY"] == 1))].shape[0]
        # Ausleihen an Stationen außerhalb des Stadtgebiets:
        rental_station_not_city_number = st.session_state.chosen_months[((st.session_state.chosen_months["RENTAL_IS_STATION"] == 1)\
                                            & (st.session_state.chosen_months["RENTAL_IS_CITY"] == 0))].shape[0]
        # Rückgaben an Stationen im Stadtgebiet:
        return_station_city_number = st.session_state.chosen_months[((st.session_state.chosen_months["RETURN_IS_STATION"] == 1)\
                                            & (st.session_state.chosen_months["RETURN_IS_CITY"] == 1))].shape[0]
        # Rückgaben an Stationen außerhalb des Stadtgebiets:
        return_station_not_city_number = st.session_state.chosen_months[((st.session_state.chosen_months["RENTAL_IS_STATION"] == 1)\
                                            & (st.session_state.chosen_months["RETURN_IS_CITY"] == 0))].shape[0]

        # Textausgabe
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Anzahl Fahrten:")
            st.write(f"Mittlere Fahrtenlänge:")
            # st.write(f"Mittlere Entfernung (Luftlinie):")
            st.write(f"Beliebtestes Startviertel:")
            st.write(f"Beliebtestes Zielviertel:")
            st.write(f"Stationsausleihen in-/außerhalb des Stadtgebiets:")
            st.write(f"Stationsrückgaben in-/außerhalb des Stadtgebiets:")
        with col2:
            st.write(f"{st.session_state.chosen_months.shape[0]}")
            st.write(f"{avg_length_str}")
            # st.write(f"{st.session_state.chosen_months["DISTANCE"].median():.1f} Kilometer")
            st.write(f"{st.session_state.chosen_months["CITY_DISTRICT_START"].mode()[0]}")
            st.write(f"{st.session_state.chosen_months["CITY_DISTRICT_END"].mode()[0]}")
            
            st.write(f"{rental_station_city_number} / {rental_station_not_city_number}")
            st.write(f"{return_station_city_number} / {return_station_not_city_number}")





# Wenn Tage ausgewertet werden sollen
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
        valid_day = False
    elif day_input_end - day_input_start >= timedelta(days=7):
        st.write("Es können nicht mehr als 7 Tage ausgewählt werden.")
        valid_day = False
    else:
        valid_day = True

    # Auswahl des Zeitraums
    daytime_input = st.slider("Wähle die Tageszeit:",
                            min_value=time(0),
                            max_value=time(23, 59, 59),
                            value=(time(0), time(23, 59, 59)),
                            step=timedelta(hours=1))

    # speichert gewünschten Zeitraum in einer Variablen
    # chosen_time = (datetime.combine(day_input_start, daytime_input[0]), datetime.combine(day_input_end, daytime_input[1]))
    if valid_day:

        st.write("Was soll auf der Karte angezeigt werden?")
        show_startpoints = st.checkbox("Startpunkte", value=st.session_state.map_config_days["show_startpoints"])
        show_endpoints = st.checkbox("Endpunkte", value=st.session_state.map_config_days["show_endpoints"])
        show_lines = st.checkbox("Strecke (Luftlinie)", value=st.session_state.map_config_days["show_lines"])
        show_city_districts = st.checkbox("Stadtviertel", value=st.session_state.map_config_days["show_city_districts"], key="districts_days")
        show_city_area = st.checkbox("Stadtbereich", value=st.session_state.map_config_days["show_city_area"],
                                    help="Bereich, in dem Fahrräder auch abseits von Stationen zurückgegeben werden können", key="area_days")


        if st.button("Hier klicken für Auswertung und Aktualisierung der Karte", key="map_days"):
            # Speichern des DataFrames im Session State
            st.session_state.chosen_days = df[(((df["STARTTIME"].dt.date >= day_input_start) & (df["STARTTIME"].dt.date <= day_input_end))\
                            | ((df["ENDTIME"].dt.date >= day_input_start) & (df["ENDTIME"].dt.date <= day_input_end)))\
                            & (((df["STARTTIME"].dt.time >= daytime_input[0]) & (df["STARTTIME"].dt.time <= daytime_input[1]))
                            | ((df["ENDTIME"].dt.time >= daytime_input[0]) & (df["ENDTIME"].dt.time <= daytime_input[1])))].dropna().copy()
            
            # Speichern der Checkbox-Werte im Session State
            st.session_state.map_config_days["show_startpoints"] = show_startpoints
            st.session_state.map_config_days["show_endpoints"] = show_endpoints
            st.session_state.map_config_days["show_lines"] = show_lines
            st.session_state.map_config_days["show_city_districts"] = show_city_districts
            st.session_state.map_config_days["show_city_area"] = show_city_area
            
            st.session_state.show_map = True
        
        # Zeige die Karte nur, wenn "show_map" True ist
        if st.session_state.show_map and st.session_state.chosen_days is not None:

            # Initialisieren der Karte
            map_center = [48.137154, 11.576124] # Munich city centre
            munich_map = folium.Map(location=map_center, zoom_start=12)
            
            for index, row in st.session_state.chosen_days.iterrows():
                start_coordinates = [row["STARTLAT"], row["STARTLON"]]
                end_coordinates = [row["ENDLAT"], row["ENDLON"]]

                # Hinzufügen der Startpunkte
                if st.session_state.map_config_days["show_startpoints"]:
                    folium.Circle(location=start_coordinates,
                                        color="purple",
                                        fill=True,
                                        fill_color="purple",
                                        fill_opacity=0.5
                                        ).add_to(munich_map)
                
                # Hinzufügen der Endpunkte
                if st.session_state.map_config_days["show_endpoints"]: 
                        folium.Circle(location=end_coordinates,
                                            radius=20,
                                            color="green",
                                            fill=True,
                                            fill_color="green",
                                            fill_opacity=0.5
                                            ).add_to(munich_map)
                
                # Hinzufügen einer Linie zwischen Start- und Rückgabeort
                if st.session_state.map_config_days["show_lines"]:
                        folium.PolyLine(locations=[start_coordinates, end_coordinates],
                                        color="grey",
                                        weight=2.5,
                                        opacity=0.3).add_to(munich_map)
                
            # Füge die Stadtviertel als GeoJSON auf der Karte hinzu
            if st.session_state.map_config_days["show_city_districts"]:
                folium.GeoJson(
                    city_districts,
                    name="Stadtviertel",
                    style_function=lambda feature: {
                        "fillColor": "lightblue",  # Füllfarbe der Stadtviertel
                        "color": "blue",
                        "weight": 3,
                        "opacity": 0.3,
                        "fillOpacity": 0.2
                    }
                ).add_to(munich_map)

                # Füge den Stadtbereich als GeoJSON auf der Karte hinzu
            if st.session_state.map_config_days["show_city_area"]:
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

            # Berechnung weiterer Informationen
            # Durchschnittliche Dauer
            avg_length = st.session_state.chosen_days["DURATION"].median().seconds // 60
            if avg_length > 60:
                avg_length_str = f"{avg_length // 60} Stunden, {avg_length%60} Minuten"
            else:
                avg_length_str = f"{avg_length} Minuten"
            # Ausleihen an Stationen im Stadtgebiet:
            rental_station_city_number = st.session_state.chosen_days[((st.session_state.chosen_days["RENTAL_IS_STATION"] == 1)\
                                             & (st.session_state.chosen_days["RENTAL_IS_CITY"] == 1))].shape[0]
            # Ausleihen an Stationen außerhalb des Stadtgebiets:
            rental_station_not_city_number = st.session_state.chosen_days[((st.session_state.chosen_days["RENTAL_IS_STATION"] == 1)\
                                             & (st.session_state.chosen_days["RENTAL_IS_CITY"] == 0))].shape[0]
            # Rückgaben an Stationen im Stadtgebiet:
            return_station_city_number = st.session_state.chosen_days[((st.session_state.chosen_days["RETURN_IS_STATION"] == 1)\
                                             & (st.session_state.chosen_days["RETURN_IS_CITY"] == 1))].shape[0]
            # Rückgaben an Stationen außerhalb des Stadtgebiets:
            return_station_not_city_number = st.session_state.chosen_days[((st.session_state.chosen_days["RENTAL_IS_STATION"] == 1)\
                                             & (st.session_state.chosen_days["RETURN_IS_CITY"] == 0))].shape[0]
            # Output
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Anzahl Fahrten:")
                st.write(f"Mittlere Fahrtenlänge:")
                # st.write(f"Mittlere Entfernung (Luftlinie):")
                st.write(f"Beliebtestes Startviertel:")
                st.write(f"Beliebtestes Zielviertel:")
                st.write(f"Stationsausleihen in-/außerhalb des Stadtgebiets:")
                st.write(f"Stationsrückgaben in-/außerhalb des Stadtgebiets:")
            with col2:
                st.write(f"{st.session_state.chosen_days.shape[0]}")
                st.write(f"{avg_length_str}")
                # st.write(f"{st.session_state.chosen_days["DISTANCE"].median():.1f} Kilometer")
                st.write(f"{st.session_state.chosen_days["CITY_DISTRICT_START"].mode()[0]}")
                st.write(f"{st.session_state.chosen_days["CITY_DISTRICT_END"].mode()[0]}")
                
                st.write(f"{rental_station_city_number} / {rental_station_not_city_number}")
                st.write(f"{return_station_city_number} / {return_station_not_city_number}")
