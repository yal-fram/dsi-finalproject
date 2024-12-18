import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from geopy.distance import geodesic

## Formatting Functions

# Funktion, um Leerzeichen zu entfernen


"""
Removes spaces from a string.

Parameters:
-string (str): The input value to process.

Returns:
str: A String without spaces.
"""
def remove_space(string: str) -> str:
    if isinstance(string, str):
        return string.strip()
    else:
        return string


# Funktion zur Formatierung von Datetime


"""
Converts STARTTIME and ENDTIME to datetime.

Parameters:
-df(pd.DataFrame): Input DataFrame.
-column_start(str): Name of column for start time.
-column_end(str): Name of column for end time.

Returns:
pd.DataFrame: Cleaned DataFrame with validated dates.
"""
def to_datetime(df, column_start="STARTTIME", column_end="ENDTIME"):
    #> datetime; ungültige Einträge: NaT 
    df[column_start] = pd.to_datetime(df[column_start], errors='coerce')
    df[column_end] = pd.to_datetime(df[column_end], errors='coerce')

    return df


# Funktion zur Entfernung ungültiger Daten
"""
Removes invalid (STARTTIME and ENDTIME)

Parameters:
-df(pd.DataFrame): Input DataFrame.
-column_start(str): Name of column for start time.
-column_end(str): Name of column for end time.

Returns:
pd.DataFrame: Cleaned DataFrame with validated dates.
"""
def remove_invalid_datetime(df, column_start="STARTTIME", column_end="ENDTIME"):
    #Entferne Zeilen, bei denen ENDTIME < STARTTIME oder einer von beiden NaT ist
    df = df[df[column_end] >= df[column_start]]
    df = df.dropna(subset=[column_start, column_end])

    return df


# Funktion zur Formatierung der Koordinaten


def format_coordinate(coordinate:str) -> float:
    """Takes unformatted longitudinal or latitudinal data and returns them as float. Sets un-float-able values to NA.

    Args:
        coordinate (str): longitudinal or latitudinal data, unformatted

    Returns:
        float: longitudinal or latitudinal data as float, or NA
    """
    if type(coordinate) == str:
        coordinate = coordinate.strip()
        if "," in coordinate:
            coordinate = coordinate.replace(",", ".")
    if coordinate == "":
        return pd.NA
    try:
        coordinate = float(coordinate)
    except ValueError:
        print(f"Formatierung nicht möglich, setze Wert auf NA: {coordinate}")
        coordinate = pd.NA
    
    return coordinate


# Plausibilitätsprüfung Koordinaten


def remove_invalid_latitudes(lat:float) -> float:
    """Sets invalid latitudes to NA. 'Invalid' is defined by spatial dimension of Munich and surroundings, i.e. max latitudinal data of the city's S-train network.

    Args:
        coordinate (float): latitudinal data, formatted as float

    Returns:
        float: nan for invalid coordinate value
    """
    if lat < 47.8 or lat > 48.5:
        return np.nan
    else:
        return lat


def remove_invalid_longitudes(lon:float) -> float:
    """Sets invalid longitudes to NA. 'Invalid' is defined by spatial dimension of Munich and surroundings, i.e. max longitudinal data of the city's S-train network.

    Args:
        coordinate (float): longitudinal data, formatted as float

    Returns:
        float: nan for invalid coordinate value
    """
    if lon < 11.1 or lon > 12:
        return np.nan
    else:
        return lon
    

# Combining coordinates functions
def handle_coordinates(df:pd.DataFrame) -> pd.DataFrame:
    """applies coordinate functions on whole DataFrame. Coordinates get formatted to floats, invalid and missing values are set to nan.

    Args:
        df (pd.DataFrame): DataFrame with STARTLAT, STARTLON, ENDLAT, ENDLON columns

    Returns:
        pd.DataFrame: modified DataFrame
    """
    df["STARTLAT"] = df["STARTLAT"].apply(format_coordinate)
    df["STARTLAT"] = df["STARTLAT"].apply(remove_invalid_latitudes)
    df["STARTLON"] = df["STARTLON"].apply(format_coordinate)
    df["STARTLON"] = df["STARTLON"].apply(remove_invalid_longitudes)
    df["ENDLAT"] = df["ENDLAT"].apply(format_coordinate)
    df["ENDLAT"] = df["ENDLAT"].apply(remove_invalid_latitudes)
    df["ENDLON"] = df["ENDLON"].apply(format_coordinate)
    df["ENDLON"] = df["ENDLON"].apply(remove_invalid_longitudes)

    return df


# Funktion zur Formatierung von IS_STATION


def format_is_station(value:str) -> int:
    """Removes spaces and casts string into boolean integer, if necessary.
    Returns NA for empty string and all numbers, that are not 0 or 1.

    Args:
        value (str): String with number in it or empty string

    Returns:
        int / nan: 0 or 1 if that number in string, NA if empty string or number not 0 or 1
    """
    if type(value) == str:
        if value.strip() == "":
            return pd.NA
        try:
            value = int(value.strip())
        except ValueError:
            print(f"Formatierung nicht möglich, setze Wert auf NA: {value}")
            value = 2
    if value not in [0, 1]:
        return pd.NA
    return value


# Combining is_station_functions

def handle_is_station(df:pd.DataFrame) -> pd.DataFrame:
    """Applies format_is_station function to whole DataFrame: Removes spaces and casts string into boolean integer, if necessary.
    Returns NA for empty string and all numbers, that are not 0 or 1. Casts values to integers.

    Args:
        df (pd.DataFrame): DataFrame with RENTAL_IS_STATION and RETURN_IS_STATION columns

    Returns:
        pd.DataFrame: modified DataFrame
    """
    df["RENTAL_IS_STATION"] = df["RENTAL_IS_STATION"].apply(format_is_station)
    df["RENTAL_IS_STATION"] = df["RENTAL_IS_STATION"].astype("Int64")
    df["RETURN_IS_STATION"] = df["RETURN_IS_STATION"].apply(format_is_station)
    df["RETURN_IS_STATION"] = df["RETURN_IS_STATION"].astype("Int64")

    return df

## Filling Values

# Funktion zur Befüllung von NaN-Werten bei IS_STATION

def fill_is_station_values(df:pd.DataFrame) -> pd.DataFrame:
    """fills missing is_station-values with 0 or 1, depending on the occurrence or non-occurrence of a related station name.
    If no station name is given,
    is_station is set to 0. If a station name is given, is_station is set to 1.

    Args:
        df (pd.DataFrame): DataFrame with columns RENTAL_IS_STATION, RENTAL_STATION_NAME and RETURN_IS_STATION, RETURN_STATION_NAME

    Returns:
        pd.DataFrame: modified DataFrame with filled is_station-values
    """
    df.loc[(df["RENTAL_IS_STATION"].isna() & (df["RENTAL_STATION_NAME"] == "")), "RENTAL_IS_STATION"] = 0
    df.loc[(df["RENTAL_IS_STATION"].isna() & (df["RENTAL_STATION_NAME"] != "")), "RENTAL_IS_STATION"] = 1
    df.loc[(df["RETURN_IS_STATION"].isna() & (df["RETURN_STATION_NAME"] == "")), "RETURN_IS_STATION"] = 0
    df.loc[(df["RETURN_IS_STATION"].isna() & (df["RETURN_STATION_NAME"] != "")), "RETURN_IS_STATION"] = 1

    return df


# Hinzufügen von Stadtvierteln, in denen die Start- und Endpunkte jeweils liegen
def add_city_district(df:pd.DataFrame) -> pd.DataFrame:
    """Takes DataFrame, adds city districts for start and end in new columns

    Args:
        df (pd.DataFrame): DataFrame with STARTLAT, STARTLON, ENDLAT, ENDLON data

    Returns:
        pd.DataFrame: modified DataFrame with added CITY_DISTRICT_START and CITY_DISTRICT_END columns
    """
    # Einlesen der 
    city_districts = gpd.read_file("neighbourhoods.geojson") # Stadtviertel-geojson, von AirBNB

    # Erstelle zwei Geometrien für Start- und Endpunkte
    geo_start = gpd.points_from_xy(x=df["STARTLON"], crs="EPSG:4326", y=df["STARTLAT"])
    geo_end = gpd.points_from_xy(x=df["ENDLON"], crs="EPSG:4326", y=df["ENDLAT"])

    # Erstelle zwei GeoDataFrames (ein GeoDataFrame nimmt immer nur eine Geometrie an) für Start- und Endpunkte
    gdf_start = gpd.GeoDataFrame(df[["STARTLAT", "STARTLON"]], geometry=geo_start)
    gdf_end = gpd.GeoDataFrame(df[["STARTLAT", "STARTLON"]], geometry=geo_end)

    # Führe einen Spatial Join durch, um das Stadtviertel für den Endpunkt zu finden
    gdf_start_join = gpd.sjoin(gdf_start, city_districts[["geometry", "neighbourhood"]], how="left", predicate="within", rsuffix="_start")
    gdf_end_join = gpd.sjoin(gdf_end, city_districts[["geometry", "neighbourhood"]], how="left", predicate="within", rsuffix="_end")

    # Füge die Resultatspalten dem DataFrame hinzu
    df["CITY_DISTRICT_START"] = gdf_start_join["neighbourhood"]
    df["CITY_DISTRICT_END"] = gdf_end_join["neighbourhood"]

    return df


## Funktion, um zu bestimmen, ob Koordinaten in Stadtgebiet liegen

def add_city_status(df:pd.DataFrame) -> pd.DataFrame:
    """Checks, if start or end is in city area or not. Takes whole dataframe and returns modified dataframe. Adds two columns, for RENTAL and RETURN stations.
    City area: bikes can be returned anywhere, not just at a station.

    Args:
        df (pd.DataFrame): dataframe with STARTLAT and STARTLON, ENDLAT and ENDLON columns.

    Returns:
        pd.DataFrame: modified dataframe with two additional columns for city status
    """
    city_area = gpd.read_file("city_area.geojson") # city-area-geojson, selbst erstellt auf geojson.io, per Augenmaß anhand der in der MVGO-App ersichtlichen Ränder der city area

    # Erstelle zwei Geometrien für Start- und Endpunkte
    geo_start = gpd.points_from_xy(x=df["STARTLON"], crs="EPSG:4326", y=df["STARTLAT"])
    geo_end = gpd.points_from_xy(x=df["ENDLON"], crs="EPSG:4326", y=df["ENDLAT"])

    # Erstelle zwei GeoDataFrames (ein GeoDataFrame nimmt immer nur eine Geometrie an) für Start- und Endpunkte
    gdf_start = gpd.GeoDataFrame(df[["STARTLAT", "STARTLON"]], geometry=geo_start)
    gdf_end = gpd.GeoDataFrame(df[["STARTLAT", "STARTLON"]], geometry=geo_end)

    # Casten der Geometrie in synchrones Format
    gdf_city_start = gdf_start.to_crs(epsg=4326)
    gdf_city_end = gdf_end.to_crs(epsg=4326)

    # Erstellen eines Shapely Polygons
    polygon = shape(city_area["geometry"][0])

    # Prüfen, ob Punkte in Polygon liegen, speichern in neuer Spalte
    gdf_city_start["RENTAL_IS_CITY"] = gdf_start["geometry"].within(polygon)
    gdf_city_end["RETURN_IS_CITY"] = gdf_end["geometry"].within(polygon)
    gdf_city_start.head()

    # Hinzufügen der Resultatspalten zu DataFrame
    df["RENTAL_IS_CITY"] = gdf_city_start["RENTAL_IS_CITY"]
    df["RETURN_IS_CITY"] = gdf_city_end["RETURN_IS_CITY"]

    # Wahrheitswerte zu Integers umcasten
    df["RENTAL_IS_CITY"] = df["RENTAL_IS_CITY"].astype("Int64")
    df["RETURN_IS_CITY"] = df["RETURN_IS_CITY"].astype("Int64") 

    return df


# Wird zur Berechnung der Distanzen benötigt:
def calculate_geodesic(row):
    """Calculates geodesic distance for two Points. Takes DataFrame row as argument and returns distance in kilometres. To be applied with df.apply.

    Args:
        row: row of Pandas DataFrame

    Returns:
        distance: geodesic distance in kilometres
    """
    start_point = (row["STARTLAT"], row["STARTLON"])  # erst Breitengrad, dann Längengrad
    end_point = (row["ENDLAT"], row["ENDLON"])
    
    return geodesic(start_point, end_point).kilometers


def calculate_distance(df:pd.DataFrame) -> pd.DataFrame:
    """Takes DataFrame with two GeoPoints and calculates distance between them in metres, convertes into kilometres. Returns modified DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with STARTLAT and STARTLON, ENDLAT and ENDLON columns

    Returns:
        pd.DataFrame: Modified DataFrame with added DISTANCE column in kilometres
    """
    df['DISTANCE'] = df.apply(calculate_geodesic, axis=1)

    return df