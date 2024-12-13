import numpy as np
import pandas as pd

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
    df['RENTAL_IS_STATION'] = df['RENTAL_IS_STATION'].astype('Int64')
    df["RETURN_IS_STATION"] = df["RETURN_IS_STATION"].apply(format_is_station)
    df['RETURN_IS_STATION'] = df['RETURN_IS_STATION'].astype('Int64')

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


## Funktion, um zu bestimmen, ob Koordinaten in Stadtgebiet liegen

def set_station_city_status(df:pd.DataFrame) -> pd.DataFrame:
    """Checks, if station is in city area or not. Takes whole dataframe and returns modified dataframe. Adds two columns, for RENTAL and RETURN stations.
    City area: bikes can be returned anywhere, not just at a station.

    Args:
        df (pd.DataFrame): dataframe with RENTAL_STATION_NAME, STARTLAT and STARTLON; RETURN_STATION_NAME, ENDLAT and ENDLON columns.

    Returns:
        pd.DataFrame: modified dataframe with two additional columns for stations' city status
    """
    # Set value to 1, where station in city
    df.loc[(df["RENTAL_STATION_NAME"] != '')\
      & (48.08470 < df["STARTLAT"]) & (df["STARTLAT"] < 48.19058) \
      & (11.50238 < df["STARTLON"]) & (df["STARTLON"] < 11.65070),\
      "RENTAL_STATION_IS_CITY"] = 1

    # Set value to 0, where station not in city
    df.loc[(df["RENTAL_STATION_NAME"] != '')\
      & df["RENTAL_STATION_IS_CITY"].isnull(),
      "RENTAL_STATION_IS_CITY"] = 0
    
    # repeat for RETURN stations
    df.loc[(df["RETURN_STATION_NAME"] != '')\
      & (48.08470 < df["ENDLAT"]) & (df["ENDLAT"] < 48.19058) \
      & (11.50238 < df["ENDLON"]) & (df["ENDLON"] < 11.65070),\
      "RETURN_STATION_IS_CITY"] = 1
    
    df.loc[(df["RETURN_STATION_NAME"] != '')\
      & df["RETURN_STATION_IS_CITY"].isnull(),
      "RETURN_STATION_IS_CITY"] = 0
    
    # handle exceptional stations, which are on the wrong side of the line (dvs. inside given coordinates, but not city area and vice versa)
    outside_stations = ["Bahnhof Perlach", "Neuperlach Süd", "Kreillerstraße", "Quiddestraße", "Plettstraße", "Karl-Marx-Zentrum"]
    inside_stations = ["Botanischer Garten", "Amalienburgstraße", "Großhesseloher Brücke", "BR Studio München-Freimann", "Am Bahnhof Unterföhring"]

    # df.loc[(df["RENTAL_STATION_NAME"].any(outside_stations)), "RENTAL_STATION_IS_CITY"] = 0
    # df.loc[(df["RETURN_STATION_NAME"] in inside_stations), "RETURN_STATION_IS_CITY"] = 0
    # df.loc[(df["RENTAL_STATION_NAME"] in outside_stations), "RENTAL_STATION_IS_CITY"] = 1
    # df.loc[(df["RETURN_STATION_NAME"] in inside_stations), "RETURN_STATION_IS_CITY"] = 1    

    return df