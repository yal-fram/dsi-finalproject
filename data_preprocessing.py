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
    

# Funktion zur Formatierung von IS_STATION


def format_is_station(value:str) -> int:
    """Removes Spaces and casts string into boolean integer, if necessary. Returns NA for empty string and all numbers, that are not 0 or 1.

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



## Filling Values

# Funktion zur Befüllung von NaN-Werten bei IS_STATION

