import requests, sys

import pandas as pd
from datetime import datetime, timedelta
from tzfpy import get_tz

sys.path.append('../emit-retr/')
from config import CONFIG


def get_emission_rate(plant_info, obs_time, lead_time):
    with open(CONFIG['campd_key'], 'r', encoding='utf-8') as file:
        API_KEY = file.read()

    laglead = timedelta(days=1)

    parameters = {
        'api_key':    API_KEY,
        'beginDate':  (obs_time-laglead).strftime('%Y-%m-%d'),
        'endDate':    (obs_time+laglead).strftime('%Y-%m-%d'),
        'facilityId': plant_info['Fac_ID']
    }

    endpoint_url = "https://api.epa.gov/easey/streaming-services/emissions/apportioned/hourly/by-facility"
    streamingResponse = requests.get(endpoint_url, params=parameters)

    # printing the response error message if the response is not successful
    if (int(streamingResponse.status_code) > 399):
        sys.exit(f"Error message: {streamingResponse.json()['error']}")

    resp_df = pd.DataFrame(streamingResponse.json())

    # combine date + hour into a single datetime column
    resp_df['datetime'] = pd.to_datetime(resp_df['date']) + pd.to_timedelta(resp_df['hour'], unit='h')

    localtime = get_tz(plant_info['LON'], plant_info['LAT'])
    
    resp_df['datetime_utc'] = (
        resp_df['datetime']
        .dt.tz_localize(localtime)     # assign the correct local timezone
        .dt.tz_convert('UTC')          # convert to UTC
    )


    mask = (resp_df['datetime_utc'] >= obs_time-lead_time) & (resp_df['datetime_utc'] <= obs_time)
    avg_nox = resp_df.loc[mask, 'noxMass'].mean()
    return avg_nox * 0.453592/3600 # Convert from lbs/hr to kg/s


def get_emissions(plant_info, obs_time, lead_time):
    with open(CONFIG['campd_key'], 'r', encoding='utf-8') as file:
        API_KEY = file.read()

    laglead = timedelta(days=1)

    parameters = {
        'api_key':    API_KEY,
        'beginDate':  (obs_time-laglead).strftime('%Y-%m-%d'),
        'endDate':    (obs_time+laglead).strftime('%Y-%m-%d'),
        'facilityId': plant_info['Fac_ID']
    }

    endpoint_url = "https://api.epa.gov/easey/streaming-services/emissions/apportioned/hourly/by-facility"
    streamingResponse = requests.get(endpoint_url, params=parameters)

    # printing the response error message if the response is not successful
    if (int(streamingResponse.status_code) > 399):
        sys.exit(f"Error message: {streamingResponse.json()['error']}")

    resp_df = pd.DataFrame(streamingResponse.json())

    # combine date + hour into a single datetime column
    resp_df['datetime'] = pd.to_datetime(resp_df['date']) + pd.to_timedelta(resp_df['hour'], unit='h')

    localtime = get_tz(plant_info['LON'], plant_info['LAT'])
    
    resp_df['datetime_utc'] = (
        resp_df['datetime']
        .dt.tz_localize(localtime)     # assign the correct local timezone
        .dt.tz_convert('UTC')          # convert to UTC
    )


    mask = (resp_df['datetime_utc'] >= obs_time-lead_time) & (resp_df['datetime_utc'] <= obs_time)
    # avg_nox = resp_df.loc[mask, 'noxMass'].mean()
    return resp_df.loc[mask, 'datetime_utc'], resp_df.loc[mask, 'noxMass'] * 0.453592/3600 # Convert from lbs/hr to kg/s