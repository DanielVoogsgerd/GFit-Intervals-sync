#!/usr/bin/env python
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

import json

from datetime import datetime, timedelta, date

import logging as log
import argparse
import configparser

import numpy as np

from typing import Iterable

from api import API

from collections import defaultdict

import os
import sys

from google_fit_api import GoogleFitAPI

# Constants
APPLICATION_NAME = "Fit_Intervals_Sync"

# XDG Home Directories
CONFIG_DIR = os.path.expanduser(f"~/.config/{APPLICATION_NAME}")
CACHE_DIR = os.path.expanduser(f"~/.cache/{APPLICATION_NAME}")
STORAGE_DIR = os.path.expanduser(f"~/.local/share/{APPLICATION_NAME}")

USER_CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, "users.conf")
USER_CONFIG_ATHLETE_ID_FIELD = "intervals_user_id"
USER_CONFIG_API_KEY_FIELD = "intervals_api_key"

USER_GOOGLE_FIT_TOKEN_PATH_FORMAT = STORAGE_DIR + "/{username}-google-api-token.json"

GFIT_CREDENTIALS_PATH = os.path.join(CONFIG_DIR, "credentials.json")

# If modifying these scopes, delete the file token.json.
SCOPES = [
    f"https://www.googleapis.com/auth/fitness.{scope}.read"
    for scope in ["heart_rate", "sleep", "body", "blood_pressure"]
]

DATA_SOURCES = {
    "heart_rate": "derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm",
    "weight": "derived:com.google.weight:com.google.android.gms:merge_weight",
}


DAYS_TO_COMPARE = 30

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--verbose', '-v', action='count', default=0)

    subparsers = parser.add_subparsers(title="Commands")

    add_account_command = subparsers.add_parser("add_account")
    add_account_command.set_defaults(func=add_account)
    add_account_command.add_argument("username")
    add_account_command.add_argument("--intervals-athlete-id")

    run_command = subparsers.add_parser("run")
    run_command.set_defaults(func=run)

    args = parser.parse_args()

    if "func" not in args:
        parser.print_help()
        sys.exit(1)


    if args.debug or args.verbose >= 2:
        log.basicConfig(level=log.DEBUG)
    elif args.verbose >= 1:
        log.basicConfig(level=log.INFO)

    args.func(args)


def add_account(args):
    user_config = parse_config(USER_CONFIG_FILE_PATH)

    if args.username in user_config.sections():
        print(f"Username {args.username} already exists")
        sys.exit(1)

    if args.intervals_athlete_id is not None:
        athlete_id = args.intervals_athlete_id
    else:
        athlete_id = input("Enter Intervals Athlete ID: ")

    api_key = input("Enter intervals API key: ")

    assert API.validate_athlete_id(athlete_id)
    assert API.validate_api_key(api_key)

    user_config.add_section(args.username)
    user_config.set(args.username, USER_CONFIG_ATHLETE_ID_FIELD, athlete_id)
    user_config.set(args.username, USER_CONFIG_API_KEY_FIELD, api_key)

    with open(USER_CONFIG_FILE_PATH, 'w') as user_config_file:
        user_config.write(user_config_file)

    print("Requesting access from Google Fit")

    token_path = USER_GOOGLE_FIT_TOKEN_PATH_FORMAT.format(username=args.username)
    get_credentials(args.username, token_path)


def run(args):
    ensure_directory(CONFIG_DIR)
    ensure_directory(STORAGE_DIR)

    user_config = parse_config(USER_CONFIG_FILE_PATH)

    users = user_config.sections()

    if len(users) == 0:
        print("No users found. Check --help how to add an account.")

    for user in users:
        user_info = user_config[user]
        user_id = user_info[USER_CONFIG_ATHLETE_ID_FIELD]
        api_key = user_info[USER_CONFIG_API_KEY_FIELD]

        try:
            sync(user, user_id, api_key)
        except:
            log.warning(f"Something went wrong for user: {user}, skipping")


def sync(user, user_id, api_key):
    user_google_fit_token_path = USER_GOOGLE_FIT_TOKEN_PATH_FORMAT.format(username=user)
    intervals_api = API(user_id, api_key)

    data = intervals_api.wellness_csv.get(oldest=date.today() - timedelta(days=DAYS_TO_COMPARE))

    data_to_update = defaultdict(dict)

    creds = get_credentials(user, user_google_fit_token_path)
    gfit = GoogleFitAPI(creds)

    # Sleep / Night HR
    missing_resting_hr_dates = set(date_from_iso_vec(data[data['restingHR'].isna()]['date']))
    missing_average_hr_dates = set(date_from_iso_vec(data[data['avgSleepingHR'].isna()]['date']))
    missing_sleep_dates = set(date_from_iso_vec(data[data['sleepSecs'].isna()]['date']))
    combined_sleep_hr = missing_sleep_dates | missing_resting_hr_dates | missing_average_hr_dates | set((date.today(),))

    if len(combined_sleep_hr) > 0:
        sleep_sessions = gfit.get_sleep_sessions(min(combined_sleep_hr), max(combined_sleep_hr))

        log.info(f"Received {len(sleep_sessions)} from the Google API")

        for sleep_session in sleep_sessions:
            if sleep_session.date in missing_sleep_dates or sleep_session.date == date.today():
                data_to_update[sleep_session.date]['sleepSecs'] = sleep_session.asleep_duration.seconds

            if sleep_session.date in missing_resting_hr_dates or sleep_session.date == date.today():
                hr_values = gfit.get_hr_values(sleep_session.start_time, sleep_session.end_time)
                data_to_update[sleep_session.date]['avgSleepingHR'] = int(np.round(np.mean(hr_values)))
                data_to_update[sleep_session.date]['restingHR'] = int(min(hr_values))

    # Weight
    missing_weight_dates = set(date_from_iso_vec(data[data['weight'].isna()]['date']))
    if len(missing_weight_dates) > 0:
        weight_values = gfit.get_daily_weight(min(missing_weight_dates), max(missing_weight_dates))

        for weight_date in missing_weight_dates:
            if weight_date in weight_values:
                data_to_update[weight_date]['weight'] = weight_values[weight_date][0]

    # Blood pressure
    missing_systolic_dates = set(date_from_iso_vec(data[data['systolic'].isna()]['date']))
    missing_diastolic_dates = set(date_from_iso_vec(data[data['diastolic'].isna()]['date']))
    missing_blood_pressure_dates = missing_systolic_dates | missing_diastolic_dates

    if len(missing_blood_pressure_dates) > 0:
        blood_pressure_by_date = gfit.get_daily_blood_pressure(min(missing_blood_pressure_dates), max(missing_blood_pressure_dates))
        for blood_pressure_date in missing_blood_pressure_dates:
            if blood_pressure_date in blood_pressure_by_date:
                blood_pressures = blood_pressure_by_date[blood_pressure_date]
                blood_pressure = blood_pressures[0]
                data_to_update[blood_pressure_date]['systolic'] = blood_pressure.systolic
                data_to_update[blood_pressure_date]['diastolic'] = blood_pressure.diastolic


    for data_date, values in data_to_update.items():
        print(f"Updating wellness data for user: {user}, for date: {data_date}")
        log.debug(f"Setting the following values: {values}")
        data = intervals_api.wellness.update(data_date, values)


def date_from_iso_vec(iso_format_dates: Iterable[str]):
    return (date.fromisoformat(str_date) for str_date in iso_format_dates)


def parse_config(config_location):
    config = configparser.ConfigParser()
    config.read(config_location)
    return config


def dates_to_string(dates: Iterable):
    dates = (str(date) for date in dates)
    return ", ".join(dates)


def get_credentials(user: str, token_path: str):
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(GFIT_CREDENTIALS_PATH, SCOPES, redirect_uri="urn:ietf:wg:oauth:2.0:oob")
            # print(f"Need Google FIT authorization for user: {user}.")
            auth_url, _ = flow.authorization_url(prompt='consent')
            creds = flow.run_console(f"Need Google Fit authorization for user: {user}. Go the the following URL: {auth_url}.")
            # creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    return creds


def ensure_directory(directory_path):
    """Ensure needed directories exist."""
    if not os.path.exists(directory_path):
        log.info(f'Directory: "{directory_path}" does not exist, creating it.')
        os.mkdir(directory_path)

if __name__ == "__main__":
    main()
