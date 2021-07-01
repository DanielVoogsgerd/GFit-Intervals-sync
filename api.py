import requests
import pandas as pd

from io import StringIO
import datetime
from typing import Tuple, Set

import logging

import re

API_URL = "https://intervals.icu"

class API:
    """A wrapper for all the Endpoints in the API."""

    def __init__(self, athlete_id, api_key):
        assert self.validate_athlete_id(athlete_id)
        assert self.validate_api_key(api_key)

        self.athlete_id = athlete_id
        self.api_key = api_key

    @property
    def events(self):
        return EventsEndpoint(self.athlete_id, self.api_key)

    @property
    def wellness(self):
        return WellnessEndpoint(self.athlete_id, self.api_key)

    @property
    def calendar(self):
        return CalendarEndpoint(self.athlete_id, self.api_key)

    @property
    def wellness_csv(self):
        return WellnessCSVEndpoint(self.athlete_id, self.api_key)

    @property
    def activities_csv(self):
        return ActivitiesCSVEndpoint(self.athlete_id, self.api_key)

    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        return bool(re.match(r'\w{24}', api_key))

    @staticmethod
    def validate_athlete_id(athlete_id: str) -> bool:
        return bool(re.match(r'i\d+', athlete_id))


class Endpoint:
    endpoint_url: str
    auth: Tuple[str, str]
    athlete_id: str

    def __init__(self, athlete_id, api_key):
        """Initialize endpoint."""
        self.auth = ("API_KEY", api_key)
        self.athlete_id = athlete_id

    def get_request(self, url, query_parameters=None):
        """Perform JSON requests on a certain endpoint."""
        r = requests.get(
            url.format(athlete_id=self.athlete_id),
            auth=self.auth,
            params=query_parameters,
        )

        logging.debug(f"Sending GET request to {r.url}")
        return r.json()

    def get_content_request(self, url, query_parameters=None):
        """Perform requests on a certain endpoint as text."""
        r = requests.get(
            url.format(athlete_id=self.athlete_id),
            auth=self.auth,
            params=query_parameters,
        )

        logging.debug(f"Sending GET request to {r.url}")
        return r.text

    def put_request(self, url, data, query_parameters=None):
        """Perform a put request on a certain endpoint."""
        r = requests.put(
            url.format(athlete_id=self.athlete_id),
            auth=self.auth,
            params=query_parameters,
            json=data
        )

        logging.debug(f"Sending PUT request to {r.url}")
        return r.json()

    def delete_request(self, url, query_parameters=None):
        """Perform a put request on a certain endpoint."""
        r = requests.put(
            url.format(athlete_id=self.athlete_id),
            auth=self.auth,
            params=query_parameters,
            json=data
        )

        logging.debug(f"Sending DELETE request to {r.url}")
        return r.json()


class CSVEndpoint(Endpoint):
    def get_request_csv(self, url, query_parameters=None):
        r = requests.get(
            url.format(athlete_id=self.athlete_id) + ".csv",
            auth=self.auth,
            params=query_parameters,
        )

        logging.debug(f"Sending GET request to {r.url}")

        response_content = r.text
        response_file = StringIO(response_content)
        return pd.read_csv(response_file)


class CalendarEndpoint(Endpoint):
    endpoint_url = API_URL + "/api/v1/athlete/{athlete_id}/calendars"

    def get(self):
        """Retreive a list of all calendars."""
        return self.get_request(self.endpoint_url)


class EventsEndpoint(Endpoint):
    endpoint_url = API_URL + "/api/v1/athlete/{athlete_id}/events"

    def list(self, oldest: datetime.date, newest: datetime.date):
        """List all the events between two dates."""
        params = {}

        assert isinstance(oldest, datetime.date)
        assert isinstance(newest, datetime.date)

        params["oldest"] = oldest.isoformat()
        params["newest"] = newest.isoformat()

        return self.get_request(self.endpoint_url, query_parameters=params)

    def create(self, event_data):
        raise NotImplementedError

    def update(self, event_id, event_data):
        raise NotImplementedError

    def delete(self, event_id: int, event_data):
        """Untested."""
        return self.delete_request(self.endpoint_url + "/" + str(event_id))

    def download(self, event_id, workout_file_type):
        raise NotImplementedError

    def create_from_workout_file(self, workout_file):
        raise NotImplementedError


class ActivitiesCSVEndpoint(CSVEndpoint):
    """Interact with the activities in bulk.

    The activities CSV endpoint is an easy way to retreive all activities in
    bulk. The normal activities API might be easier and/or more complete, but
    will require more requests when retrieving multiple activities.
    """

    endpoint_url = API_URL + "/api/v1/athlete/{athlete_id}/activities"

    def get(self):
        """Retrieve multiple activities."""
        return self.get_request_csv(
            self.endpoint_url.format(athlete_id=self.athlete_id)
        )


class WellnessEndpoint(Endpoint):
    endpoint_url = API_URL + "/api/v1/athlete/{athlete_id}/wellness"

    def get(self, date: datetime.date):
        """Get wellness info for a certain date."""
        assert isinstance(date, datetime.date)
        return self.get_request(self.endpoint_url + "/" + date.isoformat())

    def update(self, date: datetime.date, data):
        """Update wellness info for a certain date."""
        assert isinstance(date, datetime.date)
        return self.put_request(self.endpoint_url + "/" + date.isoformat(), data)


class WellnessCSVEndpoint(CSVEndpoint):
    """Interact with the wellness data in bulk.

    The wellness csv endpoint is an easy way to retreive or update the wellness
    data in bulk.  The normal wellness API might be easier for indiviual
    records, but this one will use less requests when updating a lot of data.
    """

    endpoint_url = API_URL + "/api/v1/athlete/{athlete_id}/wellness"

    def __init__(self, athlete, auth_key):
        super().__init__(athlete, auth_key)

    def get(
        self,
        oldest: datetime.date = None,
        newest: datetime.date = None,
        cols: Set[str] = None,
        **kwargs
    ):
        params = {}

        if newest is not None:
            assert isinstance(newest, datetime.date)
            params["newest"] = newest.isoformat()

        if oldest is not None:
            assert isinstance(oldest, datetime.date)
            params["oldest"] = oldest.isoformat()

        if cols is not None:
            params["cols"] = ",".join(cols)

        return self.get_request_csv(
            self.endpoint_url.format(**kwargs, athlete_id=self.athlete_id),
            query_parameters=params,
        )

    def update(self, dataframe):
        """Update multiple wellness entries"""
        raise NotImplementedError
