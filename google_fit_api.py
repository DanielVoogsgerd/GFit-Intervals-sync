from googleapiclient.discovery import build

from dataclasses import dataclass

from datetime import datetime, timedelta, date, time

from typing import Tuple, Dict, List

from enum import Enum

from typing import List, Sequence

import numpy as np

from collections import defaultdict


class SleepType(Enum):
    Awake = 1
    Sleep = 2
    Out_of_bed = 3
    Light_Sleep = 4
    Deep_Sleep = 5
    REM = 6


ASLEEP_TYPES = (SleepType.Sleep, SleepType.Light_Sleep, SleepType.Deep_Sleep, SleepType.REM)


@dataclass
class SleepSegment:
    start_time: datetime
    end_time: datetime

    sleep_type: SleepType

    @classmethod
    def from_dict(cls, data):
        """Generate Sleep segment class from dict input.

        Example dict:
        {'dataTypeName': 'com.google.sleep.segment',
            'endTimeNanos': '1625621640000000000',
            'modifiedTimeMillis': '1625639165210',
            'originDataSourceId': 'raw:com.google.sleep.segment:24:0B:99ff173c:Notify '
                                  'for Mi Band - sleep',
            'startTimeNanos': '1625620620000000000',
            'value': [{'intVal': 6, 'mapVal': []}]},
        """
        start_time = datetime.fromtimestamp(int(data['startTimeNanos']) / 1e9)
        end_time = datetime.fromtimestamp(int(data['endTimeNanos']) / 1e9)
        sleep_type = SleepType(data['value'][0]['intVal'])

        return cls(start_time, end_time, sleep_type)

    @property
    def duration(self):
        return self.end_time - self.start_time


@dataclass
class SleepSession:
    start_time: datetime
    end_time: datetime
    sleep_segments: List[SleepSegment]

    @classmethod
    def from_dict(cls, data, sleep_segments=None):
        """Generate Sleep session class from dict input.

        Example dict:
            {'activityType': 72,
            'application': {'packageName': 'com.mc.miband1'},
            'description': '',
            'endTimeMillis': '1625634840000',
            'id': '1625610960000',
            'modifiedTimeMillis': '1625639160783',
            'name': '00:36 - 07:14',
            'startTimeMillis': '1625610960000'}
        """
        assert data['activityType'] == 72

        start_time = datetime.fromtimestamp(int(data['startTimeMillis']) / 1e3)
        end_time = datetime.fromtimestamp(int(data['endTimeMillis']) / 1e3)

        if sleep_segments is None:
            sleep_segments = []

        return cls(start_time, end_time, sleep_segments)

    @property
    def asleep_duration(self):
        return self.sleep_type_duration(ASLEEP_TYPES)

    @property
    def awake_duration(self):
        return self.sleep_type_duration((SleepType.Awake,))

    def sleep_type_duration(self, sleep_types: Sequence[SleepType]):
        if self.sleep_segments is None:
            return None

        duration = timedelta()
        for segment in self.sleep_segments:
            if segment.sleep_type in sleep_types:
                duration += segment.duration

        return duration

    @property
    def date(self) -> date:
        return self.end_time.date()


@dataclass
class BloodPressure:
    time: datetime
    systolic: int
    diastolic: int

    @classmethod
    def from_dict(cls, data):
        """Create blood pressure class from dictionary.

        Example dict:
            {'dataTypeName': 'com.google.blood_pressure',
            'endTimeNanos': '1623052800000000000',
            'modifiedTimeMillis': '1623749603745',
            'originDataSourceId': 'raw:com.google.blood_pressure:com.google.android.apps.fitness:user_input',
            'startTimeNanos': '1623052800000000000',
            'value': [{'fpVal': 110, 'mapVal': []},
            {'fpVal': 70, 'mapVal': []},
            {'mapVal': []},
            {'mapVal': []}]}
        """
        time = datetime.fromtimestamp(int(data['startTimeNanos']) / 1e9)
        systolic = data['value'][0]['fpVal']
        diastolic = data['value'][1]['fpVal']

        return cls(time, systolic, diastolic)


class DataSource(Enum):
    BLOOD_PRESSURE = "derived:com.google.blood_pressure:com.google.android.gms:merged"
    HEART_RATE = "derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm"
    RESTING_HEART_RATE = "derived:com.google.heart_rate.bpm:com.google.android.gms:resting_heart_rate<-merge_heart_rate_bpm"
    SLEEP_SEGMENT = "derived:com.google.sleep.segment:com.google.android.gms:merged"
    WEIGHT = "derived:com.google.weight:com.google.android.gms:merge_weight"


class GoogleFitAPI:
    def __init__(self, credentials):
        self.fitness_service = build("fitness", "v1", credentials=credentials)

    def get_datasource(self, data_source: DataSource, start_time: datetime, end_time: datetime):
        dataset = self._get_dataset(start_time, end_time)

        entries = (
            self.fitness_service.users()
            .dataSources()
            .datasets()
            .get(userId="me", dataSourceId=data_source.value, datasetId=dataset)
            .execute()
        )["point"]

        return entries

    def get_sessions(self, start_time: datetime, end_time: datetime):
        sessions = (
            self.fitness_service.users()
            .sessions()
            .list(
                userId="me",
                # fields="session",
                startTime=start_time.isoformat("T") + "Z",
                endTime=end_time.isoformat("T") + "Z",
            )
            .execute()
        )['session']

        return sessions

    def _get_dataset(self, start_time, end_time):
        start_nanos = start_time.timestamp() * 1e9
        end_nanos = end_time.timestamp() * 1e9

        return f"{start_nanos:.0f}-{end_nanos:.0f}"

    def get_hr_values(self, start_time: datetime, end_time):
        hr_entries = self.get_datasource(DataSource.RESTING_HEART_RATE, start_time, end_time)
        hr_values = np.array([entry["value"][0]["fpVal"] for entry in hr_entries])

        return hr_values

    def get_sleep_sessions(self, start_date: date, end_date: date, segments=False) -> Tuple[SleepSession, ...]:
        sleep_day_border = time.fromisoformat("15:00:00")

        start_time = datetime.combine(start_date, sleep_day_border) - timedelta(days=1)
        end_time = datetime.combine(end_date, sleep_day_border)

        sleep_sessions = tuple((SleepSession.from_dict(sleep_session_data) for sleep_session_data in self.get_sessions(start_time, end_time)))

        if len(sleep_sessions) == 0:
            return tuple()

        first_session = sleep_sessions[0]
        last_session = sleep_sessions[-1]

        if not segments:
            return sleep_sessions

        segment_start_time = first_session.start_time
        segment_end_time = last_session.end_time

        sleep_segments = self.get_sleep_segments(segment_start_time, segment_end_time)

        session_index = 0

        for sleep_segment in sleep_segments:
            while session_index < len(sleep_sessions):
                sleep_session = sleep_sessions[session_index]

                if sleep_session.start_time <= sleep_segment.start_time <= sleep_session.end_time:
                    sleep_session.sleep_segments.append(sleep_segment)
                    break
                else:
                    session_index += 1
                    continue
            else:
                raise Exception("Illegal state")

        for sleep_session in sleep_sessions:
            if len(sleep_session.sleep_segments) == 0:
                sleep_session.sleep_segments.append(SleepSegment(sleep_session.start_time, sleep_session.end_time, SleepType.Sleep))
            else:
                if sleep_session.sleep_segments[0].start_time > sleep_session.start_time:
                    sleep_session.sleep_segments.insert(0, SleepSegment(sleep_session.start_time, sleep_session.sleep_segments[0].start_time, SleepType.Sleep))

                if sleep_session.sleep_segments[-1].end_time < sleep_session.end_time:
                    sleep_session.sleep_segments.append(SleepSegment(sleep_session.sleep_segments[-1].end_time, sleep_session.end_time, SleepType.Sleep))

        return sleep_sessions

    def get_sleep_segments(self, start_time: datetime, end_time: datetime) -> Tuple[SleepSegment, ...]:
        entries = self.get_datasource(DataSource.SLEEP_SEGMENT, start_time, end_time)

        return tuple((SleepSegment.from_dict(entry) for entry in entries))

    def get_daily_blood_pressure(self, start_time: datetime, end_time: datetime) -> Dict[date, List[BloodPressure]]:
        data = self.get_datasource(
            DataSource.BLOOD_PRESSURE,
            datetime.fromtimestamp(0),
            datetime.now(),
        )

        blood_pressure_by_date = defaultdict(list)

        for entry in data:
            blood_pressure = BloodPressure.from_dict(entry)
            blood_pressure_by_date[blood_pressure.time.date()].append(blood_pressure)

        return blood_pressure_by_date

    def get_blood_pressure(self, start_time: datetime, end_time: datetime) -> Tuple[BloodPressure, ...]:
        data = self.get_datasource(
            DataSource.BLOOD_PRESSURE,
            datetime.fromtimestamp(0),
            datetime.now(),
        )

        return tuple((BloodPressure.from_dict(entry) for entry in data))

    def get_weight(self, day: date):
        weight_entries = self.get_datasource(
            DataSource.WEIGHT,
            datetime.combine(day, datetime.min.time()),
            datetime.combine(day, datetime.max.time())
        )

        weight_values = np.array((entry["value"][0]["fpVal"] for entry in weight_entries))

        return weight_values

    def get_daily_weight(self, start_date: date, end_date: date, moment="first") -> Dict[date, List[float]]:
        weight_entries = self.get_datasource(
            DataSource.WEIGHT,
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.max.time())
        )

        weight_values_by_date = defaultdict(list)

        for weight_entry in weight_entries:
            weigh_date = datetime.fromtimestamp(int(weight_entry['startTimeNanos']) / 1e9).date()
            weight_values_by_date[weigh_date].append(weight_entry['value'][0]['fpVal'])

        return weight_values_by_date
