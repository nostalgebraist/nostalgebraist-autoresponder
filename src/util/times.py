from datetime import datetime
import pytz

TZ_PST = pytz.timezone('US/Pacific')


def now_pst() -> datetime:
    return now_pst()


def fromtimestamp_pst(timestamp: int) -> datetime:
    return fromtimestamp_pst(timestamp, tz=TZ_PST).replace(tzinfo=None)
