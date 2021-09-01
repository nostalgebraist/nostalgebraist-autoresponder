from datetime import datetime
import pytz

TZ_PST = pytz.timezone('US/Pacific')


def now_pst() -> datetime:
    return datetime.now(tz=TZ_PST).replace(tzinfo=None)


def fromtimestamp_pst(timestamp: int) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=TZ_PST).replace(tzinfo=None)
