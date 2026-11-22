"""Fixed-offset time helpers. Period membership is decided by instants."""

import re
from datetime import date, datetime, timedelta, timezone

_OFFSET_RE = re.compile(r"^([+-])(\d{2}):(\d{2})$")
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")


def parse_offset(s: str) -> timezone:
    m = _OFFSET_RE.match(s)
    if not m:
        raise ValueError("bad offset %r" % s)
    sign = -1 if m.group(1) == "-" else 1
    hours, minutes = int(m.group(2)), int(m.group(3))
    if hours > 14 or minutes >= 60:
        raise ValueError("bad offset %r" % s)
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def offset_str(hours: int) -> str:
    sign = "-" if hours < 0 else "+"
    return "%s%02d:00" % (sign, abs(hours))


def offset_hours(tz: timezone) -> int:
    return int(tz.utcoffset(None).total_seconds() // 3600)


def fmt(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("naive datetime")
    return dt.isoformat(timespec="seconds")


def parse(s: str) -> datetime:
    if not ISO_RE.match(s):
        raise ValueError("bad timestamp %r" % s)
    return datetime.fromisoformat(s)


def local_date(dt: datetime, tz: timezone) -> date:
    return dt.astimezone(tz).date()


def is_business(d: date) -> bool:
    return d.weekday() < 5


def last_business_on_or_before(d: date) -> date:
    while not is_business(d):
        d -= timedelta(days=1)
    return d


def local_midnight(d: date, tz: timezone) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=tz)


def interval_of(dt: datetime, period_start: datetime, period_end: datetime) -> str:
    if dt < period_start:
        return "pre"
    if dt < period_end:
        return "period"
    return "window"


def daterange(start: date, end_exclusive: date):
    d = start
    while d < end_exclusive:
        yield d
        d += timedelta(days=1)
