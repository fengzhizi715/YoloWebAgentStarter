from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.types import DateTime, TypeDecorator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class UTCDateTime(TypeDecorator[datetime]):
    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: object) -> datetime | None:
        normalized = ensure_utc(value)
        return normalized.replace(tzinfo=None) if normalized else None

    def process_result_value(self, value: datetime | None, dialect: object) -> datetime | None:
        return ensure_utc(value)

