from enum import Enum

from pydantic import BaseModel


class TableStatus(str, Enum):
    EMPTY_CLEAN = "empty_clean"
    GUESTS_WAITING = "guests_waiting"
    GUESTS_EATING = "guests_eating"
    NEEDS_ATTENTION = "needs_attention"
    NEEDS_CLEANING = "needs_cleaning"
    UNKNOWN = "unknown"


# Statuses that mean guests are sitting at the table.
OCCUPIED_STATUSES = {
    TableStatus.GUESTS_WAITING,
    TableStatus.GUESTS_EATING,
    TableStatus.NEEDS_ATTENTION,
}


class Zone(BaseModel):
    table_id: str
    x1: int
    y1: int
    x2: int
    y2: int


class TableReading(BaseModel):
    table_id: str
    status: TableStatus
    confidence: float
    note: str


class FrameReading(BaseModel):
    tables: list[TableReading]


class FrameResult(BaseModel):
    timestamp_s: float
    image_path: str
    readings: list[TableReading]
    # Tables the model was actually asked about; the rest were carried over unchanged.
    checked_tables: list[str] = []

    def status_of(self, table_id: str) -> TableStatus:
        for reading in self.readings:
            if reading.table_id == table_id:
                return reading.status
        return TableStatus.UNKNOWN


class RunResult(BaseModel):
    video: str
    interval_s: float
    zones: list[Zone]
    frames: list[FrameResult] = []

    @property
    def table_ids(self) -> list[str]:
        return [zone.table_id for zone in self.zones]
