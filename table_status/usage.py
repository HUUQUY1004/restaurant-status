from dataclasses import dataclass

from table_status.models import RunResult


@dataclass(frozen=True)
class ModelUsage:
    frames: int
    model_calls: int
    table_checks: int
    naive_table_checks: int

    @property
    def calls_saved_ratio(self) -> float:
        """Share of API calls avoided compared with sending every frame; this is what drives cost."""
        if self.frames == 0:
            return 0.0
        return 1 - self.model_calls / self.frames

    @property
    def checks_saved_ratio(self) -> float:
        """Share of per-table questions avoided; calls still send the full frame, so this is not cost."""
        if self.naive_table_checks == 0:
            return 0.0
        return 1 - self.table_checks / self.naive_table_checks


def model_usage(run: RunResult) -> ModelUsage:
    return ModelUsage(
        frames=len(run.frames),
        model_calls=sum(1 for frame in run.frames if frame.checked_tables),
        table_checks=sum(len(frame.checked_tables) for frame in run.frames),
        naive_table_checks=len(run.frames) * len(run.table_ids),
    )
