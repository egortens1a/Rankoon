from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_path: Path

    @computed_field  # type: ignore[prop-decorator]
    @property
    def wands_data_dir(self) -> Path:
        return self.base_path / "data" / "raw" / "dataset"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def wands_annots_path(self) -> Path:
        return self.base_path / "data" / "annots" / "candidate_pairs.csv"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processed_dir(self) -> Path:
        return self.base_path / "data" / "processed"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def splits_dir(self) -> Path:
        return self.base_path / "data" / "splits"


@lru_cache
def get_settings(base_path: Path = ROOT) -> Settings:
    return Settings(base_path=base_path)  # type: ignore[call-arg]