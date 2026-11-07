"""A collection read from one JSON file, such as the synthetic sample in examples/."""

from pathlib import Path

from pydantic import ValidationError

from selecta.domain.collection import Collection


class CollectionFileError(Exception):
    """The collection file can't be read."""


class CollectionFile:
    """Collection source that reads a `Collection` serialised as JSON."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> Collection:
        """Return the collection in the file. Raises CollectionFileError."""
        try:
            return Collection.model_validate_json(self._path.read_text())
        except (OSError, ValidationError) as e:
            raise CollectionFileError(f"can't read {self._path}: {e}") from e
