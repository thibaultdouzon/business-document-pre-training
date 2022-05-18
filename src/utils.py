import dataclasses
import json

from enum import Enum


class DataclassJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for dataclasses.

    Args:
        json.JSONEncoder: base encoder class
    """
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, Enum):
            return o.name
        return super().default(o)
