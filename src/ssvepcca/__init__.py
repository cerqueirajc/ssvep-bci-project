from dataclasses import dataclass
from numpy import float64, typing

from .runtime_configuration import RuntimeConfiguration


runtime_configuration = RuntimeConfiguration()

NDArrayFloat = typing.NDArray[float64]

@dataclass
class parameters:
    electrode_list_fbcca = (
        "PZ",
        "PO5",
        "PO3",
        "POz",
        "PO4",
        "PO6",
        "O1",
        "Oz",
        "O2",
    )
