from dataclasses import dataclass
from numpy import typing as npt, float64

from .runtime_configuration import RuntimeConfiguration

NDArrayFloat = npt.NDArray[float64]

runtime_configuration = RuntimeConfiguration()

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
