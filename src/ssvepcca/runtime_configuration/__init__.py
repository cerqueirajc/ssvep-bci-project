import yaml


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class RuntimeConfiguration(metaclass=SingletonMeta):
    num_subjects: int
    num_blocks: int
    num_targets: int
    num_samples: int
    num_electrodes: int
    sample_frequency: float
    target_frequencies: list[float]
    target_phases: list[float]
    electrodes: dict[str, int]

    def load_config(self, config):
        self.num_subjects = config["num_subjects"]
        self.num_blocks = config["num_blocks"]
        self.num_targets = config["num_targets"]
        self.num_samples = config["num_samples"]
        self.num_electrodes = config["num_electrodes"]
        self.sample_frequency = config["sample_frequency"]
        self.target_frequencies = config["target_frequencies"]
        self.target_phases = config["target_phases"]
        self.electrodes = config["electrodes"]

        assert len(self.target_frequencies) == self.num_targets
        assert len(self.target_phases) == self.num_targets
        assert len(self.electrodes) == self.num_electrodes


    def load_from_yaml(self, yaml_file_path: str) -> None:
        with open(yaml_file_path) as y:
            params = yaml.safe_load(y)
            self.load_config(params)
    

    def load_from_name(self, config_name: str) -> None:
        if config_name == "tsinghua-bci-lab":
            from .tsinghua_bci_lab import PARAMS as tsinghua_bci_lab_params
            self.load_config(tsinghua_bci_lab_params)
        else:
            raise AttributeError(f"Attribute {config_name} could not be found.")
        

    def load_from_dict(self, params) -> None:
        self.load_config(params)
