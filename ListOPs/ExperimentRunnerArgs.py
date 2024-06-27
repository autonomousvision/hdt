class ExperimentRunnerArgs(dict):
    # Need this class to run a debugger while also setting argparse arguments
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Attribute '{key}' not found")

    def _get_kwargs(self):
        return self.items()