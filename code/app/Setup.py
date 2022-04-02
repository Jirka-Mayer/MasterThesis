class Setup:
    """
    Represents a specific model + dataset + options setup
    that can be trained, evaluated and then experimented upon
    """
    def __init__(self, name: str):
        self.name = name
