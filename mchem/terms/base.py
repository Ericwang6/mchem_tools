"""Base container for force-term lists keyed by term class."""


class TermList(list):
    """
    List of force terms restricted to a single dataclass type.

    Used by :class:`mchem.system.System` to store terms per table (e.g. ``AmoebaBond``).
    """

    def __init__(self, cls):
        """
        Parameters
        ----------
        cls : type
            Dataclass type; only instances of this type can be appended.
        """
        super().__init__()
        self._cls = cls

    @property
    def cls(self):
        """Term dataclass type for this list."""
        return self._cls

    def append(self, item):
        """Append a term; must be an instance of :attr:`cls`."""
        assert isinstance(item, self.cls), f"Not {self.cls.__name__} instance"
        super().append(item)