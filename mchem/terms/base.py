class TermList(list):
    def __init__(self, cls):
        super().__init__()
        self._cls = cls
    
    @property
    def cls(self):
        return self._cls
    
    def append(self, item):
        assert isinstance(item, self.cls), f"Not {self.cls.__name__} instance"
        super().append(item)