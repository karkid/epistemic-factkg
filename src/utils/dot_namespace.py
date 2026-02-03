class DotNamespace:
    def __init__(self, items):
        self._items = list(items)
        for x in self._items:
            setattr(self, x.name, x.value)

    def __iter__(self):
        return iter(x.value for x in self._items)

    def keys(self):
        return [x.name for x in self._items]

    def values(self):
        return [x.value for x in self._items]

    def items(self):
        return [(x.name, x.value) for x in self._items]


def to_namespace(fset):
    return DotNamespace(fset)