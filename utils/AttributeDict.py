class AttributeDict(dict):
    """Convienent dict that allow accessing keys like attribute."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__