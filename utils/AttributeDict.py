class AttributeDict(dict):
    """
    Convienent dict that allow accessing keys like attribute.

    Example usage:
        d = AttrDict(foo=1, bar=2)
        d.foo # instead of d['foo']
        >> 1
        d.bar # instead of d['bar']
        >> 2
        print(d)
        >> {'bar': 2, 'foo': 1}
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__