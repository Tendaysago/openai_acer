from rogueinabox_lib import rewards


def register(cls):
    if hasattr(cls,  cls.__name__):
        raise ValueError('A class with name "%s" was already registered.' % cls.__name__)
    setattr(rewards, cls.__name__, cls)


from .simple import *
