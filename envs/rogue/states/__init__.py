from roguelib_module import states


def register(cls):
    if hasattr(cls,  cls.__name__):
        raise ValueError('A class with name "%s" was already registered.' % cls.__name__)
    setattr(states, cls.__name__, cls)


from .cropped import *
