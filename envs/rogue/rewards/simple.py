
from roguelib_module.rewards import RewardGenerator
from . import register


"""
Define in this file custom reward generators in the following way:

class GeneratorName(Superclass):  # a common superclass is roguelib_module.rewards.RewardGenerator
    # implementation
    ...

    
register(GeneratorName)  # call this function to be able to use the generator by name string

"""

