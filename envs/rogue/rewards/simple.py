
from rogueinabox_lib.rewards import RewardGenerator
from . import register


"""
Define in this file custom reward generators in the following way:

class GeneratorName(Superclass):  # a common superclass is rogueinabox_lib.rewards.RewardGenerator
    # implementation
    ...

    
register(GeneratorName)  # call this function to be able to use the generator by name string

"""

