import sys

from .ant import AntEnv
from .swimmer import SwimmerEnv

env_overwrite = {'Ant': AntEnv, 'Swimmer': SwimmerEnv}

sys.modules[__name__] = env_overwrite