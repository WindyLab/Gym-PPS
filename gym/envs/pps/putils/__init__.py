def check_python_version():
    import sys
    if sys.version_info[0] == 3 and (sys.version_info[1] == 8 or 10):
        pass
    else:
        raise ValueError('Python 3.8 or 3.10 REQUIRED !')

check_python_version()

from .param import *
from .prop import *
from .putils import *




