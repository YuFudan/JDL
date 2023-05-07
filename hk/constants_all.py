import os
import sys

path = os.getcwd()  # 获取工作路径
is_root = path.split("/")[-1] == "jd_demo1"
if is_root:
    from constants_top import *
    from hk.constants_file import *
    from hk.constants_function import *
    from hk.constants_spatial import *
else:
    sys.path.append("..")
    from constants_top import *
    from constants_file import *
    from constants_function import *
    from constants_spatial import *
    