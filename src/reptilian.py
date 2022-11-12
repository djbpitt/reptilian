# ###
# External imports
# ###
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
import graphviz
from heapq import * # priority heap, https://docs.python.org/3/library/heapq.html
from IPython.display import display, HTML, SVG
from linsuffarr import SuffixArray
from linsuffarr import UNIT_BYTE
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=2)
import re
from typing import List

# ###
# Main
# ###
print("Hi, Mom!")