import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)
