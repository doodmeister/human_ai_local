# Ensure project root is on sys.path for pytest and scripts.
import os
import sys
root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(root, 'src')
# Ensure project root and src/ both resolvable for 'import src.*' in varied runners
if root not in sys.path:
    sys.path.insert(0, root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)
