"""Put the repository root on ``sys.path`` so ``scripts/*.py`` can import ``src``.

Keeps the project runnable straight from a clone with no install step, which is
what the acceptance criteria call for.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
