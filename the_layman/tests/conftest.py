from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repository root is importable when running `pytest` directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Tests run in fallback mode unless a model is explicitly configured in environment.
os.environ.setdefault("LAYMAN_ALLOW_GROUNDED_FALLBACK", "1")
