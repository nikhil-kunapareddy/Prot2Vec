"""Make the src-layout package importable when it has not been installed.

``pip install -e ".[dev]"`` is the supported workflow, but running ``pytest``
straight from a fresh clone is a common first move, so ``src/`` is placed at
the front of ``sys.path``.

It goes at the *front* deliberately, and unconditionally. Checking
``find_spec("prot2vec")`` first is not enough: any stray ``prot2vec/``
directory in the repository root — one containing nothing but a leftover
``__pycache__``, say — is an implicit namespace package, so the check passes
while the real modules stay unreachable. Testing the working tree is also just
what you want from a checkout's own suite, whatever happens to be installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"

if (_SRC / "prot2vec" / "__init__.py").is_file():  # pragma: no cover - bootstrap
    src = str(_SRC)
    if src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)
