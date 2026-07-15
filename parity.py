"""DEPRECATED root shim — the implementation moved to the ``qwen3_vl``
package. This file exists for backward compatibility with ``docker/run.sh``,
the ``qwen3-vl`` console script, and any external ``import <name>``.
New code should import from ``qwen3_vl.<name>`` directly.
"""

from __future__ import annotations

import qwen3_vl.parity as _impl  # noqa: F401

# Re-export every public name for `from <shim> import X` callers.
__all__ = [n for n in dir(_impl) if not n.startswith('_')]  # type: ignore[list-item]
globals().update({n: getattr(_impl, n) for n in __all__})


if __name__ == '__main__':
    import sys as _sys
    raise SystemExit(_impl.main(_sys.argv[1:]))
