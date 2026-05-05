"""
conftest.py
-----------
Session-scoped module stubs installed before any test file is imported.

Both test_bug_condition.py and test_preservation.py patch heavy third-party
modules (kanade_tokenizer, kokoro_onnx, misaki, …) that are not installed in
the test environment.  pytest collects and imports test files in alphabetical
order, so test_bug_condition.py is imported before test_preservation.py has a
chance to install its sys.modules mocks.

This conftest installs minimal stubs for every module that would otherwise
cause an ImportError during collection or patching.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _stub(name: str, **attrs) -> types.ModuleType:
    """Create a stub module and register it (and any parent packages) in sys.modules."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    # Register parent packages too (e.g. "misaki.espeak" → also register "misaki")
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            sys.modules[parent] = types.ModuleType(parent)
    return mod


# ---------------------------------------------------------------------------
# kanade_tokenizer
# ---------------------------------------------------------------------------
if "kanade_tokenizer" not in sys.modules:
    _stub(
        "kanade_tokenizer",
        KanadeModel=MagicMock(),
        load_audio=MagicMock(),
        load_vocoder=MagicMock(),
        vocode=MagicMock(),
    )
else:
    # Already installed by test_preservation.py's module-level code — make sure
    # the attributes that test_bug_condition.py patches are present.
    _kt = sys.modules["kanade_tokenizer"]
    for _attr in ("KanadeModel", "load_audio", "load_vocoder", "vocode"):
        if not hasattr(_kt, _attr):
            setattr(_kt, _attr, MagicMock())

# ---------------------------------------------------------------------------
# kokoro_onnx
# ---------------------------------------------------------------------------
if "kokoro_onnx" not in sys.modules:
    _kokoro_mod = _stub("kokoro_onnx", Kokoro=MagicMock())
    # kokoro_onnx.config sub-module
    _config_mod = types.ModuleType("kokoro_onnx.config")
    _config_mod.MAX_PHONEME_LENGTH = 510
    _config_mod.SAMPLE_RATE = 24000
    sys.modules["kokoro_onnx.config"] = _config_mod
    _kokoro_mod.config = _config_mod

# ---------------------------------------------------------------------------
# misaki (and sub-modules)
# ---------------------------------------------------------------------------
if "misaki" not in sys.modules:
    _misaki = _stub("misaki")
    _espeak_mod = types.ModuleType("misaki.espeak")
    _espeak_mod.EspeakG2P = MagicMock()
    sys.modules["misaki.espeak"] = _espeak_mod
    _misaki.espeak = _espeak_mod

# ---------------------------------------------------------------------------
# huggingface_hub
# ---------------------------------------------------------------------------
if "huggingface_hub" not in sys.modules:
    _stub("huggingface_hub", hf_hub_download=MagicMock(return_value="/tmp/fake.onnx"))

# ---------------------------------------------------------------------------
# soundfile
# ---------------------------------------------------------------------------
if "soundfile" not in sys.modules:
    _stub("soundfile", write=MagicMock(), read=MagicMock())
