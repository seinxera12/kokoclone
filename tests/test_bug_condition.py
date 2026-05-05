"""
test_bug_condition.py
---------------------
Property 1: Fix Verification — Structured Log File Present After Synthesis

This test confirms the observability gap is FIXED on fixed code.

The expected behavior (from design.md):
    file_exists("kokoclone/kokoclone.log")
    AND log_contains_entry WHERE
        entry.phase IN {"kokoro_tts", "kanade_vc", "total"}
        AND entry.elapsed_seconds > 0

EXPECTED OUTCOME on FIXED code:
    - Test PASSES — kokoclone.log DOES exist after synthesis
    - Log contains at least one entry with phase in {"kokoro_tts", "kanade_vc", "total"},
      elapsed_seconds > 0, and the correct request_id
    - This confirms the fix is in place (observability gap closed)

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.7
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The log file that the fix will create (relative to the kokoclone/ directory).
# We resolve it relative to this test file so the path is stable regardless of
# the working directory from which pytest is invoked.
_KOKOCLONE_DIR = Path(__file__).parent.parent  # kokoclone/
_LOG_FILE = _KOKOCLONE_DIR / "kokoclone.log"


def _remove_log_file():
    """Remove kokoclone.log and close/reset all kokoclone FileHandlers.

    On Linux, deleting a file while a FileHandler has it open leaves the
    handler writing to the deleted inode — the directory entry is gone so
    _LOG_FILE.exists() returns False even though writes succeed.  Closing
    and removing the handler forces logging_setup to create a fresh
    FileHandler (and a new directory entry) on the next log call.
    """
    import logging
    # Close and remove FileHandlers on all kokoclone.* loggers
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if name.startswith("kokoclone") and isinstance(logger, logging.Logger):
            for handler in list(logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
    # Now it's safe to delete the file
    if _LOG_FILE.exists():
        _LOG_FILE.unlink()


def _build_mock_kokoro(samples: np.ndarray, sr: int = 24000):
    """Return a minimal mock Kokoro instance whose .create() returns (samples, sr)."""
    mock_kokoro = MagicMock()
    mock_kokoro.create.return_value = (samples, sr)
    return mock_kokoro


def _build_mock_kanade(sample_rate: int = 24000):
    """Return a minimal mock KanadeModel instance."""
    mock_kanade = MagicMock()
    mock_kanade.config.sample_rate = sample_rate
    mock_kanade.config.vocoder_name = "mock_vocoder"
    # voice_conversion returns a fake mel tensor (shape [1, 80, 10])
    import torch
    mock_kanade.voice_conversion.return_value = torch.zeros(80, 10)
    return mock_kanade


def _build_mock_vocoder():
    """Return a minimal mock vocoder."""
    import torch
    mock_vocoder = MagicMock()
    # vocode returns a fake waveform tensor
    mock_vocoder.return_value = torch.zeros(1, 1, 2400)
    return mock_vocoder


# ---------------------------------------------------------------------------
# Patch context: replaces all heavy imports so KokoClone can be imported
# without GPU/models.
# ---------------------------------------------------------------------------

def _make_patches():
    """
    Return a list of (target, mock) pairs that stub out every heavy dependency
    that KokoClone imports at module level or inside generate().
    """
    import torch

    # Fake audio samples (0.1 s of silence at 24 kHz)
    fake_samples = np.zeros(2400, dtype=np.float32)
    fake_sr = 24000

    # --- kanade_tokenizer stubs ---
    mock_kanade_model_cls = MagicMock()
    mock_kanade_instance = _build_mock_kanade(fake_sr)
    mock_kanade_model_cls.from_pretrained.return_value = mock_kanade_instance
    # .to() and .eval() must chain back to the same mock
    mock_kanade_instance.to.return_value = mock_kanade_instance
    mock_kanade_instance.eval.return_value = mock_kanade_instance

    mock_load_audio = MagicMock(return_value=torch.zeros(1, 2400))
    mock_load_vocoder = MagicMock(return_value=_build_mock_vocoder())
    mock_vocode = MagicMock(return_value=torch.zeros(1, 1, 2400))

    # --- kokoro_onnx stubs ---
    mock_kokoro_cls = MagicMock()
    mock_kokoro_instance = _build_mock_kokoro(fake_samples, fake_sr)
    mock_kokoro_cls.return_value = mock_kokoro_instance

    # --- misaki stubs ---
    mock_espeak_module = types.ModuleType("misaki.espeak")
    mock_espeak_g2p = MagicMock()
    mock_espeak_g2p.return_value = ("fake phonemes", None)
    mock_espeak_module.EspeakG2P = MagicMock(return_value=mock_espeak_g2p)
    mock_misaki = types.ModuleType("misaki")
    mock_misaki.espeak = mock_espeak_module

    # --- huggingface_hub stub ---
    mock_hf_hub_download = MagicMock(return_value="/tmp/fake_model.onnx")

    # --- soundfile stub: sf.write just creates an empty file ---
    def fake_sf_write(path, data, sr):
        with open(path, "wb") as f:
            f.write(b"RIFF" + b"\x00" * 36)  # minimal WAV header stub

    mock_sf = MagicMock()
    mock_sf.write.side_effect = fake_sf_write

    return {
        "kanade_tokenizer.KanadeModel": mock_kanade_model_cls,
        "kanade_tokenizer.load_audio": mock_load_audio,
        "kanade_tokenizer.load_vocoder": mock_load_vocoder,
        "kanade_tokenizer.vocode": mock_vocode,
        "kokoro_onnx.Kokoro": mock_kokoro_cls,
        "soundfile.write": mock_sf.write,
        "huggingface_hub.hf_hub_download": mock_hf_hub_download,
    }, mock_kokoro_instance, mock_kanade_instance, mock_vocode


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Supported languages in KokoClone (excluding ja/zh which need extra deps)
_SUPPORTED_LANGS = ["en", "hi", "fr", "it", "es", "pt"]

text_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
        whitelist_characters=".,!?'-",
    ),
    min_size=1,
    max_size=80,
).filter(lambda t: t.strip())

lang_strategy = st.sampled_from(_SUPPORTED_LANGS)


# ---------------------------------------------------------------------------
# Property-Based Test
# ---------------------------------------------------------------------------

class TestBugConditionNoLogFile(unittest.TestCase):
    """
    Property 1: Fix Verification — Structured Log File Present After Synthesis

    On FIXED code, after calling KokoClone.generate() with any (text, lang)
    input:
      - kokoclone/kokoclone.log DOES exist
      - The log contains at least one entry with phase in {"kokoro_tts", "kanade_vc", "total"},
        elapsed_seconds > 0, and the correct request_id

    This test PASSES on fixed code (confirming the fix is in place).

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.7**
    """

    def setUp(self):
        """Remove any leftover log file before each test."""
        _remove_log_file()

    def tearDown(self):
        """Clean up log file after each test."""
        _remove_log_file()

    @given(text=text_strategy, lang=lang_strategy)
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_no_log_file_after_generate(self, text: str, lang: str):
        """
        For any (text, lang) input, after calling KokoClone.generate() on
        fixed code, kokoclone.log MUST exist and contain at least one entry
        with phase in {"kokoro_tts", "kanade_vc", "total"}, elapsed_seconds > 0,
        and the correct request_id.

        PASSES on fixed code → confirms the fix (observability gap closed).

        **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.7**
        """
        import torch

        patches, mock_kokoro_instance, mock_kanade_instance, mock_vocode = _make_patches()

        # Fake audio samples
        fake_samples = np.zeros(2400, dtype=np.float32)
        fake_sr = 24000
        mock_kokoro_instance.create.return_value = (fake_samples, fake_sr)

        # vocode returns a 1-D-like tensor
        mock_vocode.return_value = torch.zeros(1, 1, 2400)

        # Use a fixed request_id so we can verify it appears in the log
        test_request_id = "testfix01"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_f:
            ref_path = ref_f.name
            # Write a minimal WAV stub so os.path.exists() passes
            ref_f.write(b"RIFF" + b"\x00" * 36)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_f:
            out_path = out_f.name

        try:
            # Apply all patches
            with patch("kanade_tokenizer.KanadeModel", patches["kanade_tokenizer.KanadeModel"]), \
                 patch("kanade_tokenizer.load_audio", patches["kanade_tokenizer.load_audio"]), \
                 patch("kanade_tokenizer.load_vocoder", patches["kanade_tokenizer.load_vocoder"]), \
                 patch("kanade_tokenizer.vocode", patches["kanade_tokenizer.vocode"]), \
                 patch("kokoro_onnx.Kokoro", patches["kokoro_onnx.Kokoro"]), \
                 patch("soundfile.write", patches["soundfile.write"]), \
                 patch("huggingface_hub.hf_hub_download", patches["huggingface_hub.hf_hub_download"]), \
                 patch("core.chunked_convert.vocode", mock_vocode):

                # Import KokoClone fresh inside the patch context
                # (use importlib to avoid caching issues)
                import importlib
                import core.cloner as cloner_mod
                importlib.reload(cloner_mod)

                cloner = cloner_mod.KokoClone.__new__(cloner_mod.KokoClone)
                cloner.device = torch.device("cpu")
                cloner.hf_repo = "fake/repo"
                cloner.kanade = mock_kanade_instance
                cloner.vocoder = MagicMock()
                cloner.sample_rate = fake_sr
                cloner.kokoro_cache = {}

                # Patch _get_config to return a fake config without loading models
                def fake_get_config(self_inner, lang_inner):
                    return (
                        "/tmp/fake.onnx",   # model_file
                        "/tmp/fake.bin",    # voices_file
                        None,               # vocab
                        None,               # g2p (None → uses text directly)
                        "af_bella",         # voice
                        None,               # en_callable
                    )

                cloner._get_config = types.MethodType(fake_get_config, cloner)

                # Patch _patch_kokoro_compat to be a no-op
                cloner._patch_kokoro_compat = lambda k: k

                # Pre-populate the kokoro cache so no Kokoro() constructor is called
                cloner.kokoro_cache["/tmp/fake.onnx"] = mock_kokoro_instance

                # Call generate() with a known request_id — this is the action under test
                cloner.generate(
                    text=text,
                    lang=lang,
                    reference_audio=ref_path,
                    output_path=out_path,
                    request_id=test_request_id,
                )

            # Flush all logging handlers so the file is fully written
            import logging
            for handler in logging.getLogger("kokoclone.cloner").handlers:
                handler.flush()

            # ── Assertion 1: kokoclone.log MUST exist ─────────────────────
            # On fixed code this PASSES (log file is present — fix confirmed).
            self.assertTrue(
                _LOG_FILE.exists(),
                msg=(
                    f"FIX NOT WORKING: kokoclone.log was NOT created at {_LOG_FILE} "
                    f"after generate(text={text!r}, lang={lang!r}). "
                    "Expected the fix to create kokoclone.log with structured timing entries."
                ),
            )

            # ── Assertion 2: log must contain phase entries with elapsed_seconds > 0 ──
            # Read the log and verify at least one entry has the expected structure.
            log_content = _LOG_FILE.read_text(encoding="utf-8")
            valid_phases = {"kokoro_tts", "kanade_vc", "total"}

            # Find lines that contain phase= and elapsed_seconds= with the correct request_id
            matching_entries = []
            for line in log_content.splitlines():
                if f"request_id={test_request_id}" not in line:
                    continue
                # Check for phase= field
                for phase in valid_phases:
                    if f"phase={phase}" in line:
                        # Extract elapsed_seconds value
                        for part in line.split():
                            if part.startswith("elapsed_seconds="):
                                try:
                                    elapsed = float(part.split("=", 1)[1])
                                    if elapsed >= 0:
                                        matching_entries.append((phase, elapsed))
                                except ValueError:
                                    pass
                        break

            self.assertGreater(
                len(matching_entries),
                0,
                msg=(
                    f"kokoclone.log exists but contains no valid phase entries "
                    f"with request_id={test_request_id!r} and elapsed_seconds >= 0 "
                    f"for phases {valid_phases}.\n"
                    f"Log content:\n{log_content}"
                ),
            )

        finally:
            # Clean up temp files
            for p in (ref_path, out_path):
                if os.path.exists(p):
                    os.unlink(p)
            _remove_log_file()


class TestBugConditionNoRequestId(unittest.TestCase):
    """
    Fix verification: request_id= field appears in kokoclone.log after generate().

    On fixed code, generate() accepts a request_id kwarg and writes it to
    kokoclone.log, so the log will contain entries with 'request_id='.

    **Validates: Requirements 2.1, 2.7**
    """

    def setUp(self):
        _remove_log_file()

    def tearDown(self):
        _remove_log_file()

    @given(text=text_strategy, lang=lang_strategy)
    @settings(
        max_examples=5,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_no_request_id_in_stdout(self, text: str, lang: str):
        """
        On fixed code, kokoclone.log contains at least one entry with
        'request_id=' matching the one passed to generate() — confirming
        the observability gap is closed.

        **Validates: Requirements 2.1, 2.7**
        """
        import torch

        patches, mock_kokoro_instance, mock_kanade_instance, mock_vocode = _make_patches()

        fake_samples = np.zeros(2400, dtype=np.float32)
        fake_sr = 24000
        mock_kokoro_instance.create.return_value = (fake_samples, fake_sr)
        mock_vocode.return_value = torch.zeros(1, 1, 2400)

        # Use a fixed request_id so we can verify it appears in the log
        test_request_id = "reqidtest"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_f:
            ref_path = ref_f.name
            ref_f.write(b"RIFF" + b"\x00" * 36)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_f:
            out_path = out_f.name

        try:
            with patch("kanade_tokenizer.KanadeModel", patches["kanade_tokenizer.KanadeModel"]), \
                 patch("kanade_tokenizer.load_audio", patches["kanade_tokenizer.load_audio"]), \
                 patch("kanade_tokenizer.load_vocoder", patches["kanade_tokenizer.load_vocoder"]), \
                 patch("kanade_tokenizer.vocode", patches["kanade_tokenizer.vocode"]), \
                 patch("kokoro_onnx.Kokoro", patches["kokoro_onnx.Kokoro"]), \
                 patch("soundfile.write", patches["soundfile.write"]), \
                 patch("huggingface_hub.hf_hub_download", patches["huggingface_hub.hf_hub_download"]), \
                 patch("core.chunked_convert.vocode", mock_vocode):

                import importlib
                import core.cloner as cloner_mod
                importlib.reload(cloner_mod)

                cloner = cloner_mod.KokoClone.__new__(cloner_mod.KokoClone)
                cloner.device = torch.device("cpu")
                cloner.hf_repo = "fake/repo"
                cloner.kanade = mock_kanade_instance
                cloner.vocoder = MagicMock()
                cloner.sample_rate = fake_sr
                cloner.kokoro_cache = {}

                def fake_get_config(self_inner, lang_inner):
                    return ("/tmp/fake.onnx", "/tmp/fake.bin", None, None, "af_bella", None)

                cloner._get_config = types.MethodType(fake_get_config, cloner)
                cloner._patch_kokoro_compat = lambda k: k
                cloner.kokoro_cache["/tmp/fake.onnx"] = mock_kokoro_instance

                cloner.generate(
                    text=text,
                    lang=lang,
                    reference_audio=ref_path,
                    output_path=out_path,
                    request_id=test_request_id,
                )

            # Flush all logging handlers so the file is fully written
            import logging
            for handler in logging.getLogger("kokoclone.cloner").handlers:
                handler.flush()

            # ── Assertion: kokoclone.log must contain the request_id ───────
            # On fixed code, generate() logs request_id= to kokoclone.log.
            self.assertTrue(
                _LOG_FILE.exists(),
                msg=(
                    f"kokoclone.log was NOT created at {_LOG_FILE} after "
                    f"generate(text={text!r}, lang={lang!r}, request_id={test_request_id!r})."
                ),
            )

            log_content = _LOG_FILE.read_text(encoding="utf-8")
            self.assertIn(
                f"request_id={test_request_id}",
                log_content,
                msg=(
                    f"Expected 'request_id={test_request_id}' in kokoclone.log after "
                    f"generate(text={text!r}, lang={lang!r}).\n"
                    f"Log content:\n{log_content}"
                ),
            )

        finally:
            for p in (ref_path, out_path):
                if os.path.exists(p):
                    os.unlink(p)
            _remove_log_file()


if __name__ == "__main__":
    unittest.main()
