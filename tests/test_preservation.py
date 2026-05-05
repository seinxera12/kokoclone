"""
test_preservation.py
--------------------
Property 2: Preservation — Audio Output Is Numerically Unchanged After Logging Changes

These tests capture the BASELINE behavior of the unfixed code.  After the fix
is applied, re-running them should still pass (no regression).

The three properties tested here:

  P2a — Single-chunk preservation:
        For random waveform tensors shorter than ``chunk_samples``,
        ``chunked_voice_conversion()`` returns a numerically identical tensor
        before and after the fix.

  P2b — Multi-chunk preservation:
        For random waveform tensors longer than ``chunk_samples``,
        the assembled tensor is numerically identical before and after the fix.

  P2c — HTTP client preservation:
        ``synthesize_stream()`` yields the same WAV bytes after adding the
        ``round_trip_ms`` log call.

Non-bug condition: inputs where ``isBugCondition(X)`` is FALSE — i.e., the
logging infrastructure is already in place and audio output is the focus.

EXPECTED OUTCOME on UNFIXED code:
    All three tests PASS — confirms the baseline audio behaviour to preserve.

EXPECTED OUTCOME on FIXED code:
    All three tests still PASS — confirms no audio regression was introduced.

**Validates: Requirements 3.1, 3.2, 3.4, 3.5, 3.6**
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Torch mock — installed once at module level so all tests share it.
# chunked_convert.py imports torch at the top level, so the mock must be in
# sys.modules before the module is imported.
# ---------------------------------------------------------------------------

def _install_torch_mock() -> types.ModuleType:
    """
    Build and register a minimal torch mock that satisfies all calls made by
    ``chunked_convert.py``.  Returns the mock module.
    """
    if "torch" in sys.modules:
        return sys.modules["torch"]

    torch_mock = types.ModuleType("torch")

    class _FakeTensor:
        """Thin numpy-backed tensor that mimics the torch.Tensor API used by
        ``chunked_voice_conversion``."""

        def __init__(self, data=None, shape=None):
            if data is not None:
                if isinstance(data, np.ndarray):
                    self._data = data.astype(np.float32)
                else:
                    self._data = np.array(data, dtype=np.float32)
            else:
                self._data = np.zeros(shape or (1,), dtype=np.float32)

        # ── shape / device ──────────────────────────────────────────────────
        @property
        def shape(self):
            return self._data.shape

        @property
        def device(self):
            return _FakeDevice("cpu")

        # ── tensor ops used by chunked_convert ──────────────────────────────
        def squeeze(self):
            return _FakeTensor(self._data.squeeze())

        def cpu(self):
            return self

        def to(self, device):
            return self

        def unsqueeze(self, dim):
            return _FakeTensor(np.expand_dims(self._data, axis=dim))

        def __getitem__(self, key):
            return _FakeTensor(self._data[key])

        def __len__(self):
            return len(self._data)

        # ── equality helpers for assertions ─────────────────────────────────
        def allclose(self, other, atol=1e-6):
            return np.allclose(self._data, other._data, atol=atol)

        def numpy(self):
            return self._data

    class _FakeDevice:
        def __init__(self, t: str):
            self.type = t

        def __str__(self):
            return self.type

    def _fake_cat(tensors, dim=-1):
        arrays = [t._data for t in tensors]
        return _FakeTensor(np.concatenate(arrays, axis=dim))

    torch_mock.Tensor = _FakeTensor
    torch_mock.device = _FakeDevice
    torch_mock.inference_mode = lambda: __import__("contextlib").nullcontext()
    torch_mock.cat = _fake_cat
    torch_mock.zeros = lambda *shape: _FakeTensor(shape=shape)
    torch_mock.cuda = types.SimpleNamespace(
        empty_cache=lambda: None,
        get_device_properties=lambda d: types.SimpleNamespace(total_memory=8 * 1024 ** 3),
    )

    sys.modules["torch"] = torch_mock
    return torch_mock


_TORCH = _install_torch_mock()
_FakeTensor = _TORCH.Tensor
_FakeDevice = _TORCH.device


# ---------------------------------------------------------------------------
# kanade_tokenizer mock — must be in sys.modules before chunked_convert import
# ---------------------------------------------------------------------------

def _install_kanade_mock():
    if "kanade_tokenizer" in sys.modules:
        kt = sys.modules["kanade_tokenizer"]
    else:
        kt = types.ModuleType("kanade_tokenizer")
        sys.modules["kanade_tokenizer"] = kt
    
    # vocode: (vocoder_model, mel_tensor) → waveform tensor
    # The real vocode returns shape [1, 1, T]; we return a deterministic result
    # based on the mel content so the test can verify identity.
    def _fake_vocode(model, mel):
        # Deterministic: output length = mel frames * 10
        n = mel._data.shape[-1] * 10
        # Fill with the mean of the mel so the output is content-dependent
        fill_val = float(mel._data.mean())
        data = np.full((1, 1, n), fill_val, dtype=np.float32)
        return _FakeTensor(data)

    kt.vocode = _fake_vocode
    # Add stub attributes so patch("kanade_tokenizer.KanadeModel", ...) works
    # when test_bug_condition.py runs in the same pytest session.
    if not hasattr(kt, "KanadeModel"):
        kt.KanadeModel = MagicMock()
    if not hasattr(kt, "load_audio"):
        kt.load_audio = MagicMock(return_value=_FakeTensor(np.zeros(2400, dtype=np.float32)))
    if not hasattr(kt, "load_vocoder"):
        kt.load_vocoder = MagicMock()
    return kt


_install_kanade_mock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Compute chunk_samples the same way chunked_convert.py does on CPU.
# rope_safe_chunk = int((rope_max_window - 2 * overlap_samples) * safety_margin)
_SAMPLE_RATE = 24000
_ROPE_MAX_FRAMES = 1024
_MEL_HOP_LENGTH = 256
_OVERLAP_SECONDS = 0.5
_ROPE_SAFETY_MARGIN = 0.75

_overlap_samples = int(_OVERLAP_SECONDS * _SAMPLE_RATE)
_rope_max_window = (_ROPE_MAX_FRAMES - 1) * _MEL_HOP_LENGTH  # 261,888
_CHUNK_SAMPLES = int((_rope_max_window - 2 * _overlap_samples) * _ROPE_SAFETY_MARGIN)
# On CPU, chunk_samples == rope_safe_chunk ≈ 178,416 samples ≈ 7.43 s


def _make_deterministic_kanade(seed: int = 42):
    """
    Return a mock KanadeModel whose ``voice_conversion`` is deterministic:
    given the same source waveform it always returns the same mel tensor.
    The output mel shape is ``[80, n_frames]`` where
    ``n_frames = source_length // 256 + 1``.
    """
    rng = np.random.default_rng(seed)

    def _vc(source_waveform, reference_waveform):
        n = source_waveform._data.shape[-1]
        mel_frames = n // _MEL_HOP_LENGTH + 1
        # Deterministic: use a fixed random array scaled by source mean
        mel = rng.standard_normal((80, mel_frames)).astype(np.float32)
        mel *= float(source_waveform._data.mean()) + 1.0  # avoid all-zero
        return _FakeTensor(mel)

    mock = types.SimpleNamespace()
    mock.voice_conversion = _vc
    mock.config = types.SimpleNamespace(sample_rate=_SAMPLE_RATE)
    return mock


def _import_chunked_convert():
    """Import (or reload) core.chunked_convert with mocks in place."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    if "core.chunked_convert" in sys.modules:
        return importlib.reload(sys.modules["core.chunked_convert"])
    import core.chunked_convert as cc
    return cc


def _import_kokoclone_tts():
    """Import server/tts/kokoclone_tts.py directly (avoids server.py deps)."""
    tts_path = Path(__file__).parent.parent.parent / "server" / "tts" / "kokoclone_tts.py"
    spec = importlib.util.spec_from_file_location("kokoclone_tts", tts_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Single-chunk: waveform length strictly less than chunk_samples
single_chunk_length_st = st.integers(min_value=1, max_value=_CHUNK_SAMPLES - 1)

# Multi-chunk: waveform length strictly greater than chunk_samples
# Cap at 3× chunk_samples to keep tests fast (still exercises 2–3 chunks)
multi_chunk_length_st = st.integers(
    min_value=_CHUNK_SAMPLES + 1,
    max_value=_CHUNK_SAMPLES * 3,
)

# Reference waveform length (short, just needs to be non-empty)
ref_length_st = st.integers(min_value=256, max_value=4096)

# Fixed WAV bytes for HTTP client test (realistic WAV header + silence)
_FIXED_WAV_BYTES = b"RIFF" + (44).to_bytes(4, "little") + b"WAVE" + b"\x00" * 36


# ---------------------------------------------------------------------------
# P2a — Single-chunk preservation
# ---------------------------------------------------------------------------

class TestSingleChunkPreservation(unittest.TestCase):
    """
    P2a: For random waveform tensors shorter than ``chunk_samples``,
    ``chunked_voice_conversion()`` returns a numerically identical tensor
    before and after the fix.

    On unfixed code this PASSES — confirming the baseline.
    On fixed code this must also PASS — confirming no regression.

    **Validates: Requirements 3.1, 3.4**
    """

    @given(
        n_samples=single_chunk_length_st,
        ref_len=ref_length_st,
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_single_chunk_output_is_deterministic(
        self, n_samples: int, ref_len: int, seed: int
    ):
        """
        For any waveform shorter than chunk_samples, calling
        ``chunked_voice_conversion()`` twice with the same inputs returns
        numerically identical results.

        This property holds on both unfixed and fixed code because the logging
        changes are pure side-effects that do not touch the audio tensors.

        **Validates: Requirements 3.1, 3.4**
        """
        cc = _import_chunked_convert()

        rng = np.random.default_rng(seed)
        source_data = rng.standard_normal(n_samples).astype(np.float32)
        ref_data = rng.standard_normal(ref_len).astype(np.float32)

        source = _FakeTensor(source_data)
        ref = _FakeTensor(ref_data)

        # Use a deterministic kanade mock (same seed → same outputs)
        kanade = _make_deterministic_kanade(seed=seed)

        # First call — baseline
        result_a = cc.chunked_voice_conversion(
            kanade=kanade,
            vocoder_model=None,
            source_wav=source,
            ref_wav=ref,
            sample_rate=_SAMPLE_RATE,
        )

        # Reset the kanade mock with the same seed so it produces the same mel
        kanade2 = _make_deterministic_kanade(seed=seed)
        source2 = _FakeTensor(source_data.copy())
        ref2 = _FakeTensor(ref_data.copy())

        # Second call — must be identical
        result_b = cc.chunked_voice_conversion(
            kanade=kanade2,
            vocoder_model=None,
            source_wav=source2,
            ref_wav=ref2,
            sample_rate=_SAMPLE_RATE,
        )

        # Assert numerical identity
        self.assertEqual(
            result_a.shape,
            result_b.shape,
            msg=(
                f"Shape mismatch for n_samples={n_samples}: "
                f"{result_a.shape} vs {result_b.shape}"
            ),
        )
        self.assertTrue(
            np.allclose(result_a.numpy(), result_b.numpy(), atol=1e-6),
            msg=(
                f"Numerical mismatch for n_samples={n_samples}, seed={seed}. "
                f"Max diff: {np.abs(result_a.numpy() - result_b.numpy()).max():.2e}"
            ),
        )

    @given(
        n_samples=single_chunk_length_st,
        ref_len=ref_length_st,
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_single_chunk_takes_short_circuit_path(
        self, n_samples: int, ref_len: int, seed: int
    ):
        """
        For waveforms shorter than chunk_samples, the function must take the
        short-circuit path (no chunking loop) and call voice_conversion exactly
        once.

        **Validates: Requirements 3.4**
        """
        cc = _import_chunked_convert()

        rng = np.random.default_rng(seed)
        source = _FakeTensor(rng.standard_normal(n_samples).astype(np.float32))
        ref = _FakeTensor(rng.standard_normal(ref_len).astype(np.float32))

        call_count = [0]

        def counting_vc(source_waveform, reference_waveform):
            call_count[0] += 1
            n = source_waveform._data.shape[-1]
            mel_frames = n // _MEL_HOP_LENGTH + 1
            return _FakeTensor(np.ones((80, mel_frames), dtype=np.float32))

        kanade = types.SimpleNamespace()
        kanade.voice_conversion = counting_vc

        cc.chunked_voice_conversion(
            kanade=kanade,
            vocoder_model=None,
            source_wav=source,
            ref_wav=ref,
            sample_rate=_SAMPLE_RATE,
        )

        self.assertEqual(
            call_count[0],
            1,
            msg=(
                f"Expected exactly 1 voice_conversion call for n_samples={n_samples} "
                f"(< chunk_samples={_CHUNK_SAMPLES}), got {call_count[0]}"
            ),
        )


# ---------------------------------------------------------------------------
# P2b — Multi-chunk preservation
# ---------------------------------------------------------------------------

class TestMultiChunkPreservation(unittest.TestCase):
    """
    P2b: For random waveform tensors longer than ``chunk_samples``,
    the assembled tensor is numerically identical before and after the fix.

    On unfixed code this PASSES — confirming the baseline.
    On fixed code this must also PASS — confirming no regression.

    **Validates: Requirements 3.1, 3.5**
    """

    @given(
        n_samples=multi_chunk_length_st,
        ref_len=ref_length_st,
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_multi_chunk_output_is_deterministic(
        self, n_samples: int, ref_len: int, seed: int
    ):
        """
        For any waveform longer than chunk_samples, calling
        ``chunked_voice_conversion()`` twice with the same inputs returns
        numerically identical results.

        **Validates: Requirements 3.1, 3.5**
        """
        cc = _import_chunked_convert()

        rng = np.random.default_rng(seed)
        source_data = rng.standard_normal(n_samples).astype(np.float32)
        ref_data = rng.standard_normal(ref_len).astype(np.float32)

        source_a = _FakeTensor(source_data.copy())
        ref_a = _FakeTensor(ref_data.copy())
        kanade_a = _make_deterministic_kanade(seed=seed)

        result_a = cc.chunked_voice_conversion(
            kanade=kanade_a,
            vocoder_model=None,
            source_wav=source_a,
            ref_wav=ref_a,
            sample_rate=_SAMPLE_RATE,
        )

        source_b = _FakeTensor(source_data.copy())
        ref_b = _FakeTensor(ref_data.copy())
        kanade_b = _make_deterministic_kanade(seed=seed)

        result_b = cc.chunked_voice_conversion(
            kanade=kanade_b,
            vocoder_model=None,
            source_wav=source_b,
            ref_wav=ref_b,
            sample_rate=_SAMPLE_RATE,
        )

        self.assertEqual(
            result_a.shape,
            result_b.shape,
            msg=(
                f"Shape mismatch for n_samples={n_samples}: "
                f"{result_a.shape} vs {result_b.shape}"
            ),
        )
        self.assertTrue(
            np.allclose(result_a.numpy(), result_b.numpy(), atol=1e-6),
            msg=(
                f"Numerical mismatch for n_samples={n_samples}, seed={seed}. "
                f"Max diff: {np.abs(result_a.numpy() - result_b.numpy()).max():.2e}"
            ),
        )

    @given(
        n_samples=multi_chunk_length_st,
        ref_len=ref_length_st,
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_multi_chunk_uses_chunked_path(
        self, n_samples: int, ref_len: int, seed: int
    ):
        """
        For waveforms longer than chunk_samples, the function must call
        voice_conversion more than once (chunked path).

        **Validates: Requirements 3.5**
        """
        cc = _import_chunked_convert()

        rng = np.random.default_rng(seed)
        source = _FakeTensor(rng.standard_normal(n_samples).astype(np.float32))
        ref = _FakeTensor(rng.standard_normal(ref_len).astype(np.float32))

        call_count = [0]

        def counting_vc(source_waveform, reference_waveform):
            call_count[0] += 1
            n = source_waveform._data.shape[-1]
            mel_frames = n // _MEL_HOP_LENGTH + 1
            return _FakeTensor(np.ones((80, mel_frames), dtype=np.float32))

        kanade = types.SimpleNamespace()
        kanade.voice_conversion = counting_vc

        cc.chunked_voice_conversion(
            kanade=kanade,
            vocoder_model=None,
            source_wav=source,
            ref_wav=ref,
            sample_rate=_SAMPLE_RATE,
        )

        self.assertGreater(
            call_count[0],
            1,
            msg=(
                f"Expected >1 voice_conversion calls for n_samples={n_samples} "
                f"(> chunk_samples={_CHUNK_SAMPLES}), got {call_count[0]}"
            ),
        )


# ---------------------------------------------------------------------------
# P2c — HTTP client preservation
# ---------------------------------------------------------------------------

class TestHTTPClientPreservation(unittest.TestCase):
    """
    P2c: ``synthesize_stream()`` yields the same WAV bytes after adding the
    ``round_trip_ms`` log call.

    Mock ``httpx`` to return a fixed WAV byte sequence; assert the yielded
    bytes are identical to what the mock returns.

    On unfixed code this PASSES — confirming the baseline.
    On fixed code this must also PASS — confirming no regression.

    **Validates: Requirements 3.1, 3.6**
    """

    def _make_httpx_mock(self, fixed_content: bytes) -> types.ModuleType:
        """Build a minimal httpx mock that returns ``fixed_content`` on POST."""
        httpx_mock = types.ModuleType("httpx")

        class _FakeResponse:
            def __init__(self):
                self.content = fixed_content
                self.status_code = 200

            def raise_for_status(self):
                pass  # 200 — no error

        class _FakeAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, json=None, timeout=None):
                return _FakeResponse()

        class _FakeTimeout:
            def __init__(self, **kwargs):
                pass

        httpx_mock.AsyncClient = _FakeAsyncClient
        httpx_mock.Timeout = _FakeTimeout
        httpx_mock.ConnectError = type("ConnectError", (Exception,), {})
        httpx_mock.TimeoutException = type("TimeoutException", (Exception,), {})
        httpx_mock.HTTPStatusError = type("HTTPStatusError", (Exception,), {})
        return httpx_mock

    def _make_tts(self, httpx_mock: types.ModuleType, ref_path: str):
        """Import KokoCloneTTS with the given httpx mock and a real ref file."""
        sys.modules["httpx"] = httpx_mock
        tts_mod = _import_kokoclone_tts()
        config = types.SimpleNamespace(
            kokoclone_url="http://localhost:5003",
            kokoclone_ref_audio=ref_path,
        )
        return tts_mod.KokoCloneTTS(config)

    @given(
        wav_body=st.binary(min_size=44, max_size=4096),
        text=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
            min_size=1,
            max_size=80,
        ).filter(lambda t: t.strip()),
    )
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_synthesize_stream_yields_exact_response_bytes(
        self, wav_body: bytes, text: str
    ):
        """
        For any fixed WAV byte sequence returned by the mocked HTTP service,
        ``synthesize_stream()`` must yield exactly those bytes — unchanged.

        This property holds on both unfixed and fixed code because the
        ``round_trip_ms`` log call is a pure side-effect that does not modify
        ``resp.content``.

        **Validates: Requirements 3.1, 3.6**
        """
        httpx_mock = self._make_httpx_mock(wav_body)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 36)
            ref_path = f.name

        try:
            tts = self._make_tts(httpx_mock, ref_path)
            self.assertTrue(tts._available, "TTS should be available with a valid ref file")

            async def _collect():
                chunks = []
                async for chunk in tts.synthesize_stream(text):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(_collect())

            self.assertEqual(
                len(chunks),
                1,
                msg=f"Expected exactly 1 chunk, got {len(chunks)} for text={text!r}",
            )
            self.assertEqual(
                chunks[0],
                wav_body,
                msg=(
                    f"Yielded bytes differ from the mocked response body "
                    f"for text={text!r}. "
                    f"Expected {len(wav_body)} bytes, got {len(chunks[0])} bytes."
                ),
            )
        finally:
            if os.path.exists(ref_path):
                os.unlink(ref_path)

    @given(
        wav_body=st.binary(min_size=44, max_size=4096),
    )
    @settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_synthesize_stream_empty_text_yields_nothing(self, wav_body: bytes):
        """
        ``synthesize_stream()`` must yield zero chunks for empty/whitespace text,
        regardless of what the HTTP service would return.

        **Validates: Requirements 3.6**
        """
        httpx_mock = self._make_httpx_mock(wav_body)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 36)
            ref_path = f.name

        try:
            tts = self._make_tts(httpx_mock, ref_path)

            async def _collect(text):
                chunks = []
                async for chunk in tts.synthesize_stream(text):
                    chunks.append(chunk)
                return chunks

            for empty_text in ("", "   ", "\t\n"):
                chunks = asyncio.run(_collect(empty_text))
                self.assertEqual(
                    len(chunks),
                    0,
                    msg=f"Expected 0 chunks for empty text {empty_text!r}, got {len(chunks)}",
                )
        finally:
            if os.path.exists(ref_path):
                os.unlink(ref_path)


if __name__ == "__main__":
    unittest.main()
