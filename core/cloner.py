import importlib.resources
import json
import os
import sys
import tempfile
import time
import types

from core.logging_setup import get_kokoclone_logger

_logger = get_kokoclone_logger("cloner")

import numpy as np
import torch
import soundfile as sf
from huggingface_hub import hf_hub_download
from kanade_tokenizer import KanadeModel, load_audio, load_vocoder, vocode
from kokoro_onnx import Kokoro
from kokoro_onnx.config import MAX_PHONEME_LENGTH, SAMPLE_RATE
from misaki import espeak
from misaki.espeak import EspeakG2P
from core.chunked_convert import chunked_voice_conversion

class KokoClone:
    def __init__(self, kanade_model="frothywater/kanade-12.5hz", hf_repo="PatnaikAshish/kokoclone"):
        # Auto-detect GPU (CUDA) or fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing KokoClone on: {self.device.type.upper()}")
        
        self.hf_repo = hf_repo
        
        # Load Kanade & Vocoder once, move to detected device
        print("Loading Kanade model...")
        self.kanade = KanadeModel.from_pretrained(kanade_model).to(self.device).eval()
        self.vocoder = load_vocoder(self.kanade.config.vocoder_name).to(self.device)
        self.sample_rate = self.kanade.config.sample_rate
        
        # Cache for Kokoro ONNX sessions (keyed by model file path)
        self.kokoro_cache = {}

        # Cache for Kanade reference (speaker) embeddings, keyed by absolute
        # reference audio path.  Encoding the reference wav is the most
        # expensive part of the VC pipeline (~0.3–1 s on GPU); caching it
        # means every request after the first one for a given reference file
        # skips that work entirely.
        self._ref_embedding_cache: dict[str, torch.Tensor] = {}

    def _get_vocab_config(self, lang):
        """Return a vocab config path compatible with the selected language/model."""
        # zh/ja model exports use the v1.1-zh vocabulary from hexgrad.
        if lang in {"zh", "ja"}:
            zh_vocab = os.path.join("model", "config-v1.1-zh.json")
            if not os.path.exists(zh_vocab):
                print("Downloading missing file 'config-v1.1-zh.json' from hexgrad/Kokoro-82M-v1.1-zh...")
                hf_hub_download(
                    repo_id="hexgrad/Kokoro-82M-v1.1-zh",
                    filename="config.json",
                    local_dir=".",
                )
                downloaded = os.path.join("config.json")
                if os.path.exists(downloaded):
                    os.replace(downloaded, zh_vocab)

            if os.path.exists(zh_vocab):
                return zh_vocab

        local_config = os.path.join("model", "config.json")
        if os.path.exists(local_config):
            try:
                with open(local_config, encoding="utf-8") as fp:
                    config = json.load(fp)
                if isinstance(config, dict) and "vocab" in config:
                    return local_config
                print("Warning: model/config.json is missing 'vocab'; using packaged kokoro_onnx config instead")
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Warning: could not read model/config.json ({exc}); using packaged kokoro_onnx config instead")

        return str(importlib.resources.files("kokoro_onnx").joinpath("config.json"))

    def _patch_kokoro_compat(self, kokoro):
        """Patch kokoro_onnx instances for model exports with mixed input conventions."""
        input_types = {input_meta.name: input_meta.type for input_meta in kokoro.sess.get_inputs()}
        if input_types.get("speed") != "tensor(float)" or "input_ids" not in input_types:
            return kokoro

        def _create_audio_compat(instance, phonemes, voice, speed):
            if len(phonemes) > MAX_PHONEME_LENGTH:
                phonemes = phonemes[:MAX_PHONEME_LENGTH]

            start_t = time.time()
            tokens = np.array(instance.tokenizer.tokenize(phonemes), dtype=np.int64)
            assert len(tokens) <= MAX_PHONEME_LENGTH, (
                f"Context length is {MAX_PHONEME_LENGTH}, but leave room for the pad token 0 at the start & end"
            )

            voice_style = voice[len(tokens)]
            inputs = {
                "input_ids": [[0, *tokens, 0]],
                "style": np.array(voice_style, dtype=np.float32),
                "speed": np.array([speed], dtype=np.float32),
            }

            audio = instance.sess.run(None, inputs)[0]
            audio_duration = len(audio) / SAMPLE_RATE
            create_duration = time.time() - start_t
            if audio_duration > 0:
                _ = create_duration / audio_duration
            return audio, SAMPLE_RATE

        kokoro._create_audio = types.MethodType(_create_audio_compat, kokoro)
        return kokoro

    def _ensure_file(self, folder, filename):
        """Auto-downloads missing models from your Hugging Face repo."""
        filepath = os.path.join(folder, filename)
        repo_filepath = f"{folder}/{filename}"
        
        if not os.path.exists(filepath):
            print(f"Downloading missing file '{filename}' from {self.hf_repo}...")
            hf_hub_download(
                repo_id=self.hf_repo,
                filename=repo_filepath,
                local_dir="." # Downloads securely into local ./model or ./voice
            )
        return filepath

    def _create_en_callable(self):
        """Create an English G2P callable for handling English tokens in non-English text."""
        en_g2p = EspeakG2P(language="en-us")
        def en_callable(text):
            try:
                phonemes, _ = en_g2p(text)
                return phonemes
            except Exception:
                return text
        return en_callable

    def _get_config(self, lang):
        """Routes the correct model, voice, and G2P based on language."""
        model_file = self._ensure_file("model", "kokoro.onnx")
        voices_file = self._ensure_file("voice", "voices-v1.0.bin")
        vocab = None
        g2p = None
        en_callable = None

        # Optimized routing: Only load the specific G2P engine requested
        if lang == "en":
            voice = "af_bella"
        elif lang == "hi":
            g2p = EspeakG2P(language="hi")
            voice = "hf_alpha"
        elif lang == "fr":
            g2p = EspeakG2P(language="fr-fr")
            voice = "ff_siwis"
        elif lang == "it":
            g2p = EspeakG2P(language="it")
            voice = "im_nicola"
        elif lang == "es":
            g2p = EspeakG2P(language="es")
            voice = "im_nicola"
        elif lang == "pt":
            g2p = EspeakG2P(language="pt-br")
            voice = "pf_dora"
        elif lang == "ja":
            from misaki import ja
            import unidic
            import subprocess
            
            # FIX: Auto-download the Japanese dictionary if it's missing!
            if not os.path.exists(unidic.DICDIR):
                print("Downloading missing Japanese dictionary (this takes a minute but only happens once)...")
                subprocess.run([sys.executable, "-m", "unidic", "download"], check=True)
                
            g2p = ja.JAG2P()
            voice = "jf_alpha"
            vocab = self._get_vocab_config(lang)
            # Provide English fallback for mixed Japanese-English text
            en_callable = self._create_en_callable()
        elif lang == "zh":
            from misaki import zh
            import re
            
            base_g2p = zh.ZHG2P(version="1.1")
            en_callable = self._create_en_callable()
            
            # Wrap ZHG2P to handle English tokens in mixed Chinese-English text.
            def mixed_g2p(text):
                # Split on English words/names and process them separately
                parts = re.split(r'([a-zA-Z]+)', text)
                phonemes_list = []
                for part in parts:
                    if part and part[0].isalpha() and part[0].isascii():
                        # English token: use English G2P
                        phonemes_list.append(en_callable(part))
                    else:
                        # Chinese token: use Chinese G2P
                        if part:
                            ph, _ = base_g2p(part)
                            phonemes_list.append(ph)
                result = "".join(phonemes_list)
                return result, text
            
            g2p = mixed_g2p
            voice = "zf_001"
            model_file = self._ensure_file("model", "kokoro-v1.1-zh.onnx")
            voices_file = self._ensure_file("voice", "voices-v1.1-zh.bin")
            vocab = self._get_vocab_config(lang)
        else:
            raise ValueError(f"Language '{lang}' not supported.")

        return model_file, voices_file, vocab, g2p, voice, en_callable

    def precompute_reference(self, reference_audio: str) -> torch.Tensor:
        """Encode *reference_audio* into a speaker embedding and cache it.

        The embedding is keyed by the absolute, normalised file path so that
        any subsequent call to :meth:`generate` or :meth:`convert` with the
        same path reuses the cached tensor without re-reading or re-encoding
        the file.

        Parameters
        ----------
        reference_audio:
            Path to the reference WAV file.

        Returns
        -------
        torch.Tensor
            The global (speaker) embedding tensor of shape ``(dim,)`` on the
            model's device.
        """
        key = os.path.realpath(reference_audio)
        if key not in self._ref_embedding_cache:
            _logger.info(f"precompute_reference path={reference_audio!r} — encoding and caching")
            ref_wav = load_audio(reference_audio, sample_rate=self.sample_rate).to(self.device)
            with torch.inference_mode():
                ref_features = self.kanade.encode(ref_wav, return_content=False, return_global=True)
            self._ref_embedding_cache[key] = ref_features.global_embedding
            _logger.info(f"precompute_reference path={reference_audio!r} — cached embedding shape={ref_features.global_embedding.shape}")
        else:
            _logger.info(f"precompute_reference path={reference_audio!r} — already cached, skipping")
        return self._ref_embedding_cache[key]

    def _get_ref_embedding(self, reference_audio: str) -> torch.Tensor:
        """Return the cached speaker embedding for *reference_audio*, computing
        it on first access.  This is the internal helper used by
        :meth:`generate` and :meth:`convert`."""
        key = os.path.realpath(reference_audio)
        if key not in self._ref_embedding_cache:
            self.precompute_reference(reference_audio)
        return self._ref_embedding_cache[key]

    def generate(self, text, lang, reference_audio, output_path="output.wav", request_id=""):
        """Generates the speech and applies the target voice."""
        model_file, voices_file, vocab, g2p, voice, en_callable = self._get_config(lang)
        
        # 1. Kokoro TTS Phase
        if model_file not in self.kokoro_cache:
            kokoro = Kokoro(model_file, voices_file, vocab_config=vocab) if vocab else Kokoro(model_file, voices_file)
            self.kokoro_cache[model_file] = self._patch_kokoro_compat(kokoro)
        
        kokoro = self.kokoro_cache[model_file]
        
        _logger.info(f"synthesizing lang={lang.upper()} request_id={request_id}")
        _t0 = time.perf_counter()
        if g2p:
            phonemes, _ = g2p(text)
            samples, sr = kokoro.create(phonemes, voice=voice, speed=1.0, is_phonemes=True)
        else:
            samples, sr = kokoro.create(text, voice=voice, speed=0.9, lang="en-us")
        _kokoro_elapsed = time.perf_counter() - _t0
        _audio_duration = len(samples) / sr
        _logger.info(f"phase=kokoro_tts request_id={request_id} elapsed_seconds={_kokoro_elapsed:.3f} audio_duration_seconds={_audio_duration:.3f} lang={lang}")

        # Use a secure temporary file for the base audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = temp_audio.name
            sf.write(temp_path, samples, sr)

        # 2. Kanade Voice Conversion Phase
        try:
            _logger.info(f"applying_voice_clone request_id={request_id}")
            # Load source audio and push to device
            source_wav = load_audio(temp_path, sample_rate=self.sample_rate).to(self.device)

            # Retrieve (or compute-and-cache) the reference speaker embedding.
            # The reference wav is encoded only once per unique file path; all
            # subsequent requests reuse the cached global embedding tensor,
            # saving ~0.3–1 s of GPU work per request.
            ref_emb = self._get_ref_embedding(reference_audio)

            _t1 = time.perf_counter()
            with torch.inference_mode():
                converted_wav = chunked_voice_conversion(
                    kanade=self.kanade,
                    vocoder_model=self.vocoder,
                    source_wav=source_wav,
                    ref_wav=None,           # not needed — embedding is pre-computed
                    sample_rate=self.sample_rate,
                    request_id=request_id,
                    ref_embedding=ref_emb,
                )
            _vc_elapsed = time.perf_counter() - _t1
            _vc_duration = converted_wav.shape[-1] / self.sample_rate
            _logger.info(f"phase=kanade_vc request_id={request_id} elapsed_seconds={_vc_elapsed:.3f} audio_duration_seconds={_vc_duration:.3f}")

            sf.write(output_path, converted_wav.numpy(), self.sample_rate)
            _logger.info(f"phase=total request_id={request_id} total_elapsed_seconds={(_kokoro_elapsed + _vc_elapsed):.3f}")
            _logger.info(f"saved output_path={output_path} request_id={request_id}")

        except Exception as exc:
            _logger.error(f"phase=kanade_vc request_id={request_id} exc_type={type(exc).__name__} message={exc}", exc_info=True)
            raise

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path) # Clean up temp file silently

    def convert(self, source_audio, reference_audio, output_path="output.wav"):
        """Re-voices source_audio to sound like reference_audio using chunking."""
        print("Applying Voice Conversion...")
        source_wav = load_audio(source_audio, sample_rate=self.sample_rate).to(self.device)
        ref_emb = self._get_ref_embedding(reference_audio)

        with torch.inference_mode():
            converted_wav = chunked_voice_conversion(
                kanade=self.kanade,
                vocoder_model=self.vocoder,
                source_wav=source_wav,
                ref_wav=None,
                sample_rate=self.sample_rate,
                ref_embedding=ref_emb,
            )

        sf.write(output_path, converted_wav.numpy(), self.sample_rate)
        print(f"Success! Saved: {output_path}")
