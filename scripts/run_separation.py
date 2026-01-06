# scripts/run_separation.py

import argparse
import sys
from pathlib import Path

import numpy as np
import librosa

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from etude.utils.logger import logger


def separate_with_spleeter(
    input_file: Path, mel_filter_bank: np.ndarray, sample_rate: int
):
    """
    Performs 5-stem source separation using Spleeter.

    Args:
        input_file: Path to the input audio file.
        mel_filter_bank: Pre-computed mel filterbank matrix.
        sample_rate: Target sample rate (44100).

    Returns:
        List of mel spectrograms for each stem.
    """
    from spleeter.separator import Separator
    from spleeter.audio.adapter import AudioAdapter

    logger.substep("Initializing Spleeter separator (5stems)...")
    separator = Separator("spleeter:5stems")

    logger.substep(f"Loading audio: {input_file.name}")
    audio_loader = AudioAdapter.default()
    waveform, _ = audio_loader.load(str(input_file), sample_rate=sample_rate)

    logger.substep("Separating audio into 5 stems...")
    separated_stems = separator.separate(waveform)

    logger.substep("Converting each stem to a dB Mel Spectrogram...")
    processed_spectrograms = []
    for key in separated_stems:
        stem_waveform = separated_stems[key]
        stft_result = separator._stft(stem_waveform)
        power_spec = np.abs(np.mean(stft_result, axis=-1)) ** 2
        mel_spec = np.dot(power_spec, mel_filter_bank)
        processed_spectrograms.append(mel_spec)

    return processed_spectrograms


def separate_with_demucs(
    input_file: Path, mel_filter_bank: np.ndarray, sample_rate: int, device: str
):
    """
    Performs 6-stem source separation using Demucs (htdemucs_6s).

    Args:
        input_file: Path to the input audio file.
        mel_filter_bank: Pre-computed mel filterbank matrix.
        sample_rate: Target sample rate (44100).
        device: Device to run inference on.

    Returns:
        List of mel spectrograms for each stem (5 stems to match Spleeter output).
    """
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    # Determine device
    # Note: MPS has limitations with large convolutions (Output channels > 65536)
    # so we fall back to CPU for Demucs on Apple Silicon
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            # MPS is not well supported for Demucs due to conv1d limitations
            device = "cpu"
    elif device == "mps":
        logger.warn("MPS has limitations with Demucs, falling back to CPU")
        device = "cpu"
    logger.substep(f"Using device: {device}")

    logger.substep("Initializing Demucs separator (htdemucs_6s)...")
    model = get_model("htdemucs_6s")
    model.to(device)

    logger.substep(f"Loading audio: {input_file.name}")
    waveform, sr = torchaudio.load(str(input_file))

    # Resample if necessary
    if sr != sample_rate:
        logger.substep(f"Resampling from {sr}Hz to {sample_rate}Hz...")
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Convert mono to stereo if necessary
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    # Add batch dimension and move to device
    waveform = waveform.unsqueeze(0).to(device)

    logger.substep("Separating audio into stems...")
    with torch.no_grad():
        sources = apply_model(model, waveform, device=device, progress=True)

    # sources shape: (batch, num_sources, channels, samples)
    # htdemucs_6s sources order: drums, bass, other, vocals, guitar, piano
    sources = sources.squeeze(0)  # Remove batch dimension

    # Select 5 stems to match original spleeter output format
    # Spleeter 5stems: vocals, drums, bass, piano, other
    # htdemucs_6s: drums(0), bass(1), other(2), vocals(3), guitar(4), piano(5)
    # Note: Spleeter's "other" includes guitar, so we need to merge other(2) + guitar(4)
    stem_configs = [
        ("vocals", [3]),  # vocals
        ("drums", [0]),  # drums
        ("bass", [1]),  # bass
        ("piano", [5]),  # piano
        ("other", [2, 4]),  # other + guitar (to match Spleeter's "other")
    ]

    logger.substep("Converting each stem to a dB Mel Spectrogram...")
    processed_spectrograms = []

    for _, indices in stem_configs:
        # Merge multiple stems if needed (e.g., other + guitar)
        stem_waveform = sum(sources[idx] for idx in indices)  # (channels, samples)
        # Convert to mono by averaging channels
        stem_mono = stem_waveform.mean(dim=0).cpu().numpy()

        # Compute STFT
        stft_result = librosa.stft(stem_mono, n_fft=4096, hop_length=1024)
        power_spec = np.abs(stft_result) ** 2

        # Apply mel filterbank
        mel_spec = np.dot(power_spec.T, mel_filter_bank)
        processed_spectrograms.append(mel_spec)

    return processed_spectrograms


def separate_with_stem_splitter(
    input_file: Path,
    mel_filter_bank: np.ndarray,
    target_sample_rate: int,
    device: str,
):
    import math
    import subprocess
    import tempfile

    import torch
    import librosa
    import torchaudio

    from stem_splitter.inference import (
        SeparationConfig,
        load_mss_model,
        resolve_device,
    )

    # -------------------------
    # config / device / dtype / model
    # -------------------------
    config = SeparationConfig(skip_existing=False)

    dev = resolve_device(config.device_preference)
    use_amp = (
        config.enable_autocast and config.use_half_precision and dev.type == "cuda"
    )
    dtype = (
        torch.float16
        if (config.use_half_precision and dev.type == "cuda")
        else torch.float32
    )

    model = load_mss_model(config, device=dev)

    # -------------------------
    # helper: resample mono (44100 -> target_sample_rate)
    # -------------------------
    def _resample_mono(x: np.ndarray) -> np.ndarray:
        if target_sample_rate == config.target_sample_rate:
            return x
        # torchaudio resample (fast, stable)
        xt = torch.from_numpy(x).unsqueeze(0)  # [1, T]
        resampler = torchaudio.transforms.Resample(
            config.target_sample_rate, target_sample_rate
        )
        yt = resampler(xt).squeeze(0)
        return yt.cpu().numpy().astype(np.float32, copy=False)

    # -------------------------
    # 1) input -> wav(44100, stereo) に揃える（確実にする）
    # -------------------------
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wav_path = input_file
        if input_file.suffix.lower() != ".wav":
            wav_path = td / "input_44100.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(input_file),
                    "-ac",
                    "2",
                    "-ar",
                    str(config.target_sample_rate),
                    str(wav_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        audio, _ = librosa.load(str(wav_path), sr=config.target_sample_rate, mono=False)
        if audio.ndim == 1:
            audio = audio[None, :]
        elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
            audio = audio.T
        audio = audio.astype(np.float32, copy=False)

        channels, total_length = audio.shape

        chunk_size = int(config.chunk_size)
        hop_size = (
            int(config.hop_size) if config.hop_size is not None else chunk_size // 2
        )
        if not (1 <= hop_size <= chunk_size):
            raise ValueError("hop_size must satisfy 1 <= hop_size <= chunk_size.")

        if total_length <= chunk_size:
            padded_length = chunk_size
        else:
            steps = math.ceil((total_length - chunk_size) / hop_size)
            padded_length = steps * hop_size + chunk_size

        if padded_length > total_length:
            audio = np.pad(
                audio, ((0, 0), (0, padded_length - total_length)), mode="constant"
            )

        # hann window
        tensor_dtype = torch.float32 if use_amp else dtype
        base_window = torch.hann_window(
            chunk_size, periodic=False, dtype=tensor_dtype, device=dev
        )
        base_window_np = base_window.to(torch.float32).cpu().numpy()

        stem_names = (
            config.stem_names
        )  # ("bass","drums","other","vocals","guitar","piano")
        num_stems = len(stem_names)

        accum = np.zeros((num_stems, channels, padded_length), dtype=np.float32)
        weight_sum = np.zeros(padded_length, dtype=np.float32)

        # -------------------------
        # overlap-add 推論
        # -------------------------
        with torch.inference_mode():
            for start in range(0, padded_length - chunk_size + 1, hop_size):
                end = start + chunk_size
                input_chunk_np = audio[:, start:end]
                if input_chunk_np.shape[1] < chunk_size:
                    input_chunk_np = np.pad(
                        input_chunk_np,
                        ((0, 0), (0, chunk_size - input_chunk_np.shape[1])),
                        mode="constant",
                    )

                input_chunk = (
                    torch.from_numpy(input_chunk_np)
                    .to(device=dev, dtype=tensor_dtype)
                    .unsqueeze(0)
                )

                with torch.amp.autocast(
                    device_type=dev.type, enabled=use_amp, dtype=torch.float16
                ):
                    output_chunk = model(input_chunk)  # (B, N, C, T)

                if not isinstance(output_chunk, torch.Tensor) or output_chunk.ndim != 4:
                    raise RuntimeError("Model output must be (B, N, C, T).")

                _, _, _, t_out = output_chunk.shape
                if t_out == chunk_size:
                    window_vec = base_window
                    window_np = base_window_np
                else:
                    window_vec = torch.hann_window(
                        t_out, periodic=False, dtype=tensor_dtype, device=dev
                    )
                    window_np = window_vec.to(torch.float32).cpu().numpy()

                out_np = (
                    (output_chunk * window_vec.view(1, 1, 1, -1))
                    .squeeze(0)
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )

                accum[:, :, start : start + t_out] += out_np
                weight_sum[start : start + t_out] += window_np

                del input_chunk, output_chunk

        weight_sum = np.maximum(weight_sum, 1e-8)
        accum /= weight_sum[None, None, :]
        accum = accum[:, :, :total_length]  # trim (N, C, T)

        # -------------------------
        # 6stem -> 5stem (vocals, drums, bass, piano, other=other+guitar)
        # -------------------------
        idx = {name: i for i, name in enumerate(stem_names)}

        def _mono(stem_cc_t: np.ndarray) -> np.ndarray:
            return stem_cc_t.mean(axis=0).astype(np.float32, copy=False)

        vocals = _mono(accum[idx["vocals"]])
        drums = _mono(accum[idx["drums"]])
        bass = _mono(accum[idx["bass"]])
        piano = _mono(accum[idx["piano"]])
        other = _mono(accum[idx["other"]]) + _mono(accum[idx["guitar"]])

        # 44100 -> target_sample_rate へ変換
        vocals = _resample_mono(vocals)
        drums = _resample_mono(drums)
        bass = _resample_mono(bass)
        piano = _resample_mono(piano)
        other = _resample_mono(other)

        def _to_mel(x_mono: np.ndarray) -> np.ndarray:
            stft_result = librosa.stft(x_mono, n_fft=4096, hop_length=1024)
            power_spec = np.abs(stft_result) ** 2
            mel_spec = np.dot(power_spec.T, mel_filter_bank)  # [T, n_mels]
            return mel_spec

        return [
            _to_mel(vocals),
            _to_mel(drums),
            _to_mel(bass),
            _to_mel(piano),
            _to_mel(other),
        ]


def separate_and_extract_features(
    input_path: str, output_path: str, backend: str = "spleeter", device: str = "auto"
):
    """
    Performs source separation and converts each stem into a dB-scaled Mel spectrogram.

    Args:
        input_path (str): Path to the source audio file.
        output_path (str): Path to save the resulting feature array as a .npy file.
        backend (str): Separation backend to use ('spleeter' or 'demucs' or 'stem-splitter').
        device (str): Device to run on (for demucs: 'cuda', 'cpu', 'mps', or 'auto').
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        logger.error(f"Input audio file not found at {input_file}")
        sys.exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Define Mel filter banks, matching the original script's parameters.
        sample_rate = 44100
        mel_filter_bank = librosa.filters.mel(
            sr=sample_rate, n_fft=4096, n_mels=128, fmin=30, fmax=11000
        ).T

        # Run separation based on backend
        if backend == "demucs":
            processed_spectrograms = separate_with_demucs(
                input_file, mel_filter_bank, sample_rate, device
            )
        elif backend == "stem-splitter":
            processed_spectrograms = separate_with_stem_splitter(
                input_file, mel_filter_bank, sample_rate, device
            )
        else:
            processed_spectrograms = separate_with_spleeter(
                input_file, mel_filter_bank, sample_rate
            )

        stacked_mel_specs = np.stack(processed_spectrograms)
        stacked_mel_specs = np.transpose(stacked_mel_specs, (0, 2, 1))

        db_specs = np.stack(
            [librosa.power_to_db(s, ref=np.max) for s in stacked_mel_specs]
        )
        final_features = np.transpose(db_specs, (0, 2, 1))

        logger.substep(f"Saving features to {output_file.name}...")
        np.save(output_file, final_features)

    except Exception as e:
        logger.error(f"An unexpected error occurred during {backend} processing: {e}")
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Audio feature extraction via source separation and Mel spectrogram conversion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to the input audio file.")
    parser.add_argument(
        "--output", required=True, help="Path for the output .npy feature file."
    )
    parser.add_argument(
        "--backend",
        default="spleeter",
        choices=["spleeter", "demucs", "stem-splitter"],
        help="Source separation backend to use.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to run inference on (only used with demucs backend).",
    )
    args = parser.parse_args()

    separate_and_extract_features(args.input, args.output, args.backend, args.device)


if __name__ == "__main__":
    main()
