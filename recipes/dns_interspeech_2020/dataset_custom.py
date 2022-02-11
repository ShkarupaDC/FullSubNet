import random
import numpy as np
import librosa
import soundfile as sf
from math import sqrt
from joblib import Parallel, delayed
import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional, TypeVar
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))

from audio_zen.dataset.base_dataset import BaseDataset
from audio_zen.acoustics.feature import (
    norm_amplitude,
    tailor_dB_FS,
    is_clipped,
)

Signal = np.ndarray
T = TypeVar("T")


def load_wav(path: str, sr: int = 16000, offset: float = 0.5) -> Signal:
    return librosa.load(path, sr=sr, mono=True, offset=offset)[0]


def subsample(signal: Signal, num_target: int) -> Signal:
    num_samples = len(signal)
    if num_samples > num_target:
        start = random.randint(0, num_samples - num_target)
        end = start + num_target
        return signal[start:end]
    return signal


def pad_signal(signal: Signal, num_target: int) -> Signal:
    num_samples = len(signal)
    if num_samples < num_target:
        num_miss = num_target - num_samples
        return np.pad(signal, (0, num_miss), mode="constant")
    return signal


def save_audio(path: str, signal: Signal, sr: int) -> None:
    sf.write(str(path), signal, samplerate=int(sr), format="wav")


class CustomDataset(BaseDataset):
    def __init__(
        self,
        stage: str,
        clean_dir: str,
        noise_dir: str,
        subsample_len: int,
        fixed_noise_scale: bool,
        noise_scale: float,
        snr_range: tuple[float, float],
        target_dbfs: float,
        target_dbfs_float: float,
        sr: int = 16000,
        offset: float = 0,  # in seconds
        preload_clean: bool = False,
        preload_noise: bool = False,
        num_workers: int = 0,
        cleaned_dir: Optional[str] = None,
    ):
        super().__init__()
        self.stage = stage
        self.sr = sr
        self.num_workers = num_workers

        self.clean_paths = list(map(str, Path(clean_dir).iterdir()))
        self.noise_paths = list(map(str, Path(noise_dir).iterdir()))

        self.clean_data = (
            self._preload_data(self.clean_paths, description="Load clean")
            if preload_clean
            else None
        )
        self.noise_data = (
            self._preload_data(self.noise_paths, description="Load noise")
            if preload_noise
            else None
        )
        self.fixed_noise_scale = fixed_noise_scale
        if not fixed_noise_scale:
            self.snr_list = self._parse_snr_range(snr_range)
        self.noise_scale = noise_scale

        self.num_samples = int(subsample_len * sr)
        self.offset = offset
        self.target_dbfs = target_dbfs
        self.target_dbfs_float = target_dbfs_float

        self.cleaned_dir = cleaned_dir
        if self.cleaned_dir is not None:

            self.cleaned_dir = Path(self.cleaned_dir)
            self.cleaned_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.clean_paths)

    def _preload_data(
        self, paths: list[str], type: str, description: str
    ) -> list[Signal]:
        waveforms = Parallel(n_jobs=self.num_workers)(
            delayed(load_wav)(
                path,
                sr=self.sr,
                offset=self.offset if type == "clean" else 0,
            )
            for path in tqdm(paths, total=len(paths), desc=description)
        )
        return waveforms

    @staticmethod
    def _select_random(paths: list[T]) -> T:
        return random.choice(paths)

    def _select_noise(self, target_len: int) -> Signal:
        rest_len = target_len
        noise = []
        while rest_len > 0:

            noise_idx = self._select_random(range(len(self.noise_paths)))
            if self.noise_data is None:
                new_noise = load_wav(self.noise_paths[noise_idx], sr=self.sr)
            else:
                new_noise = self.noise_data[noise_idx]
            noise.append(new_noise)
            rest_len -= len(new_noise)

        noise = np.concatenate(noise)
        noise_len = len(noise)

        if noise_len > target_len:
            start = random.randint(0, noise_len - target_len)
            end = start + target_len
            return noise[start:end]
        return noise

    def mix_signals(
        self, clean: np.ndarray, noise: np.ndarray, eps: float = 1e-6
    ) -> tuple[Signal, Signal]:

        clean = norm_amplitude(clean)[0]
        clean = tailor_dB_FS(clean, self.target_dbfs)[0]
        clean_rms = sqrt((clean**2).mean())

        noise = norm_amplitude(noise)[0]
        noise = tailor_dB_FS(noise, self.target_dbfs)[0]
        noise_rms = sqrt((noise**2).mean())

        if not self.fixed_noise_scale:
            snr = self._select_random(self.snr_list)
            snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
        else:
            snr_scalar = self.noise_scale
        noise *= snr_scalar
        noisy = clean + noise

        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dbfs = random.randint(
            self.target_dbfs - self.target_dbfs_float,
            self.target_dbfs + self.target_dbfs_float,
        )
        noisy, _, noisy_scalar = tailor_dB_FS(noisy, noisy_target_dbfs)
        clean *= noisy_scalar

        if is_clipped(noisy):
            noisy_scalar = np.max(np.abs(noisy)) / (0.99 - eps)
            noisy = noisy / noisy_scalar
            clean = clean / noisy_scalar
        return noisy, clean

    def __getitem__(self, index: int) -> tuple[Signal, Signal]:
        if self.clean_data is None:
            clean_path = self.clean_paths[index]
            clean = load_wav(clean_path, sr=self.sr, offset=self.offset)
        else:
            clean = self.clean_data[index]

        clean = subsample(clean, self.num_samples)
        clean = pad_signal(clean, self.num_samples)

        if self.cleaned_dir is not None:
            save_audio(
                self.cleaned_dir / Path(clean_path).name, clean, self.sr
            )
        noise = self._select_noise(len(clean))
        noisy, clean = self.mix_signals(clean=clean, noise=noise)

        noisy = noisy.astype(np.float32)
        clean = clean.astype(np.float32)

        assert len(noisy) == len(clean) and len(clean) == self.num_samples

        if self.stage == "train":
            return noisy, clean
        if self.stage == "inference":
            return noisy, Path(clean_path).stem
        return noisy, clean, f"val_{Path(clean_path).stem}", "No_reverb"


from audio_zen.utils import initialize_module
import toml

if __name__ == "__main__":
    dataset_path = "fullsubnet/dataset.toml"
    idx = 10

    config = toml.load(dataset_path)
    dataset_config = config["train_dataset"]

    dataset = initialize_module(
        path=dataset_config["path"], args=dataset_config["args"]
    )
    for idx in range(100):
        noisy, clean = dataset[idx]
        print(noisy.shape, clean.shape)
