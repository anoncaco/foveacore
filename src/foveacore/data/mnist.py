"""FoveaCore Phase 0 MNIST 원본 다운로드·SHA-256·IDX 파싱 (T3).

원본 파일은 data/raw/에 로컬 고정한다. 실제 IDX 파일을 파싱해 이미지/라벨을
추출하고, 이미지 픽셀 기반 SHA-256 및 지각 해시(pHash)를 계산한다.
"""

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

MNIST_FILES = {
    "train-images-idx3-ubyte.gz": {
        "url": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "sha256": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    },
    "train-labels-idx1-ubyte.gz": {
        "url": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "sha256": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    },
    "t10k-images-idx3-ubyte.gz": {
        "url": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "sha256": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    },
    "t10k-labels-idx1-ubyte.gz": {
        "url": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
        "sha256": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_mnist(raw_dir: Path | str, force: bool = False) -> Dict[str, str]:
    """MNIST 원본을 data/raw/에 다운로드하고 {filename: sha256}을 반환."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, str] = {}
    for filename, info in MNIST_FILES.items():
        dst = raw_dir / filename
        if dst.exists() and not force:
            actual = sha256_file(dst)
            expected = info.get("sha256")
            if expected is not None and actual != expected:
                raise RuntimeError(
                    f"MNIST 체크섬 불일치({filename}): expected={expected}, actual={actual}"
                )
            results[filename] = actual
            continue
        try:
            urllib.request.urlretrieve(info["url"], dst)
        except Exception as exc:
            raise RuntimeError(f"MNIST 다운로드 실패({filename}): {exc}") from exc
        actual = sha256_file(dst)
        expected = info.get("sha256")
        if expected is not None and actual != expected:
            raise RuntimeError(
                f"MNIST 체크섬 불일치({filename}): expected={expected}, actual={actual}"
            )
        results[filename] = actual
    return results


def verify_mnist_checksums(raw_dir: Path | str, expected: Dict[str, str] | None = None) -> List[str]:
    """예상 체크섬과 실제 파일 체크섬을 비교. 불일치 목록 반환."""
    raw_dir = Path(raw_dir)
    errors: List[str] = []
    if expected is None:
        expected = {name: info.get("sha256") for name, info in MNIST_FILES.items()}
    for filename, expected_hash in expected.items():
        dst = raw_dir / filename
        if not dst.exists():
            errors.append(f"원본 파일 누락: {filename}")
            continue
        if expected_hash is None:
            # 체크섬 미고정: 존재만 확인
            continue
        actual = sha256_file(dst)
        if actual != expected_hash:
            errors.append(
                f"체크섬 불일치({filename}): expected={expected_hash}, actual={actual}"
            )
    return errors


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def read_idx_images(path: Path | str) -> np.ndarray:
    """IDX 이미지 파일을 (N, 28, 28) uint8 배열로 반환."""
    path = Path(path)
    with _open_maybe_gzip(path) as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"잘못된 IDX 이미지 magic: {magic}")
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8).reshape(num, rows, cols)
    return arr


def read_idx_labels(path: Path | str) -> np.ndarray:
    """IDX 라벨 파일을 (N,) uint8 배열로 반환."""
    path = Path(path)
    with _open_maybe_gzip(path) as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"잘못된 IDX 라벨 magic: {magic}")
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8).reshape(num)
    return arr


def validate_mnist_data(train_images, train_labels, test_images, test_labels):
    errors = []
    if train_images.shape != (60000, 28, 28):
        errors.append(f"train_images shape 이상: {train_images.shape}")
    if test_images.shape != (10000, 28, 28):
        errors.append(f"test_images shape 이상: {test_images.shape}")
    if train_labels.shape != (60000,):
        errors.append(f"train_labels shape 이상: {train_labels.shape}")
    if test_labels.shape != (10000,):
        errors.append(f"test_labels shape 이상: {test_labels.shape}")
    if train_images.dtype != np.uint8 or test_images.dtype != np.uint8:
        errors.append("이미지 dtype은 uint8이어야 합니다.")
    if not np.all((train_labels >= 0) & (train_labels <= 9)):
        errors.append("train 라벨 범위 위반")
    if not np.all((test_labels >= 0) & (test_labels <= 9)):
        errors.append("test 라벨 범위 위반")
    if not np.all(np.isfinite(train_images)) or not np.all(np.isfinite(test_images)):
        errors.append("이미지에 NaN/Inf 포함")
    if errors:
        raise ValueError("; ".join(errors))


def load_mnist(raw_dir: Path | str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """data/raw/에서 MNIST train/test 이미지·라벨을 로드."""
    raw_dir = Path(raw_dir)
    train_images = read_idx_images(raw_dir / "train-images-idx3-ubyte.gz")
    train_labels = read_idx_labels(raw_dir / "train-labels-idx1-ubyte.gz")
    test_images = read_idx_images(raw_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = read_idx_labels(raw_dir / "t10k-labels-idx1-ubyte.gz")
    validate_mnist_data(train_images, train_labels, test_images, test_labels)
    return train_images, train_labels, test_images, test_labels


def sha256_image(img: np.ndarray) -> str:
    """이미지 픽셀 기반 SHA-256 (T3-02)."""
    arr = np.ascontiguousarray(img, dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _dct2(a: np.ndarray) -> np.ndarray:
    # 2D DCT-II (orthonormal not required; only ordering matters)
    m, n = a.shape
    # 1D DCT
    def dct1(x):
        N = x.shape[-1]
        # use numpy fft-based? simple direct
        k = np.arange(N)
        n_idx = np.arange(N)
        # cos matrix
        cos_mat = np.cos(np.pi * (2 * n_idx[None, :] + 1) * k[:, None] / (2 * N))
        return cos_mat @ x
    # apply along last axis then first
    r = np.apply_along_axis(dct1, -1, a.astype(float))
    r = np.apply_along_axis(dct1, 0, r)
    return r


def phash_image(img: np.ndarray, hash_size: int = 8) -> str:
    """간단한 pHash (DCT 기반) (T3-03). 16진 문자열 반환."""
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., 0]
    # resize to 32x32 via nearest neighbor
    h, w = img.shape
    ys = (np.arange(32) * h / 32).astype(int)
    xs = (np.arange(32) * w / 32).astype(int)
    small = img[ys][:, xs].astype(float)
    dct = _dct2(small)
    low = dct[:hash_size, :hash_size]
    med = np.median(low)
    bits = (low > med).flatten()
    # pack bits into hex
    val = 0
    for b in bits:
        val = (val << 1) | int(bool(b))
    return f"{val:0{hash_size*hash_size//4}x}"


def mnist_available(raw_dir: Path | str) -> bool:
    raw_dir = Path(raw_dir)
    required = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    return all((raw_dir / f).exists() for f in required)