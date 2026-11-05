"""音声ファイルを 16kHz・モノラル・float32 の波形にデコードする（文字起こしと話者分離で共有）。

PyAV（`av`）を直接使う。faster-whisper の `faster_whisper.audio.decode_audio` は公開 API ではなく、
PyAV の版との組み合わせで壊れたことがある（`av.open()` の `metadata_errors` 引数の削除。
実機で初めて動かして判明）。文字起こしと話者分離が同じ関数で同じ波形を作るので、
両者の時刻は原理的にずれない。

`av.open()` には、版によって消える引数を渡さない。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kijun.deps import import_optional

# 文字起こし（faster-whisper）と話者分離（pyannote）の両方が期待するサンプルレート。ここに1つだけ置く。
SAMPLE_RATE = 16000


class AudioDecodeError(RuntimeError):
    """音声をデコードできなかったときの例外。ファイルのパスを含む。"""


def decode_to_mono_16k(path: Path) -> np.ndarray:
    """path の音声を、16,000Hz・モノラル・float32・-1.0〜1.0 の1次元配列にして返す。

    元のチャンネル数とサンプルレートは問わない（PyAV の resampler で変換する）。
    """
    path = Path(path)
    if not path.is_file():
        raise AudioDecodeError(f"音声ファイルが見つかりません: {path}")
    av = import_optional("av", "transcribe", "音声のデコード")
    try:
        chunks: list[np.ndarray] = []
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise AudioDecodeError(f"音声ストリームがありません: {path}")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
            # frame=None は resampler 内に残ったサンプルを吐き出させる合図。
            frames = list(container.decode(audio=0)) + [None]
            for frame in frames:
                out = resampler.resample(frame)
                # PyAV の版によって、resample() はフレーム1つまたはフレームのリストを返す。
                if out is None:
                    continue
                for f in out if isinstance(out, list) else [out]:
                    chunks.append(f.to_ndarray().reshape(-1))
    except AudioDecodeError:
        raise
    except Exception as e:
        raise AudioDecodeError(f"音声をデコードできませんでした: {path}（{type(e).__name__}: {e}）") from e
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    return np.clip(audio, -1.0, 1.0)
