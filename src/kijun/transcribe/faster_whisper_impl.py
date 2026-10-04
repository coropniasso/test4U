"""faster-whisper による文字起こし（計画書 6-2）。

GPU と large-v3 のモデルファイルが必要。動作確認は Windows の実機で行う。
faster_whisper は `uv sync --extra transcribe` で入る。モジュールのトップレベルでは import しない。
"""

from __future__ import annotations

from pathlib import Path

from kijun.config import TranscribeConfig
from kijun.deps import import_optional
from kijun.transcribe.base import TranscribedSegment, shift_items


class FasterWhisperTranscriber:
    """Transcriber の実装。モデルは __init__ で読み込む。"""

    def __init__(self, cfg: TranscribeConfig) -> None:
        fw = import_optional("faster_whisper", "transcribe", "faster-whisper による文字起こし")
        self._cfg = cfg
        self._model = fw.WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)

    def transcribe(self, audio_paths: list[Path]) -> list[TranscribedSegment]:
        result: list[TranscribedSegment] = []
        offset = 0.0  # これまでに処理したファイルの長さの合計（秒）
        for path in audio_paths:
            segments, info = self._model.transcribe(
                str(path),
                language=self._cfg.language,
                vad_filter=self._cfg.vad_filter,
                word_timestamps=False,
            )
            # segments は generator。モデルを解放した後に評価されて落ちないよう、ここで list にする。
            one_file = [
                TranscribedSegment(start_sec=s.start, end_sec=s.end, text=s.text) for s in list(segments)
            ]
            result.extend(shift_items(one_file, offset))
            offset += info.duration
        return result
