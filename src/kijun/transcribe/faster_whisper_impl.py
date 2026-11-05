"""faster-whisper による文字起こし（計画書 6-2）。

GPU と large-v3 のモデルファイルが必要。動作確認は Windows の実機で行う。
faster_whisper は `uv sync --extra transcribe` で入る。モジュールのトップレベルでは import しない。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from kijun.config import TranscribeConfig
from kijun.deps import import_optional
from kijun.transcribe.audio_decode import SAMPLE_RATE, decode_to_mono_16k
from kijun.transcribe.base import TranscribedSegment, shift_items
from kijun.transcribe.cuda_dlls import register_nvidia_dll_directories


# faster-whisper の VadOptions の項目名と、こちらの設定名の対応。
# 版によって項目が増減するので、実際に受け付ける名前だけを渡す（知らない名前で TypeError にしない）。
VAD_PARAMETER_SOURCES = {
    "threshold": "vad_threshold",
    "min_silence_duration_ms": "vad_min_silence_duration_ms",
    "speech_pad_ms": "vad_speech_pad_ms",
    "max_speech_duration_s": "vad_max_speech_duration_sec",
}


def vad_option_names(fw) -> set[str] | None:
    """その版の VadOptions が受け付ける項目名。見つからなければ None（絞り込まない）。"""
    holder = getattr(fw, "vad", None)
    options = getattr(holder, "VadOptions", None) if holder is not None else None
    if options is None:
        options = getattr(fw, "VadOptions", None)
    if options is None:
        return None
    try:
        return {f.name for f in dataclasses.fields(options)}
    except TypeError:
        return None


def build_vad_parameters(cfg: TranscribeConfig, accepted: set[str] | None = None) -> dict:
    """faster-whisper に渡す vad_parameters を作る。

    既定のままでは区間が切れないことが実機で分かったので、設定から渡す（2026-11-04。34.2分の
    通話で 564.8 秒の区間が1つ出て、中身は3文字だった）。faster-whisper は
    max_speech_duration_s を既定で無制限にし、min_silence_duration_ms より短い無音では
    区間を切らない。PC のスピーカー出力を iPhone で録る方法では室内の音が常に入るので、
    2秒続く無音が現れず、区間が延々とつながる。

    accepted には、その版の VadOptions が受け付ける項目名を渡す。その中にある名前だけを返す。
    None なら絞り込まない。
    vad_max_speech_duration_sec が 0（上限なし）のときは max_speech_duration_s を入れない。
    """
    values: dict = {
        "threshold": cfg.vad_threshold,
        "min_silence_duration_ms": cfg.vad_min_silence_duration_ms,
        "speech_pad_ms": cfg.vad_speech_pad_ms,
    }
    if cfg.vad_max_speech_duration_sec > 0:
        values["max_speech_duration_s"] = cfg.vad_max_speech_duration_sec
    if accepted is None:
        return values
    return {k: v for k, v in values.items() if k in accepted}


class FasterWhisperTranscriber:
    """Transcriber の実装。モデルは __init__ で読み込む。"""

    def __init__(self, cfg: TranscribeConfig) -> None:
        # 順序が重要。faster_whisper を import すると ctranslate2 の拡張モジュールが読み込まれ、
        # その時点で cublas64_12.dll などの解決が行われる。DLL のディレクトリの登録は import より前に
        # 済ませないと手遅れになる。
        dll_dirs = register_nvidia_dll_directories()
        fw = import_optional("faster_whisper", "transcribe", "faster-whisper による文字起こし")
        self._cfg = cfg
        # 空のリストなら None（hotwords を渡さない）。faster-whisper の hotwords は1つの文字列を取る。
        self._hotwords: str | None = " ".join(w.strip() for w in cfg.hotwords if w.strip()) or None
        self._vad_parameters = build_vad_parameters(cfg, vad_option_names(fw))
        try:
            self._model = fw.WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
        except Exception as e:
            if "cublas" not in str(e).lower() and "cudnn" not in str(e).lower():
                raise
            listed = ", ".join(str(d) for d in dll_dirs) or "なし"
            raise RuntimeError(
                f"CUDA 12 の DLL を読み込めませんでした（{e}）。"
                "次のコマンドを実行してください: uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 。"
                "CTranslate2 は CUDA 12 の DLL（cublas64_12.dll など）を要求し、torch の CUDA 13 の DLL"
                "（cublas64_13.dll）とは名前が違うため、両方を入れて共存させる必要があります。"
                f"登録を試みたディレクトリ: {listed}"
            ) from e

    def transcribe(self, audio_paths: list[Path]) -> list[TranscribedSegment]:
        result: list[TranscribedSegment] = []
        halluc = self._cfg.hallucination_silence_threshold
        offset = 0.0  # これまでに処理したファイルの長さの合計（秒）
        for path in audio_paths:
            # 自前でデコードした配列を渡す。np.ndarray なら faster-whisper は内部の decode_audio を呼ばない
            # （その関数は非公開で、PyAV の版との組み合わせで壊れたことがある）。
            audio = decode_to_mono_16k(path)
            kwargs: dict = {
                "language": self._cfg.language,
                "vad_filter": self._cfg.vad_filter,
                "condition_on_previous_text": self._cfg.condition_on_previous_text,
                # hallucination_silence_threshold は word_timestamps=True のときだけ働く（faster-whisper の仕様）
                "word_timestamps": halluc > 0,
                "hallucination_silence_threshold": halluc if halluc > 0 else None,
                "hotwords": self._hotwords,
            }
            # vad_filter が false のときは渡さない（使われないうえ、版によっては検証される）。
            if self._cfg.vad_filter and self._vad_parameters:
                kwargs["vad_parameters"] = self._vad_parameters
            segments, _info = self._model.transcribe(audio, **kwargs)
            # segments は generator。モデルを解放した後に評価されて落ちないよう、ここで list にする。
            one_file = [
                TranscribedSegment(
                    start_sec=s.start,
                    end_sec=s.end,
                    text=s.text,
                    avg_logprob=getattr(s, "avg_logprob", None),
                )
                for s in list(segments)
            ]
            result.extend(shift_items(one_file, offset))
            # info.duration ではなく、デコードした配列の長さから秒数を出す。話者分離が同じ配列の
            # 長さでファイルをつなぐので、両者のオフセットが常に一致する。
            offset += len(audio) / SAMPLE_RATE
        return result
