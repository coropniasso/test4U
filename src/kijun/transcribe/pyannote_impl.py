"""pyannote.audio による話者分離（計画書 6-3）。

GPU、Hugging Face のトークン、モデルページでの利用条件への同意が必要。
動作確認は Windows の実機で行う。モジュールのトップレベルでは重い依存を import しない。
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import numpy as np

from kijun.config import DiarizeConfig
from kijun.deps import import_optional
from kijun.transcribe.base import SpeakerTurn

_SAMPLE_RATE = 16000

HF_TOKEN_MISSING_MESSAGE = (
    "環境変数 HF_TOKEN が設定されていません。pyannote のモデルを取得するには次の3つが必要です。\n"
    "  1. Hugging Face のアカウントでアクセストークンを作る（https://huggingface.co/settings/tokens）。\n"
    "  2. https://huggingface.co/pyannote/speaker-diarization-3.1 のページで利用条件に同意する。\n"
    "  3. https://huggingface.co/pyannote/segmentation-3.0 のページでも利用条件に同意する。\n"
    "     （2 と 3 の両方に同意しないと、モデルの取得が 401 / 403 で失敗する）\n"
    "  そのうえで、PowerShell なら `$env:HF_TOKEN = \"hf_...\"` を実行してから再度実行してください。"
)


class PyannoteDiarizer:
    """Diarizer の実装。パイプラインは __init__ で読み込む。"""

    def __init__(self, cfg: DiarizeConfig) -> None:
        pyannote_audio = import_optional("pyannote.audio", "transcribe", "pyannote による話者分離")
        torch = import_optional("torch", "transcribe", "pyannote による話者分離")
        # faster_whisper の decode_audio（PyAV）で音声を読むために使う。
        self._fw_audio = import_optional(
            "faster_whisper.audio", "transcribe", "pyannote に渡す音声の読み込み"
        )
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(HF_TOKEN_MISSING_MESSAGE)

        self._cfg = cfg
        self._torch = torch
        Pipeline = pyannote_audio.Pipeline
        # pyannote.audio 3.x の引数名は use_auth_token、4.x 以降は token。署名を見て選ぶ。
        params = inspect.signature(Pipeline.from_pretrained).parameters
        token_kwarg = "token" if "token" in params else "use_auth_token"
        try:
            pipeline = Pipeline.from_pretrained(cfg.model, **{token_kwarg: token})
        except Exception as e:
            raise RuntimeError(
                f"pyannote のモデル '{cfg.model}' を読み込めませんでした: {e}\n"
                "HF_TOKEN が有効か、speaker-diarization-3.1 と segmentation-3.0 の両方の"
                "モデルページで利用条件に同意したかを確認してください。"
            ) from e
        if pipeline is None:
            raise RuntimeError(
                f"pyannote のモデル '{cfg.model}' を取得できませんでした（None が返りました）。"
                "利用条件への同意と HF_TOKEN を確認してください。"
            )
        pipeline.to(torch.device(cfg.device))
        self._pipeline = pipeline

    def _load_waveform(self, audio_paths: list[Path]) -> dict:
        """全ファイルを 16kHz モノラルで読み込んで時間方向につなぎ、pyannote が受け取る辞書にする。

        読み込みに faster-whisper が使う PyAV を使う理由は、pyannote が内部で使う torchaudio が
        Windows で m4a を読めないことがあるため。
        複数ファイルを1本につなぐ理由は、ファイルごとに話者分離すると SPEAKER_00 が
        ファイルによって別人を指し、同じ会議の中で話者ラベルが一貫しなくなるため。
        つないだ後の時刻は、会議の先頭からの秒数になる。
        """
        arrays = [
            self._fw_audio.decode_audio(str(p), sampling_rate=_SAMPLE_RATE) for p in audio_paths
        ]
        audio = arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
        waveform = self._torch.from_numpy(audio).unsqueeze(0)  # (チャンネル数 1, サンプル数)
        return {"waveform": waveform, "sample_rate": _SAMPLE_RATE}

    def diarize(self, audio_paths: list[Path]) -> list[SpeakerTurn]:
        num_speakers = self._cfg.num_speakers or None  # 0 は「指定なし」
        output = self._pipeline(self._load_waveform(audio_paths), num_speakers=num_speakers)
        # pyannote.audio 4.x は DiarizeOutput を返し、3.x は Annotation を返す。
        annotation = getattr(output, "speaker_diarization", output)
        return [
            SpeakerTurn(start_sec=seg.start, end_sec=seg.end, speaker_label=label)
            for seg, _track, label in annotation.itertracks(yield_label=True)
        ]
