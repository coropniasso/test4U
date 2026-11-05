"""pyannote.audio による話者分離（計画書 6-3）。

GPU、Hugging Face のトークン、3つのモデルページでの利用条件への同意が必要。
動作確認は Windows の実機で行う。モジュールのトップレベルでは重い依存を import しない。
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import numpy as np

from kijun.config import DiarizeConfig
from kijun.deps import import_optional
from kijun.transcribe.audio_decode import SAMPLE_RATE, decode_to_mono_16k
from kijun.transcribe.base import SpeakerTurn

MODEL_PAGES_TEXT = (
    "     - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
    "     - https://huggingface.co/pyannote/segmentation-3.0\n"
    "     - https://huggingface.co/pyannote/speaker-diarization-community-1\n"
    "       （3つ目は pyannote.audio 4.x で必要。speaker-diarization-3.1 の内部から"
    " plda/xvec_transform.npz として読まれる）\n"
    "     （1つでも欠けると、モデルの取得が 401 / 403 で失敗する）\n"
)

HF_TOKEN_MISSING_MESSAGE = (
    "環境変数 HF_TOKEN が設定されていません。pyannote のモデルを取得するには次が必要です。\n"
    "  1. Hugging Face のアカウントでアクセストークンを作る（https://huggingface.co/settings/tokens）。\n"
    f"  2. 次の3つのページすべてで利用条件に同意する。\n{MODEL_PAGES_TEXT}"
    "  そのうえで、PowerShell なら `$env:HF_TOKEN = \"hf_...\"` を実行してから再度実行してください。"
)


class PyannoteDiarizer:
    """Diarizer の実装。パイプラインは __init__ で読み込む。"""

    def __init__(self, cfg: DiarizeConfig) -> None:
        pyannote_audio = import_optional("pyannote.audio", "transcribe", "pyannote による話者分離")
        torch = import_optional("torch", "transcribe", "pyannote による話者分離")
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
                "HF_TOKEN が有効か、次の3つのモデルページすべてで利用条件に同意したかを"
                f"確認してください。\n{MODEL_PAGES_TEXT}"
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

        読み込みに自前の PyAV デコーダ（audio_decode）を使う理由は、pyannote が内部で使う torchaudio が
        Windows で m4a を読めないことがあるため。文字起こしも同じデコーダを使うので、両者の時刻はずれない。
        複数ファイルを1本につなぐ理由は、ファイルごとに話者分離すると SPEAKER_00 が
        ファイルによって別人を指し、同じ会議の中で話者ラベルが一貫しなくなるため。
        つないだ後の時刻は、会議の先頭からの秒数になる。
        """
        arrays = [decode_to_mono_16k(p) for p in audio_paths]
        audio = arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
        waveform = self._torch.from_numpy(audio).unsqueeze(0)  # (チャンネル数 1, サンプル数)
        return {"waveform": waveform, "sample_rate": SAMPLE_RATE}

    def diarize(self, audio_paths: list[Path]) -> list[SpeakerTurn]:
        num_speakers = self._cfg.num_speakers or None  # 0 は「指定なし」
        output = self._pipeline(self._load_waveform(audio_paths), num_speakers=num_speakers)
        # pyannote.audio 4.x は DiarizeOutput を返し、3.x は Annotation を返す。
        annotation = getattr(output, "speaker_diarization", output)
        return [
            SpeakerTurn(start_sec=seg.start, end_sec=seg.end, speaker_label=label)
            for seg, _track, label in annotation.itertracks(yield_label=True)
        ]
