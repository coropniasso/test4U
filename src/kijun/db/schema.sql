CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER NOT NULL,
    applied_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id VARCHAR PRIMARY KEY,   -- 例 20260920_1930_f2f_teirei
    recorded_at     TIMESTAMP NOT NULL,    -- 束の先頭ファイルの開始時刻
    setting_format  VARCHAR NOT NULL,      -- f2f / call
    setting_kind    VARCHAR NOT NULL,      -- teirei / sagyou / zatsudan / ...
    source_files    VARCHAR[] NOT NULL,    -- 束ねた音声ファイル名（processed 配下の相対パス）
    transcript_path VARCHAR,               -- 文字起こしの JSON の相対パス
    duration_sec    DOUBLE,                -- 束ねた音声の合計の長さ
    transcribed_at  TIMESTAMP,             -- 文字起こしが完了した時刻。#9 の削除判定に使う
    created_at      TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS utterances (
    conversation_id VARCHAR NOT NULL,
    seq             INTEGER NOT NULL,      -- 会議内で 0 から始まる通し番号
    speaker_label   VARCHAR,               -- SPEAKER_00 など。話者分離が失敗した場合は NULL
    start_sec       DOUBLE NOT NULL,
    end_sec         DOUBLE NOT NULL,
    text            VARCHAR NOT NULL,
    PRIMARY KEY (conversation_id, seq)
);

-- 議事録（minutes/）。1会議につき1行。Markdown 本体は path のファイルにある。
-- discord_message_id と posted_at は Discord に投稿したときに埋める。posted_at が非 NULL なら再投稿しない。
CREATE TABLE IF NOT EXISTS minutes (
    conversation_id    VARCHAR PRIMARY KEY,
    path               VARCHAR NOT NULL,   -- 議事録の Markdown のパス（設定 paths.minutes 配下）
    model_name         VARCHAR NOT NULL,   -- 議事録の生成に使ったモデル
    created_at         TIMESTAMP NOT NULL,
    discord_message_id VARCHAR,
    posted_at          TIMESTAMP
);

-- 2026-11-06 に目的をチェックリストの作成から議事録の作成に変えた（CLAUDE.md）。以前のテーブル
-- （extracted_requests、knowledge_items、item_evidence、review_log、speaker_map、discord_posts、embeddings）
-- と、それらに依存するビューは、新しい DB には作らない。既存の DB にあるものは、データを消さないために
-- DROP せず残す（db/migrate.py）。
