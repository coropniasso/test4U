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

CREATE TABLE IF NOT EXISTS extracted_requests (
    request_id          VARCHAR PRIMARY KEY,  -- {conversation_id}#{model_name}#{連番}
    conversation_id     VARCHAR NOT NULL,
    seq_from            INTEGER NOT NULL,
    seq_to              INTEGER NOT NULL,
    speaker_from_label  VARCHAR,              -- 要求した側の話者ラベル
    speaker_to_label    VARCHAR,              -- 要求された側の話者ラベル
    direction           VARCHAR,              -- self_to_other / other_to_self。speaker_map 確定後に埋める
    counterpart_role    VARCHAR,
    artifact_type       VARCHAR,              -- 成果物の種類。例 announcement / report / analysis
    quote               VARCHAR NOT NULL,     -- 発言の引用。文字起こしの文面をそのまま入れる
    request_summary     VARCHAR NOT NULL,
    stated_reason       VARCHAR,              -- 発言中で語られた理由のみ。語られていなければ NULL
    applies_when        VARCHAR,              -- 発言中で語られた適用条件のみ。語られていなければ NULL
    reusable            BOOLEAN NOT NULL,
    pass_criterion      VARCHAR,              -- Yes/No で判定できる基準。作れなければ NULL
    source_type         VARCHAR NOT NULL,     -- live_conversation / dictated_review
    model_name          VARCHAR NOT NULL,     -- 抽出に使ったモデル。#5 の比較で使う
    extracted_at        TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_items (
    item_id         VARCHAR PRIMARY KEY,
    artifact_type   VARCHAR,
    check_text      VARCHAR NOT NULL,
    pass_criterion  VARCHAR,
    reason          VARCHAR,               -- ユーザーが後から書き足す。抽出時は NULL
    applies_when    VARCHAR,               -- ユーザーが後から書き足す。抽出時は NULL
    not_applies_when VARCHAR,
    importance      INTEGER CHECK (importance BETWEEN 1 AND 3),
    status          VARCHAR NOT NULL,      -- candidate / approved / rejected / held
    version         INTEGER NOT NULL,
    updated_at      TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS item_evidence (
    item_id    VARCHAR NOT NULL,
    request_id VARCHAR NOT NULL,
    similarity DOUBLE,                     -- 新規候補として作られた場合は NULL
    linked_at  TIMESTAMP NOT NULL,
    PRIMARY KEY (item_id, request_id)
);

CREATE TABLE IF NOT EXISTS review_log (
    log_id      BIGINT PRIMARY KEY,        -- シーケンスから採番
    item_id     VARCHAR NOT NULL,
    action      VARCHAR NOT NULL,          -- approve / reject / hold / edit_reason / ...
    before      VARCHAR,
    after       VARCHAR,
    reviewed_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS speaker_map (
    conversation_id VARCHAR NOT NULL,
    speaker_label   VARCHAR NOT NULL,
    person_label    VARCHAR NOT NULL,      -- self / kouhai_a のように英数字で持つ
    PRIMARY KEY (conversation_id, speaker_label)
);

CREATE TABLE IF NOT EXISTS discord_posts (
    message_id      VARCHAR PRIMARY KEY,
    post_type       VARCHAR NOT NULL,      -- new_candidate / contradiction / speaker_sample / bundle_result
    item_id         VARCHAR,
    conversation_id VARCHAR,
    posted_at       TIMESTAMP NOT NULL,
    processed_at    TIMESTAMP              -- 翌晩のバッチで読み取った時刻
);

CREATE TABLE IF NOT EXISTS embeddings (
    owner_type VARCHAR NOT NULL,           -- knowledge_item / extracted_request
    owner_id   VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    dim        INTEGER NOT NULL,
    vector     FLOAT[] NOT NULL,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (owner_type, owner_id, model_name)
);

CREATE SEQUENCE IF NOT EXISTS review_log_seq START 1;

-- 会議ごと・抽出モデルごとの抽出件数と reusable の割合、形式別の比較に使う。
-- model_name で分けるのは、Qwen と Claude の両方で抽出した会議が二重に数えられないようにするため。
-- 1つの会議を2モデルで抽出すると2行になり、各行がその会議を1回だけ数える。
-- 抽出が1件も無い会議は、model_name が NULL、requests が 0 の行になる。
CREATE OR REPLACE VIEW v_weekly_metrics AS
WITH per_model AS (
    SELECT
        date_trunc('week', c.recorded_at) AS week,
        c.setting_format,
        r.model_name,
        count(DISTINCT c.conversation_id)              AS conversations,
        count(r.request_id)                            AS requests,
        sum(CASE WHEN r.reusable THEN 1 ELSE 0 END)    AS reusable_requests
    FROM conversations c
    LEFT JOIN extracted_requests r USING (conversation_id)
    GROUP BY 1, 2, 3
)
SELECT
    week,
    setting_format,
    model_name,                                        -- 抽出していない会議では NULL
    conversations,
    requests,
    requests / nullif(conversations, 0) AS requests_per_conversation,
    reusable_requests / nullif(requests, 0) AS reusable_ratio
FROM per_model;
