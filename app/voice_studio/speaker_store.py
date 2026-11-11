"""Transactional SQLite persistence for profiles, sample vectors and thresholds."""

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


class ProfileError(ValueError):
    pass


class RevisionConflict(ProfileError):
    pass


def unit_vector(value):
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if not 2 <= vector.size <= 4096 or not np.isfinite(vector).all():
        raise ProfileError("声纹向量无效，请重新录音。")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ProfileError("未提取到有效声纹，请重新录音。")
    return vector / norm


def centroid(vectors):
    if not vectors:
        raise ProfileError("缺少参考录音。")
    normalized = [unit_vector(v) for v in vectors]
    if len({v.size for v in normalized}) != 1:
        raise ProfileError("声纹维度不一致，需要重新登记。")
    return unit_vector(np.mean(normalized, axis=0))


def timestamp():
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Sample:
    embedding: np.ndarray
    sha256: str
    quality: dict
    audio_path: str | None = None


class SpeakerStore:
    def __init__(self, database):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS speakers (
                    speaker_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    sample_count INTEGER NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS samples (
                    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    speaker_id TEXT NOT NULL REFERENCES speakers(speaker_id) ON DELETE CASCADE,
                    sha256 TEXT NOT NULL UNIQUE,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    quality TEXT NOT NULL,
                    audio_path TEXT
                );
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT OR IGNORE INTO settings VALUES (
                    'recognition', '{"threshold":0.6,"margin":0.05,"calibration":null}'
                );
                PRAGMA user_version=1;
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA secure_delete=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _identity(speaker_id, display_name):
        speaker_id = str(speaker_id or "").strip().lower()
        display_name = str(display_name or "").strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", speaker_id):
            raise ProfileError("用户 ID 需为 1–48 位英文字母、数字、下划线或短横线，以字母或数字开头。")
        if not display_name or len(display_name) > 60 or any(ord(c) < 32 for c in display_name):
            raise ProfileError("请输入 1–60 个字符的显示姓名。")
        return speaker_id, display_name

    @staticmethod
    def _insert_samples(db, speaker_id, samples, dimension):
        seen = set()
        for sample in samples:
            vector = unit_vector(sample.embedding)
            if vector.size != dimension:
                raise ProfileError("声纹维度不一致，需要重新登记。")
            if not re.fullmatch(r"[0-9a-f]{64}", sample.sha256) or sample.sha256 in seen:
                raise ProfileError("参考录音重复或指纹无效，请使用不同的录音。")
            seen.add(sample.sha256)
            if db.execute("SELECT 1 FROM samples WHERE sha256=?", (sample.sha256,)).fetchone():
                raise ProfileError("有录音已用于登记，请换一段新录音。")
            db.execute(
                "INSERT INTO samples(speaker_id,sha256,embedding,dimension,quality,audio_path) VALUES (?,?,?,?,?,?)",
                (
                    speaker_id,
                    sample.sha256,
                    vector.astype("<f4").tobytes(),
                    dimension,
                    json.dumps(sample.quality, ensure_ascii=False),
                    sample.audio_path,
                ),
            )

    def enroll(self, speaker_id, display_name, samples, model_id):
        speaker_id, display_name = self._identity(speaker_id, display_name)
        if len(samples) < 3 or len(samples) > 20:
            raise ProfileError("请提供 3–20 段不同的参考录音，建议每段 5–10 秒。")
        vector = centroid([sample.embedding for sample in samples])
        now = timestamp()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM speakers WHERE speaker_id=?", (speaker_id,)).fetchone():
                raise ProfileError("这个用户 ID 已存在，请选择更新档案或使用其他 ID。")
            db.execute(
                "INSERT INTO speakers VALUES (?,?,?,?,?,?,1,?,?)",
                (
                    speaker_id,
                    display_name,
                    model_id,
                    vector.astype("<f4").tobytes(),
                    vector.size,
                    len(samples),
                    now,
                    now,
                ),
            )
            self._insert_samples(db, speaker_id, samples, vector.size)
        return speaker_id

    def list_profiles(self):
        with self.connection() as db:
            rows = db.execute(
                "SELECT speaker_id,display_name,model_id,sample_count,revision,created_at,updated_at FROM speakers ORDER BY created_at,speaker_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_profile(self, speaker_id):
        profiles = self.list_profiles()
        return next((p for p in profiles if p["speaker_id"] == speaker_id), None)

    def rename(self, speaker_id, display_name, expected_revision):
        _, display_name = self._identity(speaker_id, display_name)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._check_revision(db, speaker_id, expected_revision)
            db.execute(
                "UPDATE speakers SET display_name=?,revision=revision+1,updated_at=? WHERE speaker_id=?",
                (display_name, timestamp(), speaker_id),
            )

    @staticmethod
    def _check_revision(db, speaker_id, expected_revision):
        row = db.execute("SELECT * FROM speakers WHERE speaker_id=?", (speaker_id,)).fetchone()
        if row is None:
            raise ProfileError("用户不存在，请刷新档案列表。")
        if expected_revision is None or row["revision"] != int(expected_revision):
            raise RevisionConflict("档案已被修改，请重新选择用户后再操作。")
        return row

    def update_samples(self, speaker_id, samples, model_id, expected_revision, *, replace=False):
        if not samples:
            raise ProfileError("请先提供新的参考录音。")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            profile = self._check_revision(db, speaker_id, expected_revision)
            if not replace and profile["model_id"] != model_id:
                raise ProfileError("声纹模型已变化，请选择重新录制并替换全部参考录音。")
            old = db.execute("SELECT * FROM samples WHERE speaker_id=?", (speaker_id,)).fetchall()
            old_vectors = [] if replace else [np.frombuffer(row["embedding"], dtype="<f4").copy() for row in old]
            vectors = old_vectors + [sample.embedding for sample in samples]
            if not 3 <= len(vectors) <= 20:
                raise ProfileError("更新后需要保留 3–20 段参考录音。")
            vector = centroid(vectors)
            removed_audio = [row["audio_path"] for row in old if row["audio_path"]] if replace else []
            if replace:
                db.execute("DELETE FROM samples WHERE speaker_id=?", (speaker_id,))
            self._insert_samples(db, speaker_id, samples, vector.size)
            db.execute(
                "UPDATE speakers SET model_id=?,embedding=?,dimension=?,sample_count=?,revision=revision+1,updated_at=? WHERE speaker_id=?",
                (model_id, vector.astype("<f4").tobytes(), vector.size, len(vectors), timestamp(), speaker_id),
            )
        return removed_audio

    def delete(self, speaker_id, expected_revision):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._check_revision(db, speaker_id, expected_revision)
            paths = [
                row[0]
                for row in db.execute(
                    "SELECT audio_path FROM samples WHERE speaker_id=? AND audio_path IS NOT NULL", (speaker_id,)
                )
            ]
            db.execute("DELETE FROM speakers WHERE speaker_id=?", (speaker_id,))
        return paths

    def snapshot(self):
        with self.connection() as db:
            db.execute("BEGIN")
            profiles = []
            for row in db.execute("SELECT * FROM speakers ORDER BY speaker_id").fetchall():
                profile = dict(row)
                profile["centroid"] = np.frombuffer(profile.pop("embedding"), dtype="<f4").copy()
                profile["samples"] = [
                    np.frombuffer(sample[0], dtype="<f4").copy()
                    for sample in db.execute(
                        "SELECT embedding FROM samples WHERE speaker_id=? ORDER BY sample_id", (row["speaker_id"],)
                    )
                ]
                profiles.append(profile)
        fingerprint = hashlib.sha256(
            json.dumps([(p["speaker_id"], p["revision"], p["model_id"]) for p in profiles]).encode()
        ).hexdigest()
        return profiles, fingerprint

    def get_settings(self):
        with self.connection() as db:
            return json.loads(db.execute("SELECT value FROM settings WHERE key='recognition'").fetchone()[0])

    def save_settings(self, threshold, margin, calibration=None, expected_fingerprint=None):
        threshold, margin = float(threshold), float(margin)
        if not np.isfinite([threshold, margin]).all() or not 0 <= threshold <= 1 or not 0 <= margin <= 1:
            raise ProfileError("相似度门槛和区分度门槛必须是 0–1 之间的数值。")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if expected_fingerprint:
                rows = db.execute("SELECT speaker_id,revision,model_id FROM speakers ORDER BY speaker_id").fetchall()
                fingerprint = hashlib.sha256(json.dumps([tuple(row) for row in rows]).encode()).hexdigest()
                if fingerprint != expected_fingerprint:
                    raise RevisionConflict("校准期间档案发生了变化，请重新校准。")
            value = {"threshold": threshold, "margin": margin, "calibration": calibration}
            db.execute("UPDATE settings SET value=? WHERE key='recognition'", (json.dumps(value, ensure_ascii=False),))
