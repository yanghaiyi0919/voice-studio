"""Cosine matching, unknown-speaker rejection and recording-wise calibration."""

import numpy as np

from .speaker_store import ProfileError, centroid, timestamp, unit_vector


def identify(vector, profiles, model_id, threshold, margin):
    vector = unit_vector(vector)
    usable = [p for p in profiles if p["model_id"] == model_id]
    if not profiles:
        return {"status": "empty", "reason": "尚未登记任何用户，请先到说话人管理中登记。", "speaker_id": None}
    if not usable:
        return {
            "status": "model_mismatch",
            "reason": "档案使用了其他声纹模型，请重新登记或替换参考录音。",
            "speaker_id": None,
        }
    scores = []
    for profile in usable:
        reference = unit_vector(profile["centroid"])
        if reference.size != vector.size:
            raise ProfileError("声纹模型与档案维度不一致，请重新登记。")
        score = float(np.clip(np.dot(vector, reference), -1, 1))
        scores.append((score, profile))
    scores.sort(key=lambda item: (-item[0], item[1]["speaker_id"]))
    best, profile = scores[0]
    gap = best - scores[1][0] if len(scores) > 1 else None
    accepted = best >= threshold and (gap is None or gap >= margin)
    if best < threshold:
        reason = "与已登记用户的相似度不足，暂按未登记说话人处理。"
    elif gap is not None and gap < margin:
        reason = "两个候选过于接近，暂时无法确认身份，请换一段更清晰的录音。"
    else:
        reason = "已匹配到登记档案。"
    return {
        "status": "recognized" if accepted else "unknown",
        "speaker_id": profile["speaker_id"] if accepted else None,
        "display_name": profile["display_name"] if accepted else None,
        "reason": reason,
        "score": round(best, 6),
        "threshold": threshold,
        "margin": margin,
        "score_gap": None if gap is None else round(gap, 6),
        "candidate_count": len(scores),
    }


def calibrate(profiles, model_id, fingerprint, *, target_far=0.01):
    usable = [p for p in profiles if p["model_id"] == model_id and len(p["samples"]) >= 3]
    if len(usable) < 2:
        raise ProfileError("校准至少需要两名用户，每人至少三段参考录音。")
    genuine, impostor = [], []
    for profile in usable:
        samples = [unit_vector(vector) for vector in profile["samples"]]
        others = [unit_vector(p["centroid"]) for p in usable if p["speaker_id"] != profile["speaker_id"]]
        for index, sample in enumerate(samples):
            reference = centroid([v for i, v in enumerate(samples) if i != index])
            genuine.append(float(np.dot(sample, reference)))
            impostor.append(max(float(np.dot(sample, other)) for other in others))
    genuine, impostor = np.clip(genuine, -1, 1), np.clip(impostor, -1, 1)
    values = np.unique(np.concatenate((genuine, impostor, [0.0, 1.0])))
    candidates = np.unique(np.clip(np.concatenate((values, (values[:-1] + values[1:]) / 2)), 0, 1))
    feasible = []
    ideal = (float(np.quantile(genuine, 0.1)) + float(np.quantile(impostor, 1 - target_far))) / 2
    for threshold in candidates:
        far = float(np.mean(impostor >= threshold))
        frr = float(np.mean(genuine < threshold))
        if far <= target_far:
            feasible.append((frr, far, abs(float(threshold) - ideal), float(threshold)))
    if not feasible:
        raise ProfileError("现有录音区分度不足，无法得到满足目标的门槛。请检查混入其他人的录音。")
    frr, far, _, threshold = min(feasible)
    return {
        "method": "leave-one-recording-out",
        "fingerprint": fingerprint,
        "model_id": model_id,
        "threshold": threshold,
        "estimated_far": far,
        "estimated_frr": frr,
        "target_far": target_far,
        "genuine_trials": len(genuine),
        "impostor_trials": len(impostor),
        "speakers": len(usable),
        "created_at": timestamp(),
        "limitation": "仅基于登记录音的内部验证；不代表对独立陌生人或伪造语音的表现。",
    }
