"""Small escaped HTML views; cosine scores are never called probabilities."""

from datetime import UTC, datetime, timedelta, timezone
from html import escape

BEIJING_TIME = timezone(timedelta(hours=8))


def display_time(value):
    """Render stored UTC timestamps as readable, explicitly labelled Beijing time."""
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return "—"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(BEIJING_TIME).strftime("%Y-%m-%d %H:%M:%S")


def analysis_card(identity, emotion=None, warning=None):
    accepted = identity.get("status") == "recognized"
    name = (
        identity.get("display_name")
        if accepted
        else ("尚无登记用户" if identity.get("status") == "empty" else "未登记 / 暂无法确认")
    )
    if identity.get("status") == "invalid_audio":
        name = "音频暂不适合身份识别"
    score = identity.get("score")
    details = ""
    if score is not None:
        details = f"<p>声纹相似度：<strong>{score:.3f}</strong> · 当前门槛：{identity['threshold']:.3f}</p>"
    emotion_text = "未完成"
    if emotion:
        emotion_text = f"{escape(emotion['label'])} · 模型得分 {emotion['score']:.3f}"
    warning_html = f"<p>{escape(warning)}</p>" if warning else ""
    return (
        '<div class="result-card">'
        f"<h2>说话人：{escape(str(name))}</h2>"
        f"<p>{escape(identity.get('reason', ''))}</p>"
        f"{details}<p>情绪：{emotion_text}</p>"
        f"<small>门槛状态：{escape(identity.get('calibration', '未校准'))}。相似度不等同于识别正确率。</small>"
        f"{warning_html}</div>"
    )
