"""Gradio UI for analysis, enrollment, profile management and calibration."""

import json
import logging
import os
from datetime import datetime
from functools import partial
from pathlib import Path

import gradio as gr

from .audio import capture_local_microphone, prepare_audio
from .config import Paths
from .effects import transform
from .models import ModelHub
from .presentation import analysis_card, display_time
from .speaker_service import SpeakerService
from .speaker_store import ProfileError, SpeakerStore

LOGGER = logging.getLogger(__name__)


def build_app(paths, models, store):
    service = SpeakerService(store, models, paths)

    def profile_rows():
        return [
            [p["speaker_id"], p["display_name"], p["sample_count"], display_time(p["updated_at"])]
            for p in store.list_profiles()
        ]

    def choices():
        return [(f"{p['display_name']} · {p['speaker_id']}", p["speaker_id"]) for p in store.list_profiles()]

    def summary():
        profiles, fingerprint = store.snapshot()
        settings = store.get_settings()
        report = settings.get("calibration")
        state = (
            "尚未校准，当前为初始门槛"
            if not report
            else ("内部校准有效" if report["fingerprint"] == fingerprint else "档案已变化，建议重新校准")
        )
        return f"**已登记 {len(profiles)} 人** · 相似度门槛 {settings['threshold']:.3f} · 区分度门槛 {settings['margin']:.3f} · {state}"

    def profile_result(message="", speaker_id=None):
        profile = store.get_profile(speaker_id) if speaker_id else None
        if not profile:
            speaker_id = None
        details = "选择用户后可改名、追加或替换参考录音、删除档案。"
        if profile:
            details = f"当前档案：**{profile['speaker_id']}** · {profile['sample_count']} 段参考录音 · 更新于 {display_time(profile['updated_at'])}（北京时间）"
        status_file = paths.runtime / "model-status.json"
        if status_file.is_file():
            try:
                status = json.loads(status_file.read_text(encoding="utf-8"))
                status["registered_speakers"] = len(store.list_profiles())
                status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
            except (OSError, ValueError):
                LOGGER.warning("Optional runtime status file could not be refreshed.")
        return (
            message,
            profile_rows(),
            gr.update(choices=choices(), value=speaker_id),
            profile["display_name"] if profile else "",
            profile["revision"] if profile else None,
            summary(),
            details,
        )

    def safe_error(error):
        if isinstance(error, ValueError):
            raise gr.Error(str(error)) from error
        LOGGER.exception("Operation failed")
        raise gr.Error("处理未完成，请检查音频或查看本机日志后重试。") from error

    def on_file_upload(file_path):
        if not file_path:
            return None, "<p>录音或上传音频后开始分析。</p>", {}
        try:
            normalized, identity = service.identify(file_path)
            emotion, warning = None, None
            if identity["status"] == "invalid_audio":
                warning = "建议先增强音频或重新录音，再进行身份和情绪分析。"
            else:
                try:
                    emotion = models.emotion(normalized)
                except Exception:
                    LOGGER.exception("Emotion inference failed")
                    warning = "情绪分析暂未完成，身份识别结果仍可查看。"
            result = {"speaker": identity, "emotion": emotion}
            return normalized, analysis_card(identity, emotion, warning), result
        except Exception as error:
            safe_error(error)

    def convert_speech_to_text(path):
        if not path:
            raise gr.Error("请先录音或上传音频。")
        try:
            return models.transcribe(path)
        except Exception as error:
            safe_error(error)

    def enhance_audio_wrapper(strength, path):
        if not path:
            raise gr.Error("请先录音或上传音频。")
        try:
            import enhance_audio

            wav, _, spectrogram, _ = enhance_audio.enhance_file_from_path(
                path, strength=float(strength), output_dir=paths.runtime / "outputs"
            )
            base = spectrogram.removesuffix("_enh.png")
            return wav, [base + "_orig.png", base + "_enh.png"]
        except Exception as error:
            safe_error(error)

    def process_voice_transform(method, path):
        if not path:
            raise gr.Error("请先录音或上传音频。")
        try:
            return transform(method, path, paths.runtime / "outputs")
        except Exception as error:
            safe_error(error)

    def use_enhanced(path):
        if not path:
            raise gr.Error("请先执行增强处理。")
        return path

    def record_local_audio():
        try:
            return capture_local_microphone(paths.runtime / "outputs")
        except Exception as error:
            safe_error(error)

    def add_recording(path, pending):
        try:
            pending = list(pending or [])
            if len(pending) >= 20:
                raise ProfileError("最多保留 20 段待登记录音，请先保存或清空。")
            audio = prepare_audio(path, paths.runtime / "pending", enrollment=True)
            if any(item["sha256"] == audio.sha256 for item in pending):
                audio.path.unlink(missing_ok=True)
                raise ProfileError("这一段录音已经加入，请录制另一段内容。")
            pending.append({"path": str(audio.path), "sha256": audio.sha256, "quality": audio.quality.to_dict()})
            rows = [[i, item["quality"]["duration"], item["quality"]["rms_dbfs"]] for i, item in enumerate(pending, 1)]
            return pending, rows, None, f"已加入 {len(pending)} 段录音。登记新用户或替换全部录音至少需要 3 段。"
        except Exception as error:
            safe_error(error)

    def clear_pending(pending):
        for item in pending or []:
            file = Path(item["path"]).resolve()
            if file.is_relative_to((paths.runtime / "pending").resolve()):
                file.unlink(missing_ok=True)
        return [], [], None, "待登记录音已清空。"

    def save_speaker_profile(mode, speaker_id, name, selected, revision, pending, uploaded, retain):
        files = [item["path"] for item in pending or []] + list(uploaded or [])
        try:
            if mode == "登记新用户":
                selected = service.enroll(speaker_id, name, files, bool(retain))
                message = "登记成功，新用户已立即加入识别库，重启后仍会保留。"
            else:
                if not selected:
                    raise ProfileError("请先在档案列表中选择要更新的用户。")
                warnings = service.update(
                    selected, files, revision, replace=mode == "替换全部参考录音", retain_audio=bool(retain)
                )
                message = "参考录音已更新，新的声纹已经生效。" + " ".join(warnings)
            clear_pending(pending)
            return (*profile_result(message, selected), [], [], None, None, "参考录音已保存。")
        except Exception as error:
            safe_error(error)

    def select_speaker(speaker_id):
        result = profile_result("", speaker_id)
        return result[3], result[4], result[6]

    def rename_speaker(speaker_id, name, revision):
        try:
            if not speaker_id:
                raise ProfileError("请先选择用户。")
            store.rename(speaker_id, name, revision)
            return profile_result("显示姓名已更新。", speaker_id)
        except Exception as error:
            safe_error(error)

    def delete_speaker(speaker_id, confirmed, revision):
        try:
            if not speaker_id or not confirmed:
                raise ProfileError("请选择用户，并勾选删除确认。")
            warnings = service.delete(speaker_id, revision)
            return (*profile_result("用户档案和声纹已删除。" + " ".join(warnings)), False)
        except Exception as error:
            safe_error(error)

    def save_recognition_settings(threshold, margin):
        try:
            store.save_settings(threshold, margin)
            return "识别设置已保存；手动调整后请重新校准或使用独立样本验证。", summary()
        except Exception as error:
            safe_error(error)

    def calibrate_threshold():
        try:
            report = service.calibrate()
            text = (
                f"已保存建议门槛 **{report['threshold']:.3f}**。"
                f"内部验证使用 {report['speakers']} 人、{report['genuine_trials']} 次同人比较和 "
                f"{report['impostor_trials']} 次异人比较；估计误接受率 {report['estimated_far']:.1%}，"
                f"估计误拒率 {report['estimated_frr']:.1%}。\n\n"
                "这是登记录音的留一验证，不代表独立陌生人、跨设备或伪造语音的表现；"
                "上述统计针对相似度门槛，运行时还会检查两个候选的区分度。"
            )
            return text, report["threshold"], store.get_settings()["margin"], summary(), report
        except Exception as error:
            safe_error(error)

    settings = store.get_settings()
    css = """
    .gradio-container {max-width: 1180px !important;}
    .result-card {padding: 20px; border: 1px solid var(--border-color-primary); border-radius: 12px;}
    .result-card h2 {margin-top: 0;}
    """
    with gr.Blocks(
        theme=gr.themes.Soft(), css=css, delete_cache=(600, 3600), title="Voice Studio · 语音分析与声纹管理"
    ) as demo:
        gr.Markdown("# Voice Studio\n语音分析、声纹登记与音效处理，一处完成。")
        registry_status = gr.Markdown(summary())
        current_audio = gr.State(None)
        pending_state = gr.State([])
        selected_revision = gr.State(None)
        with gr.Tab("语音分析"):
            with gr.Row():
                with gr.Column():
                    analysis_input = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="录音或上传音频（1–60 秒）",
                        max_length=60,
                    )
                    local_query_record = gr.Button("使用本机麦克风录 5 秒")
                    gr.Markdown("先到「说话人管理」登记用户。相似度不足或候选过于接近时，系统会暂不确认身份。")
                with gr.Column():
                    result_html = gr.HTML("<p>录音或上传音频后开始分析。</p>")
                    with gr.Accordion("查看本次分析详情", open=False):
                        analysis_json = gr.JSON(label="识别详情")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 语音转文字")
                    asr_button = gr.Button("开始识别字幕")
                    asr_text = gr.Textbox(label="识别结果", lines=3, show_copy_button=True)
                with gr.Column():
                    gr.Markdown("### 语音增强")
                    strength = gr.Slider(0.1, 3, value=1, step=0.1, label="增强强度")
                    enhance_button = gr.Button("执行增强处理")
                    enhanced_audio = gr.Audio(label="增强后音频", interactive=False)
                    reanalyze_button = gr.Button("使用增强结果重新分析")
            spectrograms = gr.Gallery(label="增强前后频谱对比", columns=2, height="auto")
            gr.Markdown("### 趣味变声")
            with gr.Row():
                with gr.Column():
                    sb = gr.Button("🧽 海绵宝宝")
                    sb_audio = gr.Audio(label="变声结果")
                with gr.Column():
                    fc = gr.Button("🐱 慵懒胖猫")
                    fc_audio = gr.Audio(label="变声结果")
                with gr.Column():
                    mi = gr.Button("👾 电子小黄人")
                    mi_audio = gr.Audio(label="变声结果")
        with gr.Tab("说话人管理"):
            gr.Markdown("### 用户档案\n声纹仅保存在本机。首次安装为空库，可通过下面的表单登记新用户。")
            profiles_table = gr.Dataframe(
                headers=["用户 ID", "显示姓名", "参考录音数", "更新时间（北京时间）"],
                value=profile_rows(),
                interactive=False,
                type="array",
                row_count=(0, "dynamic"),
            )
            with gr.Row():
                profile_selector = gr.Dropdown(choices=choices(), value=None, label="选择要管理的用户")
                refresh_button = gr.Button("刷新档案列表")
            profile_details = gr.Markdown("选择用户后可改名、更新参考录音或删除档案。")
            with gr.Row():
                rename_name = gr.Textbox(label="选中用户的显示姓名")
                rename_button = gr.Button("保存姓名")
            with gr.Accordion("删除选中用户", open=False):
                delete_confirm = gr.Checkbox(label="确认删除该用户的声纹及本项目保存的登记录音", value=False)
                delete_button = gr.Button("删除用户", variant="stop")
            gr.Markdown(
                "### 登记或更新参考录音\n使用同一个人在安静环境中的不同录音，建议每段 5–10 秒。新登记或全部替换需要 3–20 段；追加可提供 1 段起。"
            )
            mode = gr.Radio(["登记新用户", "追加参考录音", "替换全部参考录音"], value="登记新用户", label="操作")
            with gr.Row():
                new_id = gr.Textbox(label="新用户 ID", placeholder="例如 user_01")
                new_name = gr.Textbox(label="新用户显示姓名", placeholder="例如 示例用户")
            record_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="逐段录制或选择参考语音（2–30 秒）",
                max_length=30,
            )
            local_reference_record = gr.Button("使用本机麦克风录 5 秒（参考录音）")
            gr.Markdown("浏览器找不到麦克风时可使用本机按钮；采集后试听，再点击「加入待登记录音」。")
            with gr.Row():
                add_button = gr.Button("加入待登记录音")
                clear_button = gr.Button("清空待登记录音")
            pending_table = gr.Dataframe(
                headers=["序号", "时长（秒）", "音量（dBFS）"],
                value=[],
                row_count=(0, "dynamic"),
                interactive=False,
                type="array",
            )
            pending_note = gr.Markdown("也可直接在下面一次上传多段录音。")
            bulk_files = gr.Files(file_types=["audio"], label="批量上传参考录音", type="filepath")
            retain = gr.Checkbox(value=False, label="同时保留登记录音（默认仅长期保存声纹与质量摘要）")
            save_button = gr.Button("保存登记或更新", variant="primary")
            operation_status = gr.Markdown("")
        with gr.Tab("识别设置"):
            gr.Markdown(
                "### 陌生人拒识\n门槛越高越严格，可能拒绝更多本人录音；门槛越低越容易接受其他人的录音。默认值是初始设置，需要结合自己的录音验证。"
            )
            threshold = gr.Slider(0, 1, value=settings["threshold"], step=0.001, label="最低声纹相似度")
            margin = gr.Slider(0, 1, value=settings["margin"], step=0.001, label="第一与第二候选的最低分差")
            save_settings = gr.Button("保存识别设置")
            settings_status = gr.Markdown("")
            gr.Markdown(
                "### 内部阈值校准\n至少登记两个人，每人至少三段语音。每次留出一段录音验证，与其余参考录音比较，估计一个相似度门槛。"
            )
            calibrate_button = gr.Button("校准并应用建议门槛")
            calibration_text = gr.Markdown("")
            with gr.Accordion("校准记录", open=False):
                calibration_json = gr.JSON(value=settings.get("calibration"), label="校准结果")
        common_outputs = [
            operation_status,
            profiles_table,
            profile_selector,
            rename_name,
            selected_revision,
            registry_status,
            profile_details,
        ]
        inference = {"concurrency_id": "voice-models", "concurrency_limit": 1}
        analysis_input.change(
            on_file_upload,
            [analysis_input],
            [current_audio, result_html, analysis_json],
            api_name="on_file_upload",
            **inference,
        )
        local_query_record.click(
            record_local_audio,
            [],
            [analysis_input],
            api_name="record_local_query",
            concurrency_id="local-microphone",
            concurrency_limit=1,
        )
        local_reference_record.click(
            record_local_audio,
            [],
            [record_input],
            api_name="record_local_reference",
            concurrency_id="local-microphone",
            concurrency_limit=1,
        )
        asr_button.click(
            convert_speech_to_text, [current_audio], [asr_text], api_name="convert_speech_to_text", **inference
        )
        enhance_button.click(
            enhance_audio_wrapper,
            [strength, current_audio],
            [enhanced_audio, spectrograms],
            api_name="enhance_audio_wrapper",
        )
        reanalyze_button.click(use_enhanced, [enhanced_audio], [analysis_input], api_name="use_enhanced_audio")
        for button, name, output in [(sb, "spongebob", sb_audio), (fc, "fatcat", fc_audio), (mi, "minion", mi_audio)]:
            button.click(
                partial(process_voice_transform, name), [current_audio], [output], api_name="transform_" + name
            )
        add_button.click(
            add_recording,
            [record_input, pending_state],
            [pending_state, pending_table, record_input, pending_note],
            api_name="add_reference_recording",
        )
        clear_button.click(
            clear_pending,
            [pending_state],
            [pending_state, pending_table, record_input, pending_note],
            api_name="clear_reference_recordings",
        )
        save_button.click(
            save_speaker_profile,
            [mode, new_id, new_name, profile_selector, selected_revision, pending_state, bulk_files, retain],
            common_outputs + [pending_state, pending_table, bulk_files, record_input, pending_note],
            api_name="save_speaker_profile",
            **inference,
        )
        profile_selector.change(
            select_speaker,
            [profile_selector],
            [rename_name, selected_revision, profile_details],
            api_name="select_speaker",
            trigger_mode="once",
        )
        refresh_button.click(
            lambda: profile_result("档案列表已刷新。"), [], common_outputs, api_name="refresh_speakers"
        )
        rename_button.click(
            rename_speaker,
            [profile_selector, rename_name, selected_revision],
            common_outputs,
            api_name="rename_speaker",
        )
        delete_button.click(
            delete_speaker,
            [profile_selector, delete_confirm, selected_revision],
            common_outputs + [delete_confirm],
            api_name="delete_speaker",
        )
        save_settings.click(
            save_recognition_settings,
            [threshold, margin],
            [settings_status, registry_status],
            api_name="save_recognition_settings",
        )
        calibrate_button.click(
            calibrate_threshold,
            [],
            [calibration_text, threshold, margin, registry_status, calibration_json],
            api_name="calibrate_threshold",
        )
        mode.change(
            lambda value: (gr.update(visible=value == "登记新用户"), gr.update(visible=value == "登记新用户")),
            [mode],
            [new_id, new_name],
            api_name=False,
        )
    return demo


def main():
    paths = Paths.load()
    models = ModelHub(paths, os.environ.get("VOICE_DEVICE", "auto"))
    store = SpeakerStore(paths.database)
    if os.environ.get("VOICE_PRELOAD_MODELS", "1") == "1":
        models.warm_up()
    demo = build_app(paths, models, store)
    (paths.runtime / "model-status.json").write_text(
        json.dumps(
            {
                "device": models.device,
                "registered_speakers": len(store.list_profiles()),
                "persistent_profiles": True,
                "version": "0.2.0",
                "started_at": datetime.now().astimezone().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("VOICE_PORT", "7860")),
        share=False,
        inbrowser=False,
        show_error=True,
        allowed_paths=[str(paths.runtime / "outputs")],
        blocked_paths=[
            str(paths.private),
            str(paths.root / "models"),
            str(paths.root / "original-source"),
            str(paths.root / "app" / "data"),
            str(paths.root / "app" / "model"),
            str(paths.root / ".cache"),
            str(paths.root / ".venv"),
            str(paths.root / ".git"),
            str(paths.root / "logs"),
        ],
    )
