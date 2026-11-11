# 第三方模型与依赖

本仓库仅发布应用代码，不分发第三方模型权重、个人录音或上游完整源码副本。

| 项目 | 用途 | 上游 |
| --- | --- | --- |
| FunASR | 语音模型推理接口 | https://github.com/modelscope/FunASR |
| CAM++ / 3D-Speaker | 说话人特征提取 | https://github.com/modelscope/3D-Speaker |
| CAM++ 中文模型 | 本应用声纹 checkpoint | https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common |
| emotion2vec+ | 情绪识别 | https://github.com/ddlBoJack/emotion2vec |
| emotion2vec+ large | 本应用情绪 checkpoint | https://modelscope.cn/models/iic/emotion2vec_plus_large |
| Vosk | 本地中文转写 | https://alphacephei.com/vosk/ |
| vosk-model-small-cn-0.22 | 默认中文转写模型 | https://alphacephei.com/vosk/models |
| Gradio、PyTorch、librosa、SciPy、NumPy、Matplotlib、soundfile | UI、推理、音频处理与可视化 | 各项目官方发行版 |

Vosk 官方模型列表将 small-cn-0.22 标为 Apache 2.0。其他模型的使用、再分发和商业限制应以对应模型卡与上游许可为准；本项目 MIT 许可不覆盖这些权重。

增强与变声模块来自本项目的原始课程实现，现将数据路径、界面和声纹管理逻辑分层重构。
