# Voice Studio · 本地语音分析与声纹管理

Local speech analysis with persistent speaker enrollment, unknown-speaker rejection, transcription and audio effects.

一个基于 Python 和 Gradio 的本地语音应用。通过界面登记新说话人，保存声纹、管理档案，再对新录音进行身份和情绪分析；同时支持中文转写、降噪增强与趣味变声。

## 功能

- **动态登记**：录音或批量上传 3–20 段语音，新用户立即进入识别库。
- **持久化档案**：SQLite 保存逐段声纹、平均声纹、录音质量摘要及识别设置，重启直接加载。
- **档案管理**：改名、追加参考语音、全部替换、确认删除；使用版本检查避免覆盖已修改的档案。
- **拒识机制**：相似度不足或第一、第二候选过于接近时，输出“未登记／暂无法确认”。
- **阈值校准**：使用登记录音的留一验证估计门槛，保存校准来源；档案变化后提醒重新校准。
- **语音工作台**：CAM++ 身份识别、emotion2vec+ 情绪分析、Vosk 中文转写、谱减降噪、音高与时长变换。

首次安装是空档案库。仓库不包含任何真实姓名、个人录音或声纹；无需模型 API Key。

## 快速开始

建议使用 **Python 3.11**。首次安装需要下载模型；后续从本地加载。桌面端优先使用现代浏览器，麦克风录音需要允许浏览器访问麦克风。

### Windows

在项目目录打开 PowerShell，CPU 环境运行：

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1
~~~

有兼容 NVIDIA 显卡时，首次安装可选择 CUDA：

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Device cuda
~~~

安装完成后双击 **启动系统.cmd**。默认访问 http://127.0.0.1:7860；端口占用时会选择 7861–7879。使用完双击 **停止系统.cmd**。

### Linux

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/prepare_models.py
python app/GUI.py
~~~

应用监听本机地址，终端运行时用 Ctrl+C 停止。本机完整功能在 Windows + CUDA 上验收；Linux 的核心逻辑由 CI 配置覆盖，其他平台的完整音频链路尚未实测。

Linux 的本机麦克风备用功能需要系统 PortAudio（例如 libportaudio2）；浏览器录音和上传不依赖这个备用入口。

## 使用流程

1. 打开「说话人管理」，输入用户 ID 和显示姓名。
2. 在安静环境中录制三段不同内容的语音，每段建议 5–10 秒。逐段加入待登记列表，或一次上传多段录音。
3. 点击「保存登记或更新」。默认仅长期保存声纹和质量摘要；如需保留登记录音，可主动勾选。上传文件可能暂存于本机缓存中。
4. 到「语音分析」录音或上传文件，查看身份、情绪、转写和增强结果。
5. 至少登记两个人后，可到「识别设置」执行内部校准，再用额外的本人和陌生人录音验证门槛。

更新档案时先选择用户，再选择「追加参考录音」或「替换全部参考录音」。删除会移除档案、声纹，以及该功能保存的登记录音；不会改动原始压缩包或迁移来源文件。

浏览器不提供麦克风时，可点击“使用本机麦克风录 5 秒”。这个备用按钮采集运行程序这台电脑的默认输入设备，不会自动录音。

## 技术路线

| 层 | 实现 |
| --- | --- |
| 交互 | Gradio，多段录音队列、批量上传、档案管理、参数设置 |
| 音频处理 | NumPy、SciPy、librosa、soundfile；重采样、幅度与削波检查 |
| 身份识别 | FunASR / CAM++，逐段向量归一化、平均声纹、余弦相似度 |
| 情绪与转写 | emotion2vec+ large、Vosk 小型中文模型 |
| 持久化 | SQLite 事务、唯一录音指纹、更新版本校验、模型指纹 |
| 增强与展示 | STFT 谱减、静音裁剪、弱音增益、Matplotlib 声谱图 |

模型负责提取声纹和情绪推理；登记、匹配、拒识、校准、存储与界面流程由应用实现。没有自行训练基础模型。

## 测试

无需下载模型即可运行核心回归测试：

~~~bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
~~~

已配置 GitHub Actions，推送后会在 Ubuntu 与 Windows 上执行同一组模型无关测试。真实模型和界面验收使用本机自备录音，见 [测试说明](docs/TESTING.md)。

## 项目结构

~~~text
app/GUI.py                   本地启动入口
app/voice_studio/             音频、模型适配、声纹服务、存储、拒识与界面
app/enhance_audio.py          谱减增强与图像输出
scripts/prepare_models.py    首次模型准备及校验
tests/                       不含个人数据的自动化测试
private/                     本机 SQLite 与可选登记录音，不提交 Git
models/                      下载的模型权重，不提交 Git
runtime/                     临时音频和验收结果，不提交 Git
~~~

详细设计见 [架构说明](docs/ARCHITECTURE.md)。

## GitHub 发布

.gitignore 排除了原压缩包、个人录音、声纹数据库、模型、日志、环境和缓存。提交后可额外检查并导出源码：

~~~bash
python scripts/check_public_files.py
python scripts/export_source.py
~~~

检查与导出读取的是 Git 暂存快照，不遍历整个工作目录。修改公开文件后先 git add，再重新检查和导出。不要把本机整个目录手动打包公开。

## 边界与许可

- 这是单机声纹档案工具，用户档案不等同于登录账号。默认无网络账号认证，保持仅监听本机。
- 余弦分数不是正确率。初始门槛未经你的独立数据验证；内部校准也不代表真实陌生人测试结果。
- 当前不包含重放、合成语音或变声攻击检测，不能作为反欺骗或安全认证系统。
- 基础音频质检检查时长、音量、有效声音和削波，不保证音频一定是单人语音。
- 情绪模型的九类标签包含 happy，没有独立的 excited 标签；情绪输出仅用于实验展示。
- 本项目代码采用 MIT；第三方模型和依赖按各自许可使用，权重不随仓库分发。见 [第三方说明](THIRD_PARTY_NOTICES.md)。
