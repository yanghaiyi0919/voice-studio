# 测试与验收

## 核心测试

~~~bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
~~~

测试使用程序生成的波形、人工构造的向量和临时数据库，不包含真实说话人的录音。覆盖新增、重复 ID／录音、事务回滚、改名、追加、替换、删除、持久化、过期版本、未知与歧义拒识、模型不匹配、校准来源、静音／短音频／削波和录音清理范围。

这组测试验证逻辑正确性，不衡量 CAM++ 的识别准确率。

## 本机真实模型验收

下面的测试需要先准备模型和自己的录音：

~~~bash
python scripts/validate_deployment.py --audio /path/to/your/sample.wav
python scripts/validate_profile_api.py --audio-dir /path/to/your/reference-recordings
~~~

界面 API 测试创建临时档案，执行登记、重复登记、改名、追加、替换和删除，最后恢复原有档案数量。请在本机空闲时运行。

如需使用隔离数据库测试三个说话人的录音，可提供本机 JSON manifest：

~~~json
{
  "profiles": [
    {"folder": "/path/to/person-a"},
    {"folder": "/path/to/person-b"},
    {"folder": "/path/to/person-c"}
  ]
}
~~~

~~~bash
python scripts/validate_real_speakers.py --manifest /path/to/local-manifest.json
~~~

该测试在临时数据库登记前两人，验证登记片段的识别及第三人的拒识，再检查更名、更新、新进程读取和删除。它是功能冒烟测试，不是独立识别性能评估。

## 更完整的评估

若要报告性能，应另外收集不参与登记或校准的本人和陌生人录音，记录 FAR、FRR、身份识别准确率，以及录音时长、设备和噪声条件。不要将模型分数或本仓库的功能测试通过率当成识别准确率。
