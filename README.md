# DBT情感咨询机器人

基于 Qwen3-1.7B 的 DBT（辩证行为疗法）情感咨询机器人，使用 LoRA 微调训练，专为空巢老人提供心理支持。

## 目录

- [环境要求](#环境要求)
- [第一步：安装 Python](#第一步安装-python)
- [第二步：创建项目环境](#第二步创建项目环境)
- [第三步：安装依赖](#第三步安装依赖)
- [第四步：下载基础模型](#第四步下载基础模型)
- [第五步：开始训练](#第五步开始训练)
- [第六步：合并权重](#第六步合并权重)
- [第七步：量化模型](#第七步量化模型)
- [常见问题](#常见问题)

---

## 环境要求

| 配置 | 最低要求 |
|------|----------|
| GPU | NVIDIA RTX 3060 (12GB显存) 或更高 |
| 内存 | 16 GB |
| 硬盘 | 30 GB 可用空间 |
| 系统 | Windows 10/11 或 Linux |
| Python | 3.10 |

> ⚠️ **注意**：必须有 NVIDIA 显卡，AMD 显卡或核显无法训练。

---

## 第一步：安装 Python

### Windows

1. 下载 Python 3.10：https://www.python.org/downloads/release/python-31011/
2. 选择 **Windows installer (64-bit)**
3. 运行安装程序，**勾选 "Add Python to PATH"**
4. 点击 Install Now

### 验证安装

打开命令提示符（Win+R 输入 cmd），输入：

```bash
python --version
```

显示 `Python 3.10.x` 即成功。

---

## 第二步：创建项目环境

### 1. 克隆项目

```bash
git clone https://github.com/Hong-Guang-L/dbt-counselor-bot.git
cd dbt-counselor-bot
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
```

### 3. 激活虚拟环境

**Windows：**
```bash
.venv\Scripts\activate
```

**Linux/Mac：**
```bash
source .venv/bin/activate
```

激活成功后，命令行前面会显示 `(.venv)`。

---

## 第三步：安装依赖

### 1. 安装 PyTorch（CUDA 版本）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. 验证 CUDA

```bash
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

显示你的显卡名称即成功。

### 3. 安装其他依赖

```bash
pip install -r requirements.txt
```

---

## 第四步：下载基础模型

我们使用 ModelScope 下载 Qwen3-1.7B 模型（国内速度快）：

```bash
pip install modelscope
```

然后运行：

```bash
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B', cache_dir='./models')"
```

下载完成后，模型会保存在 `models/Qwen/Qwen3-1.7B/` 目录。

> 💡 模型约 3.4 GB，下载需要几分钟。

---

## 第五步：开始训练

### 1. 检查数据集

项目已包含训练数据 `datasets/train_final.json`，无需额外准备。

### 2. 开始训练

```bash
llamafactory-cli train train.yaml
```

### 3. 等待训练完成

训练过程示例：

```
{'loss': 2.3456, 'learning_rate': 1e-4, 'epoch': 0.1}
{'loss': 1.8765, 'learning_rate': 9.5e-5, 'epoch': 0.2}
{'loss': 1.2345, 'learning_rate': 8e-5, 'epoch': 0.3}
...
```

训练时间约 1-3 小时（取决于 GPU 性能）。

### 4. 训练输出

训练完成后，LoRA 权重保存在 `output/qwen3_dbt_lora/` 目录。

---

## 第六步：合并权重

将 LoRA 权重与基础模型合并：

```bash
llamafactory-cli export merge.yaml
```

合并后的模型保存在 `output/qwen3_dbt_merged/` 目录。

---

## 第七步：量化模型

量化可以大幅减小模型体积（从 3.4GB 到 1.2GB）。

### 1. 下载 llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

### 2. 编译 llama.cpp

**Windows（需要先安装 Visual Studio Build Tools）：**

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

> 📥 Visual Studio Build Tools 下载：https://visualstudio.microsoft.com/visual-cpp-build-tools/
> 
> 安装时勾选 "Desktop development with C++"

**Linux：**

```bash
make
```

### 3. 转换为 GGUF 格式

```bash
python convert_hf_to_gguf.py ../output/qwen3_dbt_merged --outfile dbt-counselor-f16.gguf --outtype f16
```

### 4. 量化

```bash
llama-quantize dbt-counselor-f16.gguf dbt-counselor-q4_k_m.gguf Q4_K_M
```

### 5. 完成！

量化后的模型 `dbt-counselor-q4_k_m.gguf` 约 1.2 GB，可以部署到边缘设备。

---

## 常见问题

### Q: CUDA 显示不可用？

**检查步骤：**

1. 确认有 NVIDIA 显卡
2. 安装 NVIDIA 驱动：https://www.nvidia.com/Download/index.aspx
3. 安装 CUDA Toolkit 12.1：https://developer.nvidia.com/cuda-12-1-0-download-archive

### Q: 训练时显存不足？

**解决方案：** 修改 `train.yaml`：

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
```

### Q: Windows 编译 llama.cpp 失败？

**解决方案：**

1. 安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. 安装时勾选 "Desktop development with C++"
3. 安装 [CMake](https://cmake.org/download/)
4. 重启电脑后重试

### Q: 下载模型太慢？

**解决方案：** 使用 ModelScope 而不是 HuggingFace（已在上文使用）。

---

## 项目结构

```
dbt-counselor-bot/
├── datasets/
│   └── train_final.json      # 训练数据集
├── models/
│   └── Qwen/Qwen3-1.7B/      # 基础模型（需下载）
├── output/
│   ├── qwen3_dbt_lora/       # LoRA 权重
│   ├── qwen3_dbt_merged/     # 合并后模型
│   └── Modelfile            # Ollama 配置
├── train.yaml               # 训练配置
├── merge.yaml               # 合并配置
├── requirements.txt         # 依赖列表
└── README.md
```

---

## 致谢

- [Qwen](https://github.com/QwenLM/Qwen) - 基础模型
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - 训练框架
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 量化工具

---

## 版权

版权所有 © Hong-Guang-L | MIT License
