# i2ii - 多GPU并发图像编辑服务

基于 Qwen-Image-Edit 模型的高性能多GPU并发图像编辑API服务，支持多种图像编辑任务的并发处理。

## 项目简介

i2ii 是一个基于阿里巴巴 Qwen-Image-Edit 模型构建的图像编辑服务，通过多GPU并发处理架构，能够高效处理大量图像编辑请求。服务采用 FastAPI 框架构建，支持异步请求处理和实时状态监控。

## 功能特性

- ✅ **多GPU并发处理** - 自动负载均衡，支持多GPU同时工作
- ✅ **异步请求处理** - 提交请求后立即返回，支持状态查询
- ✅ **智能任务调度** - 轮询分配策略，确保GPU负载均衡
- ✅ **实时状态监控** - 提供系统状态和处理进度查询接口
- ✅ **健康检查** - 自动监控GPU工作进程健康状态
- ✅ **错误恢复** - 异常处理和进程重启机制
- ✅ **RESTful API** - 标准HTTP接口，易于集成

## 技术栈

- **编程语言**: Python 3.8+
- **深度学习框架**: PyTorch
- **模型**: Qwen-Image-Edit (ModelScope)
- **Web框架**: FastAPI
- **图像处理**: Pillow (PIL)
- **并发处理**: Multiprocessing
- **Web服务器**: Uvicorn

## 环境要求

- Python 3.12 或更高版本
- CUDA 兼容的 NVIDIA GPU（支持多GPU） cuda>=12.4; pytorch>=2.5.1
- 至少 64GB GPU 显存（推荐 80GB+）
- 足够的磁盘空间>=80GB用于模型缓存和结果存储

## 快速开始

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/skyrover001/i2ii.git
cd i2ii

# 安装依赖包
pip install git+https://github.com/huggingface/diffusers
pip install fastapi uvicorn pillow modelscope transformer python-multipart
```

### 启动服务

```bash
# 启动服务（使用所有可用GPU）
python server.py

# 或指定特定GPU
CUDA_VISIBLE_DEVICES=0,1 python server.py
```

服务启动后将在 `http://localhost:8000` 提供API服务。

## API 使用方法

### 提交图像编辑请求

```bash
curl -X POST "http://localhost:8000/edit" \
  -F "image=@your_image.jpg" \
  -F "prompt=将图像转换为水彩画风格" \
  -F "negative_prompt=low quality, blurry" \
  -F "num_inference_steps=50" \
  -F "true_cfg_scale=4.0" \
  -F "seed=42"
```

**响应示例：**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "submitted",
  "message": "Request submitted successfully",
  "ready_workers": 2,
  "total_requests": 1,
  "estimated_position": 1
}
```

### 查询请求状态

```bash
curl "http://localhost:8000/status/{request_id}"
```

### 获取处理结果

```bash
curl "http://localhost:8000/result/{request_id}" -o result.png
```

### 系统状态监控

```bash
curl "http://localhost:8000/system/status"
```

**响应示例：**
```json
{
  "total_gpus": 2,
  "alive_processes": 2,
  "ready_workers": 2,
  "total_requests": 10,
  "completed_requests": 8,
  "failed_requests": 0,
  "queued_requests": 2
}
```

### 健康检查

```bash
curl "http://localhost:8000/health"
```

## 并发测试

项目包含完整的并发测试脚本：

```bash
# 运行并发测试
python test.py
```

测试脚本会：
- 自动创建多个测试图像
- 并发提交多个编辑请求
- 监控处理进度和GPU使用情况
- 统计处理性能和并发效率

## API 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|-----|------|------|--------|------|
| `image` | File | ✅ | - | 输入图像文件 |
| `prompt` | String | ✅ | - | 编辑描述文本 |
| `negative_prompt` | String | ❌ | "" | 负面提示词 |
| `num_inference_steps` | Integer | ❌ | 50 | 推理步数 (10-100) |
| `true_cfg_scale` | Float | ❌ | 4.0 | CFG 缩放因子 (1.0-10.0) |
| `seed` | Integer | ❌ | 0 | 随机种子 |

## 项目结构

```
i2ii/
├── server.py          # 主服务文件，多GPU并发管理
├── test.py           # 并发测试脚本
├── outputs/          # 处理结果输出目录
└── README.md         # 项目文档
```

## 性能优化

### GPU 内存管理
- 自动清理 GPU 缓存
- 限制输入图像尺寸（最大768px）
- 使用 bfloat16 精度减少显存占用

### 并发优化
- 轮询负载均衡策略
- 异步请求处理
- 智能任务调度

### 监控与恢复
- 实时进程状态监控
- 异常自动处理
- 详细日志记录

## 环境变量配置

```bash
# 模型缓存目录
export MODELSCOPE_CACHE=/path/to/cache

# GPU显存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 指定使用的GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## 常见问题

### Q: 如何处理 CUDA 内存不足？
A: 可以通过以下方式优化：
- 减少 `num_inference_steps` 参数
- 限制输入图像尺寸
- 减少同时使用的GPU数量

### Q: 为什么某些GPU工作进程失败？
A: 检查以下项目：
- GPU显存是否充足
- CUDA驱动版本是否兼容
- 查看系统日志获取详细错误信息

### Q: 如何提高处理速度？
A: 建议：
- 使用更多GPU
- 减少推理步数
- 优化图像预处理

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 联系方式

- 作者：skyrover001
- 项目链接：https://github.com/skyrover001/i2ii
- 问题反馈：请通过 GitHub Issues 提交

---

如果这个项目对您有帮助，请给个 ⭐️ 支持一下！