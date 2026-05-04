# autoannotation 自动打标工具

基于 FastAPI + 原生前端的全流程自动标注工具，支持视频抽帧、YOLO 权重模型打标、千问 VLM 零样本打标，三条流水线全并发。

## 功能概览

| 模块 | 说明 |
|------|------|
| 视频抽帧与清洗 | 上传视频 -> 按秒抽帧 -> 预览/筛选/批量删除 -> 生成干净图片集 |
| 权重模型自动打标 | 上传 YOLO .pt/.onnx -> 类别筛选与 ID 映射 -> 批量推理 -> YOLO TXT 标注 |
| 千问 VLM 零样本打标 | 文字描述 + 可选参考图 -> 千问大模型识别 -> SVG 绘图提取 -> YOLO TXT 标注 |
| 任务中心 | 全并发任务队列，实时进度，历史记录持久化 |
| 多轮打标 | append 追加模式 + 轮次标签 + 映射预设 + 回滚快照，支持多权重交替打标 |

## 快速启动

### 环境要求

- Python 3.10+
- 如需 GPU 推理：CUDA 环境 + 对应 PyTorch

### 安装与运行

```bash
cd autoannotation

# 方式一：使用 run.sh（自动创建 venv 并安装依赖）
# 开发模式（热重载 + 外网可访问）
AUTOANNOTATION_HOST=0.0.0.0 AUTOANNOTATION_RELOAD=true ./run.sh

# 生产模式（默认 127.0.0.1:8010，无热重载）
./run.sh

# 方式二：手动
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8010
```

浏览器打开 `http://127.0.0.1:8010` 即可使用。API 文档：`http://127.0.0.1:8010/docs`

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AUTOANNOTATION_CONFIG` | `./config/runtime.json` | 运行配置文件路径，支持 JSON |
| `AUTOANNOTATION_DATA_DIR` | 配置文件中的 `data_dir` 或 `./data` | 运行数据根目录 |
| `AUTOANNOTATION_FRONT_DIR` | 配置文件中的 `front_dir` 或 `./front` | 前端静态文件目录 |
| `AUTOANNOTATION_HOST` | `127.0.0.1` | 服务监听地址 |
| `AUTOANNOTATION_PORT` | `8010` | 服务端口 |
| `AUTOANNOTATION_RELOAD` | `false` | 开启热重载（开发模式设为 `true`） |
| `AUTOANNOTATION_CORS_ORIGINS` | `*` | 允许的 CORS 源，逗号分隔 |
| `AUTOANNOTATION_MAX_WORKERS` | `8` | 最大并发任务数 |
| `AUTOANNOTATION_DEFAULT_CONF` | `0.25` | YOLO 默认置信度阈值 |
| `AUTOANNOTATION_DEFAULT_IOU` | `0.45` | YOLO 默认 IoU 阈值 |
| `AUTOANNOTATION_DEFAULT_DEVICE` | _(空)_ | 推理设备（`0`=GPU0，`cpu`，`mps`） |
| `AUTOANNOTATION_ADMIN_PASSWORD` | `herefly` | 管理员密码；整站登录与管理员操作统一使用 |
| `AUTOANNOTATION_ADMIN_USERNAME` | `root` | 默认管理员账号 |
| `AUTOANNOTATION_SESSION_SECRET` | 自动派生 | 会话签名密钥；生产环境必须设置为随机长字符串 |
| `AUTOANNOTATION_MACHINE_ID` | _(空)_ | 硬件指纹读取失败时使用的人工机器 ID，适合 Docker/虚拟化部署 |
| `AUTOANNOTATION_QWEN_API_KEY` | _(空)_ | 千问 API Key |
| `AUTOANNOTATION_QWEN_API_BASE` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 千问 API 地址 |
| `AUTOANNOTATION_QWEN_MODEL` | `qwen-vl-max-latest` | 千问默认模型 |
| `AUTOANNOTATION_QWEN_TIMEOUT` | `45` | 千问单次请求超时（秒） |
| `AUTOANNOTATION_QWEN_MAX_REF_IMAGES` | `6` | 最大参考图数量 |
| `AUTOANNOTATION_MAX_MODEL_MB` | `500` | 模型文件上传大小限制（MB） |
| `AUTOANNOTATION_MAX_VIDEO_MB` | `2000` | 视频文件上传大小限制（MB） |
| `AUTOANNOTATION_MAX_IMAGE_MB` | `20` | 单张图片上传大小限制（MB） |

### 生产部署安全要求

首次商业交付或公网部署前，必须至少设置：

```bash
AUTOANNOTATION_ADMIN_USERNAME=你的管理员账号
AUTOANNOTATION_ADMIN_PASSWORD=强密码
AUTOANNOTATION_SESSION_SECRET=随机长字符串
AUTOANNOTATION_CORS_ORIGINS=http://你的域名或IP:端口
```

License 采用离线授权文件：首次启动自动进入 14 天试用；正式授权通过 `license.dat` 安装，授权内容绑定当前机器指纹并用 RSA 公钥验签。`/api/system/session` 会返回 `security_warnings`，用于提示默认密码、未显式会话密钥、开放 CORS 等风险。

### 商业化交付自检

登录后访问 `/api/system/readiness` 可查看交付自检结果：

- `operational=true`：运行目录、前端资源等基础可用。
- `production_ready=true`：基础可用，且无默认安全配置、无源码目录发布/模型大文件。
- `warnings`：列出仍需处理的生产风险或能力缺口。

也可以在源码目录执行：

```bash
scripts/commercial_acceptance.sh
```

完整交付验收步骤见 `商业化交付验收清单.md`。

## 核心工作流

### 流程一：视频抽帧 -> 权重打标

```
上传视频 -> 设置抽帧间隔 -> 获得图片集
    -> 上传 YOLO 模型 (.pt/.onnx)
    -> 选择/筛选类别、设置 ID 映射
    -> 启动打标 -> 生成 YOLO TXT 标注
    -> 对照画廊验收 -> 不满意可回滚
```

### 流程二：图片集 -> 千问零样本打标

```
选择已有图片集（或上传文件夹）
    -> 输入文字描述（如"标注所有车辆"）
    -> 可选上传参考图片
    -> 设置精度/采样参数
    -> 启动千问打标 -> 生成 YOLO TXT 标注
    -> 对照画廊验收 -> 不满意可回滚
```

### 流程三：多权重交替打标（持续产出数据集）

这是本工具的核心能力。同一个图片集可以被多个模型多轮打标，逐步丰富标注：

```
第一轮：YOLO 车辆模型打标
    -> label_mode = append, round_tag = "round-1-vehicle"
    -> 打标完成，labels/ 里有车辆框

第二轮：YOLO 行人模型打标（换模型）
    -> 选择行人模型，重新映射类别
    -> label_mode = append, round_tag = "round-2-person"
    -> 新框追加到同一批 txt 文件

第三轮：千问 VLM 补充小目标
    -> 文字描述 "标注所有交通标志"
    -> label_mode = append, round_tag = "round-3-sign"
    -> AI 识别结果追加到已有标注

每一轮都可以：
  - 在对照画廊中查看效果
  - 不满意可单独回滚该轮
  - 导出 CVAT 格式 ZIP 用于精标
```

**关键参数说明：**

| 参数 | 说明 |
|------|------|
| `label_mode = append` | 追加模式，新框加在旧框后面，不覆盖 |
| `label_mode = replace` | 覆盖模式，只保留本轮结果 |
| `update_imageset_labels = true` | 将结果回写到图片集 `labels/` 目录，下一轮可基于此继续 |
| `round_tag` | 轮次标签，用于在历史记录中区分各轮 |
| 映射预设 | 保存在浏览器 localStorage，换模型后可快速加载预设映射 |

### 类别映射 (Class ID Mapping)

支持多对一映射，适用于：
- 多个模型类别合并为统一类别（如 `car + bus + truck -> 0:vehicle`）
- 不同模型的类别 ID 不一致时统一编号

操作步骤：
1. 上传/选择模型 -> 读取模型类别
2. 手动输入目标类别（如 `vehicle / person / sign`）
3. 在 checklist 中勾选需要的来源类别，映射到目标 ID
4. 点击"确认映射（必做）"
5. 可保存为预设，下次一键加载

## 路径配置 / Docker / NFS

默认会读取 `/Users/zhao/Desktop/sourcespace/autoannotation/config/runtime.json:1`：

```json
{
  "data_dir": "./data",
  "front_dir": "./front"
}
```

说明：
- 这里的相对路径是**相对于配置文件所在目录**解析的
- 因此做 Docker 或 NFS 挂载时，只要改这个文件或环境变量，不需要改代码
- 图片集、模型、视频的元数据现在也会优先保存为**相对 `data_dir` 的路径**，换挂载点后更稳定

常见用法：

```bash
# 方式 1：直接改配置文件
{
  "data_dir": "./mounted-data",
  "front_dir": "./front"
}

# 方式 2：启动时覆盖
AUTOANNOTATION_DATA_DIR=./mounted-data ./run.sh

# 方式 3：指定另一份配置文件
AUTOANNOTATION_CONFIG=./config/runtime.docker.json ./run.sh
```

## 本地数据集导入格式

"上传本地文件夹"支持标准 YOLO 数据集：

```text
dataset/
  images/
    a.jpg
    subdir/b.jpg
  labels/
    a.txt
    subdir/b.txt
  classes.txt        # 可选
  data.yaml          # 可选，优先读取 names/task
```

导入规则：

- 支持 `images/xxx.jpg` + `labels/xxx.txt`，也支持同目录 `xxx.jpg` + `xxx.txt`。
- 子目录下同名文件按相对路径匹配，例如 `images/a/sample.jpg` 对应 `labels/a/sample.txt`。
- 类别名文件支持 `classes.txt`、`names.txt`、`obj.names`、`data.yaml` / `data.yml`。
- 上传成功后会展示导入图片数、标签数、类别数、未匹配标签数；类别文件为空或格式错误会直接提示原因。

## 发布产物与源码边界

- 源码仓库只应保存代码、配置模板、部署说明和测试。
- Docker 镜像包、`.tar`、`.zip`、模型权重等发布产物应由构建流程生成，不应长期纳入源码版本控制。
- `dist1/`、`dist2/`、`dist3/` 建议只保留 `compose.yml` 和部署说明；镜像包放到外部分发渠道或发布附件。

## API 接口概览

### 视频与抽帧
- `POST /api/videos/upload` — 上传视频
- `POST /api/extract/jobs` — 创建抽帧任务
- `GET /api/extract/jobs/{job_id}` — 查询抽帧进度

### 图片集
- `GET /api/imagesets` — 列出所有图片集
- `GET /api/imagesets/{id}/images` — 分页查看图片（支持筛选）
- `DELETE /api/imagesets/{id}` — 删除图片集
- `POST /api/imagesets/upload-folder` — 上传文件夹为图片集
- `POST /api/images/delete-batch` — 批量删除图片

### 模型
- `POST /api/models/upload` — 上传模型 + 类别文件
- `GET /api/models` — 列出已保存模型
- `GET /api/models/{id}/classes` — 获取模型类别
- `DELETE /api/models/{id}` — 删除模型

### 权重打标
- `POST /api/annotate/jobs` — 创建打标任务
- `GET /api/annotate/jobs/{job_id}` — 查询进度
- `GET /api/annotate/jobs/{job_id}/artifacts` — 获取产物链接
- `GET /api/annotate/jobs/{job_id}/preview` — 分页预览结果
- `POST /api/annotate/jobs/{job_id}/rollback` — 回滚标注

### 千问打标
- `POST /api/qwen/annotate/jobs` — 创建千问打标任务
- `GET /api/qwen/annotate/jobs/{job_id}` — 查询进度
- `POST /api/ai/qwen/suggest-classes` — 千问类别建议

### 任务管理
- `GET /api/jobs` — 活跃任务列表
- `GET /api/jobs/history` — 历史记录
- `POST /api/jobs/{job_id}/cancel` — 取消任务
- `DELETE /api/jobs/{job_id}/artifacts` — 清理产物
- `DELETE /api/jobs/history/{job_id}` — 删除历史记录

## 目录结构

```
autoannotation/
├── app/
│   ├── main.py                  # FastAPI 入口
│   ├── core/
│   │   ├── config.py            # 配置常量与环境变量
│   │   ├── models.py            # 数据模型定义
│   │   ├── state.py             # 内存状态管理
│   │   ├── utils.py             # 工具函数（路径安全/上传限制）
│   │   └── overlay_draw.py      # 绘图工具
│   ├── api/                     # REST API 路由
│   ├── services/                # 业务逻辑
│   └── schemas/                 # Pydantic 请求/响应模型
├── front/                       # 前端文件
├── data/                        # 运行时数据（自动创建）
│   ├── uploads/                 # 上传的视频和参考图
│   ├── imagesets/               # 图片集及其标注
│   ├── models/                  # 已保存的模型文件
│   ├── outputs/                 # 打标输出结果
│   └── history/                 # 任务历史记录
├── tests/                       # 测试
├── requirements.txt             # Python 依赖
└── run.sh                       # 启动脚本
```

## 输出格式

所有打标结果统一输出 YOLO TXT 格式：`<class_id> <cx> <cy> <w> <h>`（归一化坐标）

同时生成：
- `manifest.csv` — 每张图的详细打标统计
- `run_meta.json` — 本次运行的完整参数
- `overlays/` — 叠加可视化图片
- `labels_before/` — 回滚快照
- ZIP 打包 — 含完整结果和 CVAT 兼容格式

## 快捷键

| 按键 | 功能 | 生效面板 |
|------|------|----------|
| `[` | 上一页 | 视频抽帧 |
| `]` | 下一页 | 视频抽帧 |
| `A` | 全选/反选当前页 | 视频抽帧 |
| `Delete` | 批量删除选中图片 | 视频抽帧 |
| `Ctrl/Cmd + Enter` | 启动当前打标任务 | 权重打标 / 千问打标 |
