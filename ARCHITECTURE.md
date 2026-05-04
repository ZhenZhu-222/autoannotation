# 项目架构说明 (V2)

> V2 设计原则：`api` 层**严格作为 Controller**，所有业务逻辑下沉至 `services` 层。

## 一、分层总览

```
┌───────────────────────────────────────────────────────────────┐
│                          main.py                              │
│               应用入口 / 中间件 / 路由挂载                       │
└──────────────────────────┬────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      ┌─────────┐   ┌───────────┐   ┌───────────┐
      │   api   │   │  schemas  │   │  services │
      │ Controller│ │  DTO 层   │   │  业务层   │
      │ 纯转发   │   │ 请求/响应 │   │  核心逻辑 │
      │ 无逻辑   │   │           │   │           │
      └────┬────┘   └───────────┘   └─────┬─────┘
           │                              │
           └──────────────┬───────────────┘
                          ▼
                   ┌────────────┐
                   │    core    │
                   │ 基础设施层 │
                   │ 配置/状态/  │
                   │ 工具/绘图   │
                   └────────────┘
```

### 各层职责（V2 规范）

| 层级 | 职责 | 禁止事项 |
|------|------|----------|
| `api` | 接收 HTTP 参数、调用 service、异常转 HTTP 码、返回响应 | 禁止直接操作文件、禁止复杂业务逻辑、禁止内联 DTO |
| `schemas` | 定义 Pydantic 模型，用于请求体验证和响应序列化 | 禁止包含业务逻辑 |
| `services` | 实现所有业务逻辑、算法调度、数据加工 | 禁止直接处理 HTTP 相关（Request/Response） |
| `core` | 基础设施：配置、状态管理、工具函数、绘图 | 禁止依赖 services 层 |

---

## 二、文件级对应关系

### 2.1 api ↔ services 对应（V2 新增服务）

| api (Controller)           | services (Service)                      | 职责                          |
|----------------------------|-----------------------------------------|-------------------------------|
| `routes_system.py`         | —                                       | 登录/登出/会话/License        |
| `routes_videos.py`         | `extract_service.py`                    | 视频上传 + 抽帧任务             |
| `routes_imagesets.py`      | `imageset_service.py`                   | 图片集 CRUD / 上传 / 查询       |
|                            | `refine_service.py`                     | 单张图片框编辑                  |
|                            | `class_name_service.py`                 | 类别名称解析                    |
|                            | **`remap_service.py`** (V2 新增)        | 类别 ID 重映射                  |
| `routes_models.py`         | `model_service.py`                      | 模型上传 + classes 解析         |
| `routes_annotate.py`       | `annotation_service.py`                 | YOLO 模型打标（核心）           |
|                            | **`annotate_preview_service.py`** (V2) | 预览查询 / 回滚 / 产物汇总     |
| `routes_qwen_annotate.py`  | `qwen_annotation_service.py`            | Qwen 大模型打标（核心）         |
|                            | **`qwen_upload_service.py`** (V2)       | 参考图归一化 / 暂存 / 参数校验  |
| `routes_ai.py`             | `qwen_service.py`                       | Qwen 智能类别推荐               |
| `routes_train.py`          | `training_service.py`                   | YOLO 训练                     |
| `routes_jobs.py`           | `task_manager.py`                       | 任务队列/状态/取消              |
|                            | **`job_cleanup_service.py`** (V2)       | 任务产物清理 / 级联删除         |

### 2.1.1 服务分类（V2）

```
services/
├── 核心服务（主业务）
│   ├── annotation_service.py       # YOLO 打标
│   ├── qwen_annotation_service.py  # Qwen 大模型打标
│   ├── training_service.py         # YOLO 训练
│   └── extract_service.py          # 视频抽帧
│
├── 预览/清理服务（V2 拆分）
│   ├── annotate_preview_service.py   # 预览/回滚/产物
│   ├── job_cleanup_service.py        # 任务清理
│   ├── qwen_upload_service.py        # 参考图预处理
│   └── remap_service.py              # 类别重映射
│
└── 辅助服务
    ├── imageset_service.py         # 图片集管理
    ├── refine_service.py           # 单图编辑
    ├── model_service.py            # 模型/类别解析
    ├── class_name_service.py       # 类别名称解析
    ├── qwen_service.py             # Qwen API 调用
    └── inference.py                # YOLO 推理封装
```

### 2.2 api ↔ schemas 对应（V2 补全）

| api (Controller)           | schemas (DTO)                      |
|----------------------------|------------------------------------|
| `routes_system.py`         | **`system.py`** (V2 新增)          |
| `routes_videos.py`         | `extract.py` + **`video.py`** (V2) |
| `routes_imagesets.py`      | `image.py`                         |
| `routes_train.py`          | **`train.py`** (V2 新增)           |
| `routes_annotate.py`       | `annotate.py`                      |
| `routes_jobs.py`           | `common.py`                        |

#### 新增 DTO 列表（V2 从 api 层迁出）

| DTO | 原位置 | 现位置 |
|-----|--------|--------|
| `LoginRequest` | `routes_system.py` 内联 | `schemas/system.py` |
| `UnlockRequest` | `routes_system.py` 内联 | `schemas/system.py` |
| `ImportLocalVideoRequest` | `routes_videos.py` 内联 | `schemas/video.py` |
| `TrainRequest` | `routes_train.py` 内联 | `schemas/train.py` |
| `RemapClassesRequest` | `routes_imagesets.py` 内联 | `schemas/image.py` |

### 2.3 core 层各文件职责

| core 文件            | 职责                                      | 被谁使用                         |
|----------------------|-------------------------------------------|----------------------------------|
| `config.py`          | 全局配置中心：路径/密钥/限制/环境变量      | 几乎所有文件                     |
| `models.py`          | dataclass 实体定义（Video/Image/Model/Job）| `state.py`, services             |
| `state.py`           | 内存状态 + JSON 持久化（DAO/Repository）   | api 层通过依赖注入获取           |
| `auth.py`            | 鉴权：Token 签发/校验 + License 管理       | `main.py`, `routes_system.py`    |
| `utils.py`           | 通用工具：ID/时间/文件名/上传流 + **`to_data_url`/`as_bool`** (V2) | 全项目 |

### 2.3.1 V2 新增工具函数

```python
# core/utils.py 新增（从 api 层收拢）
def to_data_url(path: Path, data_dir: Path) -> str: ...
def as_bool(value, default=False) -> bool: ...
def safe_int(val, default=0) -> int: ...
def safe_float(val, default=0.0) -> float: ...
```

| `overlay_draw.py`    | 检测框绘制（Pillow 优先，OpenCV 回退）     | `annotation_service.py`, `training_service.py` |

---

## 三、api 层设计规范（V2 强制）

```python
# 标准 api 层写法（以 routes_annotate.py 为例）

@router.get("/annotate/jobs/{job_id}/preview")
def get_annotate_preview(
    job_id: str,
    page: int = Query(default=1, ge=1),
    state: AppState = Depends(get_state),
    tasks: TaskManager = Depends(get_task_manager),
) -> dict:
    try:
        return AnnotatePreviewService.build_preview(
            job_id=job_id, page=page, state=state, tasks=tasks,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="job 不存在") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
```

### 规范检查清单

- [ ] **无内联 DTO**：所有 `BaseModel` 定义必须在 `schemas/` 层
- [ ] **无业务逻辑**：路由函数体不超过 10 行（try/except + service 调用 + 异常转换）
- [ ] **无私有辅助函数**：不以 `_` 开头定义任何函数，工具函数统一收拢到 `core/utils.py`
- [ ] **异常分层**：service 抛 `KeyError/ValueError/FileNotFoundError`，api 层转 `HTTPException`
- [ ] **依赖注入**：只通过 `Depends(get_state/get_task_manager)` 获取依赖

---

## 四、数据流向

### 3.1 请求处理链路

```
浏览器 → main.py (access_guard 中间件)
       → api/routes_*.py (参数校验, schemas 验证)
       → services/*.py (业务逻辑)
       → core/state.py (数据读写)
       → JSON 文件 (磁盘持久化)
```

### 3.2 异步任务流向

```
api 层                    TaskManager                     Service
  │                          │                              │
  │  submit(runner)          │                              │
  ├─────────────────────────▶│                              │
  │                          │  线程池执行 runner            │
  │                          ├─────────────────────────────▶│
  │                          │                              │
  │                          │  ctx.set_progress(n, total)  │
  │                          │◀─────────────────────────────┤
  │                          │                              │
  │  GET /jobs/{id}          │                              │
  ├─────────────────────────▶│  返回 progress               │
  │◀─────────────────────────┤                              │
  │                          │  完成 → 写入 JSONL 历史       │
  │                          │◀─────────────────────────────┤
```

### 3.3 YOLO 打标数据流

```
图片集 (imagesets/{id}/images/)
  │
  ▼
AnnotationService.run_annotate_job()
  ├── 加载 YOLO 模型 (inference.py)
  ├── 逐图推理 → Detection 列表
  ├── 类别 ID 映射 (class_id_overrides)
  ├── 生成 YOLO TXT 标签
  ├── 绘制叠加图 (overlay_draw.py)
  ├── 写入 outputs/{job_id}/
  │     ├── manifest.csv
  │     ├── labels/
  │     ├── overlays/
  │     └── run_meta.json
  └── 可选回写到 imagesets/{id}/labels/
```

### 3.4 Qwen 打标数据流

```
图片集 + 文字描述 + 参考图
  │
  ▼
QwenAnnotationService.run_qwen_annotate_job()
  ├── 调用 Qwen-VL API (qwen_service.py)
  ├── 解析 SVG 输出 → 提取坐标
  ├── 多轮采样 → 共识融合 → 质量评分
  ├── 生成 YOLO TXT 标签
  ├── 绘制叠加图
  └── 同上输出结构
```

---

## 五、运行时目录结构

```
{DATA_DIR}/                          # 默认 ./data 或 runtime.json 配置
├── uploads/
│   ├── videos/                      # 上传的视频文件
│   └── qwen_refs/                   # Qwen 参考图（临时）
├── imagesets/
│   └── imageset_{id}/
│       ├── metadata.json            # 图片集元数据
│       ├── images/                  # 图片文件
│       ├── labels/                  # YOLO TXT 标注
│       │   └── classes.txt          # 类别名列表
│       └── data.yaml                # YOLO 训练格式
├── models/
│   └── model_{id}/
│       └── xxx.pt                   # 模型权重文件
├── outputs/
│   └── job_{id}/
│       ├── manifest.csv             # 打标清单
│       ├── run_meta.json            # 运行参数
│       ├── labels/                  # 本次生成的标签
│       ├── overlays/                # 叠加可视化图
│       ├── labels_before/           # 回滚快照
│       └── labels_before_index.json # 回滚索引
├── history/
│   └── jobs_history.jsonl           # 任务历史（JSONL）
└── system/
    └── license_state.json           # License 状态
```

---

## 六、依赖注入

```python
# api/deps.py 提供两个依赖：
def get_state(request) -> AppState       # 全局状态（DAO 层）
def get_task_manager(request) -> TaskManager  # 任务管理器

# 路由中使用：
@router.get("/imagesets")
def list_imagesets(state: AppState = Depends(get_state)):
    ...
```

`main.py` 的 `create_app()` 负责创建 `AppState` 和 `TaskManager` 实例，
挂载到 `app.state` 上，api 层通过 `Depends()` 获取。

---

## 七、中间件执行顺序

```
请求进入
  │
  ▼
CORSMiddleware          # 跨域处理
  │
  ▼
access_guard            # 自定义中间件
  ├── 公开路径？ → 直接放行
  ├── 需要登录？ → 检查 Session Cookie
  │     └── 未登录 → 401 / 跳转 auth.html
  └── License 锁定？ → 检查是否过期
        └── 已锁定 → 423 / 跳转 license.html
  │
  ▼
路由匹配 → 执行 handler
```

---

## 附录：V2 重构变更清单

### 新增服务（从 api 层下沉）

| 服务 | 来源 | 职责 |
|------|------|------|
| `job_cleanup_service.py` | `routes_jobs.py` 内联逻辑 | 产物清理、级联删除图片集 |
| `annotate_preview_service.py` | `routes_annotate.py` 内联逻辑 | 预览列表、overlay 渲染、回滚、产物汇总 |
| `remap_service.py` | `routes_imagesets.py` 内联逻辑 | 类别 ID 重映射校验与执行 |
| `qwen_upload_service.py` | `routes_qwen_annotate.py` 内联逻辑 | 参考图归一化、暂存、参数校验 |

### 新增 schemas（从 api 层迁出）

| schema 文件 | 包含 DTO | 原位置 |
|-------------|----------|--------|
| `system.py` | `LoginRequest`, `UnlockRequest` | `routes_system.py` 内联 |
| `video.py` | `ImportLocalVideoRequest` | `routes_videos.py` 内联 |
| `train.py` | `TrainRequest` | `routes_train.py` 内联 |
| `image.py` (扩展) | `RemapClassesRequest` | `routes_imagesets.py` 内联 |

### 工具函数收拢

```python
# 从各 api 文件收拢到 core/utils.py
def to_data_url(path, data_dir) -> str    # 原 routes_imagesets.py, routes_annotate.py
def as_bool(value, default=False) -> bool  # 原 routes_qwen_annotate.py
def safe_int(val, default=0) -> int       # 原各路由文件零散定义
def safe_float(val, default=0.0) -> float
```

### api 层瘦身效果

| 文件 | 重构前 | 重构后 | 降幅 |
|------|--------|--------|------|
| `routes_annotate.py` | 595 行 | 119 行 | 80% |
| `routes_imagesets.py` | 529 行 | 228 行 | 57% |
| `routes_qwen_annotate.py` | 219 行 | 141 行 | 36% |
| `routes_jobs.py` | 158 行 | 86 行 | 46% |

### V2 核心规范

1. **api 层零内联 DTO**：所有 `class X(BaseModel)` 必须在 `schemas/`
2. **api 层零私有函数**：不以 `_` 定义任何函数
3. **api 层零业务逻辑**：只保留参数接收 + service 调用 + 异常转换
4. **异常分层**：service 抛领域异常，api 层转 HTTP 状态码
