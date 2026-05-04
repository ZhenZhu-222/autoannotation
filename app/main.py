# ============================================================
# 应用启动入口
# 职责：创建 FastAPI 实例 / 注册中间件 / 挂载路由 / 静态文件服务
# 思路：
#   1. create_app() 工厂函数初始化全部组件
#   2. access_guard 中间件统一处理登录验证和 License 检查
#   3. 最后一行 app = create_app() 被 uvicorn 读取
# ============================================================
from __future__ import annotations

import sys
from pathlib import Path
# 直接 python app/main.py 时，项目根目录不在 sys.path，需手动加入
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes_assignments import router as assignments_router
from app.api.routes_ai import router as ai_router
from app.api.routes_annotate import router as annotate_router
from app.api.routes_imagesets import router as imagesets_router
from app.api.routes_jobs import router as jobs_router
from app.api.routes_models import router as models_router
from app.api.routes_qwen_annotate import router as qwen_annotate_router
from app.api.routes_system import alias_router as system_alias_router
from app.api.routes_system import router as system_router
from app.api.routes_train import router as train_router
from app.api.routes_videos import router as videos_router
from app.core.auth import get_current_user, get_license_status, is_authenticated
from app.core.config import CORS_ORIGINS, DATA_DIR, FRONT_DIR, LOGIN_REQUIRED, SESSION_COOKIE_NAME, ensure_dirs
from app.core.rbac import can_operate_imageset
from app.core.state import AppState
from app.services.task_manager import TaskManager


def create_app() -> FastAPI:
    # ---------- 初始化目录和应用实例 ----------
    ensure_dirs()

    app = FastAPI(title="autoannotation自动打标工具", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------- 挂载全局状态 ----------
    app.state.app_state = AppState()
    app.state.task_manager = TaskManager()
    app.state.session_cookie_name = SESSION_COOKIE_NAME

    # ---------- 公开路径白名单（无需登录） ----------
    public_exact_paths = {
        "/health",
        "/front/auth.html",
        "/front/license.html",
        "/api/system/session",
        "/api/system/login",
        "/api/system/logout",
        "/api/system/license/status",
        "/api/license/status",
        "/api/auth/me",
    }
    public_prefixes = (
        "/front/style.css",
        "/front/ui_utils.js",
        "/front/api.js",
        "/front/app.js",
        "/front/auth.js",
        "/front/dataset_upload.js",
        "/front/imageset_preview.js",
        "/front/gallery.js",
        "/front/class_mapping.js",
        "/front/annotate.js",
        "/front/qwen_annotate.js",
        "/front/jobs_history.js",
        "/front/refine.js",
        "/front/train.js",
        "/front/license.js",
        "/front/preview.js",
        "/front/preview.html",
        "/front/favicon",
    )

    # ---------- 统一访问控制中间件：登录 + License 检查 ----------
    # 类似 Java 网关的 GlobalFilter，所有请求必须经过这里
    # 步骤：① 白名单放行 → ② 登录检查 → ③ License 检查 → ④ 放行到路由 → ⑤ 响应加工
    @app.middleware("http")
    async def access_guard(request: Request, call_next):
        path = request.url.path

        # ① 白名单放行：登录页、静态 JS/CSS、公开 API 不需要鉴权
        if path in public_exact_paths or any(path.startswith(prefix) for prefix in public_prefixes):
            return await call_next(request)

        # ② 登录检查：从 cookie 里解析 session token，验证签名和有效期
        authed = is_authenticated(request)
        if LOGIN_REQUIRED and not authed:
            # API/数据请求返回 401 JSON；页面请求 307 跳转登录页
            if path.startswith("/api/") or path.startswith("/data/"):
                return JSONResponse(status_code=401, content={"detail": "请先登录"})
            return RedirectResponse(url="/front/auth.html", status_code=307)

        # ③ License 检查：试用到期或未激活时锁定系统
        license_status = get_license_status()
        if license_status.locked:
            # 锁定状态下只放行「安装 license」和「解锁」接口，其余全拦
            if path in {"/api/system/license/install", "/api/license/install", "/api/system/license/unlock"} and authed:
                return await call_next(request)
            if path.startswith("/api/") or path.startswith("/data/"):
                return JSONResponse(status_code=423, content={"detail": "系统已锁定，请先输入解锁密钥"})
            if path != "/front/license.html":
                return RedirectResponse(url="/front/license.html", status_code=307)

        # ④ 全部检查通过，放行到具体路由函数执行
        response = await call_next(request)

        # ⑤ 响应加工：试用期内在 header 里附带剩余天数，前端据此显示提示条
        if license_status.mode == "trial" and license_status.trial_days_left > 0:
            response.headers["X-License-Trial-Days-Left"] = str(license_status.trial_days_left)
        return response

    # ---------- 注册路由 ----------
    app.include_router(system_router)
    app.include_router(system_alias_router)
    app.include_router(videos_router)
    app.include_router(imagesets_router)
    app.include_router(assignments_router)
    app.include_router(models_router)
    app.include_router(ai_router)
    app.include_router(qwen_annotate_router)
    app.include_router(annotate_router)
    app.include_router(train_router)
    app.include_router(jobs_router)

    # ---------- 受控数据文件服务 ----------
    @app.get("/data/{resource_path:path}")
    def serve_data_file(resource_path: str, request: Request):
        # /data 不能再裸挂 StaticFiles，否则 operator 拿到 URL 就能绕过 API 权限。
        # 这里按文件类型和所属图片集做一次后端强制校验。
        rel = Path(resource_path)
        if rel.is_absolute() or ".." in rel.parts:
            return JSONResponse(status_code=404, content={"detail": "not found"})
        target = (DATA_DIR / rel).resolve()
        try:
            target.relative_to(DATA_DIR.resolve())
        except ValueError:
            return JSONResponse(status_code=404, content={"detail": "not found"})
        if not target.exists() or not target.is_file():
            return JSONResponse(status_code=404, content={"detail": "not found"})

        user = get_current_user(request)
        if not user:
            return JSONResponse(status_code=401, content={"detail": "请先登录"})

        rel_posix = rel.as_posix()
        if target.suffix.lower() in {".zip", ".pt", ".onnx", ".engine"} or rel_posix.startswith("models/"):
            # ZIP/模型权重属于外流核心资产，只允许 admin 下载。
            if user.role != "admin":
                return JSONResponse(status_code=403, content={"detail": "需要管理员权限"})
        if rel.parts and rel.parts[0] == "imagesets" and len(rel.parts) >= 2:
            # 图片集静态资源按 own/assigned 判断。
            # 本期不防 operator 批量抓取自己已授权图片集的图片。
            imageset = request.app.state.app_state.get_imageset(rel.parts[1])
            if imageset is None:
                return JSONResponse(status_code=404, content={"detail": "not found"})
            if not can_operate_imageset(user, imageset):
                return JSONResponse(status_code=403, content={"detail": "无权访问该图片集"})
        return FileResponse(target)

    app.mount("/front", StaticFiles(directory=str(FRONT_DIR)), name="front")

    # ---------- 健康检查和首页 ----------
    @app.get("/health")
    def health() -> dict:
        return {"service": "autoannotation", "status": "ok"}

    @app.get("/", response_model=None)
    def root():
        index_path = FRONT_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path, media_type="text/html; charset=utf-8")
        return RedirectResponse(url="/docs")

    return app


# ---------- uvicorn 入口 ----------
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
