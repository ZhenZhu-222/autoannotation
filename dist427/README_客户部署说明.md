# AutoAnnotation 客户部署说明（2026-04-27）

## 目录内容

| 文件 | 用途 |
|------|------|
| `autoannotation-v3.tar` | Docker 镜像包，导入后镜像名为 `autoannotation:v3` |
| `compose.yml` | 启动配置 |
| `SHA256SUMS.txt` | 文件校验 |

## 部署步骤

1. 安装 Docker Desktop。
2. 打开命令行，进入本目录。
3. 导入镜像：

```bash
docker load -i autoannotation-v3.tar
```

4. 修改 `compose.yml` 里的以下值：

```text
AUTOANNOTATION_ADMIN_PASSWORD
AUTOANNOTATION_SESSION_SECRET
AUTOANNOTATION_LICENSE_UNLOCK_KEY
```

5. 启动服务：

```bash
docker compose up -d
```

6. 浏览器打开：

```text
http://127.0.0.1:8010/
```

## 常用命令

```bash
docker compose ps
docker compose logs -f
docker compose down
docker compose up -d
```

## 导入超大数据集

直接在页面选择目录，系统会逐文件上传，避免一个超大请求断线。操作电脑和部署机器分离时，不需要填写部署机器上的文件路径。

## 升级旧版 v3

如果客户机器上原来已经有 `autoannotation:v3`，直接执行：

```bash
docker compose down
docker load -i autoannotation-v3.tar
docker compose up -d
```

`docker load` 后会把本机的 `autoannotation:v3` 标签指向新版镜像，原有 `autoannotation-data` 数据目录不变。

## 本版重点

- 修复上传 YOLO 文件夹时类别名丢失问题，支持 `classes.txt`、`obj.names`、`names.txt`、`data.yaml`。
- 修复多轮切换权重/AI 打标后类别名被 `class_0`、`class_1` 污染的主要风险。
- 数据精修支持对当前类别 ID 直接改名，并保持 label 数字 ID 不变。
- 文件夹上传增加进度与后端解析状态提示。
- 对照画廊展示标签 TXT 时带类别名，避免只看到数字 ID。
- 修复前端入口资源路径，`/front/index.html` 和服务首页均可正常加载。

## 注意

- 数据保存在 `./autoannotation-data`，升级镜像不会删除该目录。
- 如果换端口，需要同时修改 `ports` 和 `AUTOANNOTATION_CORS_ORIGINS`。
- 正式交付不要使用 `CHANGE_ME_*` 占位值。
