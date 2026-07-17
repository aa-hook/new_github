# rank_v11 部署包

该目录包含三个 V11 checkpoint、最小推理源码、常驻 HTTP 服务和单图调用客户端。

## 安装

```bash
python -m pip install -r requirements.txt
```

GitHub 托管的 `ubuntu-latest` 没有 CUDA GPU，`--device auto` 会自动使用 CPU。CUDA 模式需要带 NVIDIA GPU 的 self-hosted runner。

## 每个 job 启动一个常驻模型

```bash
python server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --device auto \
  --mode accurate \
  --cache-dir "$RUNNER_TEMP/rank_v11_cache" \
  > "$RUNNER_TEMP/rank_v11.log" 2>&1 &

python wait_ready.py --url http://127.0.0.1:8765 --timeout 180
```

`wait_ready.py` 返回时，模型加载、SHA-256 校验和首次预热都已经完成。

## 提交一张图片

```bash
ANSWER_INDEX=$(python solve.py /absolute/path/challenge.jpg)
echo "$ANSWER_INDEX"
```

默认标准输出只有一个 `0-9` 的整数，可以直接传给注册步骤。查看完整结果：

```bash
python solve.py /absolute/path/challenge.jpg --json
```

返回示例：

```json
{"ok":true,"answer_index":6,"fast_index":6,"legacy_v9_index":7,"expert_index":6,"switched_to_v10":false}
```

## 单 job 一条命令

```bash
chmod +x example_job.sh
ANSWER_INDEX=$(./example_job.sh /absolute/path/challenge.jpg)
```

## GitHub Actions 256 个隔离 job

1. 将整个目录提交到仓库根目录，名称保持为 `rank_v11/`。
2. 将 `github-actions-example.yml` 复制为 `.github/workflows/rank-v11.yml`。
3. 把示例中的图片生成占位步骤替换为注册脚本，并让每个 job 生成自己的 `2000x400` JPG。
4. `solve.py` 输出的整数就是最终候选 index。

每个 matrix job 都有独立进程、文件系统、缓存目录和网络命名空间，因此都使用端口 `8765` 也不会冲突。一个 job 内遇到多轮验证码时，重复调用 `solve.py` 即可，模型不会重新加载。

`max-parallel: 256` 只是请求上限，实际并发数量仍受 GitHub 账号、组织和 runner 配额限制。

## Docker

```bash
docker build -t rank-v11 .
docker run -d --name rank-v11-job \
  -v /host/images:/images:ro \
  rank-v11

docker exec rank-v11-job \
  python solve.py /images/challenge.jpg --url http://127.0.0.1:8765
```

NVIDIA 容器增加 `--gpus all`，并设置 `RANK_V11_DEVICE=cuda`。
