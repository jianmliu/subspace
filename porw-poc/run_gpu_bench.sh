#!/usr/bin/env bash
# PoRW P1 GPU 基准一键脚本。在 GPU 机器上运行:
#   curl -fsSL https://raw.githubusercontent.com/jianmliu/subspace/claude/subspace-consensus-dram-bandwidth-y7pgo6/porw-poc/run_gpu_bench.sh | bash
# 产出: ~/porw-bench/subspace/porw-poc/results/gpu-bench-*.txt
# 若机器上有 jianmliu/subspace 的推送凭证会自动推回分支；否则把输出贴回对话。
set -uo pipefail

BRANCH=claude/subspace-consensus-dram-bandwidth-y7pgo6
WORK=~/porw-bench
mkdir -p "$WORK" && cd "$WORK"

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "no NVIDIA GPU visible"; exit 1; }

if [ -d subspace/.git ]; then
  git -C subspace fetch origin "$BRANCH" && git -C subspace checkout -B "$BRANCH" FETCH_HEAD
else
  git clone --depth 1 -b "$BRANCH" https://github.com/jianmliu/subspace subspace
fi
cd subspace/porw-poc

echo "== Python env =="
has_cuda_torch() { "$1" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; }

PY=""
# 1) 显式指定: PORW_PYTHON=/path/to/python
if [ -n "${PORW_PYTHON:-}" ] && has_cuda_torch "$PORW_PYTHON"; then
  PY="$PORW_PYTHON"
# 2) 当前 python3 已可用
elif has_cuda_torch python3; then
  PY=python3
else
  # 3) 扫描 conda/venv 环境 (含 voicechat 这类已配好的环境)
  for p in \
    $(conda env list 2>/dev/null | awk '$NF ~ /^\// {print $NF"/bin/python"}') \
    ~/miniconda3/envs/*/bin/python ~/anaconda3/envs/*/bin/python \
    ~/*/venv/bin/python ~/venv*/bin/python "$WORK/venv/bin/python"; do
    if [ -x "$p" ] && has_cuda_torch "$p"; then PY="$p"; break; fi
  done
fi
# 4) 兜底: 新建 venv, 用 cu121 索引装 CUDA 版 torch
if [ -z "$PY" ]; then
  echo "no torch+cuda env found; creating venv (downloads CUDA wheels, ~a few GB)"
  python3 -m venv "$WORK/venv" && source "$WORK/venv/bin/activate" && PY="$WORK/venv/bin/python"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q torch numpy --index-url https://download.pytorch.org/whl/cu121
fi
echo "using python: $PY"
"$PY" -m pip install -q triton pytest numpy 2>/dev/null \
  || echo "note: pip install into $PY failed; assuming deps already present"
$PY - <<'EOF'
import torch, triton
print("torch", torch.__version__, "| triton", triton.__version__,
      "| cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
EOF

mkdir -p results
OUT="results/gpu-bench-$(date +%Y%m%d-%H%M%S).txt"
{
  echo "=== host ==="; hostname; date -u
  nvidia-smi --query-gpu=name,memory.total,clocks.max.memory --format=csv
  echo; echo "=== correctness (native GPU backend) ==="
  $PY -m pytest tests/ -q 2>&1 | tail -3
  echo; echo "=== benchmark ==="
  $PY bench_gpu.py 2>&1
} | tee "$OUT"

echo
if git add "$OUT" && git commit -q -m "Add GPU benchmark results ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))" \
   && git push -q origin "$BRANCH" 2>/dev/null; then
  echo ">> results pushed to $BRANCH: porw-poc/$OUT"
else
  git reset -q HEAD~1 2>/dev/null || true
  echo ">> push 不可用。请把上面的输出（或 porw-poc/$OUT）贴回对话。"
fi
