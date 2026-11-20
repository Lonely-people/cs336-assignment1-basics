#!/usr/bin/env bash
# ==============================================================================
# run_bpe_tinystories.sh
#
# 使用 cs336_basics/train_bpe.py 在 TinyStories valid 数据集上运行 BPE 训练。
# 不保存任何文件；可选通过 --profile 启用 cProfile，按 tottime 排序输出。
# ==============================================================================

set -euo pipefail

# --------------------
# 默认参数
# --------------------
DATA_DIR="${DATA_DIR:-data}"
DATA_FILENAME="TinyStoriesV2-GPT4-valid.txt"
DATA_URL="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/${DATA_FILENAME}"
DEFAULT_INPUT="${DATA_DIR}/${DATA_FILENAME}"

INPUT_FILE="${DEFAULT_INPUT}"
VOCAB_SIZE=10000
EXTRA_ARGS=()   # 透传给 train_bpe.py 的额外参数（如 --profile --top-n 20）

# --------------------
# 帮助信息
# --------------------
show_help() {
    cat << EOF
用法:
  ./run_bpe_tinystories.sh [输入文件] [词表大小] [--profile] [--top-n N]

参数:
  输入文件     训练文本路径 (默认: data/TinyStoriesV2-GPT4-valid.txt)
  词表大小     BPE 目标词表大小 (默认: 10000)

选项:
  --profile    启用 cProfile 性能分析，结果按 tottime 排序
  --top-n N    cProfile 显示前 N 个函数 (默认: 30)
  -h, --help   显示本帮助信息

示例:
  # 默认运行（下载并训练 TinyStories valid，不保存文件）
  ./run_bpe_tinystories.sh

  # 启用 cProfile 性能分析
  ./run_bpe_tinystories.sh --profile

  # 使用内置样例快速验证，查看前 20 个函数
  ./run_bpe_tinystories.sh tests/fixtures/tinystories_sample_5M.txt 1000 --profile --top-n 20
EOF
}

# --------------------
# 解析参数
# --------------------
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)  show_help; exit 0 ;;
        --profile)  EXTRA_ARGS+=("--profile"); shift ;;
        --top-n)    EXTRA_ARGS+=("--top-n" "$2"); shift 2 ;;
        *)          POSITIONAL+=("$1"); shift ;;
    esac
done

[[ ${#POSITIONAL[@]} -ge 1 ]] && INPUT_FILE="${POSITIONAL[0]}"
[[ ${#POSITIONAL[@]} -ge 2 ]] && VOCAB_SIZE="${POSITIONAL[1]}"

echo "============================================================"
echo " BPE 训练 — TinyStories Valid Dataset"
echo "============================================================"
echo "  输入文件:   ${INPUT_FILE}"
echo "  词表大小:   ${VOCAB_SIZE}"
echo "  特殊 Token: <|endoftext|>"
echo "  额外参数:   ${EXTRA_ARGS[*]:-（无）}"
echo "  保存文件:   否"
echo "============================================================"

# --------------------
# 自动下载数据集
# --------------------
if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "[INFO] 未找到: ${INPUT_FILE}"
    if [[ "${INPUT_FILE}" == "${DEFAULT_INPUT}" ]]; then
        echo "[INFO] 正在从 Hugging Face 下载 TinyStories valid..."
        mkdir -p "${DATA_DIR}"
        DOWNLOADED=false
        command -v wget &>/dev/null && wget -c "${DATA_URL}" -O "${INPUT_FILE}" && DOWNLOADED=true
        [[ "$DOWNLOADED" == false ]] && command -v curl &>/dev/null && curl -L "${DATA_URL}" -o "${INPUT_FILE}" && DOWNLOADED=true
        if [[ "$DOWNLOADED" == false || ! -s "${INPUT_FILE}" ]]; then
            rm -f "${INPUT_FILE}"
            echo "[错误] 下载失败。请手动下载:"
            echo "  wget ${DATA_URL} -O ${INPUT_FILE}"
            echo "或用内置样例测试:"
            echo "  ./run_bpe_tinystories.sh tests/fixtures/tinystories_sample_5M.txt 1000 --profile"
            exit 1
        fi
    else
        echo "[错误] 文件不存在: ${INPUT_FILE}"; exit 1
    fi
fi

# --------------------
# 启动训练（直接调用现有 Python 代码）
# --------------------
uv run python cs336_basics/train_bpe.py \
    --input-path "${INPUT_FILE}" \
    --vocab-size "${VOCAB_SIZE}" \
    --special-tokens "<|endoftext|>" \
    "${EXTRA_ARGS[@]}"