#!/bin/bash
# ============================================================================
# run_v2_elastic_dispatch_precision_multi_node.sh — DeepEP V2 elastic dispatch
# 多机精度测试启动器（SSH fanout）
# noqa: SIZE_OK — 单文件运维启动器：须保持单文件形态以支持"拷贝到共享挂载即用"的零依赖分发
# ============================================================================
#
# 用途：
#   在 launcher（任意一台 NPU 服务器）上运行本脚本，通过免密 SSH 扇出到
#   NODE_IPS 列表中的所有 target，每个 target 直接运行
#   test/test_v2_elastic_dispatch_precision.py，各自带正确的 NODE_RANK
#   （= 列表下标，经 export RANK=<idx> 注入）和独立的 tee 日志文件。
#
#   基于 MoonEP scripts/run_bench_dispatch_udma_multi_node.sh 的 ssh_fanout
#   master 模板移植（无 hosts.d/，所有 target 共用 DEFAULT_*），按本仓库
#   DeepEP"每节点 spawn"约定适配：
#     - 不经 torchrun 封装，每个节点只起 1 个 python 进程，进程内部
#       mp.spawn --ep-size 个 rank（= 每节点 NPU 数）；
#     - WORLD_SIZE = 节点数（脚本按 NODE_IPS 长度自动 export）；
#     - RANK = 本节点编号 = 列表下标（脚本自动 export，无需手改）；
#     - MASTER_ADDR/MASTER_PORT 供 HCCL rendezvous；
#     - --shmem-ip-port 指向节点 0（SHMEM config store server 必须先 bind）。
#
#   等价的手写单节点命令（8 节点 64 卡示例，节点 i 改 RANK=i）：
#     ACL_DEVICE_SYNC_TIMEOUT=60 ASCEND_DEEPEP_TRANSPORT=udma \
#     WORLD_SIZE=8 RANK=<i> MASTER_ADDR=141.62.17.70 MASTER_PORT=29520 \
#     python3 test/test_v2_elastic_dispatch_precision.py \
#         --ep-size 8 --num-tokens 256 --num-max-tokens-per-rank 256 \
#         --hidden 7168 --num-topk 6 --num-experts 384 --num-aiv 56 \
#         --shmem-ip-port tcp://141.62.17.70:8399 \
#         --log-dir logs/ --run-tag ep64_8n
#   本脚本把以上 env/args 全部参数化，一条命令扇出到所有节点。
#
# 执行模型：
#   - 零参数执行：编辑下方"用户配置区"（至少填 NODE_IPS）后，直接 bash 本脚本
#   - --dry-run：预览每个 target 的 EXEC 计划，不执行
#   - --log-name <name>：命令行自定义日志名后缀（覆盖配置区 LOG_NAME）
#   - --log-dir <dir>：命令行自定义 tee 日志根目录（覆盖配置区 LOG_DIR）
#   - --help：打印帮助 + 当前配置区值
#   - rank = 列表 idx（不依赖 ip→rank 反查；远端通过 export RANK=... 知道自己 rank）
#   - leader-first：rank 0 先启动（节点 0 上的 SHMEM config store server 与
#     HCCL master 必须先 bind），sleep RANK0_START_DELAY 秒后并行启动其余 rank
#   - Ctrl-C/TERM/任一节点失败：控制机按本次 RUN_ID 清理所有节点的远端进程组；
#     cleanup 控制面与外层 tee 隔离，KILL 后验证进程组消失并保留 controller 审计日志
#   - 日志目录 ${LOG_DIR}（默认 ./logs，相对各节点 WORKDIR 即仓库根/logs；
#     仅真实执行才创建，dry-run 不建目录），
#     日志文件 ${N}n${R}p<idx>s_${LOG_NAME}.log
#     （含义 {nodes}n{R}p<rank>s_ 前缀：节点数 + 全局 EP 世界大小 R
#      （= EP_SIZE × N，即全部 NPU rank 数）+ 本节点 rank）
#     例：8 节点 EP_SIZE=8 共 64 rank，rank 0 → 8n64p0s_ep64_8n.log
#   - 注意区分两层日志：本脚本的 tee 日志（每节点 1 个文件，如上）与 python
#     --log-dir/--run-tag 生成的 per-rank 日志（test 脚本内部 mp.spawn 各 rank
#     单独落盘）。共享挂载下 --run-tag 必须所有节点一致（配置区自动保证）。
#
# 设计原则（详见 MoonEP ssh_fanout master 模板 banner）：
#   1. always-ssh 主模式 + local fast-path 自动优化
#   2. rank = 列表 idx + 双保险 export
#   3. 变量名可配置（ENV_NAME_*）
#   4. per-host 覆盖机制保留（hosts.d/<ip>.sh 存在即 source 覆盖；简单情况不创建）
#   5. CLI flag 仅 --dry-run / --help / --log-name / --log-dir；
#      测试参数改配置区 TEST_ARGS，标量改 EXTRA_ENV_VARS 或对应 ${VAR:-default}
#      环境变量（MASTER_ADDR=x.x.x.x bash 本脚本 可直接覆盖配置区默认值）
#   6. 退出策略 any-fail → exit 1
#   7. 远端进程隔离为独立 session；TERM 宽限后 KILL，校验 PID 启动时间防误杀，
#      并在 KILL 后验证进程组已经退出
#
# 路径解析：
#   SCRIPT_DIR = 脚本所在目录（scripts/）
#   ROOT_DIR   = 仓库根目录（SCRIPT_DIR/..）
#   HOSTS_DIR  = per-host 配置目录，默认 $SCRIPT_DIR/hosts.d，可由 env 覆盖
#   LOG_DIR    = tee 日志根目录，默认 ./logs（远端 cd 到 WORKDIR 后解析为
#                仓库根/logs，共享挂载可见；仅真实执行时创建）
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HOSTS_DIR="${HOSTS_DIR:-$SCRIPT_DIR/hosts.d}"

# ============================================================================
# 用户配置区（DeepEP 专用 — 必填 / 选填）
# ============================================================================
# 必填：
#   NODE_IPS                  target IP 数组（每个 IP 一行；列表顺序即 RANK 顺序）
#   DEFAULT_EXEC_COMMANDS     所有 host 共用执行命令（单元素，单引号，一行）
#
# 选填（全部支持 env 覆盖，${VAR:-default} 在 LOCAL shell 展开后注入远端）：
#   DEFAULT_ENV_SETUP         所有 host 共用环境配置（如 source set_env.sh / conda）
#   DEFAULT_WORKDIR           所有 host 共用工作目录（默认 $ROOT_DIR）
#   EP_SIZE                   每节点进程/NPU 数（--ep-size；全局 EP = EP_SIZE × 节点数）
#   NUM_TOKENS / NUM_MAX_TOKENS / HIDDEN / NUM_TOPK / NUM_EXPERTS / NUM_AIV
#                             精度测试形状参数（进 TEST_ARGS）
#   MASTER_ADDR               HCCL master，默认取 NODE_IPS[0]（rank-0 节点 IP）
#   MASTER_PORT               HCCL rendezvous 端口（默认 29520）
#   SHMEM_ADDR / SHMEM_PORT   SHMEM config store tcp:// 地址（默认跟随 MASTER_ADDR / 8399）
#   TRANSPORT                 ASCEND_DEEPEP_TRANSPORT（默认 udma）
#   ACL_SYNC_TIMEOUT          ACL_DEVICE_SYNC_TIMEOUT 秒数（默认 60）
#   RUN_TAG                   test 脚本 --run-tag（默认自动 ep<R>_<N>n，如 ep64_8n）
#   TEST_ARGS                 完整测试参数串（覆盖后上面形状参数不再进命令行）
#   LOG_NAME                  tee 日志名后缀（默认取 RUN_TAG；可用 --log-name 覆盖）
#   LOG_DIR                   tee 日志根目录（默认 ./logs，即仓库根/logs；可用 --log-dir 覆盖）
#   EXTRA_ENV_VARS            额外 env vars 数组，每元素 "K=V" 形式
#   SSH_OPTS                  ssh 选项
#   PRECHECK                  主机可达性预检 on|off（默认 on）
#   LEADER_FIRST              rank-0 先行 on|off（默认 on，SHMEM config store 需要）
#   RANK0_START_DELAY         rank-0 先行等待秒数（默认 5）
#   REMOTE_KILL_GRACE_SECONDS 远端 TERM 后等待秒数（默认 5，范围 1-300）
#
# 注意：
#   - 不要在 EXTRA_ENV_VARS 里放 TEST_ARGS（脚本会在 CLI 解析后自动注入
#     TEST_ARGS=<...> 导出，且值含空格必须单独处理）
#   - python 侧约定（见 README"多机启动"）：WORLD_SIZE = 节点数、RANK = 节点编号，
#     由本脚本自动 export，不要手工写进 EXTRA_ENV_VARS / TEST_ARGS

# ---- 必填：请填入你的实际值 ----

# 示例 A：单机 1 节点（EP8 走本机回环，仅验证脚本链路）
# NODE_IPS=(
#     141.62.17.70
# )


# 141.62.17.66
# 141.62.17.62
# 141.62.17.58
# 141.62.17.30
# 141.62.17.26
# 141.62.17.22
# 141.62.17.18

# 示例 B：8 节点 x 每节点 8 进程 = 全局 EP64（列表第 1 个 IP 即 rank 0 / MASTER_ADDR）
# B04
NODE_IPS=(
    141.62.17.70
    141.62.17.66
    141.62.17.62
    141.62.17.58
    141.62.17.30
    141.62.17.26
    141.62.17.22
    141.62.17.18
)

# NODE_IPS=(
#     141.62.17.70
#     141.62.17.66
# )

# NODE_IPS=(
#     141.62.17.26
#     141.62.17.22
# )

# 141.61.54.91
# 141.61.54.139
# 141.61.54.131
# 141.61.54.127
# 141.61.54.99
# 141.61.54.95
# 141.61.54.87
# 141.61.54.135

# b07
# NODE_IPS=(
#     141.61.54.91
#     141.61.54.139
#     141.61.54.131
#     141.61.54.127
#     141.61.54.99
#     141.61.54.95
#     141.61.54.87
#     141.61.54.135
# )

# NODE_IPS=()

# 环境配置示例（CANN + conda）：
    # "pkill -9 python"
DEFAULT_ENV_SETUP=(
    "pkill -9 python"
    "source /home/pkg/b020/ascend-toolkit/set_env.sh"
    "source /home/anaconda3/etc/profile.d/conda.sh"
    "conda activate /mnt/share/t00906153/conda/envs/moonep_tsj/"
)

# DEFAULT_ENV_SETUP=()

# 工作目录：test/ 在仓库根下，远端 payload 会先 cd 到 DEFAULT_WORKDIR
DEFAULT_WORKDIR="$ROOT_DIR"

# ---- 精度测试标量（env 可覆盖）----
EP_SIZE="${EP_SIZE:-8}"                     # --ep-size：每节点进程数（= 每节点 NPU 数）
NUM_TOKENS="${NUM_TOKENS:-256}"             # --num-tokens
NUM_MAX_TOKENS="${NUM_MAX_TOKENS:-256}"     # --num-max-tokens-per-rank
HIDDEN="${HIDDEN:-7168}"                    # --hidden
NUM_TOPK="${NUM_TOPK:-16}"                   # --num-topk
NUM_EXPERTS="${NUM_EXPERTS:-384}"           # --num-experts（须能被 EP_SIZE × 节点数整除）
NUM_AIV="${NUM_AIV:-56}"                    # --num-aiv
MASTER_PORT="${MASTER_PORT:-29520}"         # HCCL rendezvous 端口
SHMEM_PORT="${SHMEM_PORT:-8399}"            # SHMEM config store 端口
TRANSPORT="${TRANSPORT:-udma}"              # ASCEND_DEEPEP_TRANSPORT
ACL_SYNC_TIMEOUT="${ACL_SYNC_TIMEOUT:-300}"  # ACL_DEVICE_SYNC_TIMEOUT（秒）

# tee 日志名后缀（默认跟随 RUN_TAG，见下）
LOG_NAME_CLI=""                             # --log-name 命令行覆盖用
LOG_DIR_CLI=""                              # --log-dir 命令行覆盖用

# ---- 组网地址 ----
# MASTER_ADDR / SHMEM_ADDR 默认都取 NODE_IPS[0]（rank-0 节点 IP）
_MASTER_ADDR_DEFAULT="${NODE_IPS[0]:-127.0.0.1}"
MASTER_ADDR="${MASTER_ADDR:-$_MASTER_ADDR_DEFAULT}"
SHMEM_ADDR="${SHMEM_ADDR:-$MASTER_ADDR}"

# RUN_TAG 默认按规模自动命名：ep<EP_SIZE×节点数>_<节点数>n（8 节点 EP_SIZE=8 → ep64_8n）
_NUM_NODES_CFG="${#NODE_IPS[@]}"
(( _NUM_NODES_CFG > 0 )) || _NUM_NODES_CFG=1
_EP_WORLD_CFG=$(( EP_SIZE * _NUM_NODES_CFG ))
RUN_TAG="${RUN_TAG:-ep${_EP_WORLD_CFG}_${_NUM_NODES_CFG}n}_$(date '+%Y%m%d_%H%M')"

LOG_NAME="${LOG_NAME:-$RUN_TAG}"

# tee 日志根目录：默认 ./logs（相对路径；远端 payload 先 cd 到 DEFAULT_WORKDIR
# 再解析，即各节点仓库根下的 logs/，共享挂载可见；可用 --log-dir 覆盖）
LOG_DIR="${LOG_DIR:-logs}"

# test_v2_elastic_dispatch_precision.py 参数（远端 export TEST_ARGS，exec 命令行引用）
# 默认即 README 的多机 64 卡精度用例形状；LOG_DIR 同时也是 python --log-dir 的根，
# python 会在其下再建 <时间戳>_<run-tag> run 子目录写 per-rank 日志。
TEST_ARGS="${TEST_ARGS:---ep-size $EP_SIZE --num-tokens $NUM_TOKENS \
--num-max-tokens-per-rank $NUM_MAX_TOKENS --hidden $HIDDEN --num-topk $NUM_TOPK \
--num-experts $NUM_EXPERTS --num-aiv $NUM_AIV  --dtype fp8 \
--shmem-ip-port tcp://$SHMEM_ADDR:$SHMEM_PORT --log-dir $LOG_DIR --run-tag $RUN_TAG \
--profile}"

# 额外 env（test 脚本直接读的环境变量；RANK/WORLD_SIZE 由脚本自动注入，勿在此手写）
EXTRA_ENV_VARS=(
    "MASTER_ADDR=$MASTER_ADDR"                  # HCCL master（rank-0 节点 IP）
    "MASTER_PORT=$MASTER_PORT"
    "ASCEND_DEEPEP_TRANSPORT=$TRANSPORT"
    "ACL_DEVICE_SYNC_TIMEOUT=$ACL_SYNC_TIMEOUT"
    "HCCL_SOCKET_IFNAME=enp35s0f2,eth2"
    "DISPATCH_RESULTS_DIR=/mnt/share/t00906153/clock_result/$RUN_TAG"
    "DISPATCH_CLOCK_DIR=/mnt/share/t00906153/clock_result/$RUN_TAG"
)

# 执行命令：单元素，单引号。TEST_ARGS 是 export 的标量（含空格），必须先用
# read -a 按空格拆成数组 TEST_ARGV，再逐元素带引号展开为多个 argv——
# 直接写 "$TEST_ARGS" 会把整串当成一个参数传给 argparse（unrecognized arguments）。
# ${TEST_ARGV[@]+"${TEST_ARGV[@]}"} 兼容 set -u 下的空数组（bash 4.3）。
# RANK/WORLD_SIZE/MASTER_* 等由 export_stmt 注入远端环境。
DEFAULT_EXEC_COMMANDS=(
    'IFS=" " read -r -a TEST_ARGV <<< "$TEST_ARGS"; python3 test/test_v2_elastic_dispatch_precision.py ${TEST_ARGV[@]+"${TEST_ARGV[@]}"}'
)

# ============================================================================
# 默认配置区（用户一般不动）
# ============================================================================
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new}"
PRECHECK="${PRECHECK:-on}"

# DeepEP test 约定：RANK = 本节点编号，WORLD_SIZE = 节点数（故 ENV_NAME_NNODES 直接叫 WORLD_SIZE）
ENV_NAME_RANK="${ENV_NAME_RANK:-RANK}"
ENV_NAME_INDEX="${ENV_NAME_INDEX:-INDEX}"
ENV_NAME_HOST="${ENV_NAME_HOST:-HOST}"
ENV_NAME_NNODES="${ENV_NAME_NNODES:-WORLD_SIZE}"

# leader-first 语义（节点 0 的 SHMEM config store server + HCCL master 必须先 bind）
LEADER_FIRST="${LEADER_FIRST:-on}"
RANK0_START_DELAY="${RANK0_START_DELAY:-5}"
REMOTE_KILL_GRACE_SECONDS="${REMOTE_KILL_GRACE_SECONDS:-5}"
# 全局 EP 世界大小 R = EP_SIZE × N：日志前缀 {N}n{R}p<idx>s_ 用（主流程计算）
R_TOTAL="1"

# 每次 launcher 使用唯一标识。远端状态文件只记录本次 run 创建的进程组，
# Ctrl-C/TERM/异常退出时据此精确清理，禁止使用会误伤其他作业的模糊 pkill。
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}${RANDOM}"
RUN_FINISHED=0
CLEANUP_STARTED=0

# 用户配置区变量若未设，给空默认（让 set -u 不报错）
[[ -z "${NODE_IPS+x}" ]] && NODE_IPS=()
[[ -z "${DEFAULT_ENV_SETUP+x}" ]] && DEFAULT_ENV_SETUP=()
[[ -z "${DEFAULT_EXEC_COMMANDS+x}" ]] && DEFAULT_EXEC_COMMANDS=()
[[ -z "${DEFAULT_WORKDIR+x}" ]] && DEFAULT_WORKDIR=""
[[ -z "${EXTRA_ENV_VARS+x}" ]] && EXTRA_ENV_VARS=()

# ============================================================================
# 实现代码（用户一般不动）
# ============================================================================

# --- 颜色与日志辅助：只向真实终端输出 ANSI，tee/重定向日志保持纯文本 ---
LOG_BLUE=""
LOG_RESET=""
ERR_RED=""
ERR_YELLOW=""
ERR_RESET=""
if [[ -z "${NO_COLOR:-}" && -t 1 ]]; then
    LOG_BLUE=$'\033[1;34m'
    LOG_RESET=$'\033[0m'
fi
if [[ -z "${NO_COLOR:-}" && -t 2 ]]; then
    ERR_RED=$'\033[1;31m'
    ERR_YELLOW=$'\033[1;33m'
    ERR_RESET=$'\033[0m'
fi
log()  { printf '\n%s=== %s ===%s\n' "$LOG_BLUE" "$*" "$LOG_RESET"; }
err()  { printf '%s[ERROR] %s%s\n' "$ERR_RED" "$*" "$ERR_RESET" >&2; }
warn() { printf '%s[WARN] %s%s\n' "$ERR_YELLOW" "$*" "$ERR_RESET" >&2; }

# --- CLI flag 解析（--dry-run / --help / --log-name / --log-dir）---
MODE="run"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) MODE="dry-run"; shift ;;
        --help|-h) MODE="help"; shift ;;
        --log-name)
            if [[ $# -lt 2 ]]; then
                err "--log-name 需要一个值（日志名后缀，如 run1）"
                exit 2
            fi
            LOG_NAME_CLI="$2"
            shift 2
            ;;
        --log-dir)
            if [[ $# -lt 2 ]]; then
                err "--log-dir 需要一个值（日志根目录路径）"
                exit 2
            fi
            LOG_DIR_CLI="$2"
            shift 2
            ;;
        *) err "Unknown argument: $1（仅支持 --dry-run / --help / --log-name <name> / --log-dir <dir>）"; exit 2 ;;
    esac
done

# CLI 覆盖配置区（优先级最高）
[[ -n "$LOG_NAME_CLI" ]] && LOG_NAME="$LOG_NAME_CLI"
[[ -n "$LOG_DIR_CLI" ]] && LOG_DIR="$LOG_DIR_CLI"

# --- help ---
print_help() {
    cat <<EOF
用法:
    bash scripts/run_v2_elastic_dispatch_precision_multi_node.sh                   # 真实执行（默认 MODE=run）
    bash scripts/run_v2_elastic_dispatch_precision_multi_node.sh --dry-run         # 预览每个 target 的 EXEC 计划
    bash scripts/run_v2_elastic_dispatch_precision_multi_node.sh --log-name <name> # 自定义 tee 日志名后缀（覆盖 LOG_NAME）
    bash scripts/run_v2_elastic_dispatch_precision_multi_node.sh --log-dir <dir>   # 自定义 tee 日志根目录（覆盖 LOG_DIR）
    bash scripts/run_v2_elastic_dispatch_precision_multi_node.sh --help            # 打印本帮助 + 当前配置区值

环境变量覆盖示例（免编辑配置区）:
    NODE_IPS 仍需编辑脚本；其余标量可 env 覆盖，如:
    EP_SIZE=4 NUM_TOKENS=128 MASTER_PORT=29530 bash scripts/run_v2_elastic_dispatch_precision_multi_node.sh --dry-run

设计原则（详见顶部 banner）:
    1. always-ssh 主模式 + local fast-path 自动优化
    2. rank = 列表 idx（不依赖 ip→rank 反查）
    3. 变量名可配置（ENV_NAME_*）+ 双保险 export（RANK=idx, WORLD_SIZE=N）
    4. per-host 覆盖机制保留（hosts.d/<ip>.sh 存在即 source 覆盖；简单情况不创建）
    5. CLI flag 仅 --dry-run / --help / --log-name / --log-dir；
       其余全靠配置区 + 环境变量驱动
    6. leader-first：rank 0 先启动（SHMEM config store + HCCL master），等待后并行其余

配置区变量清单:
    必填:
        NODE_IPS                  target IP 数组（顺序即 RANK）
        DEFAULT_EXEC_COMMANDS     所有 host 共用执行命令（单元素，单引号）
    选填:
        DEFAULT_ENV_SETUP          所有 host 共用环境配置
        DEFAULT_WORKDIR            所有 host 共用工作目录（默认 \$ROOT_DIR）
        EP_SIZE/NUM_TOKENS/NUM_MAX_TOKENS/HIDDEN/NUM_TOPK/NUM_EXPERTS/NUM_AIV
                                   精度测试形状参数（进 TEST_ARGS）
        MASTER_ADDR/MASTER_PORT    HCCL rendezvous（默认 NODE_IPS[0]:29520）
        SHMEM_ADDR/SHMEM_PORT      SHMEM config store（默认跟随 MASTER_ADDR / 8399）
        TRANSPORT                  ASCEND_DEEPEP_TRANSPORT（默认 udma）
        ACL_SYNC_TIMEOUT           ACL_DEVICE_SYNC_TIMEOUT 秒（默认 60）
        RUN_TAG                    --run-tag（默认自动 ep<全局EP>_<节点数>n）
        TEST_ARGS                  完整测试参数串（env 覆盖则整串替换）
        LOG_NAME/LOG_DIR           tee 日志名/根目录（默认 \$RUN_TAG / ./logs，即仓库根/logs）
        ENV_NAME_RANK/INDEX/HOST/N_NODES   rank 等变量名（默认 RANK/INDEX/HOST/WORLD_SIZE）
        EXTRA_ENV_VARS             额外 env vars 数组（"K=V"）
        SSH_OPTS                   ssh 选项
        HOSTS_DIR                  per-host 配置目录
        PRECHECK                   主机可达性预检 on|off
        LEADER_FIRST               rank-0 先行 on|off（默认 on）
        RANK0_START_DELAY          rank-0 先行等待秒数（默认 5）
        REMOTE_KILL_GRACE_SECONDS  TERM 后等待秒数（默认 5，超时后 KILL）

当前值:
    NODE_IPS              = ${NODE_IPS[*]:-<空>}
    DEFAULT_ENV_SETUP     = ${DEFAULT_ENV_SETUP[*]:-<空>}
    DEFAULT_EXEC_COMMANDS = ${DEFAULT_EXEC_COMMANDS[*]:-<空>}
    DEFAULT_WORKDIR       = ${DEFAULT_WORKDIR:-<空>}
    EP_SIZE               = $EP_SIZE
    NUM_TOKENS            = $NUM_TOKENS
    NUM_MAX_TOKENS        = $NUM_MAX_TOKENS
    HIDDEN                = $HIDDEN
    NUM_TOPK              = $NUM_TOPK
    NUM_EXPERTS           = $NUM_EXPERTS
    NUM_AIV               = $NUM_AIV
    MASTER_ADDR           = $MASTER_ADDR
    MASTER_PORT           = $MASTER_PORT
    SHMEM_ADDR            = $SHMEM_ADDR
    SHMEM_PORT            = $SHMEM_PORT
    TRANSPORT             = $TRANSPORT
    ACL_SYNC_TIMEOUT      = $ACL_SYNC_TIMEOUT
    RUN_TAG               = $RUN_TAG
    LOG_DIR               = $LOG_DIR
    LOG_NAME              = $LOG_NAME
    TEST_ARGS             = ${TEST_ARGS:-<空>}
    HOSTS_DIR             = $HOSTS_DIR
    ENV_NAME_RANK/INDEX/HOST/NNODES = $ENV_NAME_RANK/$ENV_NAME_INDEX/$ENV_NAME_HOST/$ENV_NAME_NNODES
    EXTRA_ENV_VARS        = ${EXTRA_ENV_VARS[*]:-<空>}
    SSH_OPTS              = $SSH_OPTS
    PRECHECK              = $PRECHECK
    LEADER_FIRST          = $LEADER_FIRST
    RANK0_START_DELAY     = $RANK0_START_DELAY
    REMOTE_KILL_GRACE_SECONDS = $REMOTE_KILL_GRACE_SECONDS
    RUN_ID                = $RUN_ID
EOF
}

# --- IPv4 格式校验 ---
is_valid_ipv4() {
    local ip="$1"
    [[ "$ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || return 1
    local IFS=.
    local a b c d
    read -r a b c d <<< "$ip"
    [[ $a -le 255 && $b -le 255 && $c -le 255 && $d -le 255 ]]
}

# --- Pre-flight 检查 ---
preflight() {
    local rc=0

    # 1. NODE_IPS 非空
    if [[ ${#NODE_IPS[@]} -eq 0 ]]; then
        err "NODE_IPS 为空（必填，请在用户配置区填入 target IP 数组）"
        rc=1
    fi

    # 2. IP 格式校验
    local ip
    for ip in "${NODE_IPS[@]:-}"; do
        [[ -z "$ip" ]] && continue
        if ! is_valid_ipv4 "$ip"; then
            err "NODE_IPS 中 '$ip' 不是合法 IPv4"
            rc=1
        fi
    done

    # 3. 配置非空（DEFAULT_EXEC_COMMANDS 或 hosts.d 任一存在）
    local has_default_cmds=0
    [[ ${#DEFAULT_EXEC_COMMANDS[@]} -gt 0 ]] && has_default_cmds=1
    local has_host_override=0
    if [[ -d "$HOSTS_DIR" ]]; then
        for ip in "${NODE_IPS[@]:-}"; do
            [[ -z "$ip" ]] && continue
            if [[ -f "$HOSTS_DIR/${ip}.sh" ]]; then
                has_host_override=1
                break
            fi
        done
    fi
    if [[ $has_default_cmds -eq 0 && $has_host_override -eq 0 ]]; then
        err "DEFAULT_EXEC_COMMANDS 为空且 hosts.d/ 下无对应配置"
        err "请在用户配置区填 DEFAULT_EXEC_COMMANDS，或为每个 target 创建 hosts.d/<ip>.sh"
        rc=1
    fi

    # 4. 可达性预检
    if [[ "$PRECHECK" == "on" && ${#NODE_IPS[@]} -gt 0 ]]; then
        log "Pre-flight: 检查 ${#NODE_IPS[@]} 个 target 可达性..."
        local unreachable=()
        for ip in "${NODE_IPS[@]:-}"; do
            [[ -z "$ip" ]] && continue
            if ! ssh $SSH_OPTS -o ConnectTimeout=5 "$ip" "true" 2>/dev/null; then
                unreachable+=("$ip")
            fi
        done
        if [[ ${#unreachable[@]} -gt 0 ]]; then
            err "以下 target 不可达（免密未配置或 sshd 未运行）:"
            for ip in "${unreachable[@]}"; do
                err "  $ip"
            done
            rc=1
        fi
    fi

    # 5. (DeepEP) 全局 EP 世界一致性：R = EP_SIZE × N ≥ 2，且 NUM_EXPERTS 能被 R
    #    整除（test 脚本 main() 有同样断言，提前在 host 侧暴露）
    if [[ "$EP_SIZE" =~ ^[0-9]+$ ]] && (( 10#$EP_SIZE >= 1 )); then
        local _world=$(( EP_SIZE * ${#NODE_IPS[@]} ))
        if [[ "$NUM_EXPERTS" =~ ^[0-9]+$ ]] && (( ${#NODE_IPS[@]} > 0 && _world >= 2 && NUM_EXPERTS % _world != 0 )); then
            err "--num-experts ($NUM_EXPERTS) 必须能被全局 EP 世界大小 R = EP_SIZE ($EP_SIZE) × 节点数 (${#NODE_IPS[@]}) = $_world 整除"
            rc=1
        fi
        if (( ${#NODE_IPS[@]} > 0 )); then
            log "Pre-flight: 全局 EP 世界大小 R = EP_SIZE ($EP_SIZE) × 节点数 (${#NODE_IPS[@]}) = $_world"
        fi
    else
        err "EP_SIZE 必须是正整数: $EP_SIZE"
        rc=1
    fi
    local port
    for port in "$MASTER_PORT" "$SHMEM_PORT"; do
        if [[ ! "$port" =~ ^[0-9]+$ ]] || (( 10#$port < 1 || 10#$port > 65535 )); then
            err "端口号必须是 [1, 65535] 内的整数: $port"
            rc=1
        fi
    done
    if [[ "$ACL_SYNC_TIMEOUT" != "auto" && ! "$ACL_SYNC_TIMEOUT" =~ ^[0-9]+$ ]]; then
        err "ACL_SYNC_TIMEOUT 必须是 'auto' 或非负整数秒: $ACL_SYNC_TIMEOUT"
        rc=1
    fi
    if (( ${#NODE_IPS[@]} > 0 )); then
        if [[ "$TEST_ARGS" != *"--run-tag"* ]]; then
            warn "TEST_ARGS 未含 --run-tag：共享挂载下各节点 per-rank 日志将落在不同时间戳目录，难以对齐"
        fi
        if [[ "$TEST_ARGS" != *"--shmem-ip-port"* ]]; then
            warn "TEST_ARGS 未含 --shmem-ip-port：多机运行将落到默认 tcp://127.0.0.1:4455，节点间连不通"
        fi
    fi

    # 6. 清理协议与 export 输入校验，避免无界等待和远端 shell 注入。
    if [[ ! "$REMOTE_KILL_GRACE_SECONDS" =~ ^[0-9]+$ ]] \
        || (( 10#$REMOTE_KILL_GRACE_SECONDS < 1 || 10#$REMOTE_KILL_GRACE_SECONDS > 300 )); then
        err "REMOTE_KILL_GRACE_SECONDS 必须是 [1, 300] 内的整数"
        rc=1
    fi
    if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
        err "内部 RUN_ID 含非法字符: $RUN_ID"
        rc=1
    fi
    local env_name
    for env_name in "$ENV_NAME_RANK" "$ENV_NAME_INDEX" "$ENV_NAME_HOST" "$ENV_NAME_NNODES"; do
        if [[ ! "$env_name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            err "环境变量名不合法: $env_name"
            rc=1
        fi
    done
    local kv
    for kv in "${EXTRA_ENV_VARS[@]:-}"; do
        [[ -z "$kv" ]] && continue
        if [[ ! "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            err "EXTRA_ENV_VARS 必须使用合法的 K=V 形式: $kv"
            rc=1
        fi
    done
    if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3) )); then
        err "控制机需要 Bash >= 4.3（使用 wait -n 做 fail-fast）"
        rc=1
    fi
    if [[ "$PRECHECK" != "on" && "$PRECHECK" != "off" ]]; then
        err "PRECHECK 只能是 on 或 off"
        rc=1
    fi
    if [[ "$LEADER_FIRST" != "on" && "$LEADER_FIRST" != "off" ]]; then
        err "LEADER_FIRST 只能是 on 或 off"
        rc=1
    fi
    if [[ ! "$RANK0_START_DELAY" =~ ^[0-9]+$ ]] || (( 10#$RANK0_START_DELAY > 300 )); then
        err "RANK0_START_DELAY 必须是 [0, 300] 内的整数"
        rc=1
    fi
    if [[ ! "$LOG_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
        err "LOG_NAME 只能包含字母、数字、点、下划线和连字符"
        rc=1
    fi
    local i j
    for ((i=0; i<${#NODE_IPS[@]}; i++)); do
        for ((j=i+1; j<${#NODE_IPS[@]}; j++)); do
            if [[ "${NODE_IPS[$i]}" == "${NODE_IPS[$j]}" ]]; then
                err "NODE_IPS 不允许重复 target: ${NODE_IPS[$i]}"
                rc=1
            fi
        done
    done

    return $rc
}

# --- 本地 IP 检测（给 is_local_host 用）---
declare -a LOCAL_IPS=()
detect_local_ips() {
    LOCAL_IPS=()
    local ips
    # hostname -I 优先
    if ips=$(hostname -I 2>/dev/null); then
        local ip
        for ip in $ips; do
            [[ "$ip" =~ ^127\. ]] && continue
            LOCAL_IPS+=("$ip")
        done
    fi
    # ip -4 -o addr show 回退
    if [[ ${#LOCAL_IPS[@]} -eq 0 ]] && command -v ip &>/dev/null; then
        while read -r _ _ ip _; do
            ip="${ip%%/*}"
            [[ "$ip" =~ ^127\. ]] && continue
            LOCAL_IPS+=("$ip")
        done < <(ip -4 -o addr show 2>/dev/null)
    fi
}

# --- 判断 target 是否本机 ---
is_local_host() {
    local target="$1"
    [[ "$target" == "localhost" || "$target" =~ ^127\. ]] && return 0
    [[ "$target" == "$(hostname 2>/dev/null)" ]] && return 0
    local ip
    for ip in "${LOCAL_IPS[@]:-}"; do
        [[ "$ip" == "$target" ]] && return 0
    done
    return 1
}

# --- 解析 per-host 配置（覆盖默认）---
resolve_host_config() {
    local host="$1"

    # 1. 重置为默认值
    ENV_SETUP=()
    local cmd
    for cmd in "${DEFAULT_ENV_SETUP[@]:-}"; do
        [[ -z "$cmd" ]] && continue
        ENV_SETUP+=("$cmd")
    done
    EXEC_COMMANDS=()
    for cmd in "${DEFAULT_EXEC_COMMANDS[@]:-}"; do
        [[ -z "$cmd" ]] && continue
        EXEC_COMMANDS+=("$cmd")
    done
    WORKDIR="${DEFAULT_WORKDIR:-$ROOT_DIR}"

    # 2. 如果有 per-host 配置文件，source 覆盖
    local host_conf="$HOSTS_DIR/${host}.sh"
    if [[ -f "$host_conf" ]]; then
        # shellcheck disable=SC1090
        source "$host_conf"
    fi
}

# --- 用 && 串联命令数组 ---
join_with_and() {
    local arr_name="$1"
    local ref="${arr_name}[@]"
    local chain=""
    local first=1
    local item
    for item in "${!ref}"; do
        [[ -z "$item" ]] && continue
        if [[ $first -eq 1 ]]; then
            chain="$item"
            first=0
        else
            chain="$chain && $item"
        fi
    done
    echo "$chain"
}

# --- 构造 export 语句（变量名可配置 + 额外变量 + TEST_ARGS）---
# DeepEP test 约定：RANK=节点编号（= 列表 idx），WORLD_SIZE=节点数（ENV_NAME_NNODES
# 默认已设为 WORLD_SIZE），MASTER_ADDR/MASTER_PORT/TRANSPORT/... 走 EXTRA_ENV_VARS。
build_export_stmt() {
    local idx="$1" host="$2" n="$3"
    local stmt="export"
    local quoted
    printf -v quoted '%q' "${ENV_NAME_RANK}=$idx"; stmt+=" $quoted"
    printf -v quoted '%q' "${ENV_NAME_INDEX}=$idx"; stmt+=" $quoted"
    printf -v quoted '%q' "${ENV_NAME_HOST}=$host"; stmt+=" $quoted"
    printf -v quoted '%q' "${ENV_NAME_NNODES}=$n"; stmt+=" $quoted"
    printf -v quoted '%q' "DEEPEP_RUN_ID=$RUN_ID"; stmt+=" $quoted"
    printf -v quoted '%q' "DEEPEP_NODE_INDEX=$idx"; stmt+=" $quoted"

    if [[ ${#EXTRA_ENV_VARS[@]} -gt 0 ]]; then
        local kv
        for kv in "${EXTRA_ENV_VARS[@]}"; do
            [[ -z "$kv" ]] && continue
            printf -v quoted '%q' "$kv"
            stmt+=" $quoted"
        done
    fi

    # TEST_ARGS 含空格，按一个 export 参数做 shell quoting。
    if [[ -n "$TEST_ARGS" ]]; then
        printf -v quoted '%q' "TEST_ARGS=$TEST_ARGS"
        stmt+=" $quoted"
    fi

    echo "$stmt"
}

# --- 日志文件名（{N}n{R}p{idx}s_{LOG_NAME}.log；R 取 R_TOTAL = EP_SIZE × N）---
log_file_for() {
    local idx="$1" n="$2"
    local base="${LOG_NAME%.log}"   # 去掉可能的 .log 后缀，统一追加
    echo "${n}n${R_TOTAL}p${idx}s_${base}.log"
}

# --- 远端受管 session：PID/启动时间原子落盘，退出时移除状态文件 ---
remote_session() {
    set -Eeuo pipefail
    local state_file="$1" cancel_file="$2" run_id="$3" payload="$4"
    local state_tmp="${state_file}.tmp.$$"
    local quoted_state_file quoted_state_tmp
    local -a stat_fields=()

    umask 077
    read -r -a stat_fields < "/proc/$$/stat"
    printf '%s %s %s\n' "$$" "$run_id" "${stat_fields[21]}" > "$state_tmp"
    mv -f -- "$state_tmp" "$state_file"

    printf -v quoted_state_tmp '%q' "$state_tmp"
    printf -v quoted_state_file '%q' "$state_file"
    # Freeze local paths now; EXIT runs after remote_session locals leave scope.
    # shellcheck disable=SC2064
    trap "rm -f -- $quoted_state_tmp $quoted_state_file" EXIT

    # cleanup 可能先于远端 session 落盘；cancel tombstone 关闭这个竞态窗口。
    if [[ -e "$cancel_file" ]]; then
        echo "[remote][$run_id] launch cancelled before payload start" >&2
        return 130
    fi

    local rc=0
    set +e
    bash -c "$payload"
    rc=$?
    set -e
    return "$rc"
}

# --- 远端 supervisor：为 payload 创建独立 session/process group ---
remote_supervisor() {
    set -Eeuo pipefail
    local run_id="$1" node_index="$2" payload="$3"
    local state_root
    state_root="${TMPDIR:-/tmp}/ascend-deepep-runs-$(id -u)"
    local state_file="$state_root/${run_id}.${node_index}.pid"
    local cancel_file="$state_root/${run_id}.${node_index}.cancel"
    local session_source rc=0

    umask 077
    if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ || ! "$node_index" =~ ^[0-9]+$ ]]; then
        echo "[remote] refusing invalid run identity" >&2
        return 2
    fi
    local required_cmd
    for required_cmd in setsid grep tr; do
        command -v "$required_cmd" >/dev/null 2>&1 || {
            echo "[remote][$run_id] $required_cmd is required for safe process-group cleanup" >&2
            return 1
        }
    done
    local setsid_help
    setsid_help="$(setsid --help 2>&1 || true)"
    if [[ "$setsid_help" != *"--wait"* ]]; then
        echo "[remote][$run_id] setsid must support --wait" >&2
        return 1
    fi
    mkdir -p -- "$state_root"
    chmod 700 "$state_root"
    if [[ -e "$cancel_file" ]]; then
        echo "[remote][$run_id] launch already cancelled" >&2
        return 130
    fi
    if [[ -e "$state_file" ]]; then
        local old_pid=""
        read -r old_pid _ _ < "$state_file" || true
        if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 -- "-$old_pid" 2>/dev/null; then
            echo "[remote][$run_id] active state already exists: $state_file" >&2
            return 1
        fi
        rm -f -- "$state_file"
    fi

    session_source="$(declare -f remote_session)"
    set +e
    setsid --wait env \
        "DEEPEP_RUN_ID=$run_id" \
        "DEEPEP_NODE_INDEX=$node_index" \
        bash -c "${session_source}"$'\n''remote_session "$@"' \
        deepep-remote-session "$state_file" "$cancel_file" "$run_id" "$payload"
    rc=$?
    set -e
    return "$rc"
}

# --- 远端幂等清理：校验 run id + PID 启动时间后终止整个进程组 ---
remote_cleanup_run() {
    set -u
    local run_id="$1" node_index="$2" grace_seconds="$3"
    local verify_seconds=5
    local state_root
    state_root="${TMPDIR:-/tmp}/ascend-deepep-runs-$(id -u)"
    local state_file="$state_root/${run_id}.${node_index}.pid"
    local cancel_file="$state_root/${run_id}.${node_index}.cancel"
    local pgid="" recorded_run="" recorded_start="" current_start=""
    local -a stat_fields=()

    umask 077
    mkdir -p -- "$state_root" || return 1
    chmod 700 "$state_root" || return 1
    : > "$cancel_file"

    [[ -e "$state_file" ]] || {
        echo "[cleanup][$run_id][node=$node_index] no active state (cancel marker installed)"
        return 0
    }
    read -r pgid recorded_run recorded_start < "$state_file" || {
        echo "[cleanup][$run_id][node=$node_index] unreadable state; refusing kill" >&2
        return 1
    }
    if [[ ! "$pgid" =~ ^[0-9]+$ || "$recorded_run" != "$run_id" \
        || ! "$recorded_start" =~ ^[0-9]+$ ]]; then
        echo "[cleanup][$run_id][node=$node_index] invalid state; refusing kill" >&2
        return 1
    fi
    if [[ ! -r "/proc/$pgid/stat" ]]; then
        rm -f -- "$state_file"
        echo "[cleanup][$run_id][node=$node_index] process group already exited"
        return 0
    fi
    read -r -a stat_fields < "/proc/$pgid/stat"
    current_start="${stat_fields[21]:-}"
    if [[ "$current_start" != "$recorded_start" ]]; then
        echo "[cleanup][$run_id][node=$node_index] PID was reused; refusing kill" >&2
        return 1
    fi
    if [[ "${stat_fields[4]:-}" != "$pgid" ]]; then
        echo "[cleanup][$run_id][node=$node_index] PID is not its process-group leader; refusing kill" >&2
        return 1
    fi
    if ! tr '\0' '\n' < "/proc/$pgid/environ" \
        | grep -Fqx "DEEPEP_RUN_ID=$run_id"; then
        echo "[cleanup][$run_id][node=$node_index] run marker mismatch; refusing kill" >&2
        return 1
    fi

    echo "[cleanup][$run_id][node=$node_index] TERM process group $pgid"
    kill -TERM -- "-$pgid" 2>/dev/null || true
    local deadline=$((SECONDS + 10#$grace_seconds))
    while kill -0 -- "-$pgid" 2>/dev/null && (( SECONDS < deadline )); do
        sleep 0.2
    done
    if kill -0 -- "-$pgid" 2>/dev/null; then
        echo "[cleanup][$run_id][node=$node_index] grace expired; KILL process group $pgid" >&2
        kill -KILL -- "-$pgid" 2>/dev/null || true
        deadline=$((SECONDS + verify_seconds))
        while kill -0 -- "-$pgid" 2>/dev/null && (( SECONDS < deadline )); do
            sleep 0.2
        done
        if kill -0 -- "-$pgid" 2>/dev/null; then
            echo "[cleanup][$run_id][node=$node_index] process group $pgid still exists after KILL" >&2
            return 1
        fi
        echo "[cleanup][$run_id][node=$node_index] KILL verified; process group $pgid exited"
    else
        echo "[cleanup][$run_id][node=$node_index] TERM verified; process group $pgid exited"
    fi
    rm -f -- "$state_file"
    return 0
}

LAST_LAUNCH_PID=""
launch_remote_function_async() {
    local function_name="$1" host="$2"
    shift 2

    if is_local_host "$host"; then
        {
            declare -f remote_session remote_supervisor remote_cleanup_run
            printf '\n%s' "$function_name"
            printf ' %q' "$@"
            printf '\n'
        } | bash -s &
    else
        {
            declare -f remote_session remote_supervisor remote_cleanup_run
            printf '\n%s' "$function_name"
            printf ' %q' "$@"
            printf '\n'
        } | ssh $SSH_OPTS "$host" 'bash -s' &
    fi
    LAST_LAUNCH_PID=$!
}

# --- 启动单个 rank（helper for run branch）---
start_rank() {
    local idx="$1"
    local host="${NODE_IPS[$idx]}"
    resolve_host_config "$host"
    local env_chain
    env_chain=$(join_with_and ENV_SETUP)
    local exec_chain
    exec_chain=$(join_with_and EXEC_COMMANDS)
    local workdir="$WORKDIR"
    local log_file
    log_file="$LOG_DIR/$(log_file_for "$idx" "$N")"
    local export_stmt
    export_stmt=$(build_export_stmt "$idx" "$host" "$N")
    local quoted_workdir quoted_log_dir quoted_log_file
    printf -v quoted_workdir '%q' "$workdir"
    printf -v quoted_log_dir '%q' "$LOG_DIR"
    printf -v quoted_log_file '%q' "$log_file"

    # payload 自带 pipefail，使 test/python 失败不会被 tee 吞掉。
    local payload="set -Eeuo pipefail; $export_stmt; cd -- $quoted_workdir"
    if [[ -n "$env_chain" ]]; then
        # CANN/conda 的环境脚本通常会读取尚未定义的变量；只在加载环境期间
        # 关闭 nounset，并显式保留环境链的失败状态。
        payload+="; set +u; if { $env_chain; }; then set -u; else "
        payload+='env_setup_rc=$?; set -u; exit "$env_setup_rc"; fi'
    fi
    payload+="; $export_stmt; mkdir -p -- $quoted_log_dir"
    [[ -n "$exec_chain" ]] && payload+="; { $exec_chain; } 2>&1 | tee -- $quoted_log_file"

    printf '[launch][node-rank=%s] starting host=%s run_id=%s command=%s\n' \
        "$idx" "$host" "$RUN_ID" "$exec_chain"
    launched_indices+=("$idx")
    launch_remote_function_async remote_supervisor "$host" "$RUN_ID" "$idx" "$payload"
    bg_pids+=("$LAST_LAUNCH_PID")
}

# --- Ctrl-C/TERM/异常退出：从控制机回收所有已启动的远端进程组 ---
declare -a bg_pids=()
declare -a launched_indices=()
cleanup_all() {
    [[ "$CLEANUP_STARTED" -eq 0 ]] || return 0
    CLEANUP_STARTED=1
    set +e
    if [[ ${#launched_indices[@]} -eq 0 ]]; then
        return 0
    fi

    # cleanup 的控制面不能继承用户的 stdout/stderr 管道。外层普通 `tee`
    # 会在 Ctrl-C 时退出；若 cleanup SSH 继续写该断管，SIGPIPE/SSH 断连可能
    # 截断 TERM -> KILL。所有节点直接追加同一个 run-scoped 审计文件，完成后再尽力回放。
    local cleanup_log="$LOG_DIR/controller_cleanup_${RUN_ID}.log"
    if ! (umask 077; : > "$cleanup_log") 2>/dev/null; then
        cleanup_log="${TMPDIR:-/tmp}/ascend-deepep-cleanup-${RUN_ID}.log"
        if ! (umask 077; : > "$cleanup_log") 2>/dev/null; then
            cleanup_log="/dev/null"
        fi
    fi
    if [[ "$cleanup_log" != "/dev/null" ]]; then
        chmod 600 -- "$cleanup_log" 2>/dev/null || true
        printf '[controller][cleanup] start run_id=%s nodes=%s grace=%ss\n' \
            "$RUN_ID" "${#launched_indices[@]}" "$REMOTE_KILL_GRACE_SECONDS" \
            >> "$cleanup_log" 2>/dev/null || true
    fi
    warn "Cleanup started for run_id=$RUN_ID; audit log: $cleanup_log" || true

    local -a cleanup_pids=()
    local idx host pid cleanup_failed=0
    for idx in "${launched_indices[@]}"; do
        host="${NODE_IPS[$idx]}"
        launch_remote_function_async remote_cleanup_run "$host" \
            "$RUN_ID" "$idx" "$REMOTE_KILL_GRACE_SECONDS" \
            >> "$cleanup_log" 2>&1
        cleanup_pids+=("$LAST_LAUNCH_PID")
    done
    for pid in "${cleanup_pids[@]}"; do
        wait "$pid" || cleanup_failed=1
    done
    # 只终止仍登记在当前 shell job table 中的 controller 子进程，避免 PID 重用误杀。
    local running_pid tracked_pid
    while read -r running_pid; do
        for tracked_pid in "${bg_pids[@]:-}"; do
            if [[ "$running_pid" == "$tracked_pid" ]]; then
                kill -TERM "$running_pid" 2>/dev/null || true
                break
            fi
        done
    done < <(jobs -pr)
    for pid in "${bg_pids[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
    if [[ "$cleanup_failed" -ne 0 ]]; then
        printf '[controller][cleanup] FAILED: at least one node was not verified clean\n' \
            >> "$cleanup_log" 2>/dev/null || true
    else
        printf '[controller][cleanup] complete run_id=%s\n' "$RUN_ID" \
            >> "$cleanup_log" 2>/dev/null || true
    fi

    # stderr 通常仍连接控制终端；即使用户使用 2>&1 | tee 导致它也断开，
    # handle_signal/handle_exit 已忽略 PIPE，审计日志仍完整保留在文件中。
    if [[ "$cleanup_log" != "/dev/null" && -r "$cleanup_log" ]]; then
        while IFS= read -r cleanup_line || [[ -n "$cleanup_line" ]]; do
            printf '%s\n' "$cleanup_line" >&2 || true
        done < "$cleanup_log"
    fi
    if [[ "$cleanup_failed" -ne 0 ]]; then
        warn "至少一个节点未确认完成安全清理；审计日志: $cleanup_log" || true
        return 1
    fi
    warn "Remote cleanup completed for run_id=$RUN_ID; audit log: $cleanup_log" || true
    return 0
}

handle_signal() {
    local signal_name="$1" exit_code=1
    [[ "$signal_name" == "HUP" ]] && exit_code=129
    [[ "$signal_name" == "INT" ]] && exit_code=130
    [[ "$signal_name" == "QUIT" ]] && exit_code=131
    [[ "$signal_name" == "TERM" ]] && exit_code=143
    [[ "$signal_name" == "PIPE" ]] && exit_code=141
    # 第二次 Ctrl-C/TERM 和已断开的 tee 都不能中断正在执行的安全清理。
    trap '' HUP INT QUIT TERM PIPE
    warn "Received $signal_name; stopping run_id=$RUN_ID" || true
    cleanup_all || true
    RUN_FINISHED=1
    trap - EXIT
    exit "$exit_code"
}

handle_exit() {
    local exit_code="$1"
    trap - EXIT
    trap '' HUP INT QUIT TERM PIPE
    if [[ "$RUN_FINISHED" -ne 1 ]]; then
        cleanup_all || true
    fi
    exit "$exit_code"
}

trap 'handle_signal HUP' HUP
trap 'handle_signal INT' INT
trap 'handle_signal QUIT' QUIT
trap 'handle_signal TERM' TERM
trap 'handle_signal PIPE' PIPE
trap 'handle_exit $?' EXIT

# ============================================================================
# 主流程
# ============================================================================

# 0. help 早退
if [[ "$MODE" == "help" ]]; then
    print_help
    exit 0
fi

# 1. 检测本地 IP
detect_local_ips
if [[ ${#LOCAL_IPS[@]} -eq 0 ]]; then
    warn "无法检测本地 IP（hostname -I 和 ip 命令都失败）"
    warn "local fast-path 不会触发，所有 target 走 ssh（包括 ssh self）"
fi

# 2. Pre-flight 检查
if ! preflight; then
    err "Pre-flight 检查失败，退出"
    exit 1
fi

N=${#NODE_IPS[@]}

# 全局 EP 世界大小 R = EP_SIZE × N（日志前缀 {N}n{R}p<idx>s_ 用）
# 注意：此处只做推导，不创建日志目录（目录仅在 run 分支 mkdir，dry-run 不落盘）
R_TOTAL=$(( EP_SIZE * N ))

# 4. dry-run 分支
if [[ "$MODE" == "dry-run" ]]; then
    log "DRY-RUN: 预览 $N 个 target 的 EXEC 计划"
    echo
    echo "全局配置:"
    echo "  NODE_IPS      = ${NODE_IPS[*]}"
    echo "  N_NODES       = $N（export 为 $ENV_NAME_NNODES）"
    echo "  R_TOTAL       = $R_TOTAL（EP_SIZE $EP_SIZE × N $N，日志前缀中的 R）"
    echo "  LOG_DIR       = $LOG_DIR"
    echo "  LOG_NAME      = $LOG_NAME"
    echo "  RUN_TAG       = $RUN_TAG"
    echo "  MASTER_ADDR   = $MASTER_ADDR:$MASTER_PORT"
    echo "  TEST_ARGS     = ${TEST_ARGS:-<空>}"
    echo "  HOSTS_DIR     = $HOSTS_DIR"
    echo "  ENV_NAME_RANK/INDEX/HOST/NNODES = $ENV_NAME_RANK/$ENV_NAME_INDEX/$ENV_NAME_HOST/$ENV_NAME_NNODES"
    echo "  EXTRA_ENV_VARS = ${EXTRA_ENV_VARS[*]:-<空>}"
    echo "  SSH_OPTS      = $SSH_OPTS"
    echo "  LOCAL_IPS     = ${LOCAL_IPS[*]:-<未检出>}"
    echo "  LEADER_FIRST  = $LEADER_FIRST"
    echo "  RANK0_START_DELAY = $RANK0_START_DELAY"
    echo "  RUN_ID        = $RUN_ID"
    echo "  REMOTE_KILL_GRACE_SECONDS = $REMOTE_KILL_GRACE_SECONDS"
    echo
    echo "Per-target EXEC 计划:"

    for idx in "${!NODE_IPS[@]}"; do
        host="${NODE_IPS[$idx]}"
        resolve_host_config "$host"
        env_chain=$(join_with_and ENV_SETUP)
        exec_chain=$(join_with_and EXEC_COMMANDS)
        workdir="$WORKDIR"
        log_file="$LOG_DIR/$(log_file_for "$idx" "$N")"
        export_stmt=$(build_export_stmt "$idx" "$host" "$N")

        target_mode="ssh"
        if is_local_host "$host"; then
            target_mode="local"
        fi

        echo "  [rank=$idx] host=$host mode=$target_mode workdir=$workdir log=$log_file"
        echo "    export_stmt: $export_stmt"
        echo "    env_chain:   ${env_chain:-<空>}"
        echo "    exec_chain:  ${exec_chain:-<空>}"
        echo
    done

    if [[ "$LEADER_FIRST" == "on" && $N -gt 1 ]]; then
        echo "  [leader-first] rank 0 将先启动，等待 ${RANK0_START_DELAY}s 后并行启动其余 rank"
        echo
    fi

    log "DRY-RUN 结束（未执行任何命令）"
    RUN_FINISHED=1
    exit 0
fi

# 5. run 分支：真实执行（此刻才创建日志目录；dry-run 不落盘）
mkdir -p "$LOG_DIR"
log "DeepEP V2 elastic dispatch 精度测试 SSH fanout 启动：$N 个 target（全局 EP$R_TOTAL），tee 日志目录 $LOG_DIR"

if [[ "$LEADER_FIRST" == "on" && $N -gt 1 ]]; then
    log "leader-first: 先启动 rank 0（SHMEM config store + HCCL master 所在节点）..."
    start_rank 0
    log "leader-first: 等待 ${RANK0_START_DELAY}s 让节点 0 的服务端口绑定..."
    sleep "$RANK0_START_DELAY"
    for ((idx=1; idx<N; idx++)); do
        start_rank "$idx"
    done
else
    for idx in "${!NODE_IPS[@]}"; do
        start_rank "$idx"
    done
fi

# 6. 等待所有后台进程；任一节点失败立即触发全局清理
log "Waiting for ${#bg_pids[@]} background job(s)..."
log "tee 日志目录: ${LOG_DIR}（test 脚本 per-rank 日志在 ${LOG_DIR} 下以 --run-tag=${RUN_TAG} 命名的 run 子目录内）"

remaining=${#bg_pids[@]}
while (( remaining > 0 )); do
    rc=0
    wait -n || rc=$?
    if [[ $rc -ne 0 ]]; then
        err "A node job exited with code $rc; cancelling the whole run"
        exit 1
    fi
    remaining=$((remaining - 1))
done

RUN_FINISHED=1
log "All jobs completed successfully (run_id=$RUN_ID)"
exit 0
