#!/usr/bin/env bash
# B04 FIFO 简化版。Linux + Bash；所有人使用同一个共享目录。
# 用法：./b04ctl.sh run [ENV=value ...] command args...
#       ./b04ctl.sh status
# 可继续使用 run_b04 / b04_status 软链接；无需先运行 init。
#
# 方法：mkdir 发号锁 -> 递增 counter -> 建立 ticket -> 按号运行。
# 最多 10 个任务等待（不含运行中）；队列满时拒绝新提交，退出码 75。
# 拿到锁后先清理并复查；通过后才执行命令，失败则退出并解锁。
# 命令结束/Ctrl+C 后再尝试清理；清理失败也解锁，下一任务重新清理。
# 首次查询为空、复查才出现的进程也会进入一次清理，并记录信号发送结果。
# kill -9/断电可能留下记录；先确认对应进程和远端任务已结束，再人工删除。
# 启动器必须前台运行并等待所有远端任务；不能启动后台任务后提前返回。
# TERM/HUP 只发给 wrapper 时，Bash 等前台命令返回后才执行退出处理。
# 升级前需等旧版任务和队列全部结束，所有人统一切换到此版本。

set -u
umask 0002
export LC_COLLATE=C
shopt -s nullglob

base="${B04_STATE_DIR:-/mnt/share/npu_b04_64p}"
batch_exec="${B04_BATCH_EXEC:-/mnt/share/batch_exec_b04.sh}"
readonly expected_nodes=8
cleanup_timeout="${B04_CLEANUP_TIMEOUT:-180}"
readonly max_waiting=10
entry='' ticket='' have_guard=0 have_run=0 command_started=0
cleanup_stage='' archive_needed=0
timestamp() { date '+%Y-%m-%d %H:%M:%S %z'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
fail() { log "[错误] $*" >&2; exit 1; }

help_text() {
    cat <<'EOF'
运行：b04ctl.sh run [ENV=value ...] command args...
查看：b04ctl.sh status
示例：b04ctl.sh run DEBUG=1 ./batch_launch.sh "python test.py ..."

默认共享状态目录：/mnt/share/npu_b04_64p（可用 B04_STATE_DIR 修改）
B04 64P 表示约定共用的整组 64 张卡；占用状态来自共享锁。
按 ticket 顺序等待，任务前台执行；Ctrl+C 取消。
最多 10 个任务等待（不含正在运行的任务）；队列满时新提交退出，返回 75。
打印提交/开始/结束时间、等待及运行时长；排队状态变化或每 30 秒更新。
启动前及命令退出后均调用 /mnt/share/batch_exec_b04.sh（可用 B04_BATCH_EXEC 修改）。
逐节点查询 npu-smi info，只清理 NPU 进程表中的宿主机 PID：
先 TERM，等待 10 秒；仍占用且 PID 身份未变时 KILL。
首次为空、复查才出现进程时也会进入一次清理；同轮不会反复追杀新增进程。
启动前必须全部 8 台、每台 0～7 号 NPU 连续两次无进程才执行命令；间隔 2 秒。
启动前清理失败：不执行命令，退出 70 并解锁；结束后清理失败也解锁。
依赖本机 timeout，以及各节点的 Bash 4+、awk、timeout、npu-smi 和杀进程权限。
每轮批量清理默认最多 180 秒，可用 B04_CLEANUP_TIMEOUT 设置秒数。
批量脚本必须同步等完所有节点，并保留 [INFO] 向 IP 发送命令 和 [IP] 输出前缀。
节点列表以 batch_exec_b04.sh 的目标声明为准，必须恰好 8 台且逐台通过复查。
正常清理只打印开始/完成提示；异常时打印错误摘要并保留清理日志。
详细日志记录查询 PID、跳过原因及信号发送结果；发送成功不等于进程已退出。
复查失败/节点失联/清理时中断：解锁并移除本次 ticket；日志归档到 failed/<ticket>/。
任务成功但结束清理失败返回 70；否则保留原任务退出码（Ctrl+C 为 130）。
检查范围是 NPU 进程占用；不要求 HBM 为 0，也不等同于设备健康检查。
没有心跳/自动回收。kill -9 后的残留需先确认进程已结束，再人工清理。
旧版残留锁不会自动解除；先确认对应调度进程已退出，再人工清理。
EOF
}

release_guard() {
    if (( have_guard )); then
        rm -f -- "$base/counter.lock/owner" "$base/counter.lock/next"
        rmdir -- "$base/counter.lock" || true
        have_guard=0
    fi
}

print_cleanup_errors() {
    # 异常摘要也不回显整段远端脚本；原始批量输出保存在 cleanup-pre/post.log。
    [[ -r "$1" ]] || return 0
    awk '
        { sub(/\r$/, "") }
        /^(\[[^]]+\][[:space:]]+)?\[(COMMAND|HOST|START|END)\]/ { next }
        /^=+ \[成功\]/ || /^\[INFO\]/ { next }
        /^\[[^]]+\] B04_RELEASE_/ { next }
        /^\[[^]]+\] \[EXIT_CODE\] 0[[:space:]]*$/ { next }
        /^\[[^]]+\][[:space:]]*$/ || /^[[:space:]]*$/ { next }
        { print }
    ' "$1" | tail -n 40 >&2
}

clean_resources() {
    local remote_script remote_command
    local marker="B04_RELEASE_${ticket}_$$_${RANDOM}"
    local stage=$1 stage_text
    local report="$base/run.lock/cleanup-$1.log"
    local batch_rc=0
    cleanup_stage=$stage
    if [[ "$stage" == pre ]]; then stage_text=启动前; else stage_text=结束后; fi
    printf 'phase=%s_cleanup\ncheck_start=%s\n' "$stage" "$(timestamp)" \
        > "$base/run.lock/release_state" || return 1
    printf '%s\n' "$marker" > "$base/run.lock/cleanup.token" || return 1
    log "[清理] ${stage_text}：正在检查并清理 B04 64P 资源，请稍候…"
    # 整段命令由现有 batch_exec_b04.sh 分发；不另建 SSH/密码/节点调度逻辑。
    # 在每台宿主机本地查询、本地 kill，不能把某台机器的 PID 发给所有机器。
    remote_script=$(cat <<'B04_REMOTE'
set -u
set -o pipefail
marker=$1
state_dir=$2
owner_ticket=$3
die() { printf '[清理失败] %s\n' "$*" >&2; exit 1; }
# 批量脚本超时/被中断后，远端命令可能尚未退出；失去本轮锁就停止发信号。
assert_cleanup_owner() {
    [[ "$(cat "$state_dir/run.lock/ticket" 2>/dev/null)" == "$owner_ticket" &&
       "$(cat "$state_dir/run.lock/cleanup.token" 2>/dev/null)" == "$marker" ]] ||
        die '本轮清理已失去锁，停止操作进程'
}
assert_cleanup_owner
for tool in npu-smi timeout awk; do
    command -v "$tool" >/dev/null || die "缺少命令：$tool"
done

# 针对附件 Ascend950DT 表格。空表、缺卡、格式不识别都不能当作空闲。
# 只读 Process id 列；绝不能使用 Process id in container 列发送信号。
snapshot() {
    local raw
    raw=$(LC_ALL=C timeout -k 5s 15s npu-smi info 2>&1) || {
        printf '[查询失败] %s\n' "$raw" >&2; return 1;
    }
    printf '%s\n' "$raw" | awk '
    function trim(s) { gsub(/^[ \t]+|[ \t\r]+$/, "", s); return s }
    /^\|/ {
        n=split($0, c, "|")
        for (i=1; i<=n; i++) c[i]=trim(c[i])
        if (c[2]=="NPU ID" && c[3]=="Name") { inventory=1; next }
        if (c[2]=="NPU ID" && c[3]=="Process id") {
            if (!inventory || process) bad=1
            process=1; next
        }
        if (!process) {
            if (inventory && c[2] ~ /^[0-9]+$/) {
                if (c[2] !~ /^[0-7]$/ || cards[c[2]]++) bad=1
            }
            next
        }
        if (c[2] ~ /^No running processes found in NPU [0-7]$/ && n==3) {
            id=substr(c[2], length(c[2]), 1)
            if (busy[id] || idle[id]++) bad=1
            seen[id]=1; last=id; next
        }
        id=c[2]; if (id=="") id=last
        if (n<6 || id !~ /^[0-7]$/ || c[3] !~ /^[0-9]+$/ ||
            length(c[3])>10 || c[3]+0<=1 || c[4]=="" || idle[id]) {
            bad=1; next
        }
        seen[id]=1; busy[id]=1; last=id; pids[c[3]]=1
    }
    END {
        for (i=0; i<8; i++) if (!cards[i] || !seen[i]) bad=1
        if (bad || !process) {
            print "[解析失败] 无法完整确认 0～7 号 NPU 的进程表" > "/dev/stderr"
            exit 1
        }
        for (pid in pids) print pid
    }' || { printf '%s\n' "$raw" >&2; return 1; }
}

process_start() {
    local stat
    local -a fields
    IFS= read -r stat 2>/dev/null < "/proc/$1/stat" || return 1
    # comm 可以含空格/括号；最后一个右括号之后才是 state 等字段。
    read -r -a fields <<< "${stat##*) }"
    [[ "${fields[19]:-}" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "${fields[19]}"
}

log_snapshot() {
    if [[ -n "$2" ]]; then
        printf '[查询] %s：NPU 宿主机 PID=%s\n' "$1" "${2//$'\n'/ }"
    else
        printf '[查询] %s：8 张 NPU 均无进程\n' "$1"
    fi
}

# 一轮批量清理最多进入这里一次；复查新增进程不反复追杀。
clear_detected_pids() {
    local pids=$1 pid start current
    local -A starts=()
    printf '[清理候选] NPU 宿主机 PID：%s\n' "${pids//$'\n'/ }"
    for pid in $pids; do
        [[ "$pid" != "$$" && "$pid" != "$PPID" ]] || die '进程表包含清理程序自身'
        if ! start=$(process_start "$pid"); then
            printf '[跳过 TERM] PID=%s：无法读取 /proc/%s/stat 身份；可能已退出、无权限或 PID 命名空间不匹配\n' "$pid" "$pid"
            continue
        fi
        starts[$pid]=$start
        assert_cleanup_owner
        current=$(process_start "$pid") || {
            printf '[跳过 TERM] PID=%s：发信号前已无法读取身份\n' "$pid"; continue;
        }
        [[ "$current" == "$start" ]] || die "TERM 前 PID=$pid 身份变化，停止清理"
        printf '[TERM] PID=%s starttime=%s；准备发送\n' "$pid" "$start"
        if kill -TERM -- "$pid"; then
            printf '[TERM已发送] PID=%s；是否退出以 NPU 复查为准\n' "$pid"
        else
            [[ "$(process_start "$pid")" != "$start" ]] || die "无法 TERM PID=$pid"
            printf '[TERM未发送] PID=%s：发送失败后原身份已不可读或已变化\n' "$pid"
        fi
    done
    printf '[等待] TERM 阶段结束，10 秒后重新查询 NPU 进程\n'
    sleep 10
    assert_cleanup_owner
    pids=$(snapshot) || die 'TERM 后查询失败'
    log_snapshot 'TERM 后复查' "$pids"
    for pid in $pids; do
        if ! current=$(process_start "$pid"); then
            printf '[跳过 KILL] PID=%s：NPU 表仍列出，但无法读取 /proc/%s/stat 身份\n' "$pid" "$pid"
            continue
        fi
        [[ -n "${starts[$pid]:-}" && "${starts[$pid]}" == "$current" ]] ||
            die "出现新进程/身份变化：PID=$pid；本轮清理失败，需检查启动器"
        assert_cleanup_owner
        printf '[KILL] PID=%s starttime=%s；准备发送\n' "$pid" "$current"
        if kill -KILL -- "$pid"; then
            printf '[KILL已发送] PID=%s；是否退出以 NPU 复查为准\n' "$pid"
        else
            [[ "$(process_start "$pid")" != "$current" ]] || die "无法 KILL PID=$pid"
            printf '[KILL未发送] PID=%s：发送失败后原身份已不可读或已变化\n' "$pid"
        fi
    done
}

pids=$(snapshot) || die '无法查询 NPU，未发送任何信号'
log_snapshot '首次查询' "$pids"
cleanup_attempted=0
if [[ -n "$pids" ]]; then
    cleanup_attempted=1
    clear_detected_pids "$pids"
fi

# 清理后允许驱动短暂延迟；必须连续两次完整查询均无进程。
empty=0
attempt=0
while (( attempt < 6 )); do
    ((attempt += 1))
    assert_cleanup_owner
    pids=$(snapshot) || die '清理后的复查失败'
    log_snapshot "复查 $attempt/6" "$pids"
    if [[ -z "$pids" ]]; then
        ((empty += 1))
    else
        empty=0
        if (( cleanup_attempted == 0 )); then
            printf '[补充清理] 首次查询为空，复查发现占用；现在执行本轮唯一一次 TERM/KILL 清理\n'
            cleanup_attempted=1
            clear_detected_pids "$pids"
            # 首次实际清理后仍给完整复查窗口；此重置整轮最多发生一次。
            attempt=0
        fi
    fi
    if (( empty >= 2 )); then
        assert_cleanup_owner
        printf '%s OK cards=8\n' "$marker"
        exit 0
    fi
    sleep 2
done
[[ -z "$pids" ]] || die "复查仍有 NPU 占用：${pids//$'\n'/ }"
die '复查未连续两次确认空闲；最后一次虽为空，但本轮不能判定通过'
B04_REMOTE
)
    printf -v remote_command 'bash -c %q b04-resource-cleanup %q %q %q' \
        "$remote_script" "$marker" "$base" "$ticket"
    # 批量脚本的普通输出只写日志；stderr 保留密码提示和即时错误。
    # stdin 仍来自调用者的终端，不影响 read -p / SSH 登录交互。
    timeout --foreground -k 5s "${cleanup_timeout}s" \
        bash -- "$batch_exec" "$remote_command" > "$report" || batch_rc=$?
    if (( batch_rc != 0 )); then
        log "[清理失败] 批量脚本退出码=$batch_rc" >&2
        print_cleanup_errors "$report"
        return 1
    fi
    # 以批量脚本声明的目标为准；不能只数成功节点，否则漏执行也可能被放行。
    # 八个目标必须各声明一次、各通过一次；缺失、重复、额外结果均视为清理失败。
    if ! awk -v marker="$marker" -v expected="$expected_nodes" '
        { sub(/\r$/, "") }
        $1=="[INFO]" && $2=="向" && $4=="发送命令" {
            if ($3 !~ /^[A-Za-z0-9_.:-]+$/) bad=1
            nodes[$3]++; next
        }
        $2==marker && $3=="OK" && $4=="cards=8" && NF==4 {
            node=$1
            if (node !~ /^\[[^]]+\]$/) { bad=1; next }
            sub(/^\[/, "", node); sub(/\]$/, "", node)
            passed[node]++
        }
        END {
            for (node in nodes) {
                count++
                if (nodes[node]!=1 || passed[node]!=1) {
                    printf "[检查失败] %s 的目标声明或通过结果缺失/重复\n", node
                    bad=1
                }
            }
            if (count!=expected) {
                printf "[检查失败] 应声明 %d 个目标节点，实际为 %d；请检查批量脚本输出\n", expected, count
                bad=1
            }
            for (node in passed) if (!(node in nodes)) {
                printf "[检查失败] 出现未声明节点的通过结果：%s\n", node
                bad=1
            }
            exit bad
        }
    ' "$report" >&2; then
        print_cleanup_errors "$report"
        return 1
    fi
    log "[确认] ${stage_text}清理完成，8 台机器、64 张 NPU 均无进程占用"
}

finish_exit() {
    local exit_code=$1 archive_path=''
    trap - EXIT
    # 文件收尾很短，避免第二次 Ctrl+C 打断删除；kill -9 仍无法捕获。
    trap '' INT TERM HUP
    if (( have_run )); then
        if [[ "$(cat "$base/run.lock/ticket" 2>/dev/null)" != "$ticket" ]]; then
            log '[错误] 运行锁归属已变化，停止解锁；未删除其他任务的锁' >&2
            (( exit_code != 0 )) || exit_code=70
            exit "$exit_code"
        fi
        printf 'wrapper_exit_code=%s\nfinished=%s\n' "$exit_code" "$(timestamp)" \
            >> "$base/run.lock/owner" || true
        # 先废止远端清理令牌，再释放运行锁，旧清理不得继续发信号。
        rm -f -- "$base/run.lock/cleanup.token" || {
            log '[错误] 无法废止清理令牌，请检查共享目录权限' >&2
            exit 70
        }
        if (( archive_needed || exit_code != 0 )); then
            archive_path="$base/failed/$ticket"
            if mkdir -p -- "$base/failed" &&
               [[ ! -e "$archive_path" ]] &&
               mv -T -- "$base/run.lock" "$archive_path"; then
                log "[检查日志] $archive_path/"
            else
                archive_path=''
                log '[错误] 日志归档失败，将尝试直接解锁；请保存终端错误信息' >&2
                (( exit_code != 0 )) || exit_code=70
            fi
        fi
        if [[ -z "$archive_path" ]]; then
            rm -f -- "$base/run.lock/owner" "$base/run.lock/ticket" \
                "$base/run.lock/release_state" "$base/run.lock/cleanup-pre.log" \
                "$base/run.lock/cleanup-post.log"
            if ! rmdir -- "$base/run.lock"; then
                log "[错误] 无法删除运行锁，请检查 $base/run.lock" >&2
                (( exit_code != 0 )) || exit_code=70
                exit "$exit_code"
            fi
        fi
        have_run=0
        log "[释放] 已释放 B04 64P 调度锁；ticket=$ticket；下一任务须先通过清理"
    fi
    if [[ -n "$entry" ]]; then
        if ! { rm -f -- "$entry/owner" && rmdir -- "$entry"; }; then
            log "[错误] 无法删除 ticket，请检查 $entry" >&2
            (( exit_code != 0 )) || exit_code=70
        fi
    fi
    release_guard
    exit "$exit_code"
}

interrupt_cleanup() {
    archive_needed=1
    if (( have_run )) && [[ "$(cat "$base/run.lock/ticket" 2>/dev/null)" == "$ticket" ]]; then
        printf 'phase=%s_cleanup_interrupted\ncheck_end=%s\n' "$cleanup_stage" "$(timestamp)" \
            > "$base/run.lock/release_state"
    fi
    log '[中断] 清理被中断；本次解锁，下一任务将重新清理' >&2
    finish_exit "$1"
}

cleanup() {
    local exit_code=$?
    trap - EXIT
    trap 'interrupt_cleanup 130' INT
    trap 'interrupt_cleanup 143' TERM
    trap 'interrupt_cleanup 129' HUP
    if (( have_run && command_started )); then
        if [[ "$(cat "$base/run.lock/ticket" 2>/dev/null)" != "$ticket" ]]; then
            log '[错误] 运行锁归属已变化，跳过清理' >&2
            finish_exit 70
        fi
        if ! clean_resources post; then
            archive_needed=1
            printf 'phase=post_cleanup_failed\ncheck_end=%s\n' "$(timestamp)" \
                > "$base/run.lock/release_state"
            log '[清理失败] 结束清理未通过，仍释放调度锁；下一任务必须重新清理' >&2
            (( exit_code != 0 )) || exit_code=70
        fi
    fi
    finish_exit "$exit_code"
}

status() {
    local path running='' position=0
    log '[状态] B04 64P（整组 64 张卡的共享使用权）'
    if [[ -d "$base/run.lock" ]]; then
        [[ ! -f "$base/run.lock/ticket" ]] || running=$(cat "$base/run.lock/ticket")
        printf '当前运行：ticket=%s\n' "${running:-正在写入}"
        cat -- "$base/run.lock/owner" 2>/dev/null || printf '  owner 尚未写入\n'
        if [[ -f "$base/run.lock/release_state" ]]; then
            printf '当前阶段（持有调度锁）：\n'
            cat -- "$base/run.lock/release_state"
        fi
    else
        printf '调度锁：空闲；实际 NPU 是否空闲将在下一任务启动前检查\n'
    fi
    printf '\n等待队列：\n'
    for path in "$base/queue"/[0-9]*; do
        [[ -d "$path" && "${path##*/}" != "$running" ]] || continue
        ((position += 1))
        printf '[%s] ticket=%s\n' "$position" "${path##*/}"
        cat -- "$path/owner" 2>/dev/null || printf '  owner 尚未写入\n'
    done
    (( position > 0 )) || printf '  （空）\n'
    printf '\n等待任务：%s/%s（不含运行中）\n' "$position" "$max_waiting"
    if [[ -d "$base/counter.lock" ]]; then
        printf '\n发号锁存在；若长期不消失，需检查其 owner：\n'
        cat -- "$base/counter.lock/owner" 2>/dev/null || true
    fi
}

case "${0##*/}" in
    run_b04) mode=run ;;
    b04_status) mode=status ;;
    *) mode="${1:-help}"; (( $# == 0 )) || shift ;;
esac
case "$mode" in
    status) (( $# == 0 )) || fail 'status 不接受额外参数'; status; exit 0 ;;
    help|-h|--help) help_text; exit 0 ;;
    init|run) ;;
    *) fail '仅支持 run 和 status，使用 --help 查看用法' ;;
esac

mkdir -p -- "$base/queue" || fail '无法创建共享目录，请检查挂载与团队写权限'
base=$(cd -- "$base" && pwd -P) || exit 1
if [[ "$mode" == init ]]; then
    printf '已就绪：%s\n' "$base"; exit 0
fi
[[ "${1:-}" != -- ]] || shift
(( $# > 0 )) || fail 'run 后需要命令'
# 防止只有 ENV=value 时 env 打印整个环境。
command_found=0
for arg in "$@"; do
    if [[ ! "$arg" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then command_found=1; break; fi
done
(( command_found )) || fail '环境变量后还需要命令'
[[ "${B04_ACTIVE_STATE_DIR:-}" != "$base" ]] || fail '同一任务链只在最外层调用一次 run_b04'
[[ -f "$batch_exec" && -r "$batch_exec" ]] || fail "无法读取资源清理用的批量脚本：$batch_exec"
command -v timeout >/dev/null || fail '缺少 timeout 命令'
[[ "$cleanup_timeout" =~ ^[1-9][0-9]*$ ]] || fail '清理超时秒数必须是正整数'

submit_seconds=$SECONDS
submitted_at=$(timestamp)
user_name=$(id -un)
host_name=$(hostname)
printf -v command_text '%q ' "$@"
command_text=${command_text% }
trap cleanup EXIT
trap 'log "[取消] 收到 Ctrl+C / SIGINT；exit_code=130"; exit 130' INT
trap 'log "[终止] 收到 SIGTERM；exit_code=143"; exit 143' TERM
trap 'log "[终止] 收到 SIGHUP；exit_code=129"; exit 129' HUP
log "[提交] 申请 B04 64P（整组 64 张卡）使用权；user=$user_name host=$host_name pid=$$"
log "[命令] $command_text"

# 发号临界区只有文件操作。异常残留不自动打破，30 秒后提示处理。
attempt=0
until mkdir -- "$base/counter.lock" 2>/dev/null; do
    ((attempt += 1))
    (( attempt <= 30 )) || fail '无法获得发号锁，请用 status 检查占用和共享目录写权限'
    (( attempt != 1 )) || log '[取号] 发号锁暂时占用，正在等待分配 ticket'
    sleep 1
done
have_guard=1
printf 'user=%s\nhost=%s\npid=%s\n' "$user_name" "$host_name" "$$" \
    > "$base/counter.lock/owner" || fail '写入发号锁 owner 失败'

# 检查容量与创建 ticket 共用发号锁，防止并发提交同时占用最后一个名额。
# 先读取运行者，再统计队列；期间任务退出/开始只会减少等待人数。
running=$(cat "$base/run.lock/ticket" 2>/dev/null || true)
waiting_count=0
for path in "$base/queue"/[0-9]*; do
    [[ -d "$path" && "${path##*/}" != "$running" ]] || continue
    ((waiting_count += 1))
done
if (( waiting_count >= max_waiting )); then
    log "[拒绝] 等待队列已满：${waiting_count}/${max_waiting}（不含运行中）；本次提交未入队；exit_code=75" >&2
    exit 75
fi

counter=0
if [[ -e "$base/counter" ]]; then
    counter=$(cat "$base/counter") || fail '读取 counter 失败'
fi
[[ "$counter" =~ ^(0|[1-9][0-9]{0,17})$ ]] || fail 'counter 内容异常，不能重置运行中的计数器'
next=$((counter + 1))
printf '%s\n' "$next" > "$base/counter.lock/next" || fail '更新 counter 失败'
mv -- "$base/counter.lock/next" "$base/counter" || fail '保存 counter 失败'
printf -v ticket '%020d' "$next"
mkdir -- "$base/queue/$ticket" || fail '创建 ticket 失败'
entry="$base/queue/$ticket"
{
    printf 'user=%s\nhost=%s\npid=%s\nsubmit=%s\ncwd=%q\ncommand=' \
        "$user_name" "$host_name" "$$" "$submitted_at" "$PWD" &&
    printf '%q ' "$@" && printf '\n'
} > "$entry/owner" || fail '写入 ticket owner 失败'
release_guard

log "[取号] ticket=$ticket"
last_wait_state=''
last_report=$SECONDS
while :; do
    [[ -d "$entry" ]] || fail '自己的 ticket 已被删除，请重新提交'
    entries=("$base/queue"/[0-9]*)
    if [[ "${entries[0]:-}" == "$entry" ]]; then
        if mkdir -- "$base/run.lock" 2>/dev/null; then
            have_run=1
            printf '%s\n' "$ticket" > "$base/run.lock/ticket" || fail '写运行锁失败'
            cp -- "$entry/owner" "$base/run.lock/owner" || fail '写运行 owner 失败'
            acquired_seconds=$SECONDS
            break
        fi
    fi
    # 只有尚未获得运行锁才打印排队；按快照计算位置，排除运行者。
    running=$(cat "$base/run.lock/ticket" 2>/dev/null || true)
    ahead=0
    for path in "${entries[@]}"; do
        [[ "$path" != "$entry" ]] || break
        if [[ -d "$path" && "${path##*/}" != "$running" ]]; then ((ahead += 1)); fi
    done
    wait_state="$ahead:$running"
    if [[ "$wait_state" != "$last_wait_state" ]] || (( SECONDS - last_report >= 30 )); then
        log "[排队] 等待获得 B04 64P 使用权；位置=$((ahead + 1))（不含运行中）；前方排队=$ahead；已等待=$((SECONDS - submit_seconds))s；Ctrl+C 可取消"
        if [[ -r "$base/run.lock/owner" ]]; then
            log "[占用者] ticket=${running:-正在写入}"
            sed -n '/^\(user\|host\|pid\|start\|command\)=/s/^/  /p' "$base/run.lock/owner" 2>/dev/null || true
        elif [[ -d "$base/run.lock" ]]; then
            log '[占用] 运行锁存在，owner 尚未写入'
        else
            log '[队列] 尚未取得运行锁，等待前方 ticket 启动或取消后重试'
        fi
        last_wait_state=$wait_state
        last_report=$SECONDS
    fi
    sleep 2
done

# 始终持锁完成启动前清理；失败只取消本任务，不能带着占用运行新命令。
log "[获得锁] ticket=$ticket；先清理资源，通过后才执行命令"
if ! clean_resources pre; then
    archive_needed=1
    printf 'phase=pre_cleanup_failed\ncheck_end=%s\n' "$(timestamp)" \
        > "$base/run.lock/release_state"
    log '[未启动] 启动前清理未通过；本任务退出并解锁，exit_code=70' >&2
    exit 70
fi
printf 'phase=running\n' > "$base/run.lock/release_state" || fail '写运行状态失败'

# 保持命令在前台；终端 Ctrl+C 同时送给 wrapper 和前台任务。
# 不 eval，不拼接命令字符串，保留 stdin、环境变量和参数边界。
started_at=$(timestamp)
start_seconds=$SECONDS
wait_seconds=$((acquired_seconds - submit_seconds))
preclean_seconds=$((start_seconds - acquired_seconds))
printf 'start=%s\nwait_seconds=%s\npreclean_seconds=%s\n' "$started_at" "$wait_seconds" "$preclean_seconds" \
    >> "$base/run.lock/owner" || fail '写开始时间失败'
printf '[%s] [开始] 已获得 B04 64P 使用权，开始执行命令；ticket=%s；等待=%ss；启动前清理=%ss\n' \
    "$started_at" "$ticket" "$wait_seconds" "$preclean_seconds"
rc=0
command_started=1
B04_ACTIVE_STATE_DIR="$base" env -- "$@" || rc=$?
run_seconds=$((SECONDS - start_seconds))
log "[命令结束] ticket=$ticket；exit_code=$rc；运行=${run_seconds}s；接下来尝试结束清理并释放调度锁"
exit "$rc"

