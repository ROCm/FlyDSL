# set -x

N="${1:-1}"
# if ! [[ "${N}" =~ ^[1-9][0-9]*$ ]]; then
#   echo "Usage: $0 [positive_repeat_count]" >&2
#   exit 2
# fi

ANA_LOG="at2.log"
: > "${ANA_LOG}"

append_interval_analysis() {
python3 - <<'PY' >> "${ANA_LOG}"
import re
from pathlib import Path

path = Path("temp.log")

summary_section = "=== All Sample Interval Cycles Summary Compare ==="
interval_section = "=== Sample Interval Summary Compare ==="
target = "d_avg_cycles_pct"
threshold = 2.0

lines = path.read_text(errors="replace").splitlines()


def print_interval_table(table_lines):
    for line in table_lines:
        print(line)


def print_filtered_interval_table(table_lines):
    header = next((line for line in table_lines[1:] if target in line), None)
    if header is None:
        return

    print()
    print(f"{interval_section} ({target} >= {threshold:g}%)")

    cols = [(m.group(), m.start()) for m in re.finditer(r"\S+", header)]
    starts = dict(cols)
    ends = {
        name: (cols[i + 1][1] if i + 1 < len(cols) else None)
        for i, (name, _) in enumerate(cols)
    }

    print(header)

    header_idx = table_lines.index(header)
    for line in table_lines[header_idx + 1 :]:
        val = line[starts[target] : ends[target]].strip().rstrip("%")
        try:
            if float(val) >= threshold:
                print(line)
        except ValueError:
            pass


i = 0
while i < len(lines):
    line = lines[i]

    if line == summary_section:
        print(line)
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith("==="):
            print(lines[i])
            i += 1
        continue

    if line == interval_section:
        interval_table = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith("==="):
            interval_table.append(lines[i])
            i += 1
        print_interval_table(interval_table)
        print_filtered_interval_table(interval_table)
        continue

    i += 1
PY
}

print_selected_summary() {
  local sample_header=" idx description                          cpp_attn fyd_attn  d_count cpp_attn_avg fyd_attn_avg  d_avg_cycles  d_avg_cycles_pct base_sum_cycles base_sum_cycles_pct other_sum_cycles other_sum_cycles_pct cpp_attn_avg_ins fyd_attn_avg_ins        d_avg_inst   d_avg_inst_pct"
  local summary_header="baseline         other             base_sum_cycles  other_sum_cycles  delta_sum_cycles  delta_sum_cycles_pct  base_sum_inst  other_sum_inst  delta_sum_inst  delta_sum_inst_pct"
  local pattern
  local patterns=(
    "Prologue"
    "Main loop Cluster 0"
    "Main loop Cluster 1"
    "Main loop Cluster 2"
    "Main loop Cluster 3"
    "Main loop Cluster 4"
    "Main loop Cluster 5"
    "Main loop Cluster 6"
    "Main loop Cluster 7"
    "Main loop to Epilogue"
    "Epilogue Cluster 0"
    "Epilogue Cluster 1 "
    "Epilogue Cluster 2"
    "Epilogue Cluster 3"
    "Epilogue Cluster 4"
    "Epilogue Cluster 5"
    "Epilogue Cluster 6"
    "Epilogue Cluster 7"
    "Epilogue Cluster 8"
    "Epilogue Cluster 9"
    "Epilogue Cluster 10"
    "Epilogue Cluster 11"
    "Epilogue Cluster 12"
    "Epilogue Cluster 13"
    "Epilogue output store"
  )

  for pattern in "${patterns[@]}"; do
    echo "${sample_header}"
    grep -F "${pattern}" "${ANA_LOG}"
  done

  echo "${summary_header}"
  grep -F "cpp_attn         fyd_attn" "${ANA_LOG}"
}


for ((iter = 1; iter <= N; iter++)); do
echo "=== ana_trace iteration ${iter}/${N} ==="
echo "=== ana_trace iteration ${iter}/${N} ===" >> "${ANA_LOG}"

./scripts/dump_opus_attn_thread_trace.sh ./input_opus_attn_thread_trace.yaml ./thread_trace/fyd_opus_b2_s1024.tar.gz
# ./scripts/dump_opus_attn_thread_trace.sh ./input_hand_asm_thread_trace.yaml ./thread_trace/fyd_opus_b2_s1024.tar.gz
# ./scripts/dump_opus_attn_thread_trace.sh ./input_opus_gqa_d128_thread_trace.yaml ./thread_trace/cpp_opus_b2_s1024.tar.gz

python3 scripts/trace_segment_cycles.py seg_asm/fyd_cpp_compare.json > temp.log
append_interval_analysis

done

print_selected_summary


# ./exp_isa/build.sh
# ./scripts/dump_opus_attn_thread_trace.sh ./input_hand_asm_thread_trace.yaml ./thread_trace/fyd_opus_b2_s1024.tar.gz
# python3 scripts/trace_segment_cycles.py --specific-part seg_asm/specific_part.json | tee test3.log

# set +x
