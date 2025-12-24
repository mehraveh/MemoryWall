import json
import pandas as pd
from pathlib import Path

LOG_PATH = "memorywall.logs.sorted_by_happenedAt.json"
GAP_THRESHOLD_SEC = 120
LONG_BLOCK_SEC = 70
EPS = 1e-9

OUT_BLOCKS_CSV  = "memorywall_blocks_times"
OUT_METRICS_CSV = "memorywall_session_metrics.csv"

def safe_json_load(x):
    if isinstance(x, dict): return x
    if isinstance(x, str):
        try: return json.loads(x)
        except: return {}
    return {}

def norm_status(d):
    s = d.get("status") or d.get("event") or d.get("type") or d.get("action") or d.get("name")
    return str(s).strip().lower() if s is not None else ""

def is_pattern(d):
    p = d.get("pattern")
    return isinstance(p, list) and len(p) > 0

def get_level(d):
    return d.get("level", d.get("Level", None))

def get_xy(d):
    x = d.get("x"); y = d.get("y")
    if isinstance(x, int) and isinstance(y, int):
        return (x, y)
    r = d.get("row"); c = d.get("col")
    if isinstance(r, int) and isinstance(c, int):
        return (r, c)
    return None

def is_selected(d):
    return norm_status(d) == "selected"

def is_hover_enter(d):
    return norm_status(d) == "hover entered"

def is_hover_exit(d):
    return norm_status(d) == "hover exit"

def get_is_correct_underline(d):
    v = d.get("is_correct", None)

    if isinstance(v, bool):
        return v

    if isinstance(v, str):
        v = v.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False

    return None

def get_response_time(d):
    v = d.get("response time", None)
    if isinstance(v, (int, float)):
        return float(v)
    return None

logs = json.loads(Path(LOG_PATH).read_text(encoding="utf-8"))

rows = []
for rec in logs:
    t = pd.to_datetime(rec.get("happened_at"), errors="coerce")
    if pd.isna(t):
        continue
    d = safe_json_load(rec.get("data", {}))
    rows.append({
        "time": t,
        "d": d,
        "level": get_level(d),
        "is_pattern": is_pattern(d),
    })

df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


df_beh = df[df["is_pattern"] == False].copy().reset_index(drop=True)

df_beh["gap_sec"] = df_beh["time"].diff().dt.total_seconds()
new_block = (df_beh["gap_sec"].fillna(0) > GAP_THRESHOLD_SEC) | (df_beh["level"] != df_beh["level"].shift())
df_beh["block_id"] = new_block.cumsum()

blocks = df_beh.groupby("block_id").agg(
    start_time=("time", "min"),
    end_time=("time", "max"),
    level=("level", lambda x: x.mode().iloc[0] if x.notna().any() else None),
    behavior_events=("time", "count"),
).reset_index()

blocks["duration_sec"] = (blocks["end_time"] - blocks["start_time"]).dt.total_seconds()
blocks["is_long_block"] = blocks["duration_sec"] >= LONG_BLOCK_SEC
blocks.to_csv(OUT_BLOCKS_CSV, index=False)


df_pat = df[df["is_pattern"] == True].copy().sort_values("time").reset_index(drop=True)


metrics = []

for bid, g in df_beh.groupby("block_id", sort=True):
    g = g.sort_values("time").reset_index(drop=True)
    binfo = blocks.loc[blocks["block_id"] == bid].iloc[0]
    dur_min = max(float(binfo["duration_sec"]) / 60.0, EPS)

    selected_count = 0
    hover_only_pairs = 0

    is_correct_true = 0
    is_correct_false = 0
    is_correct_missing = 0

    hover_state = {}  
    rt_sum = 0.0
    rt_count = 0
    pattern_in_block = df_pat[(df_pat["time"] >= binfo["start_time"]) & (df_pat["time"] <= binfo["end_time"])].copy()
    pattern_count_in_block = int(len(pattern_in_block))


    anchor = df_pat[df_pat["time"] < binfo["start_time"]].tail(1)
    pat_times = []
    if not anchor.empty:
        pat_times.append(anchor.iloc[0]["time"])
    # patternهای داخل بلاک
    pat_times += list(pattern_in_block["time"].sort_values())

    # segments between patterns
    segments_between_patterns = max(len(pat_times) - 1, 0)
    segments_with_any_ic = 0
    segments_all_true = 0


    if segments_between_patterns > 0:
        for i in range(len(pat_times) - 1):
            t0 = pat_times[i]
            t1 = pat_times[i + 1]

            seg_events = g[(g["time"] > t0) & (g["time"] < t1)]

            ic_vals = []
            for _, rr in seg_events.iterrows():
                ic = get_is_correct_underline(rr["d"])
                if ic is True:
                    ic_vals.append(True)
                elif ic is False:
                    ic_vals.append(False)

            if len(ic_vals) > 0:
                segments_with_any_ic += 1
                true_count = sum(1 for v in ic_vals if v is True)
                total_count = len(ic_vals)


                if total_count > 0:
                    ratio = true_count / total_count

                    if lvl <= 2:
                        threshold = 0.7         # 100%
                    elif lvl <= 7:
                        threshold = 0.5          # 50%
                    else:
                        threshold = 0.33         # 33%

                    if ratio >= threshold:
                        segments_all_true += 1
    for _, row in g.iterrows():
        d = row["d"]

        ic_any = get_is_correct_underline(d)
        if ic_any is True:
            is_correct_true += 1
        elif ic_any is False:
            is_correct_false += 1
        else:
            is_correct_missing += 1

        lvl = get_level(d)
        pos = get_xy(d)
        if pos is None:
            continue
        key = (lvl, pos[0], pos[1])

        if is_hover_enter(d):
            hover_state[key] = {"in_hover": True, "selected_during": False}

        elif is_selected(d):
            selected_count += 1

            rt = get_response_time(d)
            if rt is not None:
                rt_sum += rt
                rt_count += 1

            st = hover_state.get(key)
            if st and st.get("in_hover"):
                st["selected_during"] = True
                hover_state[key] = st

        elif is_hover_exit(d):
            st = hover_state.get(key)
            if st and st.get("in_hover"):
                if st.get("selected_during") is False:
                    hover_only_pairs += 1
                hover_state[key] = {"in_hover": False, "selected_during": False}

    denom_tf = max(is_correct_true + is_correct_false, EPS)

    accuracy = is_correct_true / denom_tf
    error_rate = is_correct_false / denom_tf
    net_score = is_correct_true - is_correct_false
    cognitive_load = hover_only_pairs / dur_min  
    mean_rt = rt_sum / max(rt_count, 1)
    mean_hover_before_select = hover_only_pairs / max(selected_count, 1)
    CMLI = ((segments_all_true / max(segments_between_patterns, 1)) *
    ((10*is_correct_true - 5*is_correct_false) / max(is_correct_true + is_correct_false, 1))) / (1 + hover_only_pairs)
    # NEW: segment rate
    segments_all_true_rate = segments_all_true / max(segments_with_any_ic, 1)

    metrics.append({
        "block_id": int(bid),
        "level": binfo["level"],
        "start_time": binfo["start_time"],
        "end_time": binfo["end_time"],
        "duration_sec": float(binfo["duration_sec"]),
        "behavior_events_all": int(binfo["behavior_events"]),

        # counts
        "events_selected_only": int(selected_count),
        "hover_only_pairs": int(hover_only_pairs),

        # is_correct distribution
        "is_correct_true": int(is_correct_true),
        "is_correct_false": int(is_correct_false),
        "is_correct_missing": int(is_correct_missing),

        # per-minute
        "selected_per_min": float(selected_count / dur_min),
        "hover_only_per_min": float(hover_only_pairs / dur_min),

        # 4 formulas
        "accuracy": float(accuracy) * 100,
        "error_rate": float(error_rate) * 100,
        "net_score": float(net_score),
        "cognitive_load": float(cognitive_load),

        "pattern_count_in_block": int(pattern_count_in_block)+1,
        "segments_between_patterns": int(segments_between_patterns),
        "segments_with_any_ic": int(segments_with_any_ic),
        "segments_half_true": int(segments_all_true),

        ####
        "CMLI": float(CMLI)*100,
        "mean_response_time": float(mean_rt),
        "mean_hover_before_select": float(mean_hover_before_select),
    })

metrics_df = pd.DataFrame(metrics).sort_values("block_id").reset_index(drop=True)
metrics_df.to_csv(OUT_METRICS_CSV, index=False)

print("✅ Saved blocks:", OUT_BLOCKS_CSV)
print("✅ Saved metrics:", OUT_METRICS_CSV)
print(metrics_df.head(15))