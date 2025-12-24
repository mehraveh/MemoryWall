import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


LOG_PATH = "memorywall.logs.sorted_by_happenedAt.json"
OUT_DIR = Path("viz_out_levels")
OUT_DIR.mkdir(exist_ok=True)

ONLY_LEVELS = None  # مثلا: {13,16} یا None


def safe_json_load(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            return {}
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

def is_hover_enter(d): return norm_status(d) == "hover entered"
def is_hover_exit(d):  return norm_status(d) == "hover exit"
def is_selected(d):    return norm_status(d) == "selected"

def get_is_correct(d):
    v = d.get("is_correct", None)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        vv = v.strip().lower()
        if vv == "true": return True
        if vv == "false": return False
    return None


logs = json.loads(Path(LOG_PATH).read_text(encoding="utf-8"))

events = []
for rec in logs:
    t = pd.to_datetime(rec.get("happened_at"), errors="coerce")
    if pd.isna(t):
        continue

    ca = pd.to_datetime(rec.get("created_at"), errors="coerce")
    d = safe_json_load(rec.get("data", {}))
    if not d or is_pattern(d):
        continue

    lvl = get_level(d)
    pos = get_xy(d)
    if lvl is None or pos is None:
        continue

    lvl_norm = int(lvl) if str(lvl).isdigit() else lvl
    if ONLY_LEVELS is not None and lvl_norm not in ONLY_LEVELS:
        continue

    events.append({
        "time": t,
        "created_at": ca,
        "level": lvl_norm,
        "x": int(pos[0]),
        "y": int(pos[1]),
        "status": norm_status(d),
        "is_correct": get_is_correct(d)
    })

df = pd.DataFrame(events)
if df.empty:
    raise RuntimeError("No usable behavioral events found.")

df = df.sort_values(["level", "time", "created_at"]).reset_index(drop=True)


def make_grid(df_level):
    return int(df_level["x"].max()) + 1, int(df_level["y"].max()) + 1

def plot_heatmap(mat, title, outpath):
    plt.figure()
    plt.imshow(mat, origin="lower", aspect="equal")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("y")
    plt.ylabel("x")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


summary_rows = []

for lvl, g in df.groupby("level", sort=True):
    g = g.sort_values(["time", "created_at"]).reset_index(drop=True)
    gx, gy = make_grid(g)

    selected_total   = np.zeros((gx, gy), dtype=int)
    selected_correct = np.zeros((gx, gy), dtype=int)
    selected_wrong   = np.zeros((gx, gy), dtype=int)
    selected_unknown = np.zeros((gx, gy), dtype=int)
    hover_only_pairs = np.zeros((gx, gy), dtype=int)

    # scatter data
    hover_only_before_select = []
    select_correct_flag = []

    hover_state = defaultdict(lambda: {
        "in_hover": False,
        "selected_during": False,
        "hover_only_count": 0
    })

    last_ic_by_tile = {}

    for _, r in g.iterrows():
        x, y = r["x"], r["y"]
        st = r["status"]
        key = (x, y)

  
        if st in ("hover entered", "hover exit") and r["is_correct"] is not None:
            last_ic_by_tile[key] = r["is_correct"]

        if st == "hover entered":
            hover_state[key]["in_hover"] = True
            hover_state[key]["selected_during"] = False

        elif st == "hover exit":
            if hover_state[key]["in_hover"]:
                if not hover_state[key]["selected_during"]:
                    hover_only_pairs[x, y] += 1
                    hover_state[key]["hover_only_count"] += 1
                hover_state[key]["in_hover"] = False
                hover_state[key]["selected_during"] = False

        elif st == "selected":
            selected_total[x, y] += 1

            ic = last_ic_by_tile.get(key, None)
            if ic is True:
                selected_correct[x, y] += 1
            elif ic is False:
                selected_wrong[x, y] += 1
            else:
                selected_unknown[x, y] += 1

            # scatter
            hover_only_before_select.append(hover_state[key]["hover_only_count"])
            select_correct_flag.append(ic)

            hover_state[key]["hover_only_count"] = 0
            hover_state[key]["selected_during"] = True


    lvl_dir = OUT_DIR / f"level_{lvl}"
    lvl_dir.mkdir(exist_ok=True)

    plot_heatmap(hover_only_pairs, f"Level {lvl} | Hover-only", lvl_dir / "heat_hover_only.png")
    plot_heatmap(selected_total,   f"Level {lvl} | Selected total", lvl_dir / "heat_selected_total.png")
    plot_heatmap(selected_correct, f"Level {lvl} | Selected correct", lvl_dir / "heat_selected_correct.png")
    plot_heatmap(selected_wrong,   f"Level {lvl} | Selected wrong", lvl_dir / "heat_selected_wrong.png")
    plot_heatmap(selected_unknown, f"Level {lvl} | Selected unknown", lvl_dir / "heat_selected_unknown.png")

    print(f"Level {lvl} | hover_only_before_select count =", len(hover_only_before_select))
    if hover_only_before_select:
        hc = np.array(hover_only_before_select)
        ic = np.array(select_correct_flag, dtype=object)
        y_idx = np.arange(len(hc))

        plt.figure()
        plt.scatter(hc[ic == True],  y_idx[ic == True],  c="green", label="Correct", alpha=0.7)
        plt.scatter(hc[ic == False], y_idx[ic == False], c="red",   label="Wrong", alpha=0.7)
        plt.scatter(hc[ic == None],  y_idx[ic == None],  c="gray",  label="Unknown", alpha=0.5)
        plt.xlabel("Hover-only count before select")
        plt.ylabel("Selection index")
        plt.title(f"Level {lvl} | Decision hesitation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(lvl_dir / "scatter_hover_only_before_select.png", dpi=180)
        plt.close()

    summary_rows.append({
        "level": lvl,
        "sum_hover_only": int(hover_only_pairs.sum()),
        "sum_selected_total": int(selected_total.sum()),
        "sum_selected_correct": int(selected_correct.sum()),
        "sum_selected_wrong": int(selected_wrong.sum()),
        "sum_selected_unknown": int(selected_unknown.sum()),
        "n_scatter_points": len(hover_only_before_select)
    })

summary = pd.DataFrame(summary_rows).sort_values("level")
summary.to_csv(OUT_DIR / "levels_summary.csv", index=False)

print("✅ Done. Outputs saved in:", OUT_DIR.resolve())
print(summary)