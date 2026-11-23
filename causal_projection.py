# Causal projection figure: does text generated under intervention "read like refusal" to the unmodified model?
#
# For every (prompt, response) pair we feed prompt + response to the UNMODIFIED model, take the
# residual stream at the refusal-direction layer L (hidden_states[L]) over the response tokens,
# average it, and project onto that model's unit refusal direction.
#   real Level 0 / Level 4 : baseline responses (responses_<tag>_v2.csv, cut to 100 tokens)
#   ablated Level 4        : ablation_<tag>.csv
#   steered Level 0        : actadd_<tag>_L<L>.csv, frac == 1.0 (alpha = 100% of the L0-L4 distance)
# x is normalized per model: 0 = mean of real Level 0, 1 = mean of real Level 4.
#
# usage (from project root):
#   python causal_projection.py qwen       # writes resuts-qwen-new/causal_projection_qwen_L15.csv
#   python causal_projection.py mistral    # writes results-mistral-new/causal_projection_mistral_L18.csv
#   python causal_projection.py plot       # writes figures/causal_projection_qwen_mistral.png
import os, sys, json
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
MAXTOK = 100
CFG = {
    "qwen":    {"model_id": "Qwen/Qwen2.5-7B-Instruct",           "folder": "resuts-qwen-new",     "L": 15, "label": "Qwen2.5-7B-Instruct"},
    "mistral": {"model_id": "mistralai/Mistral-7B-Instruct-v0.3", "folder": "results-mistral-new", "L": 18, "label": "Mistral-7B-Instruct-v0.3"},
}
GROUPS = ["real_L0", "real_L4", "ablated_L4", "steered_L0"]


def project(tag):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    c = CFG[tag]
    L, folder = c["L"], os.path.join(ROOT, c["folder"])
    nb = json.load(open(os.path.join(ROOT, "02_main_experiment.ipynb")))
    ns = {}
    exec("".join(nb["cells"][4]["source"]).split("levels = [")[0], ns)
    prompt_dataset = ns["prompt_dataset"]

    tok = AutoTokenizer.from_pretrained(c["model_id"])
    model = AutoModelForCausalLM.from_pretrained(c["model_id"], dtype=torch.float16).to("mps").eval()
    fmt = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)

    # refusal direction: mean(L4) - mean(L0) of last-prompt-token activations at hidden_states[L]
    dpath = os.path.join(folder, f"refusal_direction_{tag}_L{L}.pt")
    if os.path.exists(dpath):
        saved = torch.load(dpath)
        assert saved["L"] == L
        r = saved["r"].float()
    else:
        def act(p):
            with torch.no_grad():
                o = model(**tok(fmt(p), return_tensors="pt").to("mps"), output_hidden_states=True)
            return o.hidden_states[L][0, -1, :].float().cpu()
        A = lambda lv: torch.stack([act(it["prompt"]) for it in prompt_dataset if it["level"] == lv])
        r = A(4).mean(0) - A(0).mean(0)
        torch.save({"L": L, "r": r}, dpath)
    r_hat = r / r.norm()

    cut = lambda t: tok.decode(tok.encode(str(t), add_special_tokens=False)[:MAXTOK])

    def response_projection(prompt, response):
        p_ids = tok(fmt(prompt), return_tensors="pt").input_ids
        f_ids = tok(fmt(prompt) + response, return_tensors="pt").input_ids
        n = p_ids.shape[1]
        assert torch.equal(f_ids[0, :n], p_ids[0]), "prompt/response tokenization boundary mismatch"
        assert f_ids.shape[1] > n, "empty response"
        with torch.no_grad():
            hs = model(input_ids=f_ids.to("mps"), output_hidden_states=True).hidden_states[L]
        resp = hs[0, n:, :].float().cpu()
        return (resp @ r_hat).mean().item(), resp.shape[0]

    base = pd.read_csv(os.path.join(folder, f"responses_{tag}_v2.csv"))
    abl = pd.read_csv(os.path.join(ROOT, f"ablation_{tag}.csv"))
    add = pd.read_csv(os.path.join(folder, f"actadd_{tag}_L{L}.csv"))
    items = (
        [("real_L0", r) for _, r in base[base.level == 0].iterrows()]
        + [("real_L4", r) for _, r in base[base.level == 4].iterrows()]
        + [("ablated_L4", r) for _, r in abl[abl.level == 4].iterrows()]
        + [("steered_L0", r) for _, r in add[add.frac == 1.0].iterrows()]
    )
    rows = []
    for g, row in items:
        text = cut(row["response"])
        proj, ntok = response_projection(row["prompt"], text)
        rows.append({"model": tag, "L": L, "group": g, "topic": row["topic"], "prompt": row["prompt"],
                     "n_response_tokens": ntok, "projection": proj})
        print(f"{g:11s} {row['topic']:15s} ntok={ntok:3d} proj={proj:8.3f}", flush=True)
    df = pd.DataFrame(rows)
    m0 = df.loc[df.group == "real_L0", "projection"].mean()
    m4 = df.loc[df.group == "real_L4", "projection"].mean()
    df["normalized"] = (df["projection"] - m0) / (m4 - m0)
    out = os.path.join(folder, f"causal_projection_{tag}_L{L}.csv")
    df.to_csv(out, index=False)
    print("\nanchors: L0 mean", round(m0, 3), "L4 mean", round(m4, 3))
    print(df.groupby("group")["normalized"].agg(["mean", "min", "max", "count"]).round(2))
    print("saved", out)


def plot():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # categorical slots 1-4 of the reference palette (adjacent pairs CVD-validated); markers + row labels
    # carry identity too, so colour is never the only cue
    style = {
        "real_L0":    {"label": "Real Level 0",                  "color": "#2a78d6", "marker": "o"},
        "real_L4":    {"label": "Real Level 4",                  "color": "#eb6834", "marker": "s"},
        "ablated_L4": {"label": "Level 4, direction ablated",    "color": "#1baf7a", "marker": "D"},
        "steered_L0": {"label": "Level 0, direction added (100% α)", "color": "#eda100", "marker": "^"},
    }
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#0b0b0b"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    data = {t: pd.read_csv(os.path.join(ROOT, CFG[t]["folder"], f"causal_projection_{t}_L{CFG[t]['L']}.csv"))
            for t in ["qwen", "mistral"]}
    lo = min(d["normalized"].min() for d in data.values())
    hi = max(d["normalized"].max() for d in data.values())
    pad = 0.08 * (hi - lo)
    for ax, t in zip(axes, ["qwen", "mistral"]):
        d = data[t]
        for x in (0, 1):
            ax.axvline(x, color="#c3c2b7", lw=1, ls="--", zorder=0)
        for yi, g in enumerate(GROUPS):
            s = style[g]
            v = d.loc[d.group == g, "normalized"].to_numpy()
            y = yi + rng.uniform(-0.12, 0.12, len(v))
            ax.scatter(v, y, s=70 if s["marker"] == "^" else 52, color=s["color"], marker=s["marker"], edgecolor="white", linewidth=1.2, zorder=3)
            ax.plot([v.mean()] * 2, [yi - 0.32, yi + 0.32], color="#52514e", lw=1.6, zorder=2, solid_capstyle="round")
        ax.set_title(f"{CFG[t]['label']} (layer {CFG[t]['L']})", fontsize=10.5, color="#0b0b0b")
        ax.set_yticks(range(len(GROUPS)))
        ax.set_yticklabels([style[g]["label"] for g in GROUPS])
        ax.set_ylim(len(GROUPS) - 0.5, -0.5)
        ax.set_xlim(lo - pad, hi + pad)
        ax.grid(axis="x", color="#e6e5e0", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.supxlabel("Normalized projection onto refusal direction (0 = Level 0 mean, 1 = Level 4 mean)", fontsize=10)
    handles = [plt.Line2D([], [], ls="", marker=style[g]["marker"], color=style[g]["color"], markersize=7,
                          markeredgecolor="white", label=style[g]["label"]) for g in GROUPS]
    handles.append(plt.Line2D([], [], color="#52514e", lw=1.6, label="Group mean"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    out = os.path.join(ROOT, "figures", "causal_projection_qwen_mistral.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    plot() if sys.argv[1] == "plot" else project(sys.argv[1])
