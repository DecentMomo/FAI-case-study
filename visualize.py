"""
visualize.py - matplotlib replay of a finished game (simple and readable, for live demos).

Left   : the room graph.  Grey lines = corridors, dashed red = vents (only with reveal=True),
         coloured dots = agents moving room to room each tick (interpolated along corridors),
         black X = unreported/reported body, dead agents are drawn as small grey crosses.
Top-right    : suspicion HEAT-MAP.  Row = observer crewmate, column = suspect, cell = P_observer(suspect
               is the impostor); last row = aggregated (mean) suspicion.  Updates at every meeting.
Bottom-right : round-by-round outcome log (last events), plus votes / ejection banners.

The animation replays frames recorded by the engine (Config.record_frames=True), so the simulation
itself is never slowed down or altered by drawing.

Mouse wheel over the CHAT or EVENT LOG box scrolls back through history (anchored, so new lines do not
shove the view); F jumps back to live.  SPACE pause/resume.   Use --save file.gif (needs pillow) to export instead of opening a window.
"""
import math
from typing import Optional

import numpy as np


def animate(result, reveal: bool = True, interval: int = 600, save: Optional[str] = None,
            show: bool = True, loop: bool = False, frame: Optional[int] = None):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    frames = result.frames
    if not frames:
        raise ValueError("no frames recorded - run the game with Config(record_frames=True)")
    smap = result.map
    names = result.agent_order
    colors = result.agent_colors
    imp = result.impostor

    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.15, 0.55], hspace=0.5, wspace=0.1)
    axm = fig.add_subplot(gs[:, 0])
    axh = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 1])
    axl = fig.add_subplot(gs[2, 1])

    # ---------------- map (static part) ----------------
    for u in smap.rooms:
        for v in smap.adj[u]:
            if u < v:
                (x1, y1), (x2, y2) = smap.pos[u], smap.pos[v]
                axm.plot([x1, x2], [y1, y2], color="#aaaaaa", lw=3, zorder=1)
                axm.text((x1 + x2) / 2, (y1 + y2) / 2, f"{smap.adj[u][v]:.1f}", fontsize=7, color="#777",
                         ha="center", va="center", zorder=2,
                         bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    if reveal:
        for u in smap.rooms:
            for v in smap.vent_adj[u]:
                if u < v:
                    (x1, y1), (x2, y2) = smap.pos[u], smap.pos[v]
                    axm.plot([x1, x2], [y1, y2], color="#d62728", lw=1.2, ls="--", zorder=1, alpha=0.7)
    traps = set(smap.trap_rooms())
    for r, (x, y) in smap.pos.items():
        ec = "#d62728" if r in traps else "#444444"
        axm.scatter([x], [y], s=2300, c="#f4f4f4", edgecolors=ec, linewidths=2.2 if r in traps else 1.2, zorder=3)
        axm.text(x, y + 0.52, r, ha="center", va="bottom", fontsize=8.5, weight="bold", zorder=6)
    axm.text(0.01, 0.01, "red outline = dead end with no vent", transform=axm.transAxes, fontsize=7, color="#d62728")
    xs = [p[0] for p in smap.pos.values()]
    ys = [p[1] for p in smap.pos.values()]
    axm.set_xlim(min(xs) - 1, max(xs) + 1)
    axm.set_ylim(max(ys) + 1.1, min(ys) - 1)       # y grows downward like a floor plan
    axm.set_aspect("equal")
    axm.axis("off")
    title = axm.set_title("", fontsize=12, weight="bold")

    n = len(names)
    sc = axm.scatter([0] * n, [0] * n, s=170, zorder=5, edgecolors="black", linewidths=1.0)
    body_sc = axm.scatter([], [], marker="X", s=260, c="black", zorder=4)
    labels = [axm.text(0, 0, nm, fontsize=6.5, ha="center", va="top", zorder=7) for nm in names]
    banner = axm.text(0.5, 0.02, "", transform=axm.transAxes, ha="center", fontsize=11, weight="bold",
                      color="#b00020")

    # ---------------- heat-map ----------------
    rows = list(names) + ["MEAN"]
    M = np.full((len(rows), len(names)), np.nan)
    im = axh.imshow(M, vmin=0, vmax=0.7, cmap="Reds", aspect="auto")
    axh.set_xticks(range(len(names)))
    axh.set_xticklabels([("*" + x if (reveal and x == imp) else x) for x in names], rotation=45, ha="right", fontsize=7)
    axh.set_yticks(range(len(rows)))
    axh.set_yticklabels([("*" + x if (reveal and x == imp) else x) for x in rows], fontsize=7)
    axh.set_title("who does each crewmate (row) think is the impostor (column)?", fontsize=9)
    cell_txt = [[axh.text(j, i, "", ha="center", va="center", fontsize=6) for j in range(len(names))]
                for i in range(len(rows))]
    axh.axhline(len(rows) - 1.5, color="black", lw=1.5)

    axl.axis("off")
    log_txt = axl.text(0.0, 1.0, "", va="top", ha="left", fontsize=6.8, family="monospace",
                       transform=axl.transAxes)
    # chat box: one coloured Text per line (speaker colour), newest at the bottom
    axc.set_xticks([])
    axc.set_yticks([])
    axc.set_title("CHAT  (mouse-wheel over a box = scroll back,  F = follow live)", fontsize=8.5, loc="left")
    N_CHAT = 12
    chat_txt = [axc.text(0.02, 0.97 - i * 0.082, "", transform=axc.transAxes, va="top", ha="left", fontsize=7.4)
                for i in range(N_CHAT)]

    # per-name small offsets so agents sharing a room stay visible
    off = {nm: (0.34 * math.cos(2 * math.pi * i / n), 0.34 * math.sin(2 * math.pi * i / n))
           for i, nm in enumerate(names)}

    N_LOG = 7
    view = {"chat": None, "log": None, "frame": 0}     # None = follow live; else index of last visible line

    def _window(allv, end, n):
        end = max(min(end, len(allv)), 0)
        return allv[max(0, end - n):end], end

    def refresh_text(i):
        view["frame"] = i
        f = frames[i]
        chat_all, log_all = result.chat, result.events
        c_end = f["chat_n"] if view["chat"] is None else view["chat"]
        ch, c_end = _window(chat_all, c_end, N_CHAT)
        for k, tx in enumerate(chat_txt):
            if k < len(ch):
                spk, msg = ch[k]
                if spk:
                    tx.set_text(f"{spk}: {msg}"[:92])
                    tx.set_color(colors.get(spk, "black"))
                    tx.set_weight("normal")
                else:
                    tx.set_text(msg[:92])
                    tx.set_color("#b00020")
                    tx.set_weight("bold")
            else:
                tx.set_text("")
        axc.set_xlabel(f"scrolled back - line {c_end}/{f['chat_n']}  (F = live)" if view["chat"] is not None else "", fontsize=7, color="#b00020")
        l_end = f["log_n"] if view["log"] is None else view["log"]
        lines, l_end = _window(log_all, l_end, N_LOG)
        log_txt.set_text(chr(10).join(l[:80] for l in lines))
        axl.set_title(f"EVENT LOG - scrolled back (line {l_end}/{f['log_n']}, F = live)" if view["log"] is not None
                      else "EVENT LOG (scrollable)", fontsize=8, loc="left")

    def on_scroll(ev):
        key = "chat" if ev.inaxes is axc else "log" if ev.inaxes is axl else None
        if key is None:
            return
        f = frames[view["frame"]]
        cur = view[key] if view[key] is not None else f["chat_n" if key == "chat" else "log_n"]
        n = N_CHAT if key == "chat" else N_LOG
        cur = cur - 3 if ev.button == "up" else cur + 3
        total = f["chat_n" if key == "chat" else "log_n"]
        cur = max(n, min(cur, total))
        view[key] = None if cur >= total else cur
        refresh_text(view["frame"])
        fig.canvas.draw_idle()

    def draw(i):
        f = frames[i]
        pts, cols, szs = [], [], []
        for k, nm in enumerate(names):
            x, y = f["pos"][nm]
            ox, oy = off[nm]
            pts.append((x + ox, y + oy))
            alive = f["alive"][nm]
            cols.append(colors[nm] if alive else "#cccccc")
            szs.append(170 if alive else 60)
            labels[k].set_position((x + ox, y + oy + 0.13))
            labels[k].set_text(nm if alive else "")
        sc.set_offsets(pts)
        sc.set_facecolor(cols)
        sc.set_sizes(szs)
        ew = ["#d62728" if (reveal and nm == imp and f["alive"][nm]) else "black" for nm in names]
        sc.set_edgecolor(ew)
        sc.set_linewidth([3.0 if e != "black" else 1.0 for e in ew])
        bs = [smap.pos[r] for r in f["bodies"]]
        body_sc.set_offsets(bs if bs else np.empty((0, 2)))
        title.set_text(f"Round {f['round']}  |  tick {f['tick']}  |  {f['phase']}"
                       + (f"  -  {f['label']}" if f["label"] else ""))
        # heat-map
        B = f.get("beliefs")
        Mx = np.full((len(rows), len(names)), np.nan)
        if B:
            for r_i, obs in enumerate(rows[:-1]):
                if obs == imp and not reveal and B:
                    # hide the impostor: its row shows a flat decoy so the audience cannot spot it
                    cols_ = [x for x in names if x != imp and any(x in b for b in B.values())]
                    for c_i, sus_ in enumerate(names):
                        if sus_ in cols_:
                            Mx[r_i, c_i] = 1.0 / len(cols_)
                elif obs in B:
                    for c_i, sus_ in enumerate(names):
                        if sus_ in B[obs]:
                            Mx[r_i, c_i] = B[obs][sus_]
            A = f.get("agg") or {}
            for c_i, sus_ in enumerate(names):
                if sus_ in A:
                    Mx[-1, c_i] = A[sus_]
        im.set_data(np.ma.masked_invalid(Mx))
        for r_i in range(len(rows)):
            for c_i in range(len(names)):
                v = Mx[r_i, c_i]
                cell_txt[r_i][c_i].set_text("" if np.isnan(v) else f"{v:.2f}")
        refresh_text(i)
        if f.get("votes") is not None:
            e = f.get("ejected")
            banner.set_text(f"VOTE: {e} ejected" if e else "VOTE: no elimination")
        else:
            banner.set_text("BODY REPORTED" if f["label"].startswith("BODY") else "")
        return sc, body_sc, im

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=interval, blit=False, repeat=loop)
    paused = {"v": False}

    def on_key(ev):
        if ev.key in ("f", "F", "end"):
            view["chat"] = view["log"] = None
            refresh_text(view["frame"])
            fig.canvas.draw_idle()
        if ev.key == " ":
            if paused["v"]:
                anim.resume()
            else:
                anim.pause()
            paused["v"] = not paused["v"]
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.suptitle(f"Social deduction under uncertainty  -  impostor policy: {result.cfg.impostor_policy}"
                 f"  -  seed {result.cfg.seed}", fontsize=10)

    if save and save.lower().endswith(".png"):
        # single still image (default: the last frame, i.e. the final vote)
        draw(len(frames) - 1 if frame is None else frame)
        fig.savefig(save, dpi=110)
        anim._draw_was_started = True          # silence matplotlib's "animation never rendered" warning
        plt.close(fig)
        print(f"saved still to {save}")
    elif save:
        from matplotlib.animation import PillowWriter
        if save.lower().endswith(".gif"):
            anim.save(save, writer=PillowWriter(fps=max(1, int(1000 / interval))), dpi=70)
        else:
            anim.save(save, dpi=90)
        print(f"saved animation to {save}")
        plt.close(fig)
    elif show:
        plt.show()
    return anim
