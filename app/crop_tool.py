"""图片比例转换 + 去头 小工具（纯本地运行，不联网、不调用任何 API）。

流程：打开工具 → 把图片拖进窗口 → 选「只转 21:9」或「去头 + 转 21:9」→ 检查/修补 → 保存（自己选位置）。
"""
import os
import sys
import json
import queue
import threading
import traceback
import subprocess

import numpy as np
from PIL import Image, ImageTk, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
FROZEN = getattr(sys, "frozen", False)  # True when running as the packaged exe
if FROZEN:
    _cfg_dir = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "图片比例转换去头工具")
    os.makedirs(_cfg_dir, exist_ok=True)
    SETTINGS = os.path.join(_cfg_dir, "settings.json")
else:
    SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
DEFAULTS = {"ratio": "21:9", "pad": "black", "last_dir": ""}
RATIOS = ["21:9", "16:9", "2.35:1", "2.39:1", "4:3", "1:1", "9:16"]

# colours
C_BG = "#4a4d52"        # canvas background (lighter than black so the black bars show)
C_ACCENT = "#1a73e8"
C_OK = "#1e8e3e"
C_WARN = "#e8710a"
C_MUTED = "#8a8f98"
FONT = "Microsoft YaHei UI"


# ---------------------------------------------------------------- 公共

def load_settings():
    try:
        with open(SETTINGS, encoding="utf-8") as fh:
            return {**DEFAULTS, **json.load(fh)}
    except Exception:
        return dict(DEFAULTS)


def store_settings(s):
    try:
        with open(SETTINGS, "w", encoding="utf-8") as fh:
            json.dump(s, fh, ensure_ascii=False)
    except Exception:
        pass


def expand_paths(args):
    files = []
    for a in args:
        a = a.strip().strip('"')
        if os.path.isdir(a):
            for n in sorted(os.listdir(a)):
                p = os.path.join(a, n)
                if os.path.isfile(p) and os.path.splitext(n)[1].lower() in IMG_EXT:
                    files.append(p)
        elif os.path.isfile(a) and os.path.splitext(a)[1].lower() in IMG_EXT:
            files.append(a)
    return files


def ratio_tag(text):
    return text.replace("：", ":").replace(":", "比").replace("/", "比").strip()


class Doc:
    """One imported picture and everything done to it."""

    def __init__(self, path):
        self.path = path
        self.orig = None          # original pixels
        self.info = None
        self.boxes = []           # [x0, y0, x1, cut_y] in original coordinates
        self.stage = "new"        # new → loaded → boxing → done
        self.mode = None          # "pad" | "dehead"
        self.result = None        # content after head removal / repairs (not padded)
        self.history = []
        self.brush = None         # PIL "L" mask in OUTPUT (padded) coordinates
        self.saved_path = None
        self.cache = {}

    @property
    def name(self):
        return os.path.basename(self.path)


class App:
    HANDLE = 8

    def __init__(self, root, files, dnd):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.root, self.dnd = tk, ttk, root, dnd
        self.settings = load_settings()
        self.docs = []
        self.cur = -1
        self.scale, self.ox, self.oy = 1.0, 0.0, 0.0
        self.show_orig = False
        self.drag = None
        self.tkimg = None
        self.busy = False
        self.msgs = queue.Queue()
        self.toast = None
        self.toast_id = 0
        self.drop_hover = False
        self.flash_box = None

        self.ratio = tk.StringVar(value=self.settings["ratio"])
        self.pad = tk.StringVar(value=self.settings["pad"])
        self.brush_on = tk.BooleanVar(value=False)
        self.brush_px = tk.IntVar(value=24)
        self.status = tk.StringVar()

        root.title("图片比例转换 + 去头工具")
        try:
            root.iconbitmap(os.path.join(core.HERE, "icon.ico"))
        except Exception:
            pass
        root.geometry("1500x950")
        root.minsize(1100, 700)
        try:
            root.state("zoomed")
        except Exception:
            pass
        self._build_ui()
        root.after(80, self._poll)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        if files:
            self.add_files(files)
        self.update_controls()

    # ================================================================ UI
    def _build_ui(self):
        tk, ttk = self.tk, self.ttk
        k = self.k = max(1.0, self.root.winfo_fpixels("1i") / 96.0)   # 1.5 on a 150 % screen
        self.HANDLE = int(8 * k)
        style = ttk.Style()
        style.configure("Step.TLabel", font=(FONT, 12))
        style.configure("Head.TLabel", font=(FONT, 11, "bold"))
        style.configure("Hint.TLabel", foreground="#5f6368")

        # ---- step indicator
        steps = ttk.Frame(self.root, padding=(12, 8))
        steps.pack(fill="x")
        self.step_lbls = []
        for i, t in enumerate(["① 拖入图片", "② 选择处理方式", "③ 检查 / 修补", "④ 保存"]):
            if i:
                ttk.Label(steps, text="  →  ", style="Step.TLabel", foreground=C_MUTED).pack(side="left")
            lb = ttk.Label(steps, text=t, style="Step.TLabel")
            lb.pack(side="left")
            self.step_lbls.append(lb)
        ttk.Separator(self.root).pack(fill="x")

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)

        # ---- right-hand panel
        panel = ttk.Frame(body, padding=(10, 8), width=int(380 * k))
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)
        self.panel = panel

        def big_button(parent, text, cmd, color):
            b = tk.Button(parent, text=text, command=cmd, font=(FONT, 12, "bold"), fg="white", bg=color,
                          activebackground="#0b3d91", activeforeground="white", relief="raised", bd=1,
                          disabledforeground="#d0d0d0", cursor="hand2", pady=6)
            b._color = color
            return b

        def small_button(parent, text, cmd):
            return ttk.Button(parent, text=text, command=cmd, takefocus=False)

        # images
        f_img = ttk.LabelFrame(panel, text=" 图片 ", padding=6)
        f_img.pack(fill="x")
        self.listbox = tk.Listbox(f_img, height=4, activestyle="none", exportselection=False,
                                  font=(FONT, 10), selectbackground=C_ACCENT)
        self.listbox.pack(fill="x")
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)
        row = ttk.Frame(f_img)
        row.pack(fill="x", pady=(4, 0))
        self.btn_add = small_button(row, "+ 添加图片…", self.open_files)
        self.btn_add.pack(side="left")
        self.btn_remove = small_button(row, "移除这张", self.remove_current)
        self.btn_remove.pack(side="right")

        # processing choice
        f_mode = ttk.LabelFrame(panel, text=" 选择处理方式 ", padding=6)
        f_mode.pack(fill="x", pady=(10, 0))
        self.btn_pad = big_button(f_mode, "只转 21:9", self.do_pad_only, C_ACCENT)
        self.btn_pad.pack(fill="x")
        self.btn_dehead_mode = big_button(f_mode, "框选头部 → 去头 + 转 21:9", self.enter_boxing, C_ACCENT)
        self.btn_dehead_mode.pack(fill="x", pady=(6, 0))

        # boxing sub-panel (shown only while framing heads)
        self.f_box = ttk.Frame(f_mode)
        ttk.Label(self.f_box, text="在图上按住左键拖出红框，框住要去掉的头。\n框的下边（橙线）就是切口，可以框多个。\n拖框的边可微调，点框右上角 × 删除。",
                  style="Hint.TLabel", justify="left").pack(anchor="w", pady=(8, 4))
        row = ttk.Frame(self.f_box)
        row.pack(fill="x")
        self.btn_autobox = small_button(row, "自动帮我框（可选）", self.auto_boxes)
        self.btn_autobox.pack(side="left")
        self.btn_clearbox = small_button(row, "清空框", self.clear_boxes)
        self.btn_clearbox.pack(side="right")
        self.lbl_boxcount = ttk.Label(self.f_box, text="", foreground=C_WARN, font=(FONT, 10, "bold"))
        self.lbl_boxcount.pack(anchor="w", pady=(4, 0))
        self.btn_run = big_button(self.f_box, "✓ 开始去头", self.run_dehead, C_OK)
        self.btn_run.pack(fill="x", pady=(4, 0))

        # repair
        f_fix = ttk.LabelFrame(panel, text=" 修补残留（可选） ", padding=6)
        f_fix.pack(fill="x", pady=(10, 0))
        self.chk_brush = ttk.Checkbutton(f_fix, text="用笔刷涂抹残留（比如耳坠）", variable=self.brush_on,
                                         command=self.on_brush_toggle, takefocus=False)
        self.chk_brush.pack(anchor="w")
        row = ttk.Frame(f_fix)
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="笔刷粗细").pack(side="left")
        self.scale_brush = ttk.Scale(row, from_=6, to=90, variable=self.brush_px, orient="horizontal")
        self.scale_brush.pack(side="left", fill="x", expand=True, padx=6)
        row = ttk.Frame(f_fix)
        row.pack(fill="x", pady=(4, 0))
        self.btn_repair = small_button(row, "修补涂抹处", self.apply_repair)
        self.btn_repair.pack(side="left")
        self.btn_clearbrush = small_button(row, "清除涂抹", self.clear_brush)
        self.btn_clearbrush.pack(side="left", padx=4)
        self.btn_undo = small_button(row, "撤销", self.undo)
        self.btn_undo.pack(side="right")

        # output settings
        f_out = ttk.LabelFrame(panel, text=" 输出设置 ", padding=6)
        f_out.pack(fill="x", pady=(10, 0))
        row = ttk.Frame(f_out)
        row.pack(fill="x")
        ttk.Label(row, text="目标比例").pack(side="left")
        self.cmb_ratio = ttk.Combobox(row, textvariable=self.ratio, values=RATIOS, width=8)
        self.cmb_ratio.pack(side="left", padx=6)
        self.cmb_ratio.bind("<<ComboboxSelected>>", lambda e: self.on_output_setting())
        self.cmb_ratio.bind("<Return>", lambda e: self.on_output_setting())
        self.cmb_ratio.bind("<FocusOut>", lambda e: self.on_output_setting())
        row = ttk.Frame(f_out)
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="两边补边").pack(side="left")
        ttk.Radiobutton(row, text="纯黑", value="black", variable=self.pad, takefocus=False,
                        command=self.on_output_setting).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(row, text="接近背景色", value="bg", variable=self.pad, takefocus=False,
                        command=self.on_output_setting).pack(side="left", padx=6)

        # save
        self.btn_save = big_button(panel, "保存…", self.save, C_OK)
        self.btn_save.pack(fill="x", pady=(14, 0))
        row = ttk.Frame(panel)
        row.pack(fill="x", pady=(4, 0))
        self.btn_restart = small_button(row, "重新开始这张", self.restart)
        self.btn_restart.pack(side="left")
        self.btn_openfolder = small_button(row, "打开保存位置", self.open_folder)
        self.btn_openfolder.pack(side="right")

        # status + progress
        self.progress = ttk.Progressbar(panel, mode="determinate", value=0)
        self.progress.pack(fill="x", pady=(14, 4))
        ttk.Label(panel, textvariable=self.status, wraplength=int(350 * k), justify="left",
                  foreground="#174ea6", font=(FONT, 10)).pack(fill="x", anchor="w")

        # ---- canvas
        self.canvas = tk.Canvas(body, bg=C_BG, highlightthickness=0, cursor="arrow")
        self.canvas.pack(side="left", fill="both", expand=True)
        foot = ttk.Frame(self.root, padding=(8, 3))
        foot.pack(fill="x")
        ttk.Label(foot, text="滚轮缩放 · 按住右键拖动画面 · 按住空格看原图 · Ctrl+Z 撤销 · Ctrl+S 保存",
                  foreground=C_MUTED).pack(side="right")

        c = self.canvas
        c.bind("<Configure>", lambda e: self.fit() if self.tkimg is None else self.render())
        c.bind("<MouseWheel>", self.on_wheel)
        for b in ("2", "3"):
            c.bind(f"<ButtonPress-{b}>", self.pan_start)
            c.bind(f"<B{b}-Motion>", self.pan_move)
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Motion>", self.on_motion)
        c.bind("<Leave>", lambda e: c.delete("cursor"))
        r = self.root
        r.bind("<Control-z>", lambda e: self.undo())
        r.bind("<Control-s>", lambda e: self.save())
        r.bind("<KeyPress-space>", lambda e: self.set_orig(True))
        r.bind("<KeyRelease-space>", lambda e: self.set_orig(False))

        # drag & drop into the window
        if self.dnd:
            from tkinterdnd2 import DND_FILES
            for w in (c, panel, self.listbox):
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<DropEnter>>", self.on_drop_enter)
                w.dnd_bind("<<DropLeave>>", self.on_drop_leave)
                w.dnd_bind("<<Drop>>", self.on_drop)

    # ================================================================ feedback helpers
    def set_status(self, text):
        self.status.set(text)

    def show_toast(self, text, kind="ok", ms=2600):
        """A coloured banner across the top of the picture, so every action visibly 'lands'."""
        self.toast_id += 1
        self.toast = (text, {"ok": C_OK, "info": C_ACCENT, "warn": C_WARN}[kind])
        tid = self.toast_id
        self._draw_toast()
        self.root.after(ms, lambda: self._clear_toast(tid))

    def _clear_toast(self, tid):
        if tid == self.toast_id:
            self.toast = None
            self.canvas.delete("toast")

    def _draw_toast(self):
        c = self.canvas
        c.delete("toast")
        if not self.toast:
            return
        text, color = self.toast
        cw = c.winfo_width()
        t = c.create_text(cw / 2, 34, text=text, fill="white", font=(FONT, 15, "bold"), tags="toast")
        x0, y0, x1, y1 = c.bbox(t)
        r = c.create_rectangle(x0 - 22, y0 - 10, x1 + 22, y1 + 10, fill=color, outline="", tags="toast")
        c.tag_lower(r, t)

    def update_steps(self):
        d = self.doc
        if d is None or d.stage == "new":
            cur = 0
        elif d.stage in ("loaded", "boxing"):
            cur = 1
        elif d.saved_path is None:
            cur = 2
        else:
            cur = 4
        for i, lb in enumerate(self.step_lbls):
            base = lb.cget("text").lstrip("✓ ")
            if i < cur:
                lb.config(text="✓ " + base, foreground=C_OK, font=(FONT, 12))
            elif i == cur:
                lb.config(text=base, foreground=C_ACCENT, font=(FONT, 12, "bold"))
            else:
                lb.config(text=base, foreground=C_MUTED, font=(FONT, 12))

    def update_controls(self):
        d = self.doc
        ready = d is not None and d.orig is not None and not self.busy
        stage = d.stage if d else "new"
        tag = self.ratio.get()

        def en(widget, on):
            if isinstance(widget, self.tk.Button):
                widget.config(state="normal" if on else "disabled", bg=widget._color if on else "#9aa0a6")
            else:
                widget.state(["!disabled"] if on else ["disabled"])

        self.btn_pad.config(text=("✓ " if d and d.mode == "pad" and stage == "done" else "") + f"只转 {tag}")
        self.btn_dehead_mode.config(text=("✓ " if d and d.mode == "dehead" and stage in ("boxing", "done") else "")
                                    + f"框选头部 → 去头 + 转 {tag}")
        en(self.btn_pad, ready)
        en(self.btn_dehead_mode, ready)
        if stage == "boxing" and d is not None:
            if not self.f_box.winfo_ismapped():
                self.f_box.pack(fill="x")
            n = len(d.boxes)
            self.lbl_boxcount.config(text=f"已框选 {n} 个头部" if n else "还没有框（请在图上拖出红框）",
                                     foreground=C_OK if n else C_WARN)
            en(self.btn_run, ready and n > 0)
            en(self.btn_autobox, ready)
            en(self.btn_clearbox, ready and n > 0)
        elif self.f_box.winfo_ismapped():
            self.f_box.pack_forget()
        done = ready and stage == "done"
        en(self.chk_brush, done)
        en(self.scale_brush, done)
        has_brush = d is not None and d.brush is not None and d.brush.getbbox() is not None
        en(self.btn_repair, done and has_brush)
        en(self.btn_clearbrush, done and has_brush)
        en(self.btn_undo, done and (bool(d.history) or has_brush))
        en(self.btn_save, done)
        en(self.btn_restart, ready and stage in ("boxing", "done"))
        en(self.btn_openfolder, d is not None and d.saved_path is not None)
        en(self.btn_add, not self.busy)
        en(self.btn_remove, not self.busy and d is not None)
        self.listbox.config(state="disabled" if self.busy else "normal")
        if stage != "done" and self.brush_on.get():
            self.brush_on.set(False)
        self.update_steps()

    # ================================================================ background work
    def run(self, busy_text, fn, after):
        """Run fn(progress) in a thread with a visible busy state, then after(result) on the UI thread."""
        if self.busy:
            return
        self.busy = True
        self.set_status(busy_text)
        self.progress.config(mode="indeterminate")
        self.progress.start(12)
        self.root.config(cursor="watch")
        self.canvas.config(cursor="watch")
        self.update_controls()
        say = lambda m: self.msgs.put(("status", m))

        def worker():
            try:
                res = fn(say)
                self.msgs.put(("done", (after, res)))
            except Exception:
                self.msgs.put(("error", traceback.format_exc()))
        threading.Thread(target=worker, daemon=True).start()

    def _finish(self):
        self.busy = False
        self.progress.stop()
        self.progress.config(mode="determinate", value=0)   # empty bar while idle
        self.root.config(cursor="")
        self.canvas.config(cursor="arrow")
        self.update_controls()

    def _poll(self):
        try:
            while True:
                kind, payload = self.msgs.get_nowait()
                if kind == "status":
                    self.set_status(payload)
                elif kind == "done":
                    after, res = payload
                    self._finish()
                    after(res)
                    self.update_controls()
                elif kind == "error":
                    from tkinter import messagebox
                    self._finish()
                    print(payload)
                    self.set_status("出错了：" + payload.strip().splitlines()[-1])
                    self.show_toast("✗ 出错了，没有完成", "warn", 4000)
                    messagebox.showerror("出错了", payload.strip().splitlines()[-1])
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    # ================================================================ images in / out
    @property
    def doc(self):
        return self.docs[self.cur] if 0 <= self.cur < len(self.docs) else None

    def on_drop_enter(self, e):
        self.drop_hover = True
        self.render()
        return e.action

    def on_drop_leave(self, e):
        self.drop_hover = False
        self.render()
        return e.action

    def on_drop(self, e):
        self.drop_hover = False
        files = expand_paths(self.root.tk.splitlist(e.data))
        if not files:
            self.render()
            self.show_toast("✗ 没有可用的图片（支持 PNG / JPG / WEBP / BMP / TIF）", "warn")
            return e.action
        self.add_files(files)
        return e.action

    def open_files(self):
        from tkinter import filedialog
        paths = filedialog.askopenfilenames(
            title="选择图片", initialdir=self.settings.get("last_dir") or None,
            filetypes=[("图片", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("全部文件", "*.*")])
        files = expand_paths(paths)
        if files:
            self.add_files(files)

    def add_files(self, files):
        if self.busy:
            self.show_toast("正在处理中，请等一下再拖入", "warn")
            return
        known = {os.path.normcase(d.path) for d in self.docs}
        new = [f for f in files if os.path.normcase(f) not in known]
        if not new:
            self.show_toast("这些图片已经在列表里了", "info")
            return
        for f in new:
            self.docs.append(Doc(f))
            self.listbox.insert("end", "  " + os.path.basename(f))
        self.select(len(self.docs) - len(new), announce=f"✓ 已导入 {len(new)} 张图片")

    def on_list_select(self, e):
        sel = self.listbox.curselection()
        if sel and sel[0] != self.cur:
            self.select(sel[0])

    def select(self, i, announce=None):
        if self.busy:
            return
        old = self.doc
        if old is not None and old is not self.docs[i] and old.stage == "loaded":
            old.orig = None          # untouched picture: free memory, reload when needed
            old.stage = "new"
        if old is not None:
            old.cache = {}
        self.cur = i
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(i)
        self.listbox.see(i)
        d = self.doc
        self.tkimg = None
        self.brush_on.set(False)
        if d.orig is None:
            def job(say):
                return core.load_image(d.path)

            def after(res):
                d.orig, d.info = res
                d.stage = "loaded"
                self.fit()
                self.show_toast(announce or f"✓ 已打开 {d.name}")
                self.set_status(f"{d.name}（{d.orig.shape[1]}×{d.orig.shape[0]}）\n请在右边选择处理方式。")
            self.render()
            self.run(f"正在读取 {d.name} …", job, after)
        else:
            self.fit()
            if announce:
                self.show_toast(announce)
            self.describe()
        self.update_controls()

    def describe(self):
        d = self.doc
        if d is None or d.orig is None:
            return
        msgs = {"loaded": "请在右边选择处理方式。",
                "boxing": "在图上拖出红框框住要去掉的头，然后点「✓ 开始去头」。",
                "done": "检查一下效果，有残留可以用笔刷修补，满意就点「保存…」。"}
        if d.saved_path:
            self.set_status(f"{d.name}\n✓ 已保存到：{d.saved_path}\n\n可以点「打开保存位置」查看；继续修补后可以再次保存；或者在上面列表里选下一张。")
        else:
            self.set_status(f"{d.name}\n{msgs.get(d.stage, '')}")

    def remove_current(self):
        d = self.doc
        if d is None or self.busy:
            return
        if d.stage == "done" and d.saved_path is None:
            from tkinter import messagebox
            if not messagebox.askyesno("还没保存", f"「{d.name}」处理过但还没保存，确定移除吗？"):
                return
        idx = self.cur
        del self.docs[idx]
        self.listbox.delete(idx)
        self.cur = -1
        self.tkimg = None
        if self.docs:
            self.select(min(idx, len(self.docs) - 1))
        else:
            self.render()
            self.set_status("")
        self.show_toast(f"已移除 {d.name}", "info")
        self.update_controls()

    # ================================================================ the two choices
    def ratio_value(self):
        try:
            r = core.parse_ratio(self.ratio.get())
            if r <= 0:
                raise ValueError
            return r
        except Exception:
            return None

    def do_pad_only(self):
        d = self.doc
        if d is None or d.orig is None or self.busy:
            return
        if self.ratio_value() is None:
            self.show_toast("✗ 比例格式不对，请写成 21:9 这样", "warn")
            return
        d.mode, d.result, d.history, d.brush = "pad", None, [], None
        d.saved_path = None
        d.cache = {}
        spec = self.spec()

        def job(say):
            return self.output(d, spec=spec)

        def after(out):
            d.stage = "done"
            arr, (left, top) = out[0], out[1]
            self.fit()
            self.show_toast(self.pad_message(d, left, top))
            self.describe()
        self.run(f"正在转成 {self.ratio.get()} …", job, after)

    def pad_message(self, d, left, top):
        H, W = d.orig.shape[:2]
        tag = self.ratio.get()
        if left == 0 and top == 0:
            return f"✓ 图片本来就是 {tag}，不需要补边"
        color = "黑边" if self.pad.get() == "black" else "背景色边"
        if left:
            return f"✓ 已转成 {tag}：左右各补了 {left} 像素{color}"
        return f"✓ 已转成 {tag}：上下各补了 {top} 像素{color}"

    def enter_boxing(self):
        d = self.doc
        if d is None or d.orig is None or self.busy:
            return
        if self.ratio_value() is None:
            self.show_toast("✗ 比例格式不对，请写成 21:9 这样", "warn")
            return
        d.mode, d.stage = "dehead", "boxing"
        d.result, d.history, d.brush, d.saved_path = None, [], None, None
        d.cache = {}
        self.brush_on.set(False)
        self.fit()
        self.show_toast("请在图上拖出红框，框住要去掉的头", "info", 3500)
        self.describe()
        self.update_controls()

    def auto_boxes(self):
        d = self.doc
        if d is None or self.busy:
            return

        def job(say):
            return core.detect_heads(d.orig)

        def after(boxes):
            if boxes:
                d.boxes = boxes
                self.render()
                self.show_toast(f"✓ 自动框出 {len(boxes)} 个头部，请检查一下")
            else:
                self.show_toast("没找到全身人物的脸，请手动拖框", "warn", 3500)
            self.describe()
        self.run("正在自动识别头部…", job, after)

    def clear_boxes(self):
        d = self.doc
        if d and d.boxes:
            d.boxes = []
            self.render()
            self.show_toast("已清空所有框", "info")
            self.update_controls()

    def run_dehead(self):
        d = self.doc
        if d is None or not d.boxes or self.busy:
            return
        boxes = [list(b) for b in d.boxes]
        n = len(boxes)
        spec = self.spec()

        def job(say):
            d.result = core.remove_heads(d.orig, boxes, progress=say)
            d.cache = {}
            say("正在生成预览…")
            return self.output(d, spec=spec)

        def after(out):
            d.stage = "done"
            self.fit()
            tag = self.ratio.get()
            self.show_toast(f"✓ 已去掉 {n} 个头，并转成 {tag}")
            self.describe()
        self.run(f"正在去头（{n} 个）…", job, after)

    # ================================================================ output
    def spec(self):
        """Output settings read on the UI thread (worker threads must not touch Tk variables)."""
        return (self.ratio.get(), self.ratio_value(), self.pad.get())

    def output(self, d, original=False, spec=None):
        """Final padded picture (array, (left, top), PIL) for the given ratio / bar colour; cached."""
        text, value, pad = spec or self.spec()
        key = ("orig" if original else "out", text, pad)
        if key not in d.cache:
            src = d.orig if (original or d.result is None) else d.result
            arr, off = core.pad_to_ratio(src, value, pad)
            d.cache[key] = (arr, off, Image.fromarray(arr))
        return d.cache[key]

    def on_output_setting(self):
        d = self.doc
        if self.ratio_value() is None:
            self.show_toast("✗ 比例格式不对，请写成 21:9 这样", "warn")
            return
        changed = (self.ratio.get(), self.pad.get()) != (self.settings["ratio"], self.settings["pad"])
        self.settings.update(ratio=self.ratio.get(), pad=self.pad.get())
        store_settings(self.settings)
        self.update_controls()
        if not changed:
            return
        if d is not None and d.stage == "done" and not self.busy:
            d.saved_path = None
            d.brush = None          # brush strokes were drawn for the old frame size
            spec = self.spec()

            def job(say):
                return self.output(d, spec=spec)

            def after(out):
                self.fit()
                self.show_toast(self.pad_message(d, out[1][0], out[1][1]))
                self.describe()
            self.run("正在更新预览…", job, after)
        else:
            self.show_toast(f"✓ 输出改为 {self.ratio.get()}，补边：{'纯黑' if self.pad.get() == 'black' else '接近背景色'}", "info")

    def save(self):
        d = self.doc
        if d is None or d.stage != "done" or self.busy:
            if d is not None and d.stage != "done":
                self.show_toast("请先在右边选择处理方式", "warn")
            return
        from tkinter import filedialog
        stem = os.path.splitext(d.name)[0]
        default = f"{stem}_{ratio_tag(self.ratio.get())}{'_去头' if d.mode == 'dehead' else ''}.png"
        init_dir = self.settings.get("last_dir") or os.path.dirname(d.path)
        path = filedialog.asksaveasfilename(title="保存到…", initialdir=init_dir, initialfile=default,
                                            defaultextension=".png", filetypes=[("PNG 图片（无损）", "*.png")])
        if not path:
            self.show_toast("已取消保存", "info")
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        arr = self.output(d)[0]
        info = d.info

        def job(say):
            core.save_png(arr, path, info)
            return path

        def after(p):
            d.saved_path = p
            self.settings["last_dir"] = os.path.dirname(p)
            store_settings(self.settings)
            self.show_toast(f"✓ 已保存：{os.path.basename(p)}", "ok", 3500)
            self.describe()
        self.run("正在保存（大图要几秒）…", job, after)

    def open_folder(self):
        d = self.doc
        if d and d.saved_path and os.path.exists(d.saved_path):
            subprocess.Popen(["explorer", "/select,", os.path.normpath(d.saved_path)])
            self.show_toast("已打开保存位置", "info")

    def restart(self):
        d = self.doc
        if d is None or self.busy:
            return
        d.stage, d.mode, d.result, d.history, d.brush, d.saved_path = "loaded", None, None, [], None, None
        d.cache = {}
        self.brush_on.set(False)
        self.fit()
        self.show_toast("已恢复原图，可以重新选择处理方式", "info")
        self.describe()
        self.update_controls()

    # ================================================================ repair brush
    def on_brush_toggle(self):
        if self.brush_on.get():
            self.show_toast("笔刷已开启：把残留涂成红色，再点「修补涂抹处」", "info", 3500)
        else:
            self.show_toast("笔刷已关闭", "info")
        self.canvas.delete("cursor")

    def apply_repair(self):
        d = self.doc
        if d is None or d.brush is None or d.brush.getbbox() is None or self.busy:
            return
        out_arr, (left, top), _ = self.output(d)
        H, W = d.orig.shape[:2]
        mask = (np.asarray(d.brush) > 0)[top:top + H, left:left + W]
        base = d.result if d.result is not None else d.orig

        def job(say):
            return core.repair(base, mask, progress=say)

        def after(res):
            d.history.append(d.result)
            del d.history[:-8]
            d.result = res
            d.brush = None
            d.cache = {}
            d.saved_path = None
            self.render()
            self.show_toast("✓ 已修补涂抹处")
            self.describe()
        self.run("正在修补…", job, after)

    def clear_brush(self):
        d = self.doc
        if d and d.brush is not None:
            d.brush = None
            self.render()
            self.show_toast("已清除涂抹", "info")
            self.update_controls()

    def undo(self):
        d = self.doc
        if d is None or self.busy or d.stage != "done":
            return
        if d.brush is not None and d.brush.getbbox() is not None:
            d.brush = None
            self.show_toast("已撤销涂抹", "info")
        elif d.history:
            d.result = d.history.pop()
            d.cache = {}
            d.saved_path = None
            self.show_toast("✓ 已撤销上一次修补", "info")
        else:
            self.show_toast("没有可以撤销的了", "info")
        self.render()
        self.update_controls()

    # ================================================================ view
    def display(self):
        """(PIL image, (offx, offy)) currently shown; offset maps original coords into it."""
        d = self.doc
        if d is None or d.orig is None:
            return None, (0, 0)
        if d.stage == "done":
            _, off, pil = self.output(d, original=self.show_orig)
            return pil, off
        if "plain" not in d.cache:
            d.cache["plain"] = Image.fromarray(d.orig)
        return d.cache["plain"], (0, 0)

    def fit(self):
        img, _ = self.display()
        c = self.canvas
        cw, ch = max(10, c.winfo_width()), max(10, c.winfo_height())
        if img is None:
            self.render()
            return
        W, H = img.size
        self.scale = min(cw / W, (ch - 60) / H) * 0.96
        self.ox = W / 2 - cw / 2 / self.scale
        self.oy = H / 2 - (ch + 30) / 2 / self.scale
        self.render()

    def to_disp(self, sx, sy):
        return self.ox + sx / self.scale, self.oy + sy / self.scale

    def to_scr(self, dx, dy):
        return (dx - self.ox) * self.scale, (dy - self.oy) * self.scale

    def render(self):
        c = self.canvas
        c.delete("all")
        cw, ch = c.winfo_width(), c.winfo_height()
        img, (offx, offy) = self.display()
        d = self.doc
        if img is None:
            self.tkimg = None
            self._draw_dropzone(cw, ch, loading=d is not None)
            self._draw_toast()
            return
        W, H = img.size
        ix0, iy0 = max(0.0, self.ox), max(0.0, self.oy)
        ix1, iy1 = min(W, self.ox + cw / self.scale), min(H, self.oy + ch / self.scale)
        if ix1 - ix0 >= 1 and iy1 - iy0 >= 1:
            sx0, sy0 = self.to_scr(ix0, iy0)
            dw, dh = max(1, round((ix1 - ix0) * self.scale)), max(1, round((iy1 - iy0) * self.scale))
            box = (ix0, iy0, ix1, iy1)
            resample = Image.NEAREST if self.scale >= 2 else Image.BILINEAR
            view = img.resize((dw, dh), resample, box=box, reducing_gap=2.0 if self.scale < 1 else None)
            if d.stage == "done" and d.brush is not None and not self.show_orig:
                m = d.brush.resize((dw, dh), Image.NEAREST, box=box)
                red = Image.new("RGB", (dw, dh), (255, 40, 40))
                view = Image.composite(Image.blend(view, red, 0.55), view, m)
            self.tkimg = ImageTk.PhotoImage(view)
            c.create_image(sx0, sy0, anchor="nw", image=self.tkimg)
        else:
            self.tkimg = None

        if d.stage == "done":
            # outline of the final frame + where the original sits inside it
            a, b = self.to_scr(0, 0)
            e, f = self.to_scr(W, H)
            c.create_rectangle(a, b, e, f, outline="#ffffff", dash=(5, 3))
            label = f"输出 {self.ratio.get()} · {W}×{H}" + ("（原图）" if self.show_orig else "")
            c.create_text(a, b - 8, anchor="sw", text=label, fill="#ffffff", font=(FONT, 11, "bold"))
            if offx or offy:
                oh, ow = d.orig.shape[:2]
                p, q = self.to_scr(offx, offy)
                r_, s_ = self.to_scr(offx + ow, offy + oh)
                c.create_rectangle(p, q, r_, s_, outline="#ffd166", dash=(2, 4))
        if d.stage == "boxing":
            for k, bx in enumerate(d.boxes):
                self._draw_box(k, bx)
        if self.show_orig:
            c.create_text(12, ch - 12, anchor="sw", text="正在看原图（松开空格返回）", fill="#ffd166",
                          font=(FONT, 13, "bold"))
        self._draw_toast()

    def _draw_dropzone(self, cw, ch, loading=False):
        c = self.canvas
        hot = self.drop_hover
        m = 40
        c.create_rectangle(m, m, cw - m, ch - m, outline="#8ab4f8" if hot else "#9aa0a6",
                           width=4 if hot else 2, dash=() if hot else (12, 8), fill="#3c4a5e" if hot else "")
        if loading:
            c.create_text(cw / 2, ch / 2, text="正在读取图片…", fill="white", font=(FONT, 22, "bold"))
            return
        if not self.dnd:
            c.create_text(cw / 2, ch / 2 - 30, text="点右上角「+ 添加图片…」导入图片",
                          fill="white", font=(FONT, 24, "bold"))
            c.create_text(cw / 2, ch / 2 + 22, text="（这台电脑上拖放功能没有加载成功，只能用按钮添加）",
                          fill="#ffb4a9", font=(FONT, 13))
            return
        c.create_text(cw / 2, ch / 2 - 30, text="松开鼠标即可导入" if hot else "把图片拖到这里",
                      fill="white", font=(FONT, 26, "bold"))
        c.create_text(cw / 2, ch / 2 + 22, text="支持 PNG / JPG / WEBP，可以一次拖多张，也可以拖整个文件夹",
                      fill="#dadce0", font=(FONT, 13))
        c.create_text(cw / 2, ch / 2 + 54, text="或者点右上角「+ 添加图片…」", fill="#dadce0", font=(FONT, 13))

    def _draw_box(self, k, bx):
        c = self.canvas
        x0, y0, x1, y1 = bx
        a, b = self.to_scr(x0, y0)
        e, f = self.to_scr(x1, y1)
        flash = self.flash_box == k
        c.create_rectangle(a, b, e, f, outline="#ffeb3b" if flash else "#ff3b30", width=4 if flash else 2)
        c.create_line(a, f, e, f, fill="#ff9500", width=4)
        c.create_text(a + 4, f - 5, anchor="sw", text=f"头{k + 1} · 切口", fill="#ff9500",
                      font=(FONT, 10, "bold"))
        s = self.HANDLE + 3
        c.create_rectangle(e - 2 * s, b, e, b + 2 * s, fill="#ff3b30", outline="")
        c.create_text(e - s, b + s, text="×", fill="white", font=(FONT, 11, "bold"))

    def on_wheel(self, e):
        img, _ = self.display()
        if img is None:
            return
        k = 1.25 if e.delta > 0 else 1 / 1.25
        dx, dy = self.to_disp(e.x, e.y)
        self.scale = min(16.0, max(0.02, self.scale * k))
        self.ox, self.oy = dx - e.x / self.scale, dy - e.y / self.scale
        self.render()

    def pan_start(self, e):
        self._pan = (e.x, e.y, self.ox, self.oy)

    def pan_move(self, e):
        x, y, ox, oy = self._pan
        self.ox, self.oy = ox - (e.x - x) / self.scale, oy - (e.y - y) / self.scale
        self.render()

    def set_orig(self, flag):
        d = self.doc
        if d is None or d.stage != "done" or self.busy:
            flag = False
        if self.show_orig != flag:
            self.show_orig = flag
            self.render()

    # ================================================================ mouse: boxes and brush
    def _hit_box(self, sx, sy):
        """→ (index, part) where part is 'close', 'top', 'bottom', 'left', 'right' or None."""
        d = self.doc
        h = self.HANDLE
        for k in range(len(d.boxes) - 1, -1, -1):
            x0, y0, x1, y1 = d.boxes[k]
            a, b = self.to_scr(x0, y0)
            e, f = self.to_scr(x1, y1)
            s = 2 * (h + 3)
            if e - s <= sx <= e and b <= sy <= b + s:
                return k, "close"
            ix, iy = a - h <= sx <= e + h, b - h <= sy <= f + h
            if ix and abs(sy - f) <= h:
                return k, "bottom"
            if ix and abs(sy - b) <= h:
                return k, "top"
            if iy and abs(sx - a) <= h:
                return k, "left"
            if iy and abs(sx - e) <= h:
                return k, "right"
        return None

    def on_motion(self, e):
        c = self.canvas
        c.delete("cursor")
        d = self.doc
        if d is None or d.orig is None or self.busy:
            return
        if d.stage == "done" and self.brush_on.get():
            r = self.brush_px.get() / 2
            c.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, outline="#ff3b30", width=2, tags="cursor")
            c.config(cursor="tcross")
        elif d.stage == "boxing":
            hit = self._hit_box(e.x, e.y)
            if hit is None:
                c.config(cursor="crosshair")
            elif hit[1] == "close":
                c.config(cursor="hand2")
            else:
                c.config(cursor="sb_v_double_arrow" if hit[1] in ("top", "bottom") else "sb_h_double_arrow")
        else:
            c.config(cursor="arrow")

    def on_press(self, e):
        d = self.doc
        self.drag = None
        if d is None or d.orig is None or self.busy:
            return
        if d.stage == "done":
            if not self.brush_on.get():
                self.show_toast("想修补残留的话，先勾选右边「用笔刷涂抹残留」", "info")
                return
            arr, off, pil = self.output(d)
            if d.brush is None:
                d.brush = Image.new("L", pil.size, 0)
            self.drag = ("brush", self.to_disp(e.x, e.y))
            self._paint(e)
            return
        if d.stage != "boxing":
            if d.stage == "loaded":
                self.show_toast("请先在右边选择处理方式", "info")
            return
        hit = self._hit_box(e.x, e.y)
        if hit and hit[1] == "close":
            del d.boxes[hit[0]]
            self.render()
            self.show_toast(f"已删除第 {hit[0] + 1} 个框", "info")
            self.update_controls()
        elif hit:
            self.drag = ("edge", hit)
        else:
            self.drag = ("new", self.to_disp(e.x, e.y))

    def _paint(self, e):
        d = self.doc
        _, (px, py) = self.drag
        dx, dy = self.to_disp(e.x, e.y)
        r = self.brush_px.get() / 2 / self.scale
        dr = ImageDraw.Draw(d.brush)
        dr.line([(px, py), (dx, dy)], fill=255, width=max(1, int(2 * r)))
        for (x, y) in ((px, py), (dx, dy)):
            dr.ellipse([x - r, y - r, x + r, y + r], fill=255)
        self.drag = ("brush", (dx, dy))
        rs = self.brush_px.get() / 2
        self.canvas.create_oval(e.x - rs, e.y - rs, e.x + rs, e.y + rs, fill="#ff5a50", outline="",
                                stipple="gray50")

    def on_drag(self, e):
        if not self.drag:
            return
        d = self.doc
        kind = self.drag[0]
        if kind == "brush":
            self._paint(e)
        elif kind == "new":
            x0, y0 = self.to_scr(*self.drag[1])
            self.canvas.delete("rubber")
            self.canvas.create_rectangle(x0, y0, e.x, e.y, outline="#ff3b30", width=2, dash=(4, 2), tags="rubber")
        elif kind == "edge":
            k, side = self.drag[1]
            dx, dy = self.to_disp(e.x, e.y)
            H, W = d.orig.shape[:2]
            b = d.boxes[k]
            if side == "left":
                b[0] = int(min(max(0, dx), b[2] - 4))
            elif side == "right":
                b[2] = int(max(min(W, dx), b[0] + 4))
            elif side == "top":
                b[1] = int(min(max(0, dy), b[3] - 4))
            else:
                b[3] = int(max(min(H, dy), b[1] + 4))
            self.render()

    def on_release(self, e):
        if not self.drag:
            return
        d = self.doc
        kind = self.drag[0]
        if kind == "brush":
            self.render()
            self.update_controls()
            self.set_status("涂好后点右边「修补涂抹处」；涂错了点「清除涂抹」。")
        elif kind == "new":
            self.canvas.delete("rubber")
            (ax, ay), (bx, by) = self.drag[1], self.to_disp(e.x, e.y)
            H, W = d.orig.shape[:2]
            x0, x1 = sorted((int(max(0, min(W, ax))), int(max(0, min(W, bx)))))
            y0, y1 = sorted((int(max(0, min(H, ay))), int(max(0, min(H, by)))))
            if (x1 - x0) * self.scale > 10 and (y1 - y0) * self.scale > 10:
                d.boxes.append([x0, y0, x1, y1])
                self._flash(len(d.boxes) - 1)
                self.show_toast(f"✓ 已框选第 {len(d.boxes)} 个头部")
            else:
                self.show_toast("框太小了，请按住左键拖大一点", "warn")
                self.render()
        elif kind == "edge":
            k, side = self.drag[1]
            self._flash(k)
            self.show_toast(f"✓ 已调整第 {k + 1} 个框" + ("的切口" if side == "bottom" else ""))
        self.drag = None
        self.update_controls()

    def _flash(self, k):
        self.flash_box = k
        self.render()

        def off():
            if self.flash_box == k:
                self.flash_box = None
                self.render()
        self.root.after(600, off)

    # ================================================================ closing
    def on_close(self):
        unsaved = [d.name for d in self.docs if d.stage == "done" and d.saved_path is None]
        if unsaved:
            from tkinter import messagebox
            names = "\n".join("· " + n for n in unsaved[:6])
            if not messagebox.askyesno("还有图片没保存", f"下面这些图处理过但还没保存：\n{names}\n\n确定直接关闭吗？"):
                return
        self.root.destroy()


def close_splash():
    """关掉 exe 启动时的「正在启动」画面（直接用 Python 运行时没有这个画面）。"""
    try:
        import pyi_splash
        pyi_splash.close()
    except Exception:
        pass


def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)  # 高分屏下不模糊
    except Exception:
        pass
    try:
        from tkinterdnd2 import TkinterDnD
        root, dnd = TkinterDnD.Tk(), True
    except Exception:
        import tkinter as tk
        root, dnd = tk.Tk(), False
    from tkinter import font as tkfont
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
        try:
            tkfont.nametofont(name).configure(family=FONT, size=11)
        except Exception:
            pass
    App(root, expand_paths(sys.argv[1:]), dnd)
    root.after(300, close_splash)
    root.mainloop()


if __name__ == "__main__":
    main()
