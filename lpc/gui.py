"""GUI for local perspective correction (ttkbootstrap).

Layout (fixed by design):
  - Left pane: controls (image, parameters, measurement table, actions)
  - Right pane: log (upper) and errors (lower)

The angle theta is a user parameter (specification.md v0.5, D-04):
the slider defaults to 90 degrees and the user adjusts it while looking
at the rectified image.  Theta is never auto-proposed.
"""

from __future__ import annotations

import csv
import math
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox

import ttkbootstrap as tb

try:  # ttkbootstrap < 1.10
    from ttkbootstrap.scrolled import ScrolledText
except ImportError:  # newer versions expose it at top level
    from ttkbootstrap import ScrolledText

from .core import (
    InsufficientDataError,
    Measurement,
    estimate_homography,
    fill_directions,
    warp_image,
)
from .simulate import make_scene


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


class App(tb.Window):
    def __init__(self) -> None:
        super().__init__(themename="flatly")
        self.title("Local Perspective Correction")
        self.geometry("1220x740")
        self.minsize(1000, 620)

        self.image_path: str | None = None
        self.pil_image = None
        self.result = None
        self.sample_truth = None  # plane coords for synthetic sample data
        self._preview_win = None
        self._preview_photo = None

        self._build_layout()
        self.log("起動しました。左側で画像と測定データを用意し、「推定を実行」を押してください。")
        self.log("角度 θ はパラメータです（初期値 90°）。推定後にスライダーで調整できます。")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = tb.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="ns")
        right = tb.Frame(self, padding=(0, 10, 10, 10))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=1)

        self._build_image_controls(left)
        self._build_parameter_controls(left)
        self._build_measurement_controls(left)
        self._build_estimate_controls(left)

        log_frame = tb.LabelFrame(right, text="ログ", padding=5)
        log_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.log_text = ScrolledText(log_frame, height=20, autohide=True)
        self.log_text.pack(fill="both", expand=True)

        err_frame = tb.LabelFrame(right, text="エラー / 警告", padding=5)
        err_frame.grid(row=1, column=0, sticky="nsew")
        self.err_text = ScrolledText(err_frame, height=7, autohide=True)
        self.err_text.pack(fill="both", expand=True)
        self.err_text.text.tag_configure("error", foreground="#c0392b")

        self.status = tb.Label(self, text="", anchor="w", padding=(10, 4))
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _build_image_controls(self, parent) -> None:
        frame = tb.LabelFrame(parent, text="画像", padding=8)
        frame.pack(fill="x", pady=(0, 8))
        row = tb.Frame(frame)
        row.pack(fill="x")
        tb.Button(row, text="画像を開く…", bootstyle="info",
                  command=self.on_open_image).pack(side="left")
        self.image_label = tb.Label(row, text="（未選択）")
        self.image_label.pack(side="left", padx=8)
        row2 = tb.Frame(frame)
        row2.pack(fill="x", pady=(6, 0))
        self.btn_preview = tb.Button(row2, text="補正画像を表示",
                                     command=self.on_preview, state="disabled")
        self.btn_preview.pack(side="left")
        self.btn_save = tb.Button(row2, text="補正画像を保存…",
                                  command=self.on_save_warped, state="disabled")
        self.btn_save.pack(side="left", padx=6)

    def _build_parameter_controls(self, parent) -> None:
        frame = tb.LabelFrame(parent, text="パラメータ", padding=8)
        frame.pack(fill="x", pady=(0, 8))
        row = tb.Frame(frame)
        row.pack(fill="x")
        tb.Label(row, text="ピッチ P [mm]").pack(side="left")
        self.pitch_var = tk.StringVar(value="10.0")
        tb.Entry(row, textvariable=self.pitch_var, width=8).pack(side="left", padx=6)

        row2 = tb.Frame(frame)
        row2.pack(fill="x", pady=(8, 0))
        tb.Label(row2, text="角度 θ [°]").pack(side="left")
        self.theta_var = tk.DoubleVar(value=90.0)
        self.theta_label = tb.Label(row2, text="90.0°", width=7)
        self.theta_label.pack(side="right")
        slider = tb.Scale(row2, from_=30, to=150, variable=self.theta_var,
                          command=self._on_theta_changed)
        slider.pack(side="left", fill="x", expand=True, padx=6)
        slider.bind("<ButtonRelease-1>", lambda _e: self._on_theta_released())
        self.auto_reestimate = tk.BooleanVar(value=True)
        tb.Checkbutton(frame, text="推定後に θ を変えたら自動で再推定する",
                       variable=self.auto_reestimate).pack(anchor="w", pady=(4, 0))

    def _build_measurement_controls(self, parent) -> None:
        frame = tb.LabelFrame(parent, text="測定データ", padding=8)
        frame.pack(fill="x", pady=(0, 8))
        cols = ("x", "y", "period", "element", "group", "dirx", "diry")
        headers = {"x": "x", "y": "y", "period": "周期px", "element": "要素",
                   "group": "群", "dirx": "dir_x", "diry": "dir_y"}
        self.tree = tb.Treeview(frame, columns=cols, show="headings", height=9)
        for c in cols:
            self.tree.heading(c, text=headers[c])
            self.tree.column(c, width=52, anchor="center")
        self.tree.pack(fill="x")

        entry_row = tb.Frame(frame)
        entry_row.pack(fill="x", pady=(6, 0))
        self.entry_vars = {}
        for c in cols:
            v = tk.StringVar()
            self.entry_vars[c] = v
            tb.Entry(entry_row, textvariable=v, width=7).pack(side="left", padx=1)
        tb.Label(frame, text="dir_x / dir_y は空欄で構いません（同じ要素の点から自動計算します）",
                 font=("", 8)).pack(anchor="w", pady=(2, 0))
        btn_row = tb.Frame(frame)
        btn_row.pack(fill="x", pady=(6, 0))
        tb.Button(btn_row, text="追加", command=self.on_add_row).pack(side="left")
        tb.Button(btn_row, text="選択行を削除", bootstyle="danger-outline",
                  command=self.on_delete_rows).pack(side="left", padx=4)
        tb.Button(btn_row, text="クリア", bootstyle="secondary-outline",
                  command=self.on_clear_rows).pack(side="left")
        btn_row2 = tb.Frame(frame)
        btn_row2.pack(fill="x", pady=(6, 0))
        tb.Button(btn_row2, text="CSVを読込…", command=self.on_load_csv).pack(side="left")
        tb.Button(btn_row2, text="CSVを保存…", command=self.on_save_csv).pack(side="left", padx=4)
        tb.Button(btn_row2, text="合成サンプル生成", bootstyle="info-outline",
                  command=self.on_make_sample).pack(side="left")

    def _build_estimate_controls(self, parent) -> None:
        frame = tb.LabelFrame(parent, text="推定", padding=8)
        frame.pack(fill="x")
        tb.Button(frame, text="推定を実行", bootstyle="success",
                  command=self.on_estimate).pack(fill="x")
        self.summary = tb.Label(frame, text="", wraplength=400, justify="left")
        self.summary.pack(fill="x", pady=(6, 0))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log(self, msg: str) -> None:
        self.log_text.text.insert("end", f"[{_now()}] {msg}\n")
        self.log_text.text.see("end")
        self.status.configure(text=msg)

    def error(self, msg: str) -> None:
        self.err_text.text.insert("end", f"[{_now()}] {msg}\n", "error")
        self.err_text.text.see("end")
        self.status.configure(text=msg)

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------
    def on_open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="画像を開く",
            filetypes=[("画像", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("すべて", "*.*")])
        if not path:
            return
        try:
            from PIL import Image
            self.pil_image = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            self.error(f"画像を開けませんでした: {exc}")
            return
        self.image_path = path
        self.image_label.configure(text=path.split("/")[-1])
        self.log(f"画像を読み込みました: {path}（{self.pil_image.width} x {self.pil_image.height}）")
        self._update_warp_buttons()

    def _update_warp_buttons(self) -> None:
        state = "normal" if (self.result is not None and self.pil_image is not None) else "disabled"
        self.btn_preview.configure(state=state)
        self.btn_save.configure(state=state)

    def _warped_image(self):
        return warp_image(self.pil_image, self.result.H)

    def on_preview(self) -> None:
        if self.result is None or self.pil_image is None:
            return
        try:
            warped = self._warped_image()
        except Exception as exc:  # noqa: BLE001
            self.error(f"補正画像の生成に失敗しました: {exc}")
            return
        from PIL import ImageTk
        disp = warped.copy()
        disp.thumbnail((900, 700))
        if self._preview_win is None or not self._preview_win.winfo_exists():
            self._preview_win = tb.Toplevel(self)
            self._preview_win.title("補正画像")
            self._preview_label = tb.Label(self._preview_win)
            self._preview_label.pack()
        self._preview_photo = ImageTk.PhotoImage(disp)
        self._preview_label.configure(image=self._preview_photo)
        self._preview_win.lift()
        self.log(f"補正画像を表示しました（θ = {self.result.theta_deg:.1f}°、"
                 f"元サイズ {warped.width} x {warped.height}）")

    def on_save_warped(self) -> None:
        if self.result is None or self.pil_image is None:
            return
        path = filedialog.asksaveasfilename(
            title="補正画像を保存", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return
        try:
            self._warped_image().save(path)
        except Exception as exc:  # noqa: BLE001
            self.error(f"補正画像を保存できませんでした: {exc}")
            return
        self.log(f"補正画像を保存しました: {path}")

    # ------------------------------------------------------------------
    # Measurement table
    # ------------------------------------------------------------------
    def _insert_row(self, values: dict) -> None:
        self.tree.insert("", "end", values=[values[c] for c in
                                            ("x", "y", "period", "element", "group", "dirx", "diry")])

    def on_add_row(self) -> None:
        vals = {c: v.get().strip() for c, v in self.entry_vars.items()}
        try:
            float(vals["x"]); float(vals["y"]); float(vals["period"])
            if float(vals["period"]) <= 0:
                raise ValueError
            if not vals["element"] or not vals["group"]:
                raise ValueError
            if (vals["dirx"] == "") != (vals["diry"] == ""):
                raise ValueError
            if vals["dirx"] != "":
                float(vals["dirx"]); float(vals["diry"])
        except ValueError:
            self.error("入力行が不正です。x, y, 周期px は数値で周期は正の値、要素と群は必須です。"
                       "dir_x / dir_y は空欄か、両方とも数値にしてください。")
            return
        self._insert_row(vals)
        self.sample_truth = None

    def on_delete_rows(self) -> None:
        for item in self.tree.selection():
            self.tree.delete(item)
        self.sample_truth = None

    def on_clear_rows(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.sample_truth = None
        self.result = None
        self.summary.configure(text="")
        self._update_warp_buttons()
        self.log("測定データをクリアしました。")

    def _rows(self) -> list[dict]:
        rows = []
        for item in self.tree.get_children():
            v = self.tree.item(item, "values")
            rows.append({"x": v[0], "y": v[1], "period": v[2], "element": v[3],
                         "group": v[4], "dirx": v[5], "diry": v[6]})
        return rows

    def on_load_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="測定データ CSV を読込",
            filetypes=[("CSV", "*.csv"), ("すべて", "*.*")])
        if not path:
            return
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                required = {"x", "y", "period", "element", "group"}
                if reader.fieldnames is None or not required.issubset(
                        {h.strip() for h in reader.fieldnames}):
                    raise ValueError("CSV のヘッダには x,y,period,element,group が必要です"
                                     "（dirx,diry は任意）")
                rows = []
                for row in reader:
                    r = {k.strip(): (v.strip() if isinstance(v, str) else v)
                         for k, v in row.items()}
                    r.setdefault("dirx", "")
                    r.setdefault("diry", "")
                    if r["dirx"] is None:
                        r["dirx"] = ""
                    if r["diry"] is None:
                        r["diry"] = ""
                    rows.append(r)
        except Exception as exc:  # noqa: BLE001
            self.error(f"CSV を読み込めませんでした: {exc}")
            return
        self.on_clear_rows()
        for r in rows:
            self._insert_row(r)
        self.log(f"CSV を読み込みました: {path}（{len(rows)} 点）")

    def on_save_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            title="測定データを CSV に保存", defaultextension=".csv",
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        rows = self._rows()
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["x", "y", "period", "element", "group", "dirx", "diry"])
                for r in rows:
                    w.writerow([r[c] for c in
                                ("x", "y", "period", "element", "group", "dirx", "diry")])
        except Exception as exc:  # noqa: BLE001
            self.error(f"CSV を保存できませんでした: {exc}")
            return
        self.log(f"CSV を保存しました: {path}（{len(rows)} 点）")

    def on_make_sample(self) -> None:
        try:
            pitch = float(self.pitch_var.get())
        except ValueError:
            self.error("ピッチ P が数値ではありません。")
            return
        scene = make_scene(angle_deg=float(self.theta_var.get()), pitch_mm=pitch)
        self.on_clear_rows()
        truth = {}
        for idx, m in enumerate(scene.measurements):
            period = pitch / m.resolution
            self._insert_row({
                "x": f"{m.x:.2f}", "y": f"{m.y:.2f}", "period": f"{period:.3f}",
                "element": m.element_id, "group": m.group_id,
                "dirx": f"{m.dir_x:.4f}", "diry": f"{m.dir_y:.4f}"})
            if idx in scene.plane_points:
                truth[(round(m.x, 2), round(m.y, 2))] = scene.plane_points[idx]
        self.sample_truth = truth
        self.log(f"合成サンプルを生成しました（{len(scene.measurements)} 点、"
                 f"真の角度 {scene.true_angle_deg:.0f}°、ノイズなし）。"
                 "「推定を実行」で真値との比較もログに出ます。")

    # ------------------------------------------------------------------
    # Theta slider
    # ------------------------------------------------------------------
    def _on_theta_changed(self, _value: str) -> None:
        self.theta_label.configure(text=f"{self.theta_var.get():.1f}°")

    def _on_theta_released(self) -> None:
        if self.auto_reestimate.get() and self.result is not None:
            self.on_estimate(quiet=True)

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------
    def _build_measurements(self) -> list[Measurement]:
        pitch = float(self.pitch_var.get())
        if pitch <= 0:
            raise ValueError("ピッチ P は正の値にしてください")
        raw = []
        for r in self._rows():
            period = float(r["period"])
            if period <= 0:
                raise ValueError("周期px は正の値にしてください")
            has_dir = str(r["dirx"]).strip() != "" and str(r["diry"]).strip() != ""
            raw.append({
                "x": float(r["x"]), "y": float(r["y"]),
                "resolution": pitch / period,
                "element_id": str(r["element"]), "group_id": str(r["group"]),
                "dir_x": float(r["dirx"]) if has_dir else None,
                "dir_y": float(r["diry"]) if has_dir else None,
            })
        try:
            filled, counts = fill_directions(raw)
        except InsufficientDataError as exc:
            raise ValueError(str(exc)) from exc
        auto = counts["line"] + counts["vanishing_point"]
        if auto:
            self.log(f"測定方向を自動計算しました（要素の直線から {counts['line']} 点、"
                     f"消失点から {counts['vanishing_point']} 点、入力どおり {counts['given']} 点）")
        return [Measurement(
            x=f["x"], y=f["y"], resolution=f["resolution"],
            element_id=f["element_id"], group_id=f["group_id"],
            dir_x=f["dir_x"], dir_y=f["dir_y"]) for f in filled]

    def on_estimate(self, quiet: bool = False) -> None:
        try:
            measurements = self._build_measurements()
        except ValueError as exc:
            self.error(f"入力エラー: {exc}")
            return
        if not measurements:
            self.error("測定データが空です。行を追加するか、CSV 読込か合成サンプル生成を使ってください。")
            return
        theta = float(self.theta_var.get())
        try:
            result = estimate_homography(measurements, theta_deg=theta)
        except InsufficientDataError as exc:
            self.error(f"入力不足のため推定できません: {exc}")
            self.result = None
            self._update_warp_buttons()
            return
        except Exception as exc:  # noqa: BLE001
            self.error(f"推定に失敗しました: {exc}")
            self.result = None
            self._update_warp_buttons()
            return
        self.result = result
        H = result.H
        if not quiet:
            self.log(f"推定完了（θ = {theta:.1f}°、測定点 {len(measurements)} 点）")
            self.log("H =\n" + "\n".join(
                "  [" + ", ".join(f"{v: .6f}" for v in row) + "]" for row in H))
        self.log(f"残差 RMS（相対）= {result.rms_relative_residual:.4%}、"
                 f"最大 = {result.max_relative_residual:.4%}")
        for gid, rms in sorted(result.per_group_rms.items()):
            self.log(f"  方向群 {gid}: 残差 RMS = {rms:.4%}")
        self.log(f"制約ヤコビアン: ランク = {result.jacobian_rank} / 8、"
                 f"条件数 = {result.jacobian_condition:.3g}")
        if not result.identifiable:
            self.error("注意: ランクが 5 未満です。この入力では H が十分に決まっていない"
                       "可能性があります（仕様 §7.3）。測定点の分布を見直してください。")
        self._log_truth_comparison(measurements)
        self.summary.configure(
            text=f"θ = {theta:.1f}° / 残差 RMS {result.rms_relative_residual:.3%} / "
                 f"ランク {result.jacobian_rank} / 条件数 {result.jacobian_condition:.3g}")
        self._update_warp_buttons()
        if self._preview_win is not None and self._preview_win.winfo_exists() \
                and self.pil_image is not None:
            self.on_preview()

    def _log_truth_comparison(self, measurements: list[Measurement]) -> None:
        """For synthetic sample data only: distance errors vs ground truth."""
        if not self.sample_truth or self.result is None:
            return
        import numpy as np
        from .core import apply_homography
        est, tru = [], []
        for m in measurements:
            key = (round(m.x, 2), round(m.y, 2))
            if key in self.sample_truth:
                est.append(apply_homography(self.result.H, m.q))
                tru.append(self.sample_truth[key])
        if len(est) < 3:
            return
        est = np.array(est)
        tru = np.array(tru)

        def pairwise(P):
            d = P[:, None, :] - P[None, :, :]
            return np.linalg.norm(d, axis=-1)
        err = np.abs(pairwise(est) - pairwise(tru))
        iu = np.triu_indices(len(est), k=1)
        self.log(f"（参考）合成サンプルの真値との点間距離誤差: "
                 f"平均 {err[iu].mean():.3f} mm、最大 {err[iu].max():.3f} mm")


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
