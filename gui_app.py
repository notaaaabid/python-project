import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import sys

# ── Lazy imports for speed ─────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── Our modules ────────────────────────────────────────────────────────────────
from data_loader   import load_dataset
from preprocessor  import preprocess, encode_single_record
from model_trainer import train_and_evaluate, predict_single
from analytics     import (plot_disorder_distribution, plot_sleep_duration,
                            plot_correlation_heatmap, plot_confusion_matrix,
                            plot_feature_importance, plot_model_comparison,
                            plot_stress_vs_quality, plot_probability_gauge)

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (dark clinical theme)
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":        "#1A1F2E",
    "card":      "#242938",
    "panel":     "#2E3449",
    "accent":    "#4C9BE8",
    "accent2":   "#4CE89B",
    "danger":    "#E85C4C",
    "warning":   "#F0A500",
    "text":      "#E8ECF4",
    "subtext":   "#8A92A8",
    "border":    "#3A4055",
    "btn_hover": "#5AAFE0",
}

FONT_TITLE  = ("Segoe UI", 18, "bold")
FONT_HEAD   = ("Segoe UI", 12, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Courier New", 9)


# ─────────────────────────────────────────────────────────────────────────────
class SleepApneaApp(tk.Tk):
    """Root application window."""

    def __init__(self):
        super().__init__()
        self.title("Sleep Apnea & Sleep Quality Analyser")
        self.geometry("1100x750")
        self.minsize(900, 620)
        self.configure(bg=C["bg"])

        # State shared across tabs
        self.df             = None
        self.model          = None
        self.scaler         = None
        self.label_encoders = None
        self.feature_names  = None
        self.train_results  = None

        self._setup_styles()
        self._build_header()
        self._build_notebook()
        self._build_statusbar()

        # Kick off training in background so the GUI appears instantly
        self._status("Loading dataset and training model…")
        threading.Thread(target=self._init_pipeline, daemon=True).start()

    # ── Styles ────────────────────────────────────────────────────────────────
    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", background=C["bg"], foreground=C["text"],
                        font=FONT_BODY)
        style.configure("TNotebook",        background=C["bg"], borderwidth=0)
        style.configure("TNotebook.Tab",    background=C["card"],
                        foreground=C["subtext"], padding=[14, 6],
                        font=FONT_BODY)
        style.map("TNotebook.Tab",
                  background=[("selected", C["accent"])],
                  foreground=[("selected", "#ffffff")])

        style.configure("TFrame",    background=C["bg"])
        style.configure("Card.TFrame", background=C["card"],
                        relief="flat", borderwidth=0)

        style.configure("TLabel",    background=C["bg"],  foreground=C["text"])
        style.configure("Card.TLabel", background=C["card"], foreground=C["text"])
        style.configure("Sub.TLabel", background=C["card"], foreground=C["subtext"],
                        font=FONT_SMALL)

        style.configure("TCombobox", fieldbackground=C["panel"],
                        background=C["panel"], foreground=C["text"],
                        selectbackground=C["accent"], arrowcolor=C["accent"])
        style.map("TCombobox", fieldbackground=[("readonly", C["panel"])])

        style.configure("Accent.TButton", background=C["accent"],
                        foreground="#ffffff", font=("Segoe UI", 10, "bold"),
                        padding=[12, 6], relief="flat")
        style.map("Accent.TButton",
                  background=[("active", C["btn_hover"])])

        style.configure("Danger.TButton", background=C["danger"],
                        foreground="#ffffff", font=("Segoe UI", 10, "bold"),
                        padding=[12, 6], relief="flat")

        style.configure("TScrollbar", background=C["panel"],
                        troughcolor=C["bg"], arrowcolor=C["accent"])

    # ── Header bar ────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=C["card"], height=60)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        icon_lbl = tk.Label(hdr, text="🫁", font=("Segoe UI", 22),
                            bg=C["card"], fg=C["accent"])
        icon_lbl.pack(side="left", padx=(18, 6), pady=8)

        tk.Label(hdr, text="Sleep Apnea & Sleep Quality Analyser",
                 font=FONT_TITLE, bg=C["card"], fg=C["text"]).pack(
            side="left", pady=8)

        tk.Label(hdr, text="Powered by Machine Learning",
                 font=FONT_SMALL, bg=C["card"], fg=C["subtext"]).pack(
            side="right", padx=18)

    # ── Notebook ──────────────────────────────────────────────────────────────
    def _build_notebook(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=(6, 0))

        self.tab_predict  = ttk.Frame(self.nb)
        self.tab_analytics = ttk.Frame(self.nb)
        self.tab_model    = ttk.Frame(self.nb)
        self.tab_about    = ttk.Frame(self.nb)

        self.nb.add(self.tab_predict,   text="  🔍 Predict  ")
        self.nb.add(self.tab_analytics, text="  📊 Analytics  ")
        self.nb.add(self.tab_model,     text="  🤖 Model  ")
        self.nb.add(self.tab_about,     text="  ℹ️ About  ")

        self._build_predict_tab()
        self._build_analytics_tab()
        self._build_model_tab()
        self._build_about_tab()

    # ── Status bar ────────────────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = tk.Frame(self, bg=C["card"], height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self._status_var = tk.StringVar(value="Initialising…")
        tk.Label(bar, textvariable=self._status_var,
                 font=FONT_SMALL, bg=C["card"], fg=C["subtext"],
                 anchor="w").pack(side="left", padx=12)

    def _status(self, msg: str):
        self._status_var.set(msg)
        self.update_idletasks()

    # ─────────────────────────────────────────────────────────────────────────
    #  PREDICT TAB
    # ─────────────────────────────────────────────────────────────────────────
    def _build_predict_tab(self):
        outer = self.tab_predict
        outer.configure(style="TFrame")

        # ── Left panel: input form ────────────────────────────────────────────
        left = tk.Frame(outer, bg=C["card"], padx=20, pady=16)
        left.pack(side="left", fill="y", padx=(10, 5), pady=10)

        tk.Label(left, text="Patient Parameters",
                 font=FONT_HEAD, bg=C["card"], fg=C["accent"]).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        self._inputs = {}

        # Fields: (label, widget_type, options)
        # widget_type: "combo" | "entry" | "scale"
        fields = [
            ("Gender",                  "combo",  ["Male", "Female"]),
            ("Age",                     "entry",  ""),
            ("Occupation",              "combo",
             ["Nurse","Doctor","Engineer","Lawyer","Teacher",
              "Accountant","Salesperson","Software Engineer","Scientist","Manager"]),
            ("Sleep Duration (hrs)",    "entry",  ""),
            ("Quality of Sleep (1-9)",  "scale",  ""),
            ("Physical Activity (min)", "entry",  ""),
            ("Stress Level (1-9)",      "scale",  ""),
            ("BMI Category",            "combo",
             ["Normal","Normal Weight","Overweight","Obese"]),
            ("Blood Pressure",          "entry",  ""),
            ("Heart Rate (bpm)",        "entry",  ""),
            ("Daily Steps",             "entry",  ""),
        ]

        # Placeholder hint text for non-scale entry fields
        _placeholders = {
            "Age":                     "e.g. 35",
            "Sleep Duration (hrs)":    "e.g. 7.5",
            "Physical Activity (min)": "e.g. 45",
            "Blood Pressure":          "e.g. 120/80",
            "Heart Rate (bpm)":        "e.g. 72",
            "Daily Steps":             "e.g. 8000",
        }

        for i, (label, wtype, opts) in enumerate(fields, start=1):
            tk.Label(left, text=label, font=FONT_SMALL,
                     bg=C["card"], fg=C["text"], anchor="e", width=24).grid(
                row=i, column=0, sticky="e", pady=3, padx=(0, 8))

            if wtype == "combo":
                var = tk.StringVar(value=opts[0])
                w = ttk.Combobox(left, textvariable=var, values=opts,
                                 state="readonly", width=18, font=FONT_SMALL)
                w.grid(row=i, column=1, sticky="w", pady=3)

            elif wtype == "scale":
                # Integer slider 1–9 with live value readout
                var = tk.IntVar(value=5)
                row_frame = tk.Frame(left, bg=C["card"])
                row_frame.grid(row=i, column=1, sticky="w", pady=3)
                val_lbl = tk.Label(row_frame, text="5", font=FONT_SMALL,
                                   bg=C["card"], fg=C["accent"], width=2)
                val_lbl.pack(side="right", padx=(6, 0))
                scale = tk.Scale(
                    row_frame, from_=1, to=9, orient="horizontal",
                    variable=var, showvalue=False, length=140,
                    bg=C["card"], fg=C["text"], troughcolor=C["panel"],
                    activebackground=C["accent"], highlightthickness=0,
                    bd=0, sliderlength=16, sliderrelief="flat",
                    command=lambda v, lbl=val_lbl: lbl.config(text=str(int(float(v))))
                )
                scale.pack(side="left")
                w = scale

            else:  # plain entry
                var = tk.StringVar(value=opts)
                w = tk.Entry(left, textvariable=var, width=20, font=FONT_SMALL,
                             bg=C["panel"], fg=C["text"],
                             insertbackground=C["text"],
                             relief="flat", highlightthickness=1,
                             highlightbackground=C["border"],
                             highlightcolor=C["accent"])
                hint = _placeholders.get(label, "")
                if hint:
                    if not var.get():
                        var.set(hint)
                        w.config(fg=C["subtext"])
                    def _on_focus_in(e, v=var, h=hint, widget=w):
                        if v.get() == h:
                            v.set("")
                            widget.config(fg=C["text"])
                    def _on_focus_out(e, v=var, h=hint, widget=w):
                        if not v.get():
                            v.set(h)
                            widget.config(fg=C["subtext"])
                    w.bind("<FocusIn>",  _on_focus_in)
                    w.bind("<FocusOut>", _on_focus_out)
                w.grid(row=i, column=1, sticky="w", pady=3)

            self._inputs[label] = var

        # Buttons
        btn_frame = tk.Frame(left, bg=C["card"])
        btn_frame.grid(row=len(fields)+1, column=0, columnspan=2,
                       pady=(16, 4), sticky="ew")
        ttk.Button(btn_frame, text="🔍  Predict",
                   style="Accent.TButton",
                   command=self._run_prediction).pack(side="left", padx=(0, 8))
        ttk.Button(btn_frame, text="↺  Reset",
                   command=self._reset_form).pack(side="left")

        # Load CSV button
        ttk.Button(left, text="📂  Load Custom CSV",
                   command=self._load_csv).grid(
            row=len(fields)+2, column=0, columnspan=2,
            pady=(6, 0), sticky="w")

        # ── Right panel: results ──────────────────────────────────────────────
        right = tk.Frame(outer, bg=C["bg"])
        right.pack(side="right", fill="both", expand=True,
                   padx=(5, 10), pady=10)

        # ── Top row: two cards side by side ─────────────────────────────────
        top_row = tk.Frame(right, bg=C["bg"])
        top_row.pack(fill="x", pady=(0, 6))

        # Card 1 – Sleep Disorder prediction
        disorder_card = tk.Frame(top_row, bg=C["card"], padx=16, pady=14,
                                 highlightbackground=C["border"], highlightthickness=1)
        disorder_card.pack(side="left", fill="both", expand=True, padx=(0, 4))

        tk.Label(disorder_card, text="Sleep Disorder",
                 font=FONT_SMALL, bg=C["card"], fg=C["subtext"]).pack(anchor="w")
        self._pred_label = tk.Label(disorder_card, text="—",
                                    font=("Segoe UI", 20, "bold"),
                                    bg=C["card"], fg=C["text"])
        self._pred_label.pack(anchor="w", pady=(2, 0))
        self._risk_badge = tk.Label(disorder_card, text="",
                                    font=("Segoe UI", 10, "bold"),
                                    bg=C["card"], fg=C["subtext"])
        self._risk_badge.pack(anchor="w")

        # Card 2 – Sleep Quality prediction
        quality_card = tk.Frame(top_row, bg=C["card"], padx=16, pady=14,
                                highlightbackground=C["border"], highlightthickness=1)
        quality_card.pack(side="left", fill="both", expand=True, padx=(4, 0))

        tk.Label(quality_card, text="Sleep Quality",
                 font=FONT_SMALL, bg=C["card"], fg=C["subtext"]).pack(anchor="w")
        self._quality_label = tk.Label(quality_card, text="—",
                                       font=("Segoe UI", 20, "bold"),
                                       bg=C["card"], fg=C["text"])
        self._quality_label.pack(anchor="w", pady=(2, 0))
        self._quality_score = tk.Label(quality_card, text="",
                                       font=("Segoe UI", 10, "bold"),
                                       bg=C["card"], fg=C["subtext"])
        self._quality_score.pack(anchor="w")

        # Recommendation text
        rec_frame = tk.Frame(right, bg=C["card"], padx=16, pady=10)
        rec_frame.pack(fill="x", pady=(0, 6))
        self._rec_text = tk.Label(rec_frame, text="",
                                  font=FONT_SMALL, bg=C["card"],
                                  fg=C["subtext"], wraplength=420,
                                  justify="left")
        self._rec_text.pack(anchor="w")

        # Chart canvas (probability gauge)
        self._chart_frame_predict = tk.Frame(right, bg=C["bg"])
        self._chart_frame_predict.pack(fill="both", expand=True)

    def _reset_form(self):
        combo_defaults = {
            "Gender":       "Male",
            "Occupation":   "Nurse",
            "BMI Category": "Normal",
        }
        placeholders = {
            "Age":                     "e.g. 35",
            "Sleep Duration (hrs)":    "e.g. 7.5",
            "Physical Activity (min)": "e.g. 45",
            "Blood Pressure":          "e.g. 120/80",
            "Heart Rate (bpm)":        "e.g. 72",
            "Daily Steps":             "e.g. 8000",
        }
        scale_fields = {"Quality of Sleep (1-9)", "Stress Level (1-9)"}

        for label, var in self._inputs.items():
            if label in combo_defaults:
                var.set(combo_defaults[label])
            elif label in scale_fields:
                var.set(5)          # reset slider to midpoint
            elif label in placeholders:
                var.set(placeholders[label])  # show hint text again
            else:
                var.set("")

        # Clear result panel
        self._pred_label.config(text="—", fg=C["text"])
        self._risk_badge.config(text="", fg=C["subtext"])
        self._quality_label.config(text="—", fg=C["text"])
        self._quality_score.config(text="", fg=C["subtext"])
        self._rec_text.config(text="")

        # Clear probability chart
        for w in self._chart_frame_predict.winfo_children():
            w.destroy()

    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="Load Sleep Health CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        # Reset model so UI shows "training…" while re-training
        self.model  = None
        self.scaler = None
        self.label_encoders = None
        # Delete cached dataset & model so the new CSV is actually used
        for cache in ("/tmp/sleep_health_dataset.csv",
                      "/tmp/sleep_best_model.pkl",
                      "/tmp/sleep_scaler.pkl",
                      "/tmp/sleep_label_encoders.pkl"):
            if os.path.isfile(cache):
                os.remove(cache)
        self._status(f"Loading {os.path.basename(path)}…")
        threading.Thread(
            target=self._init_pipeline, args=(path,), daemon=True
        ).start()

    def _run_prediction(self):
        if self.model is None:
            messagebox.showinfo("Please wait", "Model is still training…")
            return

        raw = {k: v.get() for k, v in self._inputs.items()}

        # Strip placeholder hint text from entry fields so they count as blank
        placeholders = {
            "Age":                     "e.g. 35",
            "Sleep Duration (hrs)":    "e.g. 7.5",
            "Physical Activity (min)": "e.g. 45",
            "Blood Pressure":          "e.g. 120/80",
            "Heart Rate (bpm)":        "e.g. 72",
            "Daily Steps":             "e.g. 8000",
        }
        for field, hint in placeholders.items():
            val = raw.get(field, "")
            if isinstance(val, str) and val.strip() == hint:
                raw[field] = ""

        # ── Step 1: check entry fields are not blank (sliders are always valid) ─
        field_hints = {
            "Age":                     "Age  (e.g. 35)",
            "Sleep Duration (hrs)":    "Sleep Duration  (e.g. 7.0)",
            "Physical Activity (min)": "Physical Activity  (e.g. 45)",
            "Blood Pressure":          "Blood Pressure  (e.g. 120/80)",
            "Heart Rate (bpm)":        "Heart Rate  (e.g. 72)",
            "Daily Steps":             "Daily Steps  (e.g. 8000)",
        }
        for field, hint in field_hints.items():
            if not raw.get(field, ""):
                messagebox.showerror("Missing Input", f"Please fill in:\n{hint}")
                return

        # ── Step 2: blood pressure must contain a slash ───────────────────────
        bp_val = raw["Blood Pressure"]
        if "/" not in bp_val:
            messagebox.showerror(
                "Blood Pressure Format",
                f"Got: \"{bp_val}\"\n\nPlease use the format  systolic/diastolic\nExample: 120/80"
            )
            return

        # ── Step 3: parse all numeric fields ─────────────────────────────────
        try:
            record = {
                "Gender":                  raw["Gender"],
                "Age":                     int(float(raw["Age"])),
                "Occupation":              raw["Occupation"],
                "Sleep Duration":          float(raw["Sleep Duration (hrs)"]),
                "Quality of Sleep":        int(raw["Quality of Sleep (1-9)"]),
                "Physical Activity Level": int(float(raw["Physical Activity (min)"])),
                "Stress Level":            int(raw["Stress Level (1-9)"]),
                "BMI Category":            raw["BMI Category"],
                "Blood Pressure":          bp_val,
                "Heart Rate":              int(float(raw["Heart Rate (bpm)"])),
                "Daily Steps":             int(float(raw["Daily Steps"])),
            }
        except ValueError as e:
            messagebox.showerror("Invalid Number", f"Could not parse a value:\n{e}")
            return

        # ── Step 4: run model ─────────────────────────────────────────────────
        try:
            X = encode_single_record(record, self.scaler, self.label_encoders)
            result = predict_single(self.model, X, self.label_encoders["__target__"])
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            return

        # ── Update result widgets ─────────────────────────────────────────────
        label = result["label"]
        risk  = result["risk_level"]
        probs = result["probabilities"]

        risk_colors = {"Low": C["accent2"], "Moderate": C["warning"], "High": C["danger"]}
        disorder_colors = {"None": C["accent2"], "Sleep Apnea": C["danger"],
                           "Insomnia": C["warning"]}

        # Sleep disorder card
        self._pred_label.config(text=label, fg=disorder_colors.get(label, C["text"]))
        self._risk_badge.config(
            text=f"Risk Level: {risk}  ●",
            fg=risk_colors.get(risk, C["text"])
        )

        # ── Sleep quality rating (Good / Average / Poor) ─────────────────────
        try:
            sleep_dur   = float(raw["Sleep Duration (hrs)"])
            stress      = int(float(raw["Stress Level (1-9)"]))
            phys_act    = int(float(raw["Physical Activity (min)"]))
            daily_steps = int(float(raw["Daily Steps"]))
            bmi         = raw["BMI Category"]

            # Count how many factors are in a "bad" zone
            bad = 0
            good = 0

            # Sleep duration: 7-9 hrs is ideal
            if sleep_dur < 6 or sleep_dur > 9:
                bad += 2
            elif sleep_dur < 7:
                bad += 1
            else:
                good += 1

            # Stress: 1-4 good, 5-6 average, 7+ bad
            if stress >= 7:
                bad += 2
            elif stress >= 5:
                bad += 1
            else:
                good += 1

            # Physical activity: 45+ good, 30-44 average, <30 bad
            if phys_act >= 45:
                good += 1
            elif phys_act < 30:
                bad += 1

            # Daily steps: 7000+ good, 4000-6999 average, <4000 bad
            if daily_steps >= 7000:
                good += 1
            elif daily_steps < 4000:
                bad += 1

            # BMI
            if bmi in ("Normal", "Normal Weight"):
                good += 1
            elif bmi == "Obese":
                bad += 2
            elif bmi == "Overweight":
                bad += 1

            # Disorder
            if label == "Sleep Apnea":
                bad += 3
            elif label == "Insomnia":
                bad += 2

            # Decide rating
            if bad == 0 or (good >= 3 and bad <= 1):
                q_label, q_color = "Good 😊",    C["accent2"]
            elif bad >= 4:
                q_label, q_color = "Poor 😟",    C["danger"]
            else:
                q_label, q_color = "Average 😐", C["warning"]

            self._quality_label.config(text=q_label, fg=q_color)
            self._quality_score.config(text="", fg=q_color)
        except Exception:
            self._quality_label.config(text="—", fg=C["text"])
            self._quality_score.config(text="", fg=C["subtext"])

        # Recommendation
        recs = {
            "None":        "✅  Your sleep profile looks healthy. Maintain your current lifestyle.",
            "Sleep Apnea": "⚠️  High risk of Sleep Apnea. Consider consulting a sleep specialist. "
                           "Weight management and positional therapy may help.",
            "Insomnia":    "💤  Signs of Insomnia detected. Reduce stress, maintain a consistent "
                           "sleep schedule, and limit screen time before bed.",
        }
        self._rec_text.config(text=recs.get(label, ""))

        # Draw probability gauge
        for w in self._chart_frame_predict.winfo_children():
            w.destroy()
        fig = plot_probability_gauge(probs)
        canvas = FigureCanvasTkAgg(fig, master=self._chart_frame_predict)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        self._status(f"Prediction: {label}  |  Risk: {risk}  |  Sleep Quality: {q_label}")

    # ─────────────────────────────────────────────────────────────────────────
    #  ANALYTICS TAB
    # ─────────────────────────────────────────────────────────────────────────
    def _build_analytics_tab(self):
        tab = self.tab_analytics

        # Top controls
        ctrl = tk.Frame(tab, bg=C["card"], pady=8)
        ctrl.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(ctrl, text="Dataset Analytics", font=FONT_HEAD,
                 bg=C["card"], fg=C["accent"]).pack(side="left", padx=12)

        charts = [
            ("Disorder Distribution", "disorder"),
            ("Sleep Duration",        "duration"),
            ("Correlation Heatmap",   "heatmap"),
            ("Stress vs Quality",     "stress"),
        ]
        self._analytics_choice = tk.StringVar(value="disorder")
        for label, val in charts:
            ttk.Radiobutton(ctrl, text=label, variable=self._analytics_choice,
                            value=val,
                            command=self._refresh_analytics).pack(
                side="left", padx=6)

        ttk.Button(ctrl, text="↺ Refresh", style="Accent.TButton",
                   command=self._refresh_analytics).pack(side="right", padx=12)

        # Canvas area
        self._analytics_canvas_frame = tk.Frame(tab, bg=C["bg"])
        self._analytics_canvas_frame.pack(fill="both", expand=True, padx=10, pady=4)
        tk.Label(self._analytics_canvas_frame,
                 text="Training in progress…",
                 font=FONT_BODY, bg=C["bg"], fg=C["subtext"]).pack(pady=40)

    def _refresh_analytics(self):
        if self.df is None:
            return
        choice = self._analytics_choice.get()
        fig_map = {
            "disorder": lambda: plot_disorder_distribution(self.df),
            "duration": lambda: plot_sleep_duration(self.df),
            "heatmap":  lambda: plot_correlation_heatmap(self.df),
            "stress":   lambda: plot_stress_vs_quality(self.df),
        }
        fig = fig_map[choice]()
        self._embed_chart(fig, self._analytics_canvas_frame)

    # ─────────────────────────────────────────────────────────────────────────
    #  MODEL TAB
    # ─────────────────────────────────────────────────────────────────────────
    def _build_model_tab(self):
        tab = self.tab_model

        ctrl = tk.Frame(tab, bg=C["card"], pady=8)
        ctrl.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(ctrl, text="Model Evaluation", font=FONT_HEAD,
                 bg=C["card"], fg=C["accent"]).pack(side="left", padx=12)

        views = [
            ("Confusion Matrix", "cm"),
            ("Feature Importance", "fi"),
            ("Model Comparison",  "compare"),
            ("Report",           "report"),
        ]
        self._model_choice = tk.StringVar(value="cm")
        for label, val in views:
            ttk.Radiobutton(ctrl, text=label, variable=self._model_choice,
                            value=val, command=self._refresh_model_tab).pack(
                side="left", padx=6)

        # Accuracy badge
        self._acc_var = tk.StringVar(value="Accuracy: —")
        tk.Label(ctrl, textvariable=self._acc_var, font=("Segoe UI", 10, "bold"),
                 bg=C["card"], fg=C["accent2"]).pack(side="right", padx=12)

        # Main content area (left=chart, right=report)
        self._model_content = tk.Frame(tab, bg=C["bg"])
        self._model_content.pack(fill="both", expand=True, padx=10, pady=4)
        tk.Label(self._model_content,
                 text="Training in progress…",
                 font=FONT_BODY, bg=C["bg"], fg=C["subtext"]).pack(pady=40)

    def _refresh_model_tab(self):
        if self.train_results is None:
            return
        choice = self._model_choice.get()
        tr = self.train_results

        for w in self._model_content.winfo_children():
            w.destroy()

        if choice == "report":
            # Scrollable text widget
            txt = tk.Text(self._model_content, bg=C["card"], fg=C["text"],
                          font=FONT_MONO, relief="flat", padx=12, pady=8)
            sb = ttk.Scrollbar(self._model_content, command=txt.yview)
            txt.configure(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y")
            txt.pack(fill="both", expand=True)
            txt.insert("end", f"Best Model: {tr['best_model_name']}\n")
            txt.insert("end", f"Test Accuracy: {tr['accuracy']*100:.2f}%\n\n")
            txt.insert("end", tr["report"])
            txt.configure(state="disabled")
        else:
            fig_map = {
                "cm":      lambda: plot_confusion_matrix(
                               tr["confusion_matrix"], tr["classes"]),
                "fi":      lambda: (plot_feature_importance(tr["feature_importance"])
                                    if tr["feature_importance"] is not None
                                    else plot_model_comparison(tr["all_scores"])),
                "compare": lambda: plot_model_comparison(tr["all_scores"]),
            }
            fig = fig_map[choice]()
            self._embed_chart(fig, self._model_content)

    # ─────────────────────────────────────────────────────────────────────────
    #  ABOUT TAB
    # ─────────────────────────────────────────────────────────────────────────
    def _build_about_tab(self):
        tab = self.tab_about
        canvas = tk.Canvas(tab, bg=C["bg"], highlightthickness=0)
        sb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)

        frame = tk.Frame(canvas, bg=C["bg"])
        canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>",
                   lambda e: canvas.configure(
                       scrollregion=canvas.bbox("all")))

        modules = [
            ("📦 data_loader.py",
             "Generates or loads the Sleep Health and Lifestyle Dataset.\n"
             "Creates 400 synthetic records that mirror the Kaggle dataset's "
             "distributions and cross-variable correlations. Falls back to a "
             "user-supplied CSV if provided."),
            ("⚙️  preprocessor.py",
             "Cleans and transforms raw data into ML-ready features.\n"
             "Parses blood pressure strings, label-encodes categorical columns "
             "(Gender, Occupation, BMI Category), standardises numeric features "
             "with StandardScaler, and handles unseen labels gracefully during "
             "inference."),
            ("🤖 model_trainer.py",
             "Trains four classifiers: Random Forest, Gradient Boosting, "
             "Logistic Regression, and SVM.\n"
             "Picks the best by test accuracy, evaluates it with a classification "
             "report and confusion matrix, extracts feature importances, and "
             "persists the model to disk with joblib."),
            ("📊 analytics.py",
             "Produces all matplotlib/seaborn charts embedded in the GUI.\n"
             "Disorder distribution pie, sleep-duration histogram, correlation "
             "heatmap, stress-vs-quality scatter, confusion matrix, feature "
             "importance bar chart, model comparison, and probability gauge."),
            ("🖥️  gui_app.py",
             "Main Tkinter application tying all modules together.\n"
             "Four tabs: Predict (form → inference), Analytics (dataset charts), "
             "Model (evaluation metrics), and About (this page). Uses threading "
             "to keep the UI responsive during training."),
        ]

        tk.Label(frame, text="How This Application Works",
                 font=FONT_TITLE, bg=C["bg"], fg=C["accent"]).pack(
            anchor="w", padx=24, pady=(20, 4))
        tk.Label(frame,
                 text="Each Python file is an independent, well-documented module. "
                      "Learn one module at a time!",
                 font=FONT_BODY, bg=C["bg"], fg=C["subtext"]).pack(
            anchor="w", padx=24, pady=(0, 16))

        for title, desc in modules:
            card = tk.Frame(frame, bg=C["card"], padx=18, pady=14)
            card.pack(fill="x", padx=24, pady=5)
            tk.Label(card, text=title, font=FONT_HEAD,
                     bg=C["card"], fg=C["accent"]).pack(anchor="w")
            tk.Label(card, text=desc, font=FONT_BODY,
                     bg=C["card"], fg=C["text"],
                     wraplength=700, justify="left").pack(
                anchor="w", pady=(4, 0))

        tk.Label(frame,
                 text="Dataset: Sleep Health and Lifestyle Dataset (Kaggle) — "
                      "400 records, 13 variables",
                 font=FONT_SMALL, bg=C["bg"], fg=C["subtext"]).pack(
            anchor="w", padx=24, pady=(14, 20))

    # ─────────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _embed_chart(self, fig, container: tk.Frame):
        for w in container.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  PIPELINE (runs in background thread)
    # ─────────────────────────────────────────────────────────────────────────
    def _init_pipeline(self, csv_path: str | None = None):
        try:
            self.df = load_dataset(csv_path)
            self._status("Preprocessing…")

            X_tr, X_te, y_tr, y_te, self.scaler, self.label_encoders, \
                self.feature_names = preprocess(self.df)

            self._status("Training models (this may take ~10 seconds)…")
            self.train_results = train_and_evaluate(
                X_tr, X_te, y_tr, y_te,
                self.feature_names,
                self.label_encoders["__target__"]
            )
            self.model = self.train_results["best_model"]

            # Update UI from main thread
            self.after(0, self._on_training_complete)

        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Pipeline Error", str(exc)))
            self.after(0, lambda: self._status(f"Error: {exc}"))

    def _on_training_complete(self):
        tr = self.train_results
        acc = tr["accuracy"] * 100
        self._acc_var.set(f"Accuracy: {acc:.1f}%")
        self._status(
            f"Ready  |  Best model: {tr['best_model_name']}  |  "
            f"Accuracy: {acc:.1f}%  |  Dataset: {len(self.df)} records"
        )
        self._refresh_analytics()
        self._refresh_model_tab()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SleepApneaApp()
    app.mainloop()