"""
Tkinter desktop app for the Bag-of-Words + TF-IDF text analyzer.

The UI only handles input, display and threading; all analysis lives in the
models package and the report layout lives in utils/report.py.
"""

import os

import queue
import re
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

from models import analyzer
from models.sentiment import SentimentModel
from utils.report import build_report, write_report

FILE_LIMIT = analyzer.MAX_FILES
PLACEHOLDER = (
    "Add up to 3 .txt files, then press Analyze.\n\n"
    "The report shows word and sentence counts, phrases, Bag-of-Words and\n"
    "TF-IDF terms, an extractive summary and the sentiment of each file."
)


class AnalyzerApp(tk.Tk):
    """Main window: file selection, analysis and the full report."""

    def __init__(self):
        super().__init__()
        self.title("NLP Text Analyzer - Bag-of-Words & TF-IDF")
        self.geometry("1150x780")
        self.minsize(880, 560)

        self.paths = []
        self.model = None
        self.report_text = ""
        self.queue = queue.Queue()
        self.model_ready = threading.Event()

        self._build_widgets()
        self._show_text(PLACEHOLDER)
        self.after(100, self._poll_queue)

        # Training reads 2,000 reviews, so it runs in the background while
        # the user is still choosing files.
        threading.Thread(target=self._train_model, daemon=True).start()

    # ------------------------------------------------------------------- layout

    def _build_widgets(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        files = ttk.LabelFrame(
            self, text=f"Input files (.txt only, maximum {FILE_LIMIT})", padding=8
        )
        files.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        files.columnconfigure(0, weight=1)

        self.file_list = tk.Listbox(files, height=4, selectmode="extended")
        self.file_list.grid(row=0, column=0, rowspan=3, sticky="ew")
        file_scroll = ttk.Scrollbar(files, orient="vertical", command=self.file_list.yview)
        file_scroll.grid(row=0, column=1, rowspan=3, sticky="ns")
        self.file_list.configure(yscrollcommand=file_scroll.set)

        ttk.Button(files, text="Add files", command=self.add_files).grid(
            row=0, column=2, padx=(8, 0), sticky="ew"
        )
        ttk.Button(files, text="Remove selected", command=self.remove_selected).grid(
            row=1, column=2, padx=(8, 0), pady=4, sticky="ew"
        )
        ttk.Button(files, text="Clear all", command=self.clear_files).grid(
            row=2, column=2, padx=(8, 0), sticky="ew"
        )

        actions = ttk.Frame(self)
        actions.grid(row=1, column=0, sticky="ew", padx=10)
        actions.columnconfigure(2, weight=1)

        self.analyze_button = ttk.Button(actions, text="Analyze", command=self.analyze)
        self.analyze_button.grid(row=0, column=0)
        self.save_button = ttk.Button(
            actions, text="Save report (.txt)", command=self.save_report, state="disabled"
        )
        self.save_button.grid(row=0, column=1, padx=6)

        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=160)
        self.progress.grid(row=0, column=2, padx=8, sticky="w")

        self.status = tk.StringVar(value="Starting…")
        ttk.Label(actions, textvariable=self.status).grid(row=0, column=3, sticky="e")

        output = ttk.LabelFrame(self, text="Output", padding=(6, 4))
        output.grid(row=2, column=0, sticky="nsew", padx=10, pady=(5, 10))
        output.columnconfigure(0, weight=1)
        output.rowconfigure(0, weight=1)

        self.output = tk.Text(
            output, wrap="none", font=("Consolas", 10), padx=8, pady=6, background="white"
        )
        self.output.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(output, orient="vertical", command=self.output.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(output, orient="horizontal", command=self.output.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.output.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.output.tag_configure("title", font=("Consolas", 10, "bold"))
        self.output.tag_configure(
            "heading", font=("Consolas", 10, "bold"), foreground="#1f4e79"
        )
        self.output.tag_configure("rule", foreground="#999999")
        self.output.configure(state="disabled")

    # -------------------------------------------------------------- file input

    def add_files(self):
        selected = filedialog.askopenfilenames(
            title="Select text files", filetypes=[("Text files", "*.txt")]
        )
        rejected = 0
        for path in selected:
            if not path.lower().endswith(".txt"):
                rejected += 1
                continue
            if path in self.paths:
                continue
            if len(self.paths) >= FILE_LIMIT:
                messagebox.showwarning(
                    "Limit reached", f"You can analyze at most {FILE_LIMIT} files."
                )
                break
            self.paths.append(path)

        if rejected:
            messagebox.showerror(
                "Unsupported file", "Only .txt files can be analyzed."
            )
        self._refresh_files()

    def remove_selected(self):
        for index in reversed(self.file_list.curselection()):
            self.paths.pop(index)
        self._refresh_files()

    def clear_files(self):
        self.paths.clear()
        self._refresh_files()

    def _refresh_files(self):
        self.file_list.delete(0, "end")
        for position, path in enumerate(self.paths, start=1):
            self.file_list.insert("end", f"{position}. {os.path.basename(path)}")
        if self.paths:
            self.status.set(f"{len(self.paths)} of {FILE_LIMIT} file(s) selected.")
        else:
            self.status.set("No files selected.")

    # ----------------------------------------------------------------- analysis

    def analyze(self):
        if not self.paths:
            messagebox.showinfo("No files", "Add at least one .txt file first.")
            return

        self.report_text = ""
        self.analyze_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.progress.start(12)
        self._status("Reading files…")
        self._show_text("Analyzing, please wait…")

        paths = list(self.paths)
        threading.Thread(target=self._run_analysis, args=(paths,), daemon=True).start()

    def _run_analysis(self, paths):
        try:
            if not self.model_ready.is_set():
                self._status("Waiting for the classifier to finish training…")
                self.model_ready.wait()
            result = analyzer.analyze_files(paths, self.model, status=self._status)
            self.queue.put(("report", build_report(result)))
        except Exception as exc:  # show the real reason in the UI
            self.queue.put(("error", str(exc)))

    def _train_model(self):
        try:
            self._status("Preparing sentiment classifier…")
            model = SentimentModel().load_or_train(status=self._status)
            self.model = model
            self._status(f"Classifier ready ({model.accuracy:.1%} cross-validated accuracy).")
        except Exception as exc:
            self._status("Classifier unavailable - sentiment will be skipped.")
            self.queue.put(("warning", str(exc)))
        finally:
            self.model_ready.set()

    # ------------------------------------------------------------------ saving

    def save_report(self):
        if not self.report_text:
            messagebox.showinfo("Nothing to save", "Run an analysis first.")
            return

        default_name = f"analysis_report_{datetime.now():%Y%m%d_%H%M%S}.txt"
        path = filedialog.asksaveasfilename(
            title="Save report",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text file", "*.txt")],
        )
        if not path:
            return

        try:
            write_report(self.report_text, path)
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        self.status.set(f"Saved: {os.path.basename(path)}")
        messagebox.showinfo("Report saved", f"Saved to:\n{path}")

    # ------------------------------------------------------------- UI plumbing

    def _status(self, message):
        """Called from worker threads; the queue keeps Tkinter single-threaded."""
        self.queue.put(("status", message))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "status":
                    self.status.set(payload)
                elif kind == "report":
                    self.report_text = payload
                    self._show_text(payload)
                    self.status.set("Analysis complete.")
                    self._finish()
                elif kind == "warning":
                    messagebox.showwarning("Sentiment classifier", payload)
                elif kind == "error":
                    messagebox.showerror("Analysis failed", payload)
                    self.status.set("Analysis failed.")
                    self._finish()
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _finish(self):
        self.progress.stop()
        self.analyze_button.configure(state="normal")
        if self.report_text:
            self.save_button.configure(state="normal")

    def _show_text(self, text):
        """Replace the output pane and colour the report's headings."""
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        for line in text.splitlines():
            if line.startswith("=") or line.startswith("-" * 20):
                tag = "rule"
            elif re.match(r"^\[\d+\]", line):
                tag = "title"
            elif re.match(r"^\d+\. [A-Z]", line):
                tag = "heading"
            else:
                tag = ""
            self.output.insert("end", line + "\n", tag)
        self.output.configure(state="disabled")
        self.output.yview_moveto(0)
        self.output.xview_moveto(0)
