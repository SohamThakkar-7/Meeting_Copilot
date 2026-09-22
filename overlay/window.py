import tkinter as tk
import queue
import threading


try:
    import win32con
    import win32gui

    _HAS_WIN32 = True
except ImportError:
    _HAS_WIN32 = False


class OverlayWindow:
    def __init__(self, width=420, height=140, margin=24):
        self._root = tk.Tk()

        self._root.overrideredirect(True)          # no title bar, no border
        self._root.attributes("-topmost", True)    # sit above everything
        self._root.attributes("-alpha", 0.92)      # slight transparency
        self._root.configure(bg="#12141a")

        self._fg = "#e8eaf0"
        self._dim = "#7c8496"

        self._label = tk.Label(
            self._root,
            text="",
            font=("Segoe UI", 12),
            fg=self._fg,
            bg="#12141a",
            wraplength=width - 32,
            justify="left",
            anchor="nw",
            padx=16,
            pady=12,
        )
        self._label.pack(fill="both", expand=True)

        # Worker threads post here; the UI thread drains it on a timer.
        self._queue = queue.Queue()
        self._tick_ms = 33          # ~30 checks/sec

        # Every suggestion gets a number. Tokens from a superseded one are
        # dropped at the consumer rather than chased down at the producer.
        self._generation = 0
        self._gen_lock = threading.Lock()

        self._hide_after_ms = 12_000
        self._hide_job = None

        self._closing = threading.Event()

        # Right-click to close. Keyboard bindings are useless once the window
        # is non-activating -- it can never receive a keypress.
        self._root.bind("<Button-3>", lambda _event: self.close())

        self._place(width, height, margin)
        self._make_non_activating()
        self._root.withdraw()        # start hidden

    def _make_non_activating(self):
        """Tell Windows this window must never take the foreground."""
        if not _HAS_WIN32:
            print("WARNING: pywin32 missing -- overlay WILL steal focus. "
                  "Are you running the venv Python?")
            return

        # winfo_id() is only valid once the window has actually been realized.
        self._root.update_idletasks()

        hwnd = self._root.winfo_id()
        parent = win32gui.GetParent(hwnd)
        if parent:
            hwnd = parent          # tkinter nests a frame inside the real window

        styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        styles |= win32con.WS_EX_NOACTIVATE   # clicking never gives us focus
        styles |= win32con.WS_EX_TOOLWINDOW   # stay out of Alt+Tab
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)

    def _place(self, width, height, margin):
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        x = screen_w - width - margin
        y = screen_h - height - margin * 4     # clear of the taskbar
        self._root.geometry(f"{width}x{height}+{x}+{y}")

    def begin(self):
        """Start a new suggestion, invalidating anything still streaming.
        Returns the generation id to pass to append() and finish()."""
        with self._gen_lock:
            self._generation += 1
            generation = self._generation
        self._queue.put((generation, "begin", ""))
        return generation

    def append(self, generation, text):
        self._queue.put((generation, "append", text))

    def finish(self, generation, note=""):
        self._queue.put((generation, "finish", note))

    def dismiss(self, generation):
        """Hide immediately -- the model had nothing worth saying.

        Distinct from finish(): finish leaves the suggestion up for its
        normal dwell time. An overlay that interrupts you to show nothing is
        worse than one that stays hidden.
        """
        self._queue.put((generation, "dismiss", ""))

    def close(self):
        """Safe from any thread. Ends run() on the next tick."""
        self._closing.set()

    def _drain(self):
        """Apply everything the worker threads posted since the last tick."""
        if self._closing.is_set():
            # quit() ends mainloop so the caller's cleanup runs. destroy()
            # would tear down the widgets and leave callbacks firing into
            # a dead interpreter.
            self._root.quit()
            return

        try:
            while True:
                generation, action, payload = self._queue.get_nowait()
                # A superseded suggestion still trickling out. Drop it.
                # Reading an int attribute is atomic under the GIL, so no
                # lock is needed here -- only the increment needs one.
                if generation != self._generation:
                    continue
                self._apply(action, payload)
        except queue.Empty:
            pass

        self._root.after(self._tick_ms, self._drain)

    def _apply(self, action, payload):
        if action == "begin":
            self._label.config(text="...", fg=self._dim)
            self._show()
        elif action == "append":
            current = self._label.cget("text")
            if current == "...":
                current = ""
            self._label.config(text=current + payload, fg=self._fg)
        elif action == "finish":
            if payload:
                self._label.config(text=payload, fg=self._dim)
            self._schedule_hide()
        elif action == "dismiss":
            if self._hide_job is not None:
                self._root.after_cancel(self._hide_job)
                self._hide_job = None
            self._label.config(text="")
            self._root.withdraw()

    def _show(self):
        if self._hide_job is not None:
            self._root.after_cancel(self._hide_job)
            self._hide_job = None
        self._root.deiconify()
        self._root.attributes("-topmost", True)   # deiconify can drop this

    def _schedule_hide(self):
        if self._hide_job is not None:
            self._root.after_cancel(self._hide_job)
        self._hide_job = self._root.after(self._hide_after_ms, self._hide)

    def _hide(self):
        self._hide_job = None
        self._root.withdraw()

    def run(self):
        self._root.after(self._tick_ms, self._drain)
        self._root.mainloop()

        # mainloop() has returned, so no callback is running and it's safe to
        # tear the interpreter down. quit() alone only ends the loop -- it
        # leaks the Tcl interpreter, which the process exiting normally hides.
        # A second Tk() in the same process then fails to initialise.
        try:
            self._root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    import time

    overlay = OverlayWindow()

    def second(generation):
        for word in "NEW: ask what their budget actually is".split():
            overlay.append(generation, word + " ")
            time.sleep(0.12)
        overlay.finish(generation)

    def demo():
        time.sleep(0.8)
        gen1 = overlay.begin()
        for word in "this suggestion is about to be superseded".split():
            overlay.append(gen1, word + " ")
            time.sleep(0.12)
            if word == "about":
                gen2 = overlay.begin()
                threading.Thread(target=second, args=(gen2,), daemon=True).start()
        overlay.finish(gen1)

    threading.Thread(target=demo, daemon=True).start()
    overlay.run()
