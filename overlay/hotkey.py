import threading
import time

try:
    import pywintypes
    import win32con
    import win32gui

    _HAS_WIN32 = True
except ImportError:
    _HAS_WIN32 = False

# Stops the hotkey auto-repeating while the key is held down. Without it,
# holding the chord fires an LLM request every few milliseconds.
_MOD_NOREPEAT = getattr(win32con, "MOD_NOREPEAT", 0x4000) if _HAS_WIN32 else 0


class GlobalHotkey:
    """A system-wide hotkey. Fires `on_press` no matter which app has focus."""

    def __init__(self, on_press, modifiers=None, vk=ord("J"), hotkey_id=1, poll_ms=30):
        self._on_press = on_press
        if modifiers is None and _HAS_WIN32:
            modifiers = win32con.MOD_CONTROL | win32con.MOD_ALT
        self._modifiers = (modifiers or 0) | _MOD_NOREPEAT
        self._vk = vk
        self._id = hotkey_id
        self._poll = poll_ms / 1000.0

        self._stop = threading.Event()
        self._thread = None
        self.registered = False

    def start(self):
        if not _HAS_WIN32:
            print("WARNING: pywin32 missing -- no global hotkey.")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="hotkey")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self):
        # RegisterHotKey binds the hotkey to the CALLING THREAD. WM_HOTKEY is
        # posted to this thread's message queue -- so registering here and
        # pumping somewhere else would silently never fire.
        try:
            win32gui.RegisterHotKey(None, self._id, self._modifiers, self._vk)
        except pywintypes.error as exc:
            print(f"WARNING: hotkey not registered ({exc.strerror}). "
                  f"Another app probably owns that combination.")
            return

        self.registered = True
        try:
            while not self._stop.is_set():
                result, message = win32gui.PeekMessage(None, 0, 0, win32con.PM_REMOVE)
                if result and message[1] == win32con.WM_HOTKEY:
                    try:
                        self._on_press()
                    except Exception as exc:
                        print(f"hotkey handler raised: {exc}")
                else:
                    # Only sleep when the queue was empty, so a burst drains
                    # immediately instead of one message per poll.
                    time.sleep(self._poll)
        finally:
            win32gui.UnregisterHotKey(None, self._id)
            self.registered = False


if __name__ == "__main__":
    presses = [0]

    def on_press():
        presses[0] += 1
        print(f"hotkey pressed ({presses[0]})")

    hotkey = GlobalHotkey(on_press)
    hotkey.start()
    print("Press Ctrl+Alt+J anywhere. Ctrl+C here to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        hotkey.stop()
