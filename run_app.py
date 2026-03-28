import subprocess, sys, importlib, importlib.util
 
REQUIRED = ["sklearn", "pandas", "numpy", "matplotlib", "seaborn", "joblib"]
 
def _install_missing():
    missing = [pkg for pkg in REQUIRED
               if importlib.util.find_spec(pkg) is None]
    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            *[p.replace("sklearn", "scikit-learn") for p in missing],
            "--break-system-packages", "-q"
        ])
 
if __name__ == "__main__":
    _install_missing()
 
    import tkinter as tk
    # Quick check that Tkinter works
    try:
        root = tk.Tk()
        root.withdraw()
        root.destroy()
    except Exception:
        print("ERROR: Tkinter is not available on this system.\n"
              "Install it with:  sudo apt install python3-tk   (Linux)\n"
              "On macOS/Windows Tkinter is included with Python.")
        sys.exit(1)
 
    from gui_app import SleepApneaApp
    app = SleepApneaApp()
    app.mainloop()