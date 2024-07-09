import tkinter as tk
from tkinter import filedialog, messagebox
from ttkbootstrap import Style
from ttkbootstrap import ttk
import threading
import os
import shutil
import cv2
from main import process_video_stream, stop_video_stream

class AdManagerApp:
    def __init__(self, root):
        self.root = root
        self.style = Style(theme="flatly")  # Choose a theme from ttkbootstrap
        self.root.title("Ad Manager")
        self.root.geometry("800x600")  # Increased window size
        button_style = {
            'style': 'success.TButton',
            'padding': (15, 15),
            'width': 30
        }

        self.start_button = ttk.Button(root, text="Start System", command=self.start_system, **button_style)
        self.start_button.pack(pady=10)

        button_style['style'] = 'danger.TButton'
        self.stop_button = ttk.Button(root, text="Stop System", command=self.stop_system, **button_style)
        self.stop_button.pack(pady=10)

        button_style['style'] = 'primary.TButton'
        self.open_dashboard_button = ttk.Button(root, text="Open Dashboard", command=self.open_dashboard, **button_style)
        self.open_dashboard_button.pack(pady=10)

        button_style['style'] = 'info.TButton'
        self.upload_ad_button = ttk.Button(root, text="Upload New Ad", command=self.upload_ad, **button_style)
        self.upload_ad_button.pack(pady=10)

        button_style['style'] = 'warning.TButton'
        self.remove_ad_button = ttk.Button(root, text="Remove Ad", command=self.remove_ad, **button_style)
        self.remove_ad_button.pack(pady=10)

        self.system_running = False
        self.system_thread = None

    def start_system(self):
        if not self.system_running:
            self.system_thread = threading.Thread(target=process_video_stream, daemon=True)
            self.system_thread.start()
            self.system_running = True
            messagebox.showinfo("Info", "System started in the background.")
        else:
            messagebox.showinfo("Info", "System is already running.")

    def stop_system(self):
        if self.system_running:
            stop_video_stream()
            self.system_running = False
            messagebox.showinfo("Info", "System stopped.")
        else:
            messagebox.showinfo("Info", "System is not running.")

    def open_dashboard(self):
        # Open the PHP dashboard in a web browser
        os.system("start http://localhost/dashboard")

    def upload_ad(self):
        ad_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
        if ad_path:
            shutil.copy(ad_path, "AD_Videos")
            messagebox.showinfo("Info", "Ad uploaded successfully.")

    def remove_ad(self):
        ad_path = filedialog.askopenfilename(initialdir="AD_Videos", filetypes=[("Video files", "*.mp4")])
        if ad_path:
            os.remove(ad_path)
            messagebox.showinfo("Info", "Ad removed successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdManagerApp(root)
    root.mainloop()
    stop_video_stream()
