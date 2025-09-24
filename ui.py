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
        """Upload a new advertisement video with error handling."""
        try:
            ad_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
            if ad_path:
                if not os.path.exists(ad_path):
                    messagebox.showerror("Error", "Selected file does not exist.")
                    return

                # Ensure AD_Videos directory exists
                os.makedirs("AD_Videos", exist_ok=True)

                filename = os.path.basename(ad_path)
                destination = os.path.join("AD_Videos", filename)

                # Check if file already exists
                if os.path.exists(destination):
                    response = messagebox.askyesno("File Exists",
                        f"File {filename} already exists. Overwrite?")
                    if not response:
                        return

                shutil.copy2(ad_path, destination)
                messagebox.showinfo("Success", "Ad uploaded successfully.")
        except PermissionError:
            messagebox.showerror("Error", "Permission denied. Cannot upload file.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to upload ad: {str(e)}")

    def remove_ad(self):
        """Remove an advertisement video with error handling."""
        try:
            ad_path = filedialog.askopenfilename(initialdir="AD_Videos",
                                                filetypes=[("Video files", "*.mp4")])
            if ad_path:
                if not os.path.exists(ad_path):
                    messagebox.showerror("Error", "Selected file does not exist.")
                    return

                filename = os.path.basename(ad_path)
                response = messagebox.askyesno("Confirm Delete",
                    f"Are you sure you want to delete {filename}?")

                if response:
                    os.remove(ad_path)
                    messagebox.showinfo("Success", "Ad removed successfully.")
        except PermissionError:
            messagebox.showerror("Error", "Permission denied. Cannot delete file.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove ad: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdManagerApp(root)
    root.mainloop()
    stop_video_stream()
