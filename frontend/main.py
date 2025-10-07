import tkinter as tk
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screens.upload_files_screen import UploadFilesScreen
from screens.train_model_screen import TrainModelScreen
from screens.find_cluster_screen import FindClusterScreen


class AcademetricsApp(tk.Tk):
    
    def __init__(self):
        super().__init__()
        
        self.title("ACADEMETRICS")
        self.geometry("1020x800")
        self.minsize(900, 600)
        
        self.current_screen = None
        
        self.show_upload_files()
        
    def clear_screen(self):
        if self.current_screen:
            self.current_screen.pack_forget()
            self.current_screen.destroy()
            
    def show_upload_files(self):
        self.clear_screen()
        self.current_screen = UploadFilesScreen(self, on_navigate=self.navigate)
        self.current_screen.pack(fill=tk.BOTH, expand=True)
        
    def show_train_model(self):
        self.clear_screen()
        self.current_screen = TrainModelScreen(self, on_navigate=self.navigate)
        self.current_screen.pack(fill=tk.BOTH, expand=True)
        
    def show_find_cluster(self):
        self.clear_screen()
        self.current_screen = FindClusterScreen(self, on_navigate=self.navigate)
        self.current_screen.pack(fill=tk.BOTH, expand=True)
        
    def navigate(self, screen_name):
        if screen_name == "Upload Files":
            self.show_upload_files()
        elif screen_name == "Train Model":
            self.show_train_model()
        elif screen_name == "Find Cluster":
            self.show_find_cluster()


def main():
    app = AcademetricsApp()
    app.mainloop()


if __name__ == "__main__":
    main()
