import tkinter as tk
from tkinter import filedialog, messagebox
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layout.sidebar import Sidebar


class UploadFilesScreen(tk.Frame):
    "Upload Screen"
    
    def __init__(self, parent, on_navigate=None, **kwargs):
        super().__init__(parent, bg="#FBF8F5", **kwargs)
        
        self.on_navigate = on_navigate
        
        self.dataset_path = None
        self.embedding_path = None
        self.is_converting = False
        self.conversion_success = False
        
        self._create_upload_screen()
        
    def _create_upload_screen(self):
        container = tk.Frame(self, bg="#FBF8F5")
        container.pack(fill=tk.BOTH, expand=True)
        
        self.sidebar = Sidebar(container, width=200)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        self.sidebar.add_menu_item("Upload Files", is_active=True)
        self.sidebar.add_menu_item("Train Model", command=self._on_train_model)
        self.sidebar.add_menu_item("Find Cluster", command=self._on_find_cluster)
        
        self.content_area = tk.Frame(container, bg="#FBF8F5")
        self.content_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=40, pady=40)
        

        self._create_content()
        
    def _create_content(self):
        title = tk.Label(
            self.content_area,
            text="Upload Files",
            font=("Arial", 28, "bold"),
            bg="#FBF8F5",
            fg="#6B3642",
            anchor="w"
        )
        title.pack(fill=tk.X, pady=(0, 30))
        
        self._create_upload_section(
            "Upload Dataset",
            "(.csv, .xlsx, .sql)",
            self._on_upload_dataset
        )
        
        self._create_upload_section(
            "Upload Embedding",
            "(.npy, .json, .csv)",
            self._on_upload_embedding
        )
        
        self.status_frame = tk.Frame(self.content_area, bg="#FBF8F5")
        self.status_frame.pack(fill=tk.X, pady=(30, 20))
        
        self.converting_label = tk.Label(
            self.status_frame,
            text="",
            font=("Arial", 12, "italic"),
            bg="#FBF8F5",
            fg="#6B3642",
            anchor="w"
        )
        self.converting_label.pack(fill=tk.X)
        
        self.success_label = tk.Label(
            self.status_frame,
            text="",
            font=("Arial", 12),
            bg="#FBF8F5",
            fg="#4A7C59",
            anchor="w"
        )
        self.success_label.pack(fill=tk.X, pady=(5, 0))
        
        self.continue_btn = tk.Button(
            self.content_area,
            text="Continue  →",
            font=("Arial", 12, "bold"),
            bg="#A15264",
            fg="white",
            activebackground="#8B4555",
            activeforeground="white",
            relief=tk.FLAT,
            padx=30,
            pady=12,
            cursor="hand2",
            command=self._on_continue
        )
        
    def _create_upload_section(self, title, formats, command):
        """Create an upload section with title, input field, and button
        
        Args:
            title (str): Section title
            formats (str): Supported file formats
            command (callable): Function to call when upload button is clicked
        """
        section_frame = tk.Frame(self.content_area, bg="#FBF8F5")
        section_frame.pack(fill=tk.X, pady=(0, 30))

        header_frame = tk.Frame(section_frame, bg="#FBF8F5")
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(
            header_frame,
            text=title,
            font=("Arial", 16, "bold"),
            bg="#FBF8F5",
            fg="#6B3642"
        )
        title_label.pack(side=tk.LEFT)
        
        format_label = tk.Label(
            header_frame,
            text=f"  {formats}",
            font=("Arial", 12),
            bg="#FBF8F5",
            fg="#8B7378"
        )
        format_label.pack(side=tk.LEFT)
        

        input_frame = tk.Frame(section_frame, bg="#FBF8F5")
        input_frame.pack(fill=tk.X)
        
        input_entry = tk.Entry(
            input_frame,
            font=("Arial", 11),
            bg="white",
            fg="#333",
            relief=tk.SOLID,
            borderwidth=1,
            highlightthickness=1,
            highlightcolor="#A15264",
            highlightbackground="#CCC"
        )
        input_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, ipady=8)
        

        upload_btn = tk.Button(
            input_frame,
            text="Upload",
            font=("Arial", 11, "bold"),
            bg="#A15264",
            fg="white",
            activebackground="#8B4555",
            activeforeground="white",
            relief=tk.FLAT,
            padx=25,
            pady=10,
            cursor="hand2",
            command=lambda: command(input_entry)
        )
        upload_btn.pack(side=tk.LEFT, padx=(10, 0))
        
    def _on_upload_dataset(self, entry_widget):
        """Handle dataset file upload"""
        filetypes = [
            ("Supported files", "*.csv *.xlsx *.sql"),
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx"),
            ("SQL files", "*.sql"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=filetypes
        )
        
        if filename:
            self.dataset_path = filename
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, filename)
            self._check_and_process()
            
    def _on_upload_embedding(self, entry_widget):
        """Handle embedding file upload"""
        filetypes = [
            ("Supported files", "*.npy *.json *.csv"),
            ("NumPy files", "*.npy"),
            ("JSON files", "*.json"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Embedding File",
            filetypes=filetypes
        )
        
        if filename:
            self.embedding_path = filename
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, filename)
            self._check_and_process()
            
    def _check_and_process(self):
        """Check if both files are uploaded and simulate processing"""
        if self.dataset_path and self.embedding_path:

            self.converting_label.config(text="Converting embedding...")
            self.is_converting = True
            
            self.after(2000, self._show_success)
            
    def _show_success(self):
        """Show success message and continue button"""
        self.converting_label.config(text="")
        self.success_label.config(text="Upload and conversion was successful!")
        self.conversion_success = True
        
        self.continue_btn.pack(pady=(10, 0), anchor="w")
        
    def _on_continue(self):
        """Handle continue button click"""
        if self.on_navigate:
            self.on_navigate("Train Model")
        else:
            messagebox.showinfo("Continue", "Proceeding to next step...")
        
    def _on_train_model(self):
        """Handle Train Model menu click"""
        if self.on_navigate:
            self.on_navigate("Train Model")
        else:
            messagebox.showinfo("Navigation", "Navigate to Train Model screen")
        
    def _on_find_cluster(self):
        """Handle Find Cluster menu click"""
        if self.on_navigate:
            self.on_navigate("Find Cluster")
        else:
            messagebox.showinfo("Navigation", "Navigate to Find Cluster screen")


class UploadFilesApp(tk.Tk):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        self.title("ACADEMETRICS - Upload Files")
        self.geometry("1000x700")
        self.minsize(800, 600)
        
        # Create and pack the upload screen
        screen = UploadFilesScreen(self)
        screen.pack(fill=tk.BOTH, expand=True)


def main():
    """Run the application"""
    app = UploadFilesApp()
    app.mainloop()


if __name__ == "__main__":
    main()
