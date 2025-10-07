import tkinter as tk
from tkinter import scrolledtext
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layout.sidebar import Sidebar


class FindClusterScreen(tk.Frame):
    
    def __init__(self, parent, on_navigate=None, **kwargs):
        super().__init__(parent, bg="#EDE3E4", **kwargs)
        
        self.on_navigate = on_navigate
        
        self._create_ui()
        
    def _create_ui(self):
        container = tk.Frame(self, bg="#EDE3E4")
        container.pack(fill=tk.BOTH, expand=True)
        
        self.sidebar = Sidebar(container, width=200)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        self.sidebar.add_menu_item("Upload Files", command=lambda: self._navigate("Upload Files"))
        self.sidebar.add_menu_item("Train Model", command=lambda: self._navigate("Train Model"))
        self.sidebar.add_menu_item("Find Cluster", is_active=True)
        
        self.content_area = tk.Frame(container, bg="#EDE3E4")
        self.content_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=40, pady=40)
        
        self._create_content()
        
    def _create_content(self):
        title = tk.Label(
            self.content_area,
            text="Find Cluster",
            font=("Arial", 28, "bold"),
            bg="#EDE3E4",
            fg="#6B3642",
            anchor="w"
        )
        title.pack(fill=tk.X, pady=(0, 30))
        
        buttons_frame = tk.Frame(self.content_area, bg="#EDE3E4")
        buttons_frame.pack(fill=tk.X, pady=(0, 20))
        
        silhouette_btn = tk.Button(
            buttons_frame,
            text="Find Silhoutte Score",
            font=("Arial", 12, "bold"),
            bg="#A15264",
            fg="white",
            activebackground="#8B4555",
            activeforeground="white",
            relief=tk.FLAT,
            padx=30,
            pady=12,
            cursor="hand2",
            command=self._on_find_silhouette
        )
        silhouette_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        elbow_btn = tk.Button(
            buttons_frame,
            text="Find Elbow Method",
            font=("Arial", 12, "bold"),
            bg="#A15264",
            fg="white",
            activebackground="#8B4555",
            activeforeground="white",
            relief=tk.FLAT,
            padx=30,
            pady=12,
            cursor="hand2",
            command=self._on_find_elbow
        )
        elbow_btn.pack(side=tk.LEFT)
        
        results_container = tk.Frame(
            self.content_area,
            bg="white",
            relief=tk.SOLID,
            borderwidth=1
        )
        results_container.pack(fill=tk.BOTH, expand=True)
        
        results_text = scrolledtext.ScrolledText(
            results_container,
            font=("Arial", 10),
            bg="white",
            fg="#333",
            relief=tk.FLAT,
            wrap=tk.WORD,
            padx=15,
            pady=15,
            state=tk.DISABLED
        )
        results_text.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        insights_label = tk.Label(
            self.content_area,
            text="Textual Insights",
            font=("Arial", 11, "bold"),
            bg="#EDE3E4",
            fg="#6B3642",
            anchor="w"
        )
        insights_label.pack(fill=tk.X, pady=(20, 10))
        
        insights_text = tk.Text(
            self.content_area,
            font=("Arial", 10),
            bg="white",
            fg="#333",
            relief=tk.SOLID,
            borderwidth=1,
            wrap=tk.WORD,
            padx=15,
            pady=15,
            height=6,
            state=tk.NORMAL
        )
        insights_text.pack(fill=tk.X)
        
        sample_insights = """This is a test text for textual insights. The clustering analysis will provide valuable information about the data patterns and groupings. 

The optimal number of clusters can be determined using both the Silhouette Score and Elbow Method. These metrics help in understanding the quality of clustering and finding the best separation between clusters."""
        
        insights_text.insert("1.0", sample_insights)
        insights_text.config(state=tk.DISABLED)
        
    def _on_find_silhouette(self):
        print("Finding Silhouette Score...")
        
    def _on_find_elbow(self):
        print("Finding Elbow Method...")
        
    def _navigate(self, screen_name):
        if self.on_navigate:
            self.on_navigate(screen_name)


class FindClusterApp(tk.Tk):
    
    def __init__(self):
        super().__init__()
        
        self.title("ACADEMETRICS - Find Cluster")
        self.geometry("1100x700")
        self.minsize(900, 600)
        
        screen = FindClusterScreen(self)
        screen.pack(fill=tk.BOTH, expand=True)


def main():
    app = FindClusterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
