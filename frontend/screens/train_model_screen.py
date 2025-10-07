import tkinter as tk
from tkinter import messagebox
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layout.sidebar import Sidebar


class TrainModelScreen(tk.Frame):
    
    def __init__(self, parent, on_navigate=None, **kwargs):
        super().__init__(parent, bg="#EDE3E4", **kwargs)
        
        self.on_navigate = on_navigate
        
        self.logistic_balance_var = tk.BooleanVar()
        self.logistic_stratify_var = tk.BooleanVar()
        self.xgboost_balance_var = tk.BooleanVar()
        self.xgboost_stratify_var = tk.BooleanVar()
        
        self._create_ui()
        
    def _create_ui(self):
        container = tk.Frame(self, bg="#EDE3E4")
        container.pack(fill=tk.BOTH, expand=True)
        
        self.sidebar = Sidebar(container, width=200)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        self.sidebar.add_menu_item("Upload Files", command=self._on_upload_files)
        self.sidebar.add_menu_item("Train Model", is_active=True)
        self.sidebar.add_menu_item("Find Cluster", command=self._on_find_cluster)
        
        self.content_area = tk.Frame(container, bg="#EDE3E4")
        self.content_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=40, pady=40)
        
        self._create_content()
        
    def _create_content(self):
        title = tk.Label(
            self.content_area,
            text="Train Model",
            font=("Arial", 28, "bold"),
            bg="#EDE3E4",
            fg="#6B3642",
            anchor="w"
        )
        title.pack(fill=tk.X, pady=(0, 30))
        
        models_container = tk.Frame(self.content_area, bg="#EDE3E4")
        models_container.pack(fill=tk.BOTH, expand=True)
        
        models_container.grid_columnconfigure(0, weight=1)
        models_container.grid_columnconfigure(1, weight=1)
        models_container.grid_rowconfigure(0, weight=1)
        
        self._create_training_card(
            models_container,
            "Train Logistic Regression",
            self.logistic_balance_var,
            self.logistic_stratify_var,
            self._on_train_logistic,
            column=0
        )
        
        self._create_training_card(
            models_container,
            "Train XGBoost Classifcation",
            self.xgboost_balance_var,
            self.xgboost_stratify_var,
            self._on_train_xgboost,
            column=1
        )
        
    def _create_training_card(self, parent, title, balance_var, stratify_var, train_command, column):
        card_frame = tk.Frame(parent, bg="white", relief=tk.SOLID, borderwidth=1)
        card_frame.grid(row=0, column=column, sticky="nsew", padx=(0, 20) if column == 0 else (10, 0))
        
        header = tk.Label(
            card_frame,
            text=title,
            font=("Arial", 14, "bold"),
            bg="#A15264",
            fg="white",
            anchor="w",
            padx=20,
            pady=15
        )
        header.pack(fill=tk.X)
        
        options_frame = tk.Frame(card_frame, bg="white")
        options_frame.pack(fill=tk.X, padx=20, pady=20)
        
        balance_check = tk.Checkbutton(
            options_frame,
            text="Balance",
            variable=balance_var,
            font=("Arial", 11),
            bg="white",
            fg="#333",
            activebackground="white",
            selectcolor="white",
            cursor="hand2"
        )
        balance_check.pack(anchor="w", pady=(0, 10))
        
        stratify_check = tk.Checkbutton(
            options_frame,
            text="Stratify",
            variable=stratify_var,
            font=("Arial", 11),
            bg="white",
            fg="#333",
            activebackground="white",
            selectcolor="white",
            cursor="hand2"
        )
        stratify_check.pack(anchor="w")
        
        train_btn = tk.Button(
            options_frame,
            text="Train",
            font=("Arial", 11, "bold"),
            bg="#A15264",
            fg="white",
            activebackground="#8B4555",
            activeforeground="white",
            relief=tk.FLAT,
            padx=40,
            pady=8,
            cursor="hand2",
            command=train_command
        )
        train_btn.pack(anchor="e", pady=(15, 0))
        
        separator = tk.Frame(card_frame, bg="#E0D0D5", height=1)
        separator.pack(fill=tk.X, padx=20)
        
        results_frame = tk.Frame(card_frame, bg="white")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        results_label = tk.Label(
            results_frame,
            text="Results",
            font=("Arial", 11),
            bg="white",
            fg="#6B3642",
            anchor="w"
        )
        results_label.pack(anchor="w", pady=(0, 10))
        
        results_text = tk.Text(
            results_frame,
            font=("Arial", 10),
            bg="white",
            fg="#333",
            relief=tk.FLAT,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        results_text.pack(fill=tk.BOTH, expand=True)
        
    def _on_train_logistic(self):
        balance = "Yes" if self.logistic_balance_var.get() else "No"
        stratify = "Yes" if self.logistic_stratify_var.get() else "No"
        messagebox.showinfo(
            "Train Logistic Regression",
            f"Training Logistic Regression\nBalance: {balance}\nStratify: {stratify}"
        )
        
    def _on_train_xgboost(self):
        balance = "Yes" if self.xgboost_balance_var.get() else "No"
        stratify = "Yes" if self.xgboost_stratify_var.get() else "No"
        messagebox.showinfo(
            "Train XGBoost Classification",
            f"Training XGBoost Classification\nBalance: {balance}\nStratify: {stratify}"
        )
        
    def _on_upload_files(self):
        if self.on_navigate:
            self.on_navigate("Upload Files")
        else:
            messagebox.showinfo("Navigation", "Navigate to Upload Files screen")
        
    def _on_find_cluster(self):
        if self.on_navigate:
            self.on_navigate("Find Cluster")
        else:
            messagebox.showinfo("Navigation", "Navigate to Find Cluster screen")


class TrainModelApp(tk.Tk):
    
    def __init__(self):
        super().__init__()
        
        self.title("ACADEMETRICS - Train Model")
        self.geometry("1100x700")
        self.minsize(900, 600)
        
        screen = TrainModelScreen(self)
        screen.pack(fill=tk.BOTH, expand=True)


def main():
    app = TrainModelApp()
    app.mainloop()


if __name__ == "__main__":
    main()
