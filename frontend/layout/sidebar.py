
import tkinter as tk
from tkinter import font


class Sidebar(tk.Frame):
    "Sidebar navigation component"
    
    def __init__(self, parent, bg_color="#974858", width=200, **kwargs):
        super().__init__(parent, bg=bg_color, width=width, **kwargs)
        self.bg_color = bg_color
        self.pack_propagate(False)
        
        # Callbacks for menu items
        self.callbacks = {}
        
        self._create_logo()
        
       
        self.menu_container = tk.Frame(self, bg=self.bg_color)
        self.menu_container.pack(fill=tk.BOTH, expand=True, pady=20)
        
    def _create_logo(self):
        """Create the logo section at the top of sidebar"""
        logo_frame = tk.Frame(self, bg=self.bg_color, height=100)
        logo_frame.pack(fill=tk.X, pady=(30, 20))
        logo_frame.pack_propagate(False)
        
        #PLACEHOLDER LOGO
        logo_label = tk.Label(
            logo_frame,
            text="📚",
            font=("Arial", 40),
            bg=self.bg_color,
            fg="white"
        )
        logo_label.pack(pady=(10, 5))

        logo_text = tk.Label(
            logo_frame,
            text="ACADEMETRICS",
            font=("Arial", 14, "bold"),
            bg=self.bg_color,
            fg="white"
        )
        logo_text.pack()
        
    def add_menu_item(self, text, command=None, is_active=False):
        """Add a menu item to the sidebar
        
        Args:
            text (str): Text to display for the menu item
            command (callable): Function to call when item is clicked
            is_active (bool): Whether this item is currently active
        """
        
        btn_frame = tk.Frame(self.menu_container, bg=self.bg_color)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Determine colors based on active state
        fg_color = "white" if is_active else "#E8D5D9"
        
        btn = tk.Label(
            btn_frame,
            text=text,
            font=("Arial", 12),
            bg=self.bg_color,
            fg=fg_color,
            anchor="w",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        btn.pack(fill=tk.X)
        
        # Store callback
        if command:
            self.callbacks[text] = command
            btn.bind("<Button-1>", lambda e: command())
            
        # Hover effects
        def on_enter(e):
            if not is_active:
                btn.config(fg="white")
                
        def on_leave(e):
            if not is_active:
                btn.config(fg="#E8D5D9")
                
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn
    
    def clear_menu_items(self):
        """Clear all menu items from the sidebar"""
        for widget in self.menu_container.winfo_children():
            widget.destroy()
