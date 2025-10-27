import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sys
import os
import pandas as pd
import pickle
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layout.sidebar import Sidebar


class UploadFilesScreen(tk.Frame):
    "Upload Screen"
    
    def __init__(self, parent, on_navigate=None, **kwargs):
        super().__init__(parent, bg="#FBF8F5", **kwargs)
        
        self.on_navigate = on_navigate
        
        self.dataset_path = None
        self.model_path = None
        self.is_converting = False
        self.conversion_success = False
        self.dataset_df = None
        self.showing_preview = False
        self.showing_model_info = False
        self.showing_shap_results = False
        self.model_info = None
        
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
        # Create upload view
        self.upload_view = tk.Frame(self.content_area, bg="#FBF8F5")
        self.upload_view.pack(fill=tk.BOTH, expand=True)
        
        title = tk.Label(
            self.upload_view,
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
            "Upload Model",
            "(folder or .pkl, .joblib, .h5, .pt)",
            self._on_upload_model
        )
        
        self.status_frame = tk.Frame(self.upload_view, bg="#FBF8F5")
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
        
        # Button frame for continue and interpret buttons
        self.button_frame = tk.Frame(self.upload_view, bg="#FBF8F5")
        
        self.continue_btn = tk.Button(
            self.button_frame,
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
        
        self.interpret_btn = tk.Button(
            self.button_frame,
            text="Interpret with SHAP",
            font=("Arial", 12, "bold"),
            bg="#4A7C59",
            fg="white",
            activebackground="#3D6849",
            activeforeground="white",
            relief=tk.FLAT,
            padx=30,
            pady=12,
            cursor="hand2",
            command=self._on_interpret
        )
        
        # Create preview view (hidden initially)
        self.preview_view = tk.Frame(self.content_area, bg="#FBF8F5")
        self._create_preview_content()
        
        # Create model info view (hidden initially)
        self.model_view = tk.Frame(self.content_area, bg="#FBF8F5")
        self._create_model_view_content()
        
        # Create SHAP results view (hidden initially)
        self.shap_view = tk.Frame(self.content_area, bg="#FBF8F5")
        self._create_shap_view_content()
        
    def _create_upload_section(self, title, formats, command):
        """Create an upload section with title, input field, and button"""
        section_frame = tk.Frame(self.upload_view, bg="#FBF8F5")
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
        
    def _create_preview_content(self):
        """Create the dataset preview interface"""
        # Header with back button
        header_frame = tk.Frame(self.preview_view, bg="#FBF8F5")
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        back_btn = tk.Button(
            header_frame,
            text="←  Back to Upload",
            font=("Arial", 11, "bold"),
            bg="#A15264",
            fg="white",
            activebackground="#8B4555",
            activeforeground="white",
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor="hand2",
            command=self._show_upload_view
        )
        back_btn.pack(side=tk.LEFT)
        
        title = tk.Label(
            header_frame,
            text="Dataset Preview",
            font=("Arial", 28, "bold"),
            bg="#FBF8F5",
            fg="#6B3642"
        )
        title.pack(side=tk.LEFT, padx=(20, 0))
        
        # Dataset info
        self.info_label = tk.Label(
            self.preview_view,
            text="",
            font=("Arial", 12),
            bg="#FBF8F5",
            fg="#6B3642",
            anchor="w"
        )
        self.info_label.pack(fill=tk.X, pady=(0, 15))
        
        # Table frame with scrollbars
        table_frame = tk.Frame(self.preview_view, bg="white", relief=tk.SOLID, borderwidth=1)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical")
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        hsb = ttk.Scrollbar(table_frame, orient="horizontal")
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Treeview
        self.tree = ttk.Treeview(
            table_frame,
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
            show='tree headings',
            selectmode='browse'
        )
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        
        # Style the treeview
        style = ttk.Style()
        style.configure("Treeview", 
                       background="white",
                       foreground="#333",
                       rowheight=25,
                       fieldbackground="white")
        style.configure("Treeview.Heading",
                       font=("Arial", 11, "bold"),
                       background="#A15264",
                       foreground="white")
        style.map('Treeview', background=[('selected', '#D4A5AE')])
    
    def _create_model_view_content(self):
        """Create the model info display interface"""
        # Header with back button
        header_frame = tk.Frame(self.model_view, bg="#FBF8F5")
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        back_btn = tk.Button(
            header_frame,
            text="←  Back to Upload",
            font=("Arial", 11, "bold"),
            bg="#A15264",
            fg="white",
            activebackground="#8B4555",
            activeforeground="white",
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor="hand2",
            command=self._show_upload_view
        )
        back_btn.pack(side=tk.LEFT)
        
        title = tk.Label(
            header_frame,
            text="Model Information",
            font=("Arial", 28, "bold"),
            bg="#FBF8F5",
            fg="#6B3642"
        )
        title.pack(side=tk.LEFT, padx=(20, 0))
        
        # Scrollable content frame
        canvas = tk.Canvas(self.model_view, bg="#FBF8F5", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.model_view, orient="vertical", command=canvas.yview)
        self.model_content_frame = tk.Frame(canvas, bg="#FBF8F5")
        
        self.model_content_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.model_content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_shap_view_content(self):
        """Create the SHAP results display interface"""
        # Header with back button
        header_frame = tk.Frame(self.shap_view, bg="#FBF8F5")
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        back_btn = tk.Button(
            header_frame,
            text="←  Back to Upload",
            font=("Arial", 11, "bold"),
            bg="#A15264",
            fg="white",
            activebackground="#8B4555",
            activeforeground="white",
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor="hand2",
            command=self._show_upload_view
        )
        back_btn.pack(side=tk.LEFT)
        
        title = tk.Label(
            header_frame,
            text="SHAP Interpretation Results",
            font=("Arial", 28, "bold"),
            bg="#FBF8F5",
            fg="#6B3642"
        )
        title.pack(side=tk.LEFT, padx=(20, 0))
        
        # Scrollable content frame
        canvas = tk.Canvas(self.shap_view, bg="#FBF8F5", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.shap_view, orient="vertical", command=canvas.yview)
        self.shap_content_frame = tk.Frame(canvas, bg="#FBF8F5")
        
        self.shap_content_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.shap_content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
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
            
            # Load and display dataset
            try:
                if filename.endswith('.csv'):
                    self.dataset_df = pd.read_csv(filename)
                elif filename.endswith('.xlsx'):
                    self.dataset_df = pd.read_excel(filename)
                elif filename.endswith('.sql'):
                    messagebox.showwarning("SQL Files", "SQL file loading requires database connection. Please use CSV or Excel for preview.")
                    return
                
                # Show preview
                self._show_preview_view()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")
                return
            
            self._check_and_process()
            
    def _on_upload_model(self, entry_widget):
        """Handle model folder/file upload"""
        # Ask user if they want to upload a folder or file
        response = messagebox.askquestion(
            "Upload Model",
            "Is your model a folder (e.g., BERT from Hugging Face)?\n\n"
            "Click 'Yes' for folder\n"
            "Click 'No' for single file (.pkl, .joblib, etc.)"
        )
        
        if response == 'yes':
            # Upload folder
            folder_path = filedialog.askdirectory(
                title="Select Model Folder"
            )
            
            if folder_path:
                self.model_path = folder_path
                entry_widget.delete(0, tk.END)
                entry_widget.insert(0, folder_path)
                
                # Load and show model info immediately
                self._load_and_show_model_info()
                
                self._check_and_process()
        else:
            # Upload single file
            filetypes = [
                ("Model files", "*.pkl *.joblib *.h5 *.pt *.pth *.json *.model"),
                ("Pickle files", "*.pkl"),
                ("Joblib files", "*.joblib"),
                ("Keras/TensorFlow", "*.h5"),
                ("PyTorch files", "*.pt *.pth"),
                ("XGBoost files", "*.model *.json"),
                ("All files", "*.*")
            ]
            
            filename = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=filetypes
            )
            
            if filename:
                self.model_path = filename
                entry_widget.delete(0, tk.END)
                entry_widget.insert(0, filename)
                
                # Load and show model info immediately
                self._load_and_show_model_info()
                
                self._check_and_process()
    
    def _load_and_show_model_info(self):
        """Load model information and immediately show the model view"""
        if not self.model_path:
            return
        
        try:
            is_folder = os.path.isdir(self.model_path)
            
            if is_folder:
                folder_name = os.path.basename(self.model_path)
                files_in_folder = os.listdir(self.model_path)
                
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(self.model_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        total_size += os.path.getsize(filepath)
                
                folder_size_mb = total_size / (1024 * 1024)
                
                model_type = "Unknown Model"
                model_details = {}
                
                config_exists = 'config.json' in files_in_folder
                has_tokenizer_files = any(f.startswith('tokenizer') or f in ['vocab.txt', 'vocab.json', 'tokenizer.json'] for f in files_in_folder)
                has_model_weights = any(f.endswith(('.bin', '.safetensors', '.h5', '.pt', '.pth')) for f in files_in_folder)

                if 'config.json' in files_in_folder:
                    model_type = "Transformer Model (BERT/RoBERTa/etc.)"
                    try:
                        config_path = os.path.join(self.model_path, 'config.json')
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        
                        model_details = {
                            'model_type': config.get('model_type', 'N/A'),
                            'hidden_size': config.get('hidden_size', 'N/A'),
                            'num_hidden_layers': config.get('num_hidden_layers', 'N/A'),
                            'num_attention_heads': config.get('num_attention_heads', 'N/A'),
                            'vocab_size': config.get('vocab_size', 'N/A'),
                            'max_position_embeddings': config.get('max_position_embeddings', 'N/A')
                        }

                    except Exception as e:
                        model_details['config_error'] = f"Could not read config: {str(e)}"
                

                readiness = []
                if config_exists:
                    readiness.append("✓ config.json")
                else:
                    readiness.append("✗ config.json (MISSING)")
                    
                if has_tokenizer_files:
                    readiness.append("✓ tokenizer files")
                else:
                    readiness.append("✗ tokenizer files (MISSING)")
                    
                if has_model_weights:
                    readiness.append("✓ model weights")
                else:
                    readiness.append("✗ model weights (MISSING)")
                
                model_details['readiness'] = " | ".join(readiness)

                important_files = [f for f in files_in_folder if f.endswith(
                ('.json', '.bin', '.safetensors', '.txt', '.h5', '.pt', '.pth', '.model')
                ) or f.startswith('tokenizer') or f in ['vocab.txt', 'config.json']]

                self.model_info = {
                'filename': folder_name,
                'type': model_type,
                'size_mb': round(folder_size_mb, 2),
                'is_folder': True,
                'details': model_details,
                'files': important_files,
                'is_ready': config_exists and has_tokenizer_files and has_model_weights
                }

                # # Check for model weight files
                # weight_files = []
                # if any('.bin' in f for f in files_in_folder):
                #     weight_files.append('.bin files')
                # if any('.safetensors' in f for f in files_in_folder):
                #     weight_files.append('.safetensors files')
                # if any('.h5' in f for f in files_in_folder):
                #     weight_files.append('.h5 files')
                
                # if weight_files:
                #     model_details['weight_format'] = ', '.join(weight_files)
                
                # # List important files
                # important_files = [f for f in files_in_folder if f.endswith(
                #     ('.json', '.bin', '.safetensors', '.txt', '.h5')
                # )]
                
                # # Store model info
                # self.model_info = {
                #     'filename': folder_name,
                #     'type': model_type,
                #     'size_mb': round(folder_size_mb, 2),
                #     'is_folder': True,
                #     'details': model_details,
                #     'files': important_files
                # }
                
            else:
                # Handle single model file
                file_size = os.path.getsize(self.model_path)
                file_size_mb = file_size / (1024 * 1024)
                filename = os.path.basename(self.model_path)
                
                # Detect model type and load info
                model_type = "Unknown"
                model_details = {}
                
                if filename.endswith(('.pkl', '.joblib')):
                    # Try to load pickle/joblib file
                    try:
                        if filename.endswith('.pkl'):
                            with open(self.model_path, 'rb') as f:
                                model = pickle.load(f)
                        else:
                            import joblib
                            model = joblib.load(self.model_path)
                        
                        # Detect model type
                        model_class = type(model).__name__
                        if 'XGB' in model_class or 'xgboost' in str(type(model).__module__):
                            model_type = "XGBoost"
                            if hasattr(model, 'get_params'):
                                params = model.get_params()
                                model_details = {
                                    'n_estimators': params.get('n_estimators', 'N/A'),
                                    'max_depth': params.get('max_depth', 'N/A'),
                                    'learning_rate': params.get('learning_rate', 'N/A'),
                                    'objective': params.get('objective', 'N/A')
                                }
                        elif 'Logistic' in model_class:
                            model_type = "Logistic Regression"
                            if hasattr(model, 'get_params'):
                                params = model.get_params()
                                model_details = {
                                    'penalty': params.get('penalty', 'N/A'),
                                    'C': params.get('C', 'N/A'),
                                    'solver': params.get('solver', 'N/A'),
                                    'max_iter': params.get('max_iter', 'N/A')
                                }
                            if hasattr(model, 'n_features_in_'):
                                model_details['n_features'] = model.n_features_in_
                        else:
                            model_type = model_class
                            
                    except Exception as e:
                        model_details['error'] = f"Could not load model: {str(e)}"
                        
                elif filename.endswith(('.h5', '.keras')):
                    model_type = "Keras/TensorFlow Model"
                    model_details['framework'] = 'TensorFlow/Keras'
                    
                elif filename.endswith(('.pt', '.pth')):
                    model_type = "PyTorch Model"
                    model_details['framework'] = 'PyTorch'
                    
                elif filename.endswith('.model'):
                    model_type = "XGBoost Binary Model"
                    model_details['format'] = 'XGBoost binary format'
                    
                elif filename.endswith('.json'):
                    try:
                        with open(self.model_path, 'r') as f:
                            config = json.load(f)
                        model_type = "Model Config (JSON)"
                        if 'model_type' in config:
                            model_type = config.get('model_type', model_type)
                        model_details = {k: v for k, v in list(config.items())[:5]}
                    except:
                        pass
                
                # Store model info
                self.model_info = {
                    'filename': filename,
                    'type': model_type,
                    'size_mb': round(file_size_mb, 2),
                    'is_folder': False,
                    'details': model_details
                }
            
            # Show model view immediately
            self._show_model_view()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error reading model:\n{str(e)}")
    
    def _show_preview_view(self):
        """Show the dataset preview"""
        if self.dataset_df is None:
            return
            
        self.showing_preview = True
        self.showing_model_info = False
        self.showing_shap_results = False
        self.upload_view.pack_forget()
        self.model_view.pack_forget()
        self.shap_view.pack_forget()
        self.preview_view.pack(fill=tk.BOTH, expand=True)
        
        # Update info label
        rows, cols = self.dataset_df.shape
        self.info_label.config(text=f"Dataset: {os.path.basename(self.dataset_path)}  |  Rows: {rows}  |  Columns: {cols}")
        
        # Clear existing tree
        self.tree.delete(*self.tree.get_children())
        
        # Configure columns
        columns = list(self.dataset_df.columns)
        self.tree['columns'] = columns
        
        # Format column #0 (index column)
        self.tree.column('#0', width=50, anchor='center')
        self.tree.heading('#0', text='#')
        
        # Format data columns
        for col in columns:
            self.tree.column(col, width=150, anchor='w')
            self.tree.heading(col, text=col)
        
        # Add data rows (limit to first 1000 rows for performance)
        display_limit = min(1000, len(self.dataset_df))
        for idx, row in self.dataset_df.head(display_limit).iterrows():
            values = [str(val) for val in row.values]
            self.tree.insert('', 'end', text=str(idx), values=values)
        
        if len(self.dataset_df) > display_limit:
            messagebox.showinfo("Info", f"Displaying first {display_limit} rows of {len(self.dataset_df)} total rows.")
    
    def _show_upload_view(self):
        """Show the upload view"""
        self.showing_preview = False
        self.showing_model_info = False
        self.showing_shap_results = False
        self.preview_view.pack_forget()
        self.model_view.pack_forget()
        self.shap_view.pack_forget()
        self.upload_view.pack(fill=tk.BOTH, expand=True)
    
    def _show_model_view(self):
        """Show the model information view"""
        if self.model_info is None:
            return
        
        self.showing_model_info = True
        self.showing_preview = False
        self.showing_shap_results = False
        self.upload_view.pack_forget()
        self.preview_view.pack_forget()
        self.shap_view.pack_forget()
        self.model_view.pack(fill=tk.BOTH, expand=True)
        
        # Clear previous content
        for widget in self.model_content_frame.winfo_children():
            widget.destroy()
        
        # Create info cards
        info = self.model_info
        
        # Main info card
        main_card = tk.Frame(self.model_content_frame, bg="white", relief=tk.SOLID, borderwidth=1)
        main_card.pack(fill=tk.X, pady=(0, 20), padx=5)
        
        # Title
        title_label = tk.Label(
            main_card,
            text=info['filename'],
            font=("Arial", 18, "bold"),
            bg="white",
            fg="#6B3642",
            anchor="w"
        )
        title_label.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        # Type and size
        type_label = tk.Label(
            main_card,
            text=f"Type: {info['type']}",
            font=("Arial", 12),
            bg="white",
            fg="#333",
            anchor="w"
        )
        type_label.pack(fill=tk.X, padx=20, pady=5)
        
        size_label = tk.Label(
            main_card,
            text=f"Size: {info['size_mb']} MB",
            font=("Arial", 12),
            bg="white",
            fg="#333",
            anchor="w"
        )
        size_label.pack(fill=tk.X, padx=20, pady=(0, 5))

        if self.model_info.get('is_folder', False):
            readiness = self.model_info.get('is_ready', False)
            status_color = "#4A7C59" if readiness else "#A15264"
            status_text = "✓ READY FOR SHAP" if readiness else "✗ MISSING FILES"
            
            status_label = tk.Label(
                main_card,
                text=status_text,
                font=("Arial", 12, "bold"),
                bg="white",
                fg=status_color,
                anchor="w"
            )
            status_label.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        if info['is_folder']:
            files_count = len(info.get('files', []))
            files_label = tk.Label(
                main_card,
                text=f"Files: {files_count} key files found",
                font=("Arial", 12),
                bg="white",
                fg="#333",
                anchor="w"
            )
            files_label.pack(fill=tk.X, padx=20, pady=(0, 20))
        else:
            tk.Label(main_card, bg="white", height=1).pack()
        
        # Model details card
        if info['details']:
            details_card = tk.Frame(self.model_content_frame, bg="white", relief=tk.SOLID, borderwidth=1)
            details_card.pack(fill=tk.X, pady=(0, 20), padx=5)
            
            details_title = tk.Label(
                details_card,
                text="Model Details",
                font=("Arial", 14, "bold"),
                bg="white",
                fg="#6B3642",
                anchor="w"
            )
            details_title.pack(fill=tk.X, padx=20, pady=(20, 15))
            
            for key, value in info['details'].items():
                detail_frame = tk.Frame(details_card, bg="white")
                detail_frame.pack(fill=tk.X, padx=20, pady=3)
                
                key_label = tk.Label(
                    detail_frame,
                    text=f"{key}:",
                    font=("Arial", 11, "bold"),
                    bg="white",
                    fg="#6B3642",
                    width=25,
                    anchor="w"
                )
                key_label.pack(side=tk.LEFT)
                
                value_label = tk.Label(
                    detail_frame,
                    text=str(value),
                    font=("Arial", 11),
                    bg="white",
                    fg="#333",
                    anchor="w"
                )
                value_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            tk.Label(details_card, bg="white", height=1).pack()
        
        # Files list card (for folders)
        if info['is_folder'] and info.get('files'):
            files_card = tk.Frame(self.model_content_frame, bg="white", relief=tk.SOLID, borderwidth=1)
            files_card.pack(fill=tk.X, pady=(0, 20), padx=5)
            
            files_title = tk.Label(
                files_card,
                text="Key Files",
                font=("Arial", 14, "bold"),
                bg="white",
                fg="#6B3642",
                anchor="w"
            )
            files_title.pack(fill=tk.X, padx=20, pady=(20, 15))
            
            for file in info['files'][:15]:  # Show first 15 files
                file_label = tk.Label(
                    files_card,
                    text=f"• {file}",
                    font=("Arial", 11),
                    bg="white",
                    fg="#333",
                    anchor="w"
                )
                file_label.pack(fill=tk.X, padx=30, pady=2)
            
            if len(info['files']) > 15:
                more_label = tk.Label(
                    files_card,
                    text=f"... and {len(info['files']) - 15} more files",
                    font=("Arial", 11, "italic"),
                    bg="white",
                    fg="#8B7378",
                    anchor="w"
                )
                more_label.pack(fill=tk.X, padx=30, pady=(5, 20))
            else:
                tk.Label(files_card, bg="white", height=1).pack()
        
        # Add Interpret button at the bottom (only if dataset is uploaded)
        if self.dataset_path:
            button_container = tk.Frame(self.model_content_frame, bg="#FBF8F5")
            button_container.pack(fill=tk.X, pady=(10, 0))
            
            interpret_btn = tk.Button(
                button_container,
                text="Interpret with SHAP",
                font=("Arial", 12, "bold"),
                bg="#4A7C59",
                fg="white",
                activebackground="#3D6849",
                activeforeground="white",
                relief=tk.FLAT,
                padx=30,
                pady=12,
                cursor="hand2",
                command=self._on_interpret
            )
            interpret_btn.pack(pady=10)
            
    def _check_and_process(self):
        """Check if both files are uploaded and simulate processing"""
        if self.dataset_path and self.model_path:
            self.converting_label.config(text="Processing files...")
            self.is_converting = True
            
            self.after(2000, self._show_success)
            
    def _show_success(self):
        """Show success message and continue button"""
        self.converting_label.config(text="")
        self.success_label.config(text="Upload and processing was successful!")
        self.conversion_success = True
        
        # Show both buttons
        self.button_frame.pack(pady=(10, 0), anchor="w")
        self.continue_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.interpret_btn.pack(side=tk.LEFT)
        
    def _on_continue(self):
        """Handle continue button click"""
        if self.on_navigate:
            self.on_navigate("Train Model")
        else:
            messagebox.showinfo("Continue", "Proceeding to next step...")
    
    def _on_interpret(self):
        """Handle interpret button click - ask for row range"""
        if not self.dataset_path or not self.model_path:
            messagebox.showerror("Error", "Please upload both dataset and model first!")
            return
        
        # Create dialog window
        dialog = tk.Toplevel(self)
        dialog.title("SHAP Interpretation")
        dialog.geometry("400x250")
        dialog.configure(bg="#FBF8F5")
        dialog.resizable(False, False)
        
        # Center the dialog
        dialog.transient(self)
        dialog.grab_set()
        
        # Title
        title = tk.Label(
            dialog,
            text="Select Rows to Interpret",
            font=("Arial", 16, "bold"),
            bg="#FBF8F5",
            fg="#6B3642"
        )
        title.pack(pady=(20, 10))
        
        # Info label
        total_rows = len(self.dataset_df) if self.dataset_df is not None else 0
        info = tk.Label(
            dialog,
            text=f"Total rows in dataset: {total_rows}",
            font=("Arial", 11),
            bg="#FBF8F5",
            fg="#333"
        )
        info.pack(pady=(0, 20))
        
        # Input frame
        input_frame = tk.Frame(dialog, bg="#FBF8F5")
        input_frame.pack(pady=10, padx=30, fill=tk.X)
        
        label = tk.Label(
            input_frame,
            text="Enter row range (e.g., 1-5 or 10-20):",
            font=("Arial", 11),
            bg="#FBF8F5",
            fg="#333"
        )
        label.pack(anchor="w", pady=(0, 10))
        
        entry = tk.Entry(
            input_frame,
            font=("Arial", 12),
            bg="white",
            fg="#333",
            relief=tk.SOLID,
            borderwidth=1
        )
        entry.pack(fill=tk.X, ipady=8)
        entry.insert(0, "0-4")  # Default: first 5 rows
        entry.focus()
        
        # Button frame
        btn_frame = tk.Frame(dialog, bg="#FBF8F5")
        btn_frame.pack(pady=20)
        
        def on_start():
            row_range = entry.get().strip()
            if not row_range:
                messagebox.showerror("Error", "Please enter a row range!")
                return
            
            # Parse row range
            try:
                if '-' in row_range:
                    start, end = row_range.split('-')
                    start_idx = int(start.strip())
                    end_idx = int(end.strip())
                else:
                    start_idx = int(row_range)
                    end_idx = start_idx
                
                # Validate range
                if start_idx < 0 or end_idx >= total_rows:
                    messagebox.showerror("Error", f"Row range must be between 0 and {total_rows-1}!")
                    return
                
                if start_idx > end_idx:
                    messagebox.showerror("Error", "Start index must be less than or equal to end index!")
                    return
                
                dialog.destroy()
                self._run_shap_interpretation(start_idx, end_idx)
                
            except ValueError:
                messagebox.showerror("Error", "Invalid format! Use format like '1-5' or '10-20'")
        
        start_btn = tk.Button(
            btn_frame,
            text="Start Interpretation",
            font=("Arial", 11, "bold"),
            bg="#4A7C59",
            fg="white",
            activebackground="#3D6849",
            activeforeground="white",
            relief=tk.FLAT,
            padx=25,
            pady=10,
            cursor="hand2",
            command=on_start
        )
        start_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tk.Button(
            btn_frame,
            text="Cancel",
            font=("Arial", 11, "bold"),
            bg="#8B7378",
            fg="white",
            activebackground="#6B5358",
            activeforeground="white",
            relief=tk.FLAT,
            padx=25,
            pady=10,
            cursor="hand2",
            command=dialog.destroy
        )
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Bind Enter key
        entry.bind('<Return>', lambda e: on_start())
    
    def _run_shap_interpretation(self, start_idx, end_idx):
        """Run SHAP interpretation with GPU optimization"""
        try:
            # Validate that dataset and model are loaded
            if self.dataset_df is None:
                messagebox.showerror("Error", "No dataset loaded. Please upload a dataset first.")
                return
            
            if not self.model_path:
                messagebox.showerror("Error", "No model loaded. Please upload a model first.")
                return
            
            # Validate path exists
            if not os.path.exists(self.model_path):
                messagebox.showerror("Error", f"Model path does not exist:\n{self.model_path}")
                return
            
            # Show progress window
            progress_window = tk.Toplevel(self)
            progress_window.title("Interpreting...")
            progress_window.geometry("500x200")
            progress_window.configure(bg="#FBF8F5")
            progress_window.resizable(False, False)
            progress_window.transient(self)
            progress_window.grab_set()
            
            # Center the window
            progress_window.update_idletasks()
            x = (progress_window.winfo_screenwidth() // 2) - (500 // 2)
            y = (progress_window.winfo_screenheight() // 2) - (200 // 2)
            progress_window.geometry(f"+{x}+{y}")
            
            status_label = tk.Label(
                progress_window,
                text="Initializing SHAP...",
                font=("Arial", 12),
                bg="#FBF8F5",
                fg="#6B3642",
                wraplength=450
            )
            status_label.pack(expand=True, pady=20)
            
            device_label = tk.Label(
                progress_window,
                text="",
                font=("Arial", 10, "italic"),
                bg="#FBF8F5",
                fg="#4A7C59"
            )
            device_label.pack()
            
            def run_interpretation():
                try:
                    # Test imports
                    import json
                    import sys
                    missing_packages = []
                    
                    try:
                        import shap
                    except ImportError as e:
                        missing_packages.append(f"shap: {str(e)}")
                    
                    try:
                        import numpy as np
                    except ImportError as e:
                        missing_packages.append(f"numpy: {str(e)}")
                    
                    try:
                        import matplotlib
                        matplotlib.use('Agg')  # Use non-interactive backend
                        import matplotlib.pyplot as plt
                    except ImportError as e:
                        missing_packages.append(f"matplotlib: {str(e)}")
                    
                    try:
                        from PIL import Image
                    except ImportError as e:
                        missing_packages.append(f"PIL/Pillow: {str(e)}")
                    
                    if missing_packages:
                        progress_window.destroy()
                        error_details = "\n".join(missing_packages)
                        messagebox.showerror(
                            "Missing Dependencies",
                            f"The following packages are missing:\n\n{error_details}\n\n"
                            f"Python: {sys.executable}\n\n"
                            f"Install: pip install shap matplotlib pillow scikit-learn transformers torch"
                        )
                        return
                    
                    # Check for GPU
                    device = 'cpu'
                    try:
                        import torch
                        if torch.cuda.is_available():
                            device = 'cuda'
                            device_label.config(text=f"✓ GPU: {torch.cuda.get_device_name(0)}")
                        else:
                            device_label.config(text="Using CPU")
                    except:
                        device_label.config(text="Using CPU")
                    
                    progress_window.update()
                    status_label.config(text=f"Loading model ({device.upper()})...")
                    progress_window.update()
                    
                    # Load model
                    model = None
                    tokenizer = None
                    is_transformer = False
                    
                    # Check if it's a folder (likely transformer) or file
                    if os.path.isdir(self.model_path):
                        # BERT/Transformer model
                        try:
                            from transformers import AutoModelForSequenceClassification, AutoTokenizer
                            
                            status_label.config(text="Loading BERT/Transformer model...")
                            progress_window.update()

                            print(f"DEBUG: Loading BERT model from: {self.model_path}")
                            print(f"DEBUG: Files in directory: {os.listdir(self.model_path)}")
                            
                            # Initialize tokenizer
                            tokenizer = None
                            
                            # First try to load tokenizer from local folder
                            try:
                                tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                                print("DEBUG: Tokenizer loaded successfully from local folder")
                            except Exception as e:
                                print(f"DEBUG: Local tokenizer loading failed: {e}")
                            
                            # If tokenizer is still None, download it
                            if tokenizer is None:
                                status_label.config(text="Tokenizer missing, downloading...")
                                progress_window.update()
                                
                                try:
                                    # Read config to determine model type
                                    config_path = os.path.join(self.model_path, 'config.json')
                                    if os.path.exists(config_path):
                                        with open(config_path, 'r') as f:
                                            config = json.load(f)
                                        
                                        model_type = config.get('model_type', '')
                                        architecture = config.get('architectures', [''])[0] if config.get('architectures') else ''
                                        print(f"DEBUG: Model type from config: {model_type}")
                                        print(f"DEBUG: Architecture: {architecture}")
                                        
                                        # Map to standard Hugging Face model names
                                        model_name_map = {
                                            'bert': 'bert-base-uncased',
                                            'roberta': 'roberta-base',
                                            'distilbert': 'distilbert-base-uncased',
                                            'albert': 'albert-base-v2'
                                        }
                                        
                                        base_model_name = None

                                        if 'bert' in architecture.lower():
                                            if 'distil' in architecture.lower():
                                                base_model_name = 'distilbert-base-uncased'
                                            elif 'roberta' in architecture.lower():
                                                base_model_name = 'roberta-base'
                                            else:
                                                base_model_name = 'bert-base-uncased'
                                        else:
                                            # Fallback to model_type
                                            base_model_name = model_name_map.get(model_type, 'bert-base-uncased')
                                        
                                        print(f"DEBUG: Trying to download tokenizer for: {base_model_name}")

                                        # Download and save tokenizer
                                        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                                        tokenizer.save_pretrained(self.model_path)
                                        print("DEBUG: Tokenizer downloaded and saved to model folder")
                                        print(f"DEBUG: New files in directory: {os.listdir(self.model_path)}")
                                        
                                    else:
                                        raise Exception("config.json not found in model folder")
                                        
                                except Exception as download_error:
                                    progress_window.destroy()
                                    messagebox.showerror(
                                        "Tokenizer Error",
                                        f"Failed to download tokenizer:\n\n{str(download_error)}\n\n"
                                        "Please ensure:\n"
                                        "1. You have internet connection\n"
                                        "2. Your model folder is writable\n"
                                        "3. config.json exists in your model folder\n\n"
                                        f"Current files: {os.listdir(self.model_path)}"
                                    )
                                    return
                            
                            # Now load the model
                            try:
                                model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                                print("DEBUG: Model loaded successfully")
                            except Exception as e:
                                progress_window.destroy()
                                messagebox.showerror(
                                    "Model Loading Error", 
                                    f"Failed to load model:\n{str(e)}\n\n"
                                    "This might be due to:\n"
                                    "1. Model architecture mismatch\n"
                                    "2. Corrupted model files\n"
                                    "3. Missing dependencies\n\n"
                                    f"Files found: {os.listdir(self.model_path)}"
                                )
                                return
                            
                            if device == 'cuda':
                                import torch
                                model = model.to(device)
                            model.eval()
                            is_transformer = True
                            
                        except ImportError:
                            progress_window.destroy()
                            messagebox.showerror(
                                "Missing Libraries",
                                "Transformers and/or PyTorch not installed.\n\n"
                                f"Install with: pip install transformers torch"
                            )
                            return
                        except Exception as e:
                            progress_window.destroy()
                            messagebox.showerror(
                                "Model Loading Error",
                                f"Failed to load transformer model:\n\n{str(e)}\n\n"
                                f"Path: {self.model_path}\n"
                                f"Files: {os.listdir(self.model_path) if os.path.exists(self.model_path) else 'PATH_NOT_FOUND'}"
                            )
                            return
                    
                    elif self.model_path.endswith(('.pkl', '.joblib')):
                        # Scikit-learn style models
                        if self.model_path.endswith('.pkl'):
                            with open(self.model_path, 'rb') as f:
                                model = pickle.load(f)
                        else:
                            import joblib
                            model = joblib.load(self.model_path)
                        
                        # Move XGBoost to GPU if available
                        if 'XGB' in type(model).__name__ and device == 'cuda':
                            try:
                                model.set_param({'device': 'cuda'})
                            except:
                                pass
                    else:
                        progress_window.destroy()
                        messagebox.showerror(
                            "Unsupported Model Format",
                            f"Model format not supported: {os.path.basename(self.model_path)}\n\n"
                            "Supported formats:\n"
                            "- Transformer models (folder with config.json)\n"
                            "- Pickle files (.pkl)\n"
                            "- Joblib files (.joblib)"
                        )
                        return
                    
                    # Get data subset
                    status_label.config(text=f"Processing rows {start_idx}-{end_idx}...")
                    progress_window.update()
                    
                    X_subset = self.dataset_df.iloc[start_idx:end_idx+1].copy()
                    
                    # Compute SHAP values
                    status_label.config(text="Computing SHAP values (may take a while)...")
                    progress_window.update()
                    
                    shap_values = None
                    feature_names = None
                    
                    if is_transformer:
                        # SHAP for transformers
                        import torch
                        import gc
                        
                        # Clear memory
                        gc.collect()
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                        
                        # Find text column
                        text_column = None
                        for col in ['text', 'sentence', 'review', 'comment', 'content']:
                            if col in X_subset.columns:
                                text_column = col
                                break
                        
                        if text_column is None:
                            # Use first string column
                            for col in X_subset.columns:
                                if X_subset[col].dtype == 'object':
                                    text_column = col
                                    break
                        
                        if text_column is None:
                            progress_window.destroy()
                            messagebox.showerror(
                                "No Text Column",
                                "Could not find a text column in the dataset.\n\n"
                                "Transformer models require text data.\n"
                                "Expected column names: 'text', 'sentence', 'review', etc."
                            )
                            return
                        
                        texts = X_subset[text_column].tolist()
                        
                        # Limit number of samples for SHAP (transformers are memory-intensive)
                        max_samples = min(5, len(texts))
                        texts = texts[:max_samples]
                        
                        status_label.config(text=f"Computing SHAP for {max_samples} samples (this may take several minutes)...")
                        progress_window.update()
                        
                        def predict_fn(text_list):
                            """Prediction function for SHAP"""
                            # Handle both single strings and lists
                            if isinstance(text_list, str):
                                text_list = [text_list]
                            
                            inputs = tokenizer(
                                text_list, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True, 
                                max_length=128  # Reduced for speed
                            )
                            if device == 'cuda':
                                inputs = {k: v.to(device) for k, v in inputs.items()}
                            
                            with torch.no_grad():
                                outputs = model(**inputs)
                            
                            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                            return probs
                        
                        try:
                            # Method 1: Try using Pipeline explainer (most compatible)
                            try:
                                from transformers import pipeline
                                
                                status_label.config(text="Using pipeline-based explainer...")
                                progress_window.update()
                                
                                # Create a pipeline
                                pipe = pipeline(
                                    "text-classification",
                                    model=model,
                                    tokenizer=tokenizer,
                                    device=0 if device == 'cuda' else -1,
                                    return_all_scores=True
                                )
                                
                                # Create explainer using the pipeline
                                explainer = shap.Explainer(pipe)
                                
                                # Compute SHAP values - pass as list of strings
                                shap_values = explainer(texts)
                                feature_names = texts
                                
                            except Exception as pipeline_error:
                                print(f"Pipeline method failed: {pipeline_error}")
                                
                                # Method 2: Manual tokenization approach
                                status_label.config(text="Using manual tokenization approach...")
                                progress_window.update()
                                
                                def f(x):
                                    """Wrapper that handles tokenization internally"""
                                    # x will be a numpy array of strings
                                    if isinstance(x, np.ndarray):
                                        text_list = x.tolist()
                                    elif isinstance(x, list):
                                        text_list = x
                                    else:
                                        text_list = [str(x)]
                                    
                                    # Tokenize
                                    inputs = tokenizer(
                                        text_list,
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True,
                                        max_length=128
                                    )
                                    
                                    if device == 'cuda':
                                        inputs = {k: v.to(device) for k, v in inputs.items()}
                                    
                                    # Get predictions
                                    with torch.no_grad():
                                        outputs = model(**inputs)
                                    
                                    return torch.softmax(outputs.logits, dim=1).cpu().numpy()
                                
                                # Convert texts to numpy array
                                texts_array = np.array(texts)
                                
                                # Use KernelExplainer with string data
                                background_data = texts_array[:min(2, len(texts_array))]
                                explainer = shap.KernelExplainer(f, background_data)
                                
                                # Explain
                                shap_values = explainer.shap_values(texts_array)
                                
                                # Handle multi-class output
                                if isinstance(shap_values, list):
                                    shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                                
                                feature_names = texts
                            
                        except Exception as e:
                            progress_window.destroy()
                            import traceback
                            error_trace = traceback.format_exc()
                            messagebox.showerror(
                                "SHAP Error",
                                f"SHAP analysis failed for transformer model:\n\n{str(e)}\n\n"
                                "Tips:\n"
                                "- Transformer SHAP is very memory-intensive\n"
                                "- Try with just 1-2 rows\n"
                                "- Ensure your GPU has enough memory\n"
                                "- Or use CPU mode (slower but more stable)\n\n"
                                f"Full error:\n{error_trace[:500]}"
                            )
                            return
                        
                    else:
                        # SHAP for tabular models
                        # Remove target columns
                        target_columns = ['target', 'label', 'class', 'y']
                        for col in target_columns:
                            if col in X_subset.columns:
                                X_subset = X_subset.drop(columns=[col])
                        
                        # Select numeric columns
                        X_array = X_subset.select_dtypes(include=[np.number]).values
                        feature_names = list(X_subset.select_dtypes(include=[np.number]).columns)
                        
                        if X_array.shape[1] == 0:
                            progress_window.destroy()
                            messagebox.showerror(
                                "No Numeric Data",
                                "No numeric columns found in dataset.\n\n"
                                "SHAP analysis requires numeric features for tabular models."
                            )
                            return
                        
                        # Compute SHAP
                        if 'XGB' in type(model).__name__:
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_array)
                        else:
                            try:
                                explainer = shap.LinearExplainer(model, X_array)
                                shap_values = explainer.shap_values(X_array)
                            except:
                                # Fallback to KernelExplainer
                                status_label.config(text="Using KernelExplainer (slower but works for any model)...")
                                progress_window.update()
                                background = X_array[:min(50, len(X_array))]
                                explainer = shap.KernelExplainer(model.predict_proba, background)
                                shap_values = explainer.shap_values(X_array)
                        
                        # Handle multi-class
                        if isinstance(shap_values, list) and len(shap_values) > 1:
                            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                    
                    # Generate plots
                    status_label.config(text="Generating visualizations...")
                    progress_window.update()
                    
                    import matplotlib.pyplot as plt
                    
                    # Output directory
                    if self.dataset_path and os.path.exists(self.dataset_path):
                        output_dir = os.path.dirname(self.dataset_path)
                    else:
                        output_dir = os.path.expanduser("~")
                    
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Plot 1: Bar plot
                    plt.figure(figsize=(12, 8))
                    if is_transformer:
                        shap.plots.bar(shap_values, show=False)
                    else:
                        shap.summary_plot(shap_values, X_array if not is_transformer else None, 
                                        feature_names=feature_names, show=False, plot_type="bar")
                    plt.title(f"SHAP Feature Importance (Rows {start_idx}-{end_idx})", 
                            fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    output_path1 = os.path.join(output_dir, f"shap_interpretation_{start_idx}_{end_idx}.png")
                    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Plot 2: Summary plot
                    plt.figure(figsize=(12, 10))
                    if is_transformer:
                        shap.plots.beeswarm(shap_values, show=False)
                    else:
                        shap.summary_plot(shap_values, X_array, feature_names=feature_names, show=False)
                    plt.title(f"SHAP Summary Plot (Rows {start_idx}-{end_idx})", 
                            fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    output_path2 = os.path.join(output_dir, f"shap_summary_{start_idx}_{end_idx}.png")
                    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    progress_window.destroy()
                    
                    # Show results
                    self._show_shap_results(start_idx, end_idx, output_path1, output_path2, device)
                    
                except ImportError as e:
                    progress_window.destroy()
                    import sys
                    messagebox.showerror(
                        "Missing Dependencies",
                        f"Import Error: {str(e)}\n\n"
                        f"Python: {sys.executable}\n\n"
                        f"Install: pip install shap matplotlib pillow scikit-learn transformers torch"
                    )
                except Exception as e:
                    progress_window.destroy()
                    import traceback
                    error_details = traceback.format_exc()
                    messagebox.showerror("Error", 
                        f"SHAP interpretation failed:\n\n{str(e)}\n\n"
                        f"Details:\n{error_details[:800]}")
            
            # Run in thread
            import threading
            thread = threading.Thread(target=run_interpretation, daemon=True)
            thread.start()
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            messagebox.showerror("Error", 
                f"Failed to start:\n\n{str(e)}\n\n"
                f"Details:\n{error_details[:800]}")
               
    def _show_shap_results(self, start_idx, end_idx, plot_path1, plot_path2, device):
        """Display SHAP interpretation results"""
        self.showing_shap_results = True
        self.showing_preview = False
        self.showing_model_info = False
        self.upload_view.pack_forget()
        self.preview_view.pack_forget()
        self.model_view.pack_forget()
        self.shap_view.pack(fill=tk.BOTH, expand=True)
        
        # Clear previous content
        for widget in self.shap_content_frame.winfo_children():
            widget.destroy()
        
        # Info card
        info_card = tk.Frame(self.shap_content_frame, bg="white", relief=tk.SOLID, borderwidth=1)
        info_card.pack(fill=tk.X, pady=(0, 20), padx=5)
        
        title_label = tk.Label(
            info_card,
            text=f"SHAP Interpretation: Rows {start_idx} to {end_idx}",
            font=("Arial", 18, "bold"),
            bg="white",
            fg="#6B3642",
            anchor="w"
        )
        title_label.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        device_label = tk.Label(
            info_card,
            text=f"Computation Device: {device.upper()}",
            font=("Arial", 12),
            bg="white",
            fg="#4A7C59",
            anchor="w"
        )
        device_label.pack(fill=tk.X, padx=20, pady=5)
        
        rows_label = tk.Label(
            info_card,
            text=f"Rows Analyzed: {end_idx - start_idx + 1}",
            font=("Arial", 12),
            bg="white",
            fg="#333",
            anchor="w"
        )
        rows_label.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Display plots
        try:
            from PIL import Image, ImageTk
            
            # Plot 1: Feature Importance
            plot1_frame = tk.Frame(self.shap_content_frame, bg="white", relief=tk.SOLID, borderwidth=1)
            plot1_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20), padx=5)
            
            plot1_title = tk.Label(
                plot1_frame,
                text="Feature Importance",
                font=("Arial", 14, "bold"),
                bg="white",
                fg="#6B3642"
            )
            plot1_title.pack(pady=(15, 10))
            
            img1 = Image.open(plot_path1)
            img1.thumbnail((900, 600))
            photo1 = ImageTk.PhotoImage(img1)
            
            label1 = tk.Label(plot1_frame, image=photo1, bg="white")
            label1.image = photo1  # Keep reference
            label1.pack(padx=20, pady=(0, 15))
            
            # Plot 2: Summary Plot
            plot2_frame = tk.Frame(self.shap_content_frame, bg="white", relief=tk.SOLID, borderwidth=1)
            plot2_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20), padx=5)
            
            plot2_title = tk.Label(
                plot2_frame,
                text="Detailed Summary Plot",
                font=("Arial", 14, "bold"),
                bg="white",
                fg="#6B3642"
            )
            plot2_title.pack(pady=(15, 10))
            
            img2 = Image.open(plot_path2)
            img2.thumbnail((900, 600))
            photo2 = ImageTk.PhotoImage(img2)
            
            label2 = tk.Label(plot2_frame, image=photo2, bg="white")
            label2.image = photo2  # Keep reference
            label2.pack(padx=20, pady=(0, 15))
            
        except ImportError:
            error_label = tk.Label(
                self.shap_content_frame,
                text="PIL/Pillow not installed. Cannot display images.\n\n"
                     f"Images saved to:\n{plot_path1}\n{plot_path2}",
                font=("Arial", 11),
                bg="#FBF8F5",
                fg="#333",
                justify=tk.LEFT
            )
            error_label.pack(pady=20)
        except Exception as e:
            error_label = tk.Label(
                self.shap_content_frame,
                text=f"Error displaying images: {str(e)}\n\n"
                     f"Images saved to:\n{plot_path1}\n{plot_path2}",
                font=("Arial", 11),
                bg="#FBF8F5",
                fg="#333",
                justify=tk.LEFT
            )
            error_label.pack(pady=20)
        
        # Buttons
        button_frame = tk.Frame(self.shap_content_frame, bg="#FBF8F5")
        button_frame.pack(pady=20)
        
        open_btn = tk.Button(
            button_frame,
            text="Open Image Files",
            font=("Arial", 11, "bold"),
            bg="#4A7C59",
            fg="white",
            activebackground="#3D6849",
            activeforeground="white",
            relief=tk.FLAT,
            padx=25,
            pady=10,
            cursor="hand2",
            command=lambda: self._open_files([plot_path1, plot_path2])
        )
        open_btn.pack(side=tk.LEFT, padx=5)
        
        new_interpretation_btn = tk.Button(
            button_frame,
            text="New Interpretation",
            font=("Arial", 11, "bold"),
            bg="#A15264",
            fg="white",
            activebackground="#8B4555",
            activeforeground="white",
            relief=tk.FLAT,
            padx=25,
            pady=10,
            cursor="hand2",
            command=self._on_interpret
        )
        new_interpretation_btn.pack(side=tk.LEFT, padx=5)
    
    def _open_files(self, file_paths):
        """Open image files in default viewer"""
        import subprocess
        import platform
        
        for path in file_paths:
            try:
                if platform.system() == 'Windows':
                    os.startfile(path)
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.call(['open', path])
                else:  # Linux
                    subprocess.call(['xdg-open', path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file:\n{path}\n\n{str(e)}")
        
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