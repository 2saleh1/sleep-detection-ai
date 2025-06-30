import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageOps
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import tensorflow as tf
from tensorflow.keras.models import load_model

class SleepDetector:
    def __init__(self):
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)
        
        # Load the model
        self.model = load_model("keras_Model.h5", compile=False)
        
        # Load the labels
        with open("labels.txt", "r") as f:
            self.class_names = f.readlines()
    
    def predict_image(self, image_path):
        # Create the array of the right shape to feed into the keras model
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        
        # Load and process the image
        image = Image.open(image_path).convert("RGB")
        
        # Resize the image to 224x224 and crop from center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.asarray(image)
        
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # Load the image into the array
        data[0] = normalized_image_array
        
        # Make prediction
        prediction = self.model.predict(data, verbose=0)
        index = np.argmax(prediction)
        class_name = self.class_names[index].strip()
        confidence_score = prediction[0][index]
        
        return class_name[2:], confidence_score

class ModernButton(tk.Button):
    def __init__(self, parent, text, command=None, bg_color="#4285F4", hover_color="#3367D6", 
                 text_color="white", **kwargs):
        super().__init__(parent, text=text, command=command, 
                        bg=bg_color, fg=text_color, relief="flat", 
                        font=("Segoe UI", 11, "bold"), cursor="hand2",
                        padx=25, pady=15, border=0, **kwargs)
        
        self.bg_color = bg_color
        self.hover_color = hover_color
        
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
    
    def on_enter(self, e):
        self.config(bg=self.hover_color)
    
    def on_leave(self, e):
        self.config(bg=self.bg_color)

class SleepDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Sleep Detection System")
        self.root.geometry("950x800")
        self.root.resizable(False, False)
        
        # Dark theme color scheme
        self.colors = {
            'primary': '#2196F3',
            'primary_dark': '#1976D2',
            'primary_light': '#64B5F6',
            'success': '#4CAF50',
            'success_light': '#81C784',
            'warning': '#FF9800',
            'danger': '#F44336',
            'background': '#1E1E1E',  # Dark background
            'surface': '#2D2D2D',     # Dark surface
            'surface_alt': '#3A3A3A', # Lighter dark surface
            'surface_light': '#2A2A2A', # Slightly lighter surface
            'card_bg': '#252525',     # Dark card background
            'input_bg': '#2F2F2F',    # Dark input background
            'text_primary': '#FFFFFF', # White text
            'text_secondary': '#B0B0B0', # Light gray text
            'text_light': '#808080',   # Medium gray text
            'border': '#404040',       # Dark border
            'border_light': '#505050', # Lighter dark border
            'shadow': '#1A1A1A'       # Very dark shadow
        }
        
        self.root.configure(bg=self.colors['background'])
        
        # Initialize the sleep detector
        self.detector = None
        self.selected_image_path = None
        self.init_detector()
        self.setup_ui()
    
    def init_detector(self):
        try:
            self.detector = SleepDetector()
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load AI model:\n{str(e)}")
    
    def setup_ui(self):
        # Main container with padding and background
        main_frame = tk.Frame(self.root, bg=self.colors['background'])
        main_frame.pack(fill='both', expand=True, padx=30, pady=30)
        
        # Header section
        self.create_header(main_frame)
        
        # Content section
        content_frame = tk.Frame(main_frame, bg=self.colors['background'])
        content_frame.pack(fill='both', expand=True, pady=(30, 0))
        
        # Left panel - Image selection and display
        left_panel = tk.Frame(content_frame, bg=self.colors['card_bg'], relief='solid', bd=1)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        # Right panel - Controls and results
        right_panel = tk.Frame(content_frame, bg=self.colors['card_bg'], relief='solid', bd=1)
        right_panel.pack(side='right', fill='y', padx=(15, 0))
        right_panel.config(width=340)
        right_panel.pack_propagate(False)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
    
    def create_header(self, parent):
        # Dark themed header
        header_frame = tk.Frame(parent, bg=self.colors['primary'], height=80)
        header_frame.pack(fill='x', pady=(0, 25))
        header_frame.pack_propagate(False)
        
        # Add subtle shadow
        shadow_header = tk.Frame(parent, bg=self.colors['shadow'], height=2)
        shadow_header.place(in_=header_frame, x=0, y=80, relwidth=1)
        header_frame.lift()
        
        # Header content - centered title only
        header_content = tk.Frame(header_frame, bg=self.colors['primary'])
        header_content.pack(expand=True)
        
        # Title centered and larger
        title_label = tk.Label(
            header_content,
            text="🧠 AI Sleep Detection System",
            font=("Segoe UI", 28, "bold"),
            bg=self.colors['primary'],
            fg='white'
        )
        title_label.pack(expand=True)
        
        # Status indicator in corner
        status_frame = tk.Frame(header_frame, bg=self.colors['primary'])
        status_frame.place(relx=0.95, rely=0.1, anchor='ne')
        
        model_status = "🟢 Ready" if self.detector else "🔴 Error"
        status_label = tk.Label(
            status_frame,
            text=model_status,
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['primary'],
            fg='white'
        )
        status_label.pack()
    
    def setup_left_panel(self, parent):
        # Panel with dark background
        panel_bg = tk.Frame(parent, bg=self.colors['surface_light'])
        panel_bg.pack(fill='both', expand=True, padx=4, pady=4)
        
        # Panel title
        title_frame = tk.Frame(panel_bg, bg=self.colors['surface_light'])
        title_frame.pack(fill='x', padx=25, pady=(25, 20))
        
        tk.Label(
            title_frame,
            text="📷 Image Preview",
            font=("Segoe UI", 18, "bold"),
            bg=self.colors['surface_light'],
            fg=self.colors['text_primary']
        ).pack(anchor='w')
        
        # Image display area
        self.image_container = tk.Frame(
            panel_bg, 
            bg=self.colors['border'], 
            relief='solid', 
            bd=1
        )
        self.image_container.pack(fill='both', expand=True, padx=25, pady=(0, 20))
        
        # Inner frame with dark background
        image_inner = tk.Frame(self.image_container, bg=self.colors['input_bg'])
        image_inner.pack(fill='both', expand=True, padx=3, pady=3)
        
        self.image_label = tk.Label(
            image_inner,
            text="No image selected\n\n📁 Click 'Select Image' to choose a photo\n\n💡 Supported: JPG, PNG, BMP",
            font=("Segoe UI", 13),
            bg=self.colors['input_bg'],
            fg=self.colors['text_secondary'],
            justify='center'
        )
        self.image_label.pack(fill='both', expand=True)
        
        # Image info with dark background
        info_frame = tk.Frame(panel_bg, bg=self.colors['surface_alt'], relief='solid', bd=1)
        info_frame.pack(fill='x', padx=25, pady=(0, 25))
        
        self.image_info_label = tk.Label(
            info_frame,
            text="",
            font=("Segoe UI", 10),
            bg=self.colors['surface_alt'],
            fg=self.colors['text_secondary'],
            pady=12
        )
        self.image_info_label.pack()
    
    def setup_right_panel(self, parent):
        
        # Panel with dark background
        panel_bg = tk.Frame(parent, bg=self.colors['surface_light'])
        panel_bg.pack(fill='both', expand=True, padx=4, pady=4)
        
        # Panel title
        title_frame = tk.Frame(panel_bg, bg=self.colors['surface_light'])
        title_frame.pack(fill='x', padx=25, pady=(25, 25))
        
        tk.Label(
            title_frame,
            text="⚙️ Controls",
            font=("Segoe UI", 18, "bold"),
            bg=self.colors['surface_light'],
            fg=self.colors['text_primary']
        ).pack(anchor='w')
        
        # Buttons container
        buttons_frame = tk.Frame(panel_bg, bg=self.colors['surface_light'])
        buttons_frame.pack(fill='x', padx=25, pady=(0, 30))
        
        # Select image button
        self.select_btn = ModernButton(
            buttons_frame,
            text="📁 Select Image",
            command=self.select_image,
            bg_color=self.colors['primary'],
            hover_color=self.colors['primary_dark']
        )
        self.select_btn.pack(fill='x', pady=(0, 15))
        
        # Analyze button
        self.analyze_btn = ModernButton(
            buttons_frame,
            text="🔍 Analyze Image",
            command=self.predict_sleep,
            bg_color=self.colors['success'],
            hover_color='#388E3C',
            state='disabled'
        )
        self.analyze_btn.pack(fill='x')
        
        # Results section
        results_title_frame = tk.Frame(panel_bg, bg=self.colors['surface_light'])
        results_title_frame.pack(fill='x', padx=25, pady=(25, 15))
        
        tk.Label(
            results_title_frame,
            text="📊 Analysis Results",
            font=("Segoe UI", 18, "bold"),
            bg=self.colors['surface_light'],
            fg=self.colors['text_primary']
        ).pack(anchor='w')
        
        # Results container with much larger height to ensure confidence is visible
        self.results_container = tk.Frame(
            panel_bg, 
            bg=self.colors['surface_alt'], 
            relief='solid', 
            bd=1,
            height=280  # Increased from 220 to 280 for more space
        )
        self.results_container.pack(fill='x', padx=25, pady=(0, 25))
        self.results_container.pack_propagate(False)  # Prevent shrinking
        
        # Status result with compact spacing
        self.status_frame = tk.Frame(self.results_container, bg=self.colors['surface_alt'])
        self.status_frame.pack(fill='x', padx=20, pady=(20, 15))
        
        tk.Label(
            self.status_frame,
            text="Sleep Status:",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['surface_alt'],
            fg=self.colors['text_secondary']
        ).pack(anchor='w')
        
        self.result_label = tk.Label(
            self.status_frame,
            text="Awaiting analysis...",
            font=("Segoe UI", 16, "bold"),  # Reduced from 18 to 16
            bg=self.colors['surface_alt'],
            fg=self.colors['text_light']
        )
        self.result_label.pack(anchor='w', pady=(8, 0))
        
        # Separator
        separator = tk.Frame(self.results_container, height=2, bg=self.colors['border'])
        separator.pack(fill='x', padx=20, pady=10)
        
        # Confidence result with guaranteed space
        self.confidence_frame = tk.Frame(self.results_container, bg=self.colors['surface_alt'])
        self.confidence_frame.pack(fill='x', padx=20, pady=(5, 20))
        
        tk.Label(
            self.confidence_frame,
            text="Confidence Level:",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['surface_alt'],
            fg=self.colors['text_secondary']
        ).pack(anchor='w')
        
        self.confidence_label = tk.Label(
            self.confidence_frame,
            text="---%",
            font=("Segoe UI", 20, "bold"),  # Reduced from 22 to 20
            bg=self.colors['surface_alt'],
            fg=self.colors['text_light']
        )
        self.confidence_label.pack(anchor='w', pady=(10, 0))
        
        # Progress bar (initially hidden)
        self.progress_frame = tk.Frame(panel_bg, bg=self.colors['surface_light'])
        
        tk.Label(
            self.progress_frame,
            text="🤖 Analyzing image with AI...",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors['surface_light'],
            fg=self.colors['text_primary']
        ).pack(pady=(0, 15))
        
        self.progress = ttk.Progressbar(
            self.progress_frame,
            mode='indeterminate',
            length=290
        )
        self.progress.pack()
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image for sleep detection analysis",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.selected_image_path = file_path
            self.display_image(file_path)
            self.analyze_btn.config(state='normal')
            self.clear_results()
    
    def display_image(self, image_path):
        try:
            # Open and process image
            image = Image.open(image_path)
            
            # Get image info
            width, height = image.size
            file_size = os.path.getsize(image_path) / 1024  # KB
            file_name = os.path.basename(image_path)
            
            # Update info label
            self.image_info_label.config(
                text=f"📁 {file_name} | 📐 {width} × {height}px | 💾 {file_size:.1f}KB"
            )
            
            # Resize for display
            container_width = 520
            container_height = 420
            
            image.thumbnail((container_width, container_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load image:\n{str(e)}")
            self.image_info_label.config(text="❌ Failed to load image")
    
    def predict_sleep(self):
        if not self.selected_image_path:
            messagebox.showwarning("No Image", "Please select an image first!")
            return
        
        if not self.detector:
            messagebox.showerror("Model Error", "AI model is not loaded!")
            return
        
        # Show progress
        self.progress_frame.pack(fill='x', padx=25, pady=(0, 25))
        self.progress.start()
        self.analyze_btn.config(state='disabled', text="🔄 Analyzing...")
        self.root.update()
        
        try:
            # Make prediction
            class_name, confidence = self.detector.predict_image(self.selected_image_path)
            
            # Hide progress
            self.progress.stop()
            self.progress_frame.pack_forget()
            self.analyze_btn.config(state='normal', text="🔍 Analyze Image")
            
            # Display results
            self.display_results(class_name, confidence)
            
        except Exception as e:
            self.progress.stop()
            self.progress_frame.pack_forget()
            self.analyze_btn.config(state='normal', text="🔍 Analyze Image")
            messagebox.showerror("Analysis Error", f"Failed to analyze image:\n{str(e)}")
    
    def display_results(self, class_name, confidence):
        # Determine result styling
        if class_name.lower() == 'sleep':
            result_color = self.colors['danger']
            status_text = "😴 SLEEPING"
        else:
            result_color = self.colors['success']
            status_text = "😊 AWAKE"
        
        # Update result labels
        self.result_label.config(text=status_text, fg=result_color)
        
        confidence_percentage = confidence * 100
        
        # Color code confidence for dark theme
        if confidence_percentage >= 85:
            conf_color = self.colors['success_light']
        elif confidence_percentage >= 70:
            conf_color = self.colors['warning']
        else:
            conf_color = self.colors['danger']
        
        self.confidence_label.config(
            text=f"{confidence_percentage:.1f}%",
            fg=conf_color
        )
        
        # Subtle flash effect for dark theme
        self.flash_results()
    
    def flash_results(self):
        # Dark theme flash animation
        original_bg = self.results_container.cget('bg')
        self.results_container.config(bg=self.colors['primary_dark'])
        self.root.after(200, lambda: self.results_container.config(bg=original_bg))
    
    def clear_results(self):
        self.result_label.config(text="Awaiting analysis...", fg=self.colors['text_light'])
        self.confidence_label.config(text="---%", fg=self.colors['text_light'])

def main():
    root = tk.Tk()
    app = SleepDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()