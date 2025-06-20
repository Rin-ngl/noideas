# main.py
# man, I can't code. if this works, it's a miracle.

import customtkinter as ctk
from customtkinter import CTk, CTkFrame, CTkButton, CTkLabel, CTkEntry, CTkTextbox, CTkOptionMenu
import transformers
import torch
import os
import threading
import configparser # To remember settings

# --- Configuration ---
# This is where the app will look for your downloaded models.
# Make sure you create this folder next to the script.
MODELS_DIR = "models" 
CONFIG_FILE = "config.ini"


class App(CTk):
    def __init__(self):
        super().__init__()

        # --- Basic Window Setup ---
        self.title("noideas")
        self.geometry("800x600")
        ctk.set_appearance_mode("dark")
        
        # --- App State Variables ---
        # These variables will hold the state of the application
        self.model = None
        self.tokenizer = None
        self.model_name = None # To track which model is currently loaded
        self.is_generating = False
        self.available_models = self.find_available_models()

        # --- UI Setup ---
        self.setup_ui()
        
        # --- Load last used settings ---
        self.load_settings()

        # --- Save settings when the window is closed ---
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        """Creates and arranges all the visual components of the app."""
        # Configure the grid layout to make widgets resize nicely
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1) # The input textbox will grow
        self.grid_rowconfigure(3, weight=3) # The output textbox will grow more

        # --- Model Selection ---
        model_frame = CTkFrame(self)
        model_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        model_frame.grid_columnconfigure(1, weight=1)

        CTkLabel(model_frame, text="Model:", font=("Arial", 14, "bold")).grid(row=0, column=0, padx=10, pady=10)
        
        # The dropdown menu for choosing a model
        self.model_selector = CTkOptionMenu(
            model_frame, 
            values=self.available_models if self.available_models else ["No models found"],
            command=self.on_model_select
        )
        self.model_selector.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        if not self.available_models:
            self.model_selector.configure(state="disabled")

        # --- User Input Box ---
        CTkLabel(self, text="Your half-baked idea:", font=("Arial", 14, "bold")).grid(row=1, column=0, padx=10, pady=(5,0), sticky="nw")
        self.input_box = CTkTextbox(self, height=100, font=("Arial", 14))
        self.input_box.grid(row=1, column=0, padx=10, pady=(0,10), sticky="nsew")
        self.input_box.insert("1.0", "A dating app for ghosts.") # A fun default prompt

        # --- Generate Button ---
        self.generate_button = CTkButton(
            self, 
            text="✨ Spiral into Nonsense ✨", 
            font=("Arial", 16, "bold"),
            command=self.start_generation_thread
        )
        self.generate_button.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        # --- Output Box ---
        CTkLabel(self, text="Result:", font=("Arial", 14, "bold")).grid(row=3, column=0, padx=10, pady=(5,0), sticky="nw")
        self.output_box = CTkTextbox(self, font=("Arial", 14))
        self.output_box.grid(row=3, column=0, padx=10, pady=(0,10), sticky="nsew")
        self.output_box.configure(state="disabled") # read-only

        # --- Status Bar ---
        self.status_label = CTkLabel(self, text="Ready. Select a model to begin.", anchor="w")
        self.status_label.grid(row=4, column=0, padx=10, pady=5, sticky="ew")

    def find_available_models(self):
        """Scans the MODELS_DIR for valid model folders."""
        if not os.path.isdir(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            return []
        
        models = []
        for item in os.listdir(MODELS_DIR):
            path = os.path.join(MODELS_DIR, item)
            # A simple check: if it's a directory and has a config.json, it's probably a model
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
                models.append(item)
        
        if not models:
            self.update_status(f"Error: No models found in '{MODELS_DIR}' folder. Download a model from HuggingFace.", "error")
        return models

    def on_model_select(self, selected_model):
        """Called when a user selects a new model from the dropdown."""
        if selected_model and selected_model != self.model_name:
            # Load the model in a separate thread to not freeze the UI
            threading.Thread(target=self.load_model, args=(selected_model,), daemon=True).start()

    def load_model(self, model_name):
        """Loads the selected LLM and tokenizer from disk."""
        try:
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache() # Free up VRAM

            self.update_status(f"Loading {model_name}... this may take a moment.")
            model_path = os.path.join(MODELS_DIR, model_name)
            
            # This is the magic from the transformers library
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16, # Use float16 for less VRAM usage
                device_map="auto" # Automatically use GPU if available
            )
            
            self.model_name = model_name
            self.update_status(f"Model '{model_name}' loaded successfully. Ready to generate!", "success")
        except Exception as e:
            self.update_status(f"Error loading model: {e}", "error")
            self.model_name = None

    def start_generation_thread(self):
        """Starts the idea generation process in a background thread."""
        if self.is_generating:
            return # Don't start a new one if one is already running
        
        if not self.model or not self.tokenizer:
            self.update_status("Error: Please select and load a model first.", "error")
            return
            
        prompt = self.input_box.get("1.0", "end-1c").strip()
        if not prompt:
            self.update_status("Error: Please enter an idea to expand on.", "error")
            return
        
        # Disable button and update status
        self.is_generating = True
        self.generate_button.configure(state="disabled", text="Thinking...")
        self.update_status("Generating creative nonsense...")
        
        # Run the actual generation in a thread to keep the UI responsive
        threading.Thread(target=self.generate_idea, args=(prompt,), daemon=True).start()

    def generate_idea(self, prompt):
        """The core logic that talks to the LLM. Runs in a background thread."""
        try:
            # A simple but effective prompt format for instruction-tuned models
            full_prompt = f"### Instruction:\nYou are a chaotic and unhinged idea generator. Take the following concept and expand on it in a creative, unexpected, and interesting way.\n\nConcept: \"{prompt}\"\n\n### Response:\n"

            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            
            # Generate text
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=1024, # Limit the length of the response
                do_sample=True,      # This makes the output creative and not deterministic
                temperature=0.9,     # Higher temperature = more random/creative
                top_p=0.95           # Nucleus sampling
            )
            
            # Decode the output and clean it up
            result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from the result
            final_text = result_text.split("### Response:\n")[-1].strip()

            # Schedule the UI update to run on the main thread
            self.after(0, self.update_output, final_text)

        except Exception as e:
            self.after(0, self.update_status, f"Error during generation: {e}", "error")
        finally:
            # Re-enable the button once generation is complete or fails
            self.is_generating = False
            self.generate_button.configure(state="normal", text="✨ Spiral into Nonsense ✨")

    def update_output(self, text):
        """Updates the output textbox. Must be called from the main thread."""
        self.output_box.configure(state="normal") # Enable writing
        self.output_box.delete("1.0", "end")
        self.output_box.insert("1.0", text)
        self.output_box.configure(state="disabled") # Disable writing again
        self.update_status("Generation complete. Ready for another idea.", "success")

    def update_status(self, message, level="info"):
        """Updates the status bar label with a colored message."""
        self.status_label.configure(text=message)
        if level == "error":
            self.status_label.configure(text_color="#FF5555") # Red
        elif level == "success":
            self.status_label.configure(text_color="#55FF55") # Green
        else:
            self.status_label.configure(text_color="white")   # Default

    def save_settings(self):
        """Saves the currently selected model to the config file."""
        config = configparser.ConfigParser()
        config['Settings'] = {
            'last_model': self.model_selector.get() if self.model_selector.get() != "No models found" else ""
        }
        with open(CONFIG_FILE, 'w') as configfile:
            config.write(configfile)

    def load_settings(self):
        """Loads the last used model from the config file and applies it."""
        if not os.path.exists(CONFIG_FILE):
            return

        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)
        
        last_model = config.get('Settings', 'last_model', fallback=None)
        
        if last_model and last_model in self.available_models:
            self.model_selector.set(last_model)
            self.on_model_select(last_model) # Trigger loading the model
            
    def on_closing(self):
        """Called when the application window is closed."""
        self.save_settings()
        self.destroy()


# --- Main entry point of the application ---
if __name__ == "__main__":
    app = App()
    app.mainloop()