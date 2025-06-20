# main.py
# I've updated the script to handle both normal transformers models and .gguf files.
# I've also added comments to show what's new.

import customtkinter as ctk
from customtkinter import CTk, CTkFrame, CTkButton, CTkLabel, CTkEntry, CTkTextbox, CTkOptionMenu
import transformers
import torch
import os
import threading
import configparser # To remember settings

## NEW - Import the library for GGUF models
try:
    from llama_cpp import Llama
except ImportError:
    # This makes the app still run if llama-cpp-python is not installed,
    # but GGUF models won't work.
    Llama = None 


# --- Configuration ---
# This is where the app will look for your downloaded models.
# Make sure you create this folder next to the script.
MODELS_DIR = "models" 
CONFIG_FILE = "config.ini"


class App(CTk):
    def __init__(self):
        super().__init__()

        # --- Basic Window Setup ---
        self.title("noideas (GGUF Edition)")
        self.geometry("800x600")
        ctk.set_appearance_mode("dark")
        
        # --- App State Variables ---
        self.model = None
        self.tokenizer = None # Only used for transformers models
        self.model_name = None 
        self.is_generating = False
        ## NEW ## - A variable to track what kind of model is loaded ('transformers' or 'gguf')
        self.model_type = None

        # Check for Llama library
        if Llama is None:
            print("WARNING: 'llama-cpp-python' is not installed. GGUF models will not be available.")
            print("Install it with: pip install llama-cpp-python")

        self.available_models = self.find_available_models()

        # --- UI Setup ---
        self.setup_ui()
        self.load_settings()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        """Creates and arranges all the visual components of the app."""
        # ... (This function is unchanged)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=3)
        model_frame = CTkFrame(self)
        model_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        model_frame.grid_columnconfigure(1, weight=1)
        CTkLabel(model_frame, text="Model:", font=("Arial", 14, "bold")).grid(row=0, column=0, padx=10, pady=10)
        self.model_selector = CTkOptionMenu(
            model_frame, 
            values=self.available_models if self.available_models else ["No models found"],
            command=self.on_model_select
        )
        self.model_selector.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        if not self.available_models:
            self.model_selector.configure(state="disabled")
        CTkLabel(self, text="Your half-baked idea:", font=("Arial", 14, "bold")).grid(row=1, column=0, padx=10, pady=(5,0), sticky="nw")
        self.input_box = CTkTextbox(self, height=100, font=("Arial", 14))
        self.input_box.grid(row=1, column=0, padx=10, pady=(0,10), sticky="nsew")
        self.input_box.insert("1.0", "A dating app for ghosts.")
        self.generate_button = CTkButton(self, text="✨ Spiral into Nonsense ✨", font=("Arial", 16, "bold"), command=self.start_generation_thread)
        self.generate_button.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        CTkLabel(self, text="Result:", font=("Arial", 14, "bold")).grid(row=3, column=0, padx=10, pady=(5,0), sticky="nw")
        self.output_box = CTkTextbox(self, font=("Arial", 14))
        self.output_box.grid(row=3, column=0, padx=10, pady=(0,10), sticky="nsew")
        self.output_box.configure(state="disabled")
        self.status_label = CTkLabel(self, text="Ready. Select a model to begin.", anchor="w")
        self.status_label.grid(row=4, column=0, padx=10, pady=5, sticky="ew")


    ## MODIFIED ## - Now finds both model folders AND .gguf files
    def find_available_models(self):
        """Scans the MODELS_DIR for valid model folders and .gguf files."""
        if not os.path.isdir(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            return []
        
        models = []
        for item in os.listdir(MODELS_DIR):
            path = os.path.join(MODELS_DIR, item)
            # Find folders for transformers models
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
                models.append(item)
            # Find GGUF files directly
            elif os.path.isfile(path) and item.lower().endswith(".gguf") and Llama is not None:
                models.append(item)
        
        if not models:
            msg = f"Error: No models found in '{MODELS_DIR}' folder."
            if Llama is None:
                msg += " GGUF support disabled."
            self.update_status(msg, "error")
        return models

    def on_model_select(self, selected_model):
        """Called when a user selects a new model from the dropdown."""
        if selected_model and selected_model != self.model_name:
            threading.Thread(target=self.load_model, args=(selected_model,), daemon=True).start()

    ## MODIFIED ## - Now has two ways to load models: transformers or llama_cpp
    def load_model(self, model_name):
        """Loads the selected LLM from disk, detecting its type."""
        try:
            self.model = None
            self.tokenizer = None
            self.model_type = None
            torch.cuda.empty_cache() # Free up VRAM

            self.update_status(f"Loading {model_name}... this may take a moment.")
            model_path = os.path.join(MODELS_DIR, model_name)

            # --- GGUF Model Loader ---
            if model_name.lower().endswith(".gguf"):
                self.update_status(f"Loading GGUF model: {model_name}...")
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=4096,        # Context window size
                    n_gpu_layers=-1,   # Offload all possible layers to GPU
                    verbose=False      # Don't print llama.cpp's own logs
                )
                self.model_type = "gguf"

            # --- Transformers Model Loader ---
            else:
                self.update_status(f"Loading transformers model: {model_name}...")
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
                self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16, 
                    device_map="auto"
                )
                self.model_type = "transformers"
            
            self.model_name = model_name
            self.update_status(f"Model '{model_name}' loaded successfully. Ready to generate!", "success")

        except Exception as e:
            self.update_status(f"Error loading model: {e}", "error")
            self.model_name = None
            self.model_type = None

    def start_generation_thread(self):
        """Starts the idea generation process in a background thread."""
        if self.is_generating: return
        
        ## MODIFIED ## - Simplified the check
        if not self.model:
            self.update_status("Error: Please select and load a model first.", "error")
            return
            
        prompt = self.input_box.get("1.0", "end-1c").strip()
        if not prompt:
            self.update_status("Error: Please enter an idea to expand on.", "error")
            return
        
        self.is_generating = True
        self.generate_button.configure(state="disabled", text="Thinking...")
        self.update_status("Generating creative nonsense...")
        
        threading.Thread(target=self.generate_idea, args=(prompt,), daemon=True).start()

    ## MODIFIED ## - Now has two ways to generate text, depending on the model type
    def generate_idea(self, prompt):
        """The core logic that talks to the LLM. Runs in a background thread."""
        try:
            # The prompt format is the same for both model types, which is nice.
            full_prompt = f"### Instruction:\nYou are a chaotic and unhinged idea generator. Take the following concept and expand on it in a creative, unexpected,and interesting way.\n\nConcept: \"{prompt}\"\n\n### Response:\n"
            final_text = ""

            # --- Generation for Transformers Models ---
            if self.model_type == "transformers":
                inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95
                )
                result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                final_text = result_text.split("### Response:\n")[-1].strip()

            # --- Generation for GGUF Models ---
            elif self.model_type == "gguf":
                output = self.model(
                    prompt=full_prompt,
                    max_tokens=1024,
                    temperature=0.9,
                    top_p=0.95,
                    stop=["### Instruction:"] # Stop it from generating a new instruction
                )
                final_text = output['choices'][0]['text'].strip()

            self.after(0, self.update_output, final_text)

        except Exception as e:
            self.after(0, self.update_status, f"Error during generation: {e}", "error")
        finally:
            self.is_generating = False
            self.generate_button.configure(state="normal", text="✨ Spiral into Nonsense ✨")

    # --- The rest of the functions are unchanged ---
    
    def update_output(self, text):
        """Updates the output textbox. Must be called from the main thread."""
        self.output_box.configure(state="normal")
        self.output_box.delete("1.0", "end")
        self.output_box.insert("1.0", text)
        self.output_box.configure(state="disabled")
        self.update_status("Generation complete. Ready for another idea.", "success")

    def update_status(self, message, level="info"):
        """Updates the status bar label with a colored message."""
        self.status_label.configure(text=message)
        if level == "error":
            self.status_label.configure(text_color="#FF5555")
        elif level == "success":
            self.status_label.configure(text_color="#55FF55")
        else:
            self.status_label.configure(text_color="white")

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
        if not os.path.exists(CONFIG_FILE): return

        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)
        
        last_model = config.get('Settings', 'last_model', fallback=None)
        
        if last_model and last_model in self.available_models:
            self.model_selector.set(last_model)
            self.on_model_select(last_model)
            
    def on_closing(self):
        """Called when the application window is closed."""
        self.save_settings()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()