import tkinter as tk
from tkinter import scrolledtext, messagebox
from PIL import Image, ImageTk
import latex_model
import pyperclip

class LatexUI:
    def __init__(self, root):
        self.root = root
        self.root.title("WriTex")
        self.root.geometry("1200x400")  # Set initial window size

        self.input_label = tk.Label(root, text="Enter Latex String:", font=("Arial", 14))
        self.input_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.input_text = scrolledtext.ScrolledText(root, width=50, height=5, font=("Arial", 12))
        self.input_text.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        self.submit_button = tk.Button(root, text="Submit", command=self.process_latex, font=("Arial", 12))
        self.submit_button.grid(row=0, column=2, padx=10, pady=10, sticky="w")

        self.copy_button = tk.Button(root, text="Copy to Clipboard", command=self.copy_to_clipboard, font=("Arial", 12))
        self.copy_button.grid(row=0, column=3, padx=10, pady=10, sticky="w")

        self.min_length_label = tk.Label(root, text="Min Length:", font=("Arial", 14))
        self.min_length_label.grid(row=0, column=4, padx=10, pady=10, sticky="w")

        self.min_length_entry = tk.Entry(root, font=("Arial", 12))
        self.min_length_entry.insert(0, '5')  # Default min length
        self.min_length_entry.grid(row=0, column=5, padx=10, pady=10, sticky="w")

        self.max_length_label = tk.Label(root, text="Max Length:", font=("Arial", 14))
        self.max_length_label.grid(row=1, column=4, padx=10, pady=10, sticky="w")

        self.max_length_entry = tk.Entry(root, font=("Arial", 12))
        self.max_length_entry.insert(0, '20')  # Default max length
        self.max_length_entry.grid(row=1, column=5, padx=10, pady=10, sticky="w")

        self.required_char_label = tk.Label(root, text="Required Character:", font=("Arial", 14))
        self.required_char_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.required_char_entry = tk.Entry(root, font=("Arial", 12))
        self.required_char_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        self.output_container = tk.Frame(root)
        self.output_container.grid(row=2, column=0, columnspan=6, padx=10, pady=10)

        self.output_frames = []
        for i in range(5):
            frame = tk.Frame(self.output_container)
            frame.grid(row=0, column=i, padx=10, pady=10)

            output_label = tk.Label(frame, text=f"Output {i+1}:", font=("Arial", 14))
            output_label.pack()

            output_text = tk.Text(frame, width=20, height=3, font=("Arial", 12))
            output_text.pack()

            output_image = tk.Label(frame)
            output_image.pack()

            self.output_frames.append((frame, output_text, output_image))


    def process_latex(self):
        input_latex = self.input_text.get("1.0", tk.END).strip()
        min_length = int(self.min_length_entry.get().strip())
        max_length = int(self.max_length_entry.get().strip())
        required_char = self.required_char_entry.get().strip()

        # Check if input or required character contains invalid syntax
        if not latex_model.check_vocab(input_latex):
            messagebox.showerror("Error", "Input string contains invalid syntax.")
            return
        
        if not latex_model.check_vocab(required_char):
            messagebox.showerror("Error", "Required character contains invalid syntax.")
            return

        if not input_latex:
            input_latex = ""

        if not required_char:
            required_char = ""

        try:
            print(input_latex, min_length, max_length, required_char)
            outputs = latex_model.generate(input_latex, min_length, max_length, required_char)
            if len(outputs) != 5:
                raise ValueError("Model did not return 5 outputs.")

            for i, (output_text, output_image_path) in enumerate(outputs):
                self.output_frames[i][1].delete("1.0", tk.END)
                self.output_frames[i][1].insert(tk.END, output_text)

                img = Image.open(output_image_path)
                # Calculate new dimensions while maintaining aspect ratio
                aspect_ratio = img.width / img.height
                new_width = 200  # Define desired width
                new_height = int(new_width / aspect_ratio)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.output_frames[i][2].config(image=photo)
                self.output_frames[i][2].image = photo

                self.output_frames[i][2].bind("<Button-1>", lambda event, text=output_text: self.change_input_text(text))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def change_input_text(self, text):
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert(tk.END, text)

    def copy_to_clipboard(self):
        input_text = self.input_text.get("1.0", tk.END).strip()
        pyperclip.copy(input_text)



if __name__ == "__main__":
    root = tk.Tk()
    app = LatexUI(root)
    root.mainloop()
