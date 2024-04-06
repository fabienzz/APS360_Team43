import tkinter as tk
from PIL import Image, ImageTk
import latex_model



def display_images(num_images=5):
    input_text = entry.get()  # Get input from entry widget
    # Call your source code function to generate images
    latex_model.generate_images(latex_model.model, input_text, num_images)
    # Display up to 5 images
    for i in range(5):
        latex_model.crop_image("img/", i)
        img = Image.open("img/" + str(i) + ".png")
        img.thumbnail((200, 200))  # Resize image
        photo = ImageTk.PhotoImage(img)
        img_label = tk.Label(root, image=photo)
        img_label.image = photo  # Keep reference to image to avoid garbage collection
        img_label.grid(row=2, column=i, padx=5, pady=5)

# Create the main window
root = tk.Tk()
root.title("Image Generator")

# Create input entry widget
entry = tk.Entry(root, width=40)
entry.grid(row=0, column=0, padx=10, pady=10)

# Create generate button
generate_button = tk.Button(root, text="Generate Images", command=display_images)
generate_button.grid(row=0, column=1, padx=10, pady=10)

# Run the Tkinter event loop
root.mainloop()
