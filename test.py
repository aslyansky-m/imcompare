import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

class ImageLabelerApp:
    def __init__(self, root):
        self.root = root
        self.current_index = 0
        self.image_paths = []
        self.load_csv_button = tk.Button(root, text="Load CSV", command=self.load_csv)
        self.load_csv_button.pack()

        self.save_results_button = tk.Button(root, text="Save Results", command=self.save_results)
        self.save_results_button.pack()

        self.next_image_button = tk.Button(root, text="Next Image", command=self.next_image)
        self.next_image_button.pack()

        self.image_list_frame = tk.Frame(root)
        self.image_list_frame.pack()

        self.image_list_scrollbar = tk.Scrollbar(self.image_list_frame)
        self.image_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_listbox = tk.Listbox(self.image_list_frame, yscrollcommand=self.image_list_scrollbar.set)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_listbox.bind('<<ListboxSelect>>', self.select_image)

        self.image_list_scrollbar.config(command=self.image_listbox.yview)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, 'r') as file:
                for line in file:
                    image_path, _ = line.strip().split(',')[:2]  # Assuming 2 columns in CSV
                    self.image_paths.append(image_path)
                    self.image_listbox.insert(tk.END, image_path)

    def save_results(self):
        # Add saving logic here
        pass

    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.current_index)
        self.image_listbox.see(self.current_index)

    def select_image(self, event):
        self.current_index = self.image_listbox.curselection()[0]
        # Logic to load and display selected image
        print(self.image_paths[self.current_index])

def main():
    root = tk.Tk()
    app = ImageLabelerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()