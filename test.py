import tkinter as tk
from tkinter import ttk

def main():
    root = tk.Tk()
    root.title("Tkinter Text Box with Scrollbar")

    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Create a Text widget
    text_box = tk.Text(frame, wrap='word', height=15, width=50)
    text_box.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Create a vertical Scrollbar
    scrollbar = ttk.Scrollbar(frame, orient='vertical', command=text_box.yview)
    scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

    # Configure the Text widget to use the scrollbar
    text_box['yscrollcommand'] = scrollbar.set

    # Make the frame expandable
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)

    # Example text
    for i in range(1, 101):
        text_box.insert(tk.END, f"Line {i}\n")

    root.mainloop()

if __name__ == "__main__":
    main()
