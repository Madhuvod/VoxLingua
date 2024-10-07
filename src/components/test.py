import tkinter as tk

root = tk.Tk()
root.title("Test Window")
root.geometry("200x100")
label = tk.Label(root, text="Tkinter is working!")
label.pack()
root.mainloop()