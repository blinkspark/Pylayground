import customtkinter as ctk
import tkinter as tk

ctk.set_appearance_mode("light")  # Modes: system (default), light, dark
# ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green
# ctk.set_window_scaling(2)
# ctk.set_widget_scaling(2)
app = ctk.CTk()  # create CTk window like you do with the Tk window
app.geometry("400x240")


def button_function():
  print("button pressed")


app.rowconfigure(0, weight=1)
app.rowconfigure(2, weight=2)
app.columnconfigure(0, weight=1)

# Use CTkButton instead of tkinter Button
button = ctk.CTkButton(master=app, text="CTkButton", command=button_function)

button.grid(row=0, column=0, stick=tk.NSEW)

l = ctk.CTkButton(app, text='BTN')
l.grid(row=1, column=0, stick=tk.NSEW)

app.mainloop()