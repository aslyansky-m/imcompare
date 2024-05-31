import tkinter as tk
import math

class LogScale(tk.Scale):
    def __init__(self, master=None, **kwargs):
        self.log_min = kwargs.pop('log_min', 1)
        self.log_max = kwargs.pop('log_max', 100)
        self.logarithmic_min = math.log10(self.log_min)
        self.logarithmic_max = math.log10(self.log_max)
        kwargs['from_'] = 0
        kwargs['to'] = 1000
        super().__init__(master, **kwargs, command=self.update_value)
        self.set(self.log_to_linear(self.log_min))
    
    def log_to_linear(self, value):
        return (math.log10(value) - self.logarithmic_min) / (self.logarithmic_max - self.logarithmic_min) * 1000
    
    def linear_to_log(self, value):
        return 10 ** (value / 1000 * (self.logarithmic_max - self.logarithmic_min) + self.logarithmic_min)
    
    def update_value(self, value):
        log_value = self.linear_to_log(float(value))
        self.set_label(log_value)
    
    def set_label(self, log_value):
        self.label.config(text=f'{log_value:.2f}')

def main():
    root = tk.Tk()
    root.title("Logarithmic Scale Slider")
    value_label = tk.Label(root)
    value_label.pack()
    log_slider = LogScale(root, log_min=1, log_max=10000, orient='horizontal')
    log_slider.label = value_label
    log_slider.pack()
    root.mainloop()

if __name__ == "__main__":
    main()
