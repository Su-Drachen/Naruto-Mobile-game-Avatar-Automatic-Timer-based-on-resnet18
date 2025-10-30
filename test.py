import tkinter as tk
from tkinter import messagebox, Label
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import ImageGrab, Image, ImageTk
import threading
import time
import ctypes
from ctypes import wintypes


# --------------------------
# 核心优化：精确DPI感知与坐标转换
# --------------------------
def set_high_dpi_awareness():
    """设置高DPI感知，避免系统自动缩放坐标（仅Windows）"""
    try:
        PROCESS_PER_MONITOR_DPI_AWARE = 2
        ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
    except:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except:
            pass


def get_window_dpi(hwnd):
    """获取指定窗口所在显示器的DPI"""
    try:
        monitor = ctypes.windll.user32.MonitorFromWindow(hwnd, 2)
        dpi_x = wintypes.UINT()
        result = ctypes.windll.shcore.GetDpiForMonitor(monitor, 0, ctypes.byref(dpi_x), None)
        if result == 0 and dpi_x.value > 0:
            return dpi_x.value
        else:
            return 96
    except:
        return 96


def get_scaling_factor(hwnd):
    """获取窗口所在显示器的缩放比例"""
    dpi = get_window_dpi(hwnd)
    scaling = dpi / 96.0
    return scaling if scaling > 0 else 1.0


set_high_dpi_awareness()


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')


# 加载模型（关键修改：适配ResNet18）
def load_model(model_path='best_model.pth'):
    # 改为ResNet18，与训练代码一致
    model = models.resnet18(pretrained=False)
    # 修改全连接层（ResNet18的分类头是fc，输入特征数与训练时一致）
    model.fc = nn.Linear(model.fc.in_features, 7)  # 7分类
    # 加载训练好的权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# 图像预处理 - 与训练时保持一致（无需修改）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 预测函数（无需修改）
def predict_image(model, image):
    if not isinstance(image, Image.Image):
        raise ValueError("输入必须是PIL图像")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(1).item()

    return pred


class ScreenRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("屏幕区域实时识别")
        self.root.geometry("800x600")

        self.hwnd = self.root.winfo_id()
        self.scaling_factor = get_scaling_factor(self.hwnd)
        print(f"主窗口缩放比例: {self.scaling_factor:.2f}")

        # 模型加载
        try:
            self.model = load_model()
            self.status = "模型加载成功"
        except Exception as e:
            self.model = None
            self.status = f"模型加载失败: {str(e)}"

        self.selected_region = None
        self.is_selecting = False
        self.start_x_phys = 0
        self.start_y_phys = 0

        self.create_widgets()

        self.running = False
        self.capture_thread = None

        # 可根据实际类别名称修改（训练代码中是['0','1',...,'6']）
        self.class_names = [f"类别{i}" for i in range(7)]  # 若有具体名称可替换，如['猫','狗',...]

    def create_widgets(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10, fill=tk.X, padx=10)

        self.select_btn = tk.Button(top_frame, text="选择区域", command=self.start_selection)
        self.select_btn.pack(side=tk.LEFT, padx=5)

        self.start_btn = tk.Button(top_frame, text="开始识别", command=self.start_recognition, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(top_frame, text="停止识别", command=self.stop_recognition, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = Label(top_frame, text=self.status, fg="blue")
        self.status_label.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)

        self.display_frame = tk.Frame(self.root, bd=2, relief=tk.SUNKEN)
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.capture_label = Label(self.display_frame, text="请先选择一个区域")
        self.capture_label.pack(fill=tk.BOTH, expand=True)

        self.result_frame = tk.Frame(self.root, height=50)
        self.result_frame.pack(fill=tk.X, padx=10, pady=10)

        self.result_label = Label(self.result_frame, text="识别结果将显示在这里", font=("Arial", 14))
        self.result_label.pack()

    def start_selection(self):
        if hasattr(self, 'select_window') and self.select_window.winfo_exists():
            self.select_window.destroy()

        self.select_window = tk.Toplevel(self.root)
        self.select_window.attributes("-fullscreen", True)
        self.select_window.attributes("-alpha", 0.2)
        self.select_window.attributes("-topmost", True)
        self.select_window.overrideredirect(True)

        self.select_window_hwnd = self.select_window.winfo_id()
        self.select_scaling = get_scaling_factor(self.select_window_hwnd)
        print(f"选择窗口缩放比例: {self.select_scaling:.2f}")

        self.canvas = tk.Canvas(self.select_window, cursor="cross", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.select_window.bind("<Escape>", lambda e: self.select_window.destroy())

        self.canvas.create_text(
            self.select_window.winfo_screenwidth() // 2,
            self.select_window.winfo_screenheight() // 2,
            text="拖动鼠标选择区域，ESC取消",
            font=("Arial", 16),
            fill="red"
        )

    def on_press(self, event):
        self.is_selecting = True
        self.start_x_phys = int(event.x_root * self.select_scaling)
        self.start_y_phys = int(event.y_root * self.select_scaling)
        self.rect = self.canvas.create_rectangle(
            0, 0, 0, 0, outline="red", width=2, dash=(5, 2)
        )

    def on_drag(self, event):
        if not self.is_selecting:
            return
        current_x = event.x
        current_y = event.y
        self.canvas.coords(self.rect,
                          self.start_x_phys/self.select_scaling - self.select_window.winfo_rootx(),
                          self.start_y_phys/self.select_scaling - self.select_window.winfo_rooty(),
                          current_x, current_y)

    def on_release(self, event):
        self.is_selecting = False
        if not hasattr(self, 'rect'):
            self.select_window.destroy()
            return

        end_x_phys = int(event.x_root * self.select_scaling)
        end_y_phys = int(event.y_root * self.select_scaling)

        self.selected_region = (
            min(self.start_x_phys, end_x_phys),
            min(self.start_y_phys, end_y_phys),
            max(self.start_x_phys, end_x_phys),
            max(self.start_y_phys, end_y_phys)
        )

        width = self.selected_region[2] - self.selected_region[0]
        height = self.selected_region[3] - self.selected_region[1]
        if width < 10 or height < 10:
            messagebox.showwarning("警告", "选择的区域太小，请重新选择（至少10x10像素）")
            self.selected_region = None
            self.select_window.destroy()
            return

        self.select_window.destroy()
        self.start_btn.config(state=tk.NORMAL)
        self.status_label.config(
            text=f"已选择区域: 宽{width}px, 高{height}px | "
                 f"坐标: ({self.selected_region[0]},{self.selected_region[1]})-"
                 f"({self.selected_region[2]},{self.selected_region[3]})"
        )

    def start_recognition(self):
        if not self.selected_region:
            messagebox.showwarning("警告", "请先选择区域")
            return
        if not self.model:
            messagebox.showerror("错误", "模型加载失败，无法进行识别")
            return

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.select_btn.config(state=tk.DISABLED)

        self.capture_thread = threading.Thread(target=self.capture_and_recognize)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def stop_recognition(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.select_btn.config(state=tk.NORMAL)

    def capture_and_recognize(self):
        while self.running:
            try:
                x1, y1, x2, y2 = self.selected_region
                screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))

                self.update_display(screenshot)

                pred = predict_image(self.model, screenshot)
                result_text = f"识别结果: {self.class_names[pred]} (类别ID: {pred})"
                self.root.after(0, lambda t=result_text: self.result_label.config(text=t, fg="green"))

                time.sleep(1)
            except Exception as e:
                error_msg = f"识别出错: {str(e)}"
                self.root.after(0, lambda t=error_msg: self.status_label.config(text=t, fg="red"))
                time.sleep(1)

    def update_display(self, image):
        display_width = self.display_frame.winfo_width() or 640
        display_height = self.display_frame.winfo_height() or 480
        image.thumbnail((display_width, display_height))
        photo = ImageTk.PhotoImage(image)
        self.root.after(0, lambda p=photo: self.capture_label.config(image=p))
        self.capture_label.image = photo

    def on_close(self):
        self.running = False
        if hasattr(self, 'select_window') and self.select_window.winfo_exists():
            self.select_window.destroy()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ScreenRecognitionApp(root)
    root.bind("<Escape>", lambda e: app.on_close())
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()