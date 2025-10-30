import tkinter as tk
from tkinter import messagebox, Label, Entry, Button
from PIL import ImageGrab, Image
import threading
import time
import os
from datetime import datetime
import ctypes
from ctypes import wintypes


# 高DPI感知设置（解决坐标偏移）
def set_high_dpi_awareness():
	try:
		PROCESS_PER_MONITOR_DPI_AWARE = 2
		ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
	except:
		try:
			ctypes.windll.user32.SetProcessDPIAware()
		except:
			pass


def get_window_dpi(hwnd):
	try:
		monitor = ctypes.windll.user32.MonitorFromWindow(hwnd, 2)
		dpi_x = wintypes.UINT()
		result = ctypes.windll.shcore.GetDpiForMonitor(monitor, 0, ctypes.byref(dpi_x), None)
		return dpi_x.value if result == 0 and dpi_x.value > 0 else 96
	except:
		return 96


def get_scaling_factor(hwnd):
	dpi = get_window_dpi(hwnd)
	scaling = dpi / 96.0
	return scaling if scaling > 0 else 1.0


set_high_dpi_awareness()


class ScreenCaptureApp:
	def __init__(self, root):
		self.root = root
		self.root.title("屏幕区域连续截图工具")
		self.root.geometry("800x600")

		# 初始化参数
		self.hwnd = self.root.winfo_id()
		self.scaling_factor = get_scaling_factor(self.hwnd)
		print(f"屏幕缩放比例: {self.scaling_factor:.2f}")

		self.selected_region = None  # 选定区域（物理坐标）
		self.is_selecting = False
		self.start_x_phys = 0
		self.start_y_phys = 0
		self.save_path = r"D:\Users\Wald\Desktop\Screenshot-Recognition-Countdown\data\2"  # 默认保存路径
		self.capture_count = 10  # 连续截图数量
		self.running = False
		self.capture_thread = None

		# 创建UI
		self.create_widgets()

	def create_widgets(self):
		# 顶部控制区域
		top_frame = tk.Frame(self.root)
		top_frame.pack(pady=10, fill=tk.X, padx=10)

		# 保存路径设置
		path_frame = tk.Frame(top_frame)
		path_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

		Label(path_frame, text="保存路径:").pack(side=tk.LEFT)
		self.path_entry = Entry(path_frame)
		self.path_entry.insert(0, self.save_path)
		self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

		browse_btn = Button(path_frame, text="浏览", command=self.browse_path)
		browse_btn.pack(side=tk.LEFT, padx=5)

		# 控制按钮
		self.select_btn = Button(top_frame, text="选择区域", command=self.start_selection)
		self.select_btn.pack(side=tk.LEFT, padx=5)

		self.start_btn = Button(top_frame, text="开始截图", command=self.start_capture, state=tk.DISABLED)
		self.start_btn.pack(side=tk.LEFT, padx=5)

		self.stop_btn = Button(top_frame, text="停止", command=self.stop_capture, state=tk.DISABLED)
		self.stop_btn.pack(side=tk.LEFT, padx=5)

		# 中间预览区域
		self.preview_frame = tk.Frame(self.root, bd=2, relief=tk.SUNKEN)
		self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

		self.preview_label = Label(self.preview_frame, text="请先选择截图区域")
		self.preview_label.pack(fill=tk.BOTH, expand=True)

		# 底部状态区域
		self.status_frame = tk.Frame(self.root, height=50)
		self.status_frame.pack(fill=tk.X, padx=10, pady=10)

		self.status_label = Label(self.status_frame, text="就绪", font=("Arial", 12))
		self.status_label.pack()

	def browse_path(self):
		"""选择保存路径"""
		from tkinter import filedialog
		path = filedialog.askdirectory(title="选择保存文件夹")
		if path:
			self.save_path = path
			self.path_entry.delete(0, tk.END)
			self.path_entry.insert(0, path)

	def start_selection(self):
		"""开始选择截图区域"""
		if hasattr(self, 'select_window') and self.select_window.winfo_exists():
			self.select_window.destroy()

		self.select_window = tk.Toplevel(self.root)
		self.select_window.attributes("-fullscreen", True)
		self.select_window.attributes("-alpha", 0.2)
		self.select_window.attributes("-topmost", True)
		self.select_window.overrideredirect(True)

		self.select_window_hwnd = self.select_window.winfo_id()
		self.select_scaling = get_scaling_factor(self.select_window_hwnd)

		self.canvas = tk.Canvas(self.select_window, cursor="cross", highlightthickness=0)
		self.canvas.pack(fill=tk.BOTH, expand=True)

		# 绑定鼠标事件
		self.canvas.bind("<ButtonPress-1>", self.on_press)
		self.canvas.bind("<B1-Motion>", self.on_drag)
		self.canvas.bind("<ButtonRelease-1>", self.on_release)
		self.select_window.bind("<Escape>", lambda e: self.select_window.destroy())

		# 绘制提示文本
		self.canvas.create_text(
			self.select_window.winfo_screenwidth() // 2,
			self.select_window.winfo_screenheight() // 2,
			text="拖动鼠标选择截图区域，ESC取消",
			font=("Arial", 16),
			fill="red"
		)

	def on_press(self, event):
		"""鼠标按下时记录起点"""
		self.is_selecting = True
		self.start_x_phys = int(event.x_root * self.select_scaling)
		self.start_y_phys = int(event.y_root * self.select_scaling)
		self.rect = self.canvas.create_rectangle(
			0, 0, 0, 0, outline="red", width=2, dash=(5, 2)
		)

	def on_drag(self, event):
		"""鼠标拖动时更新选择框"""
		if not self.is_selecting:
			return
		current_x = event.x
		current_y = event.y
		self.canvas.coords(
			self.rect,
			self.start_x_phys / self.select_scaling - self.select_window.winfo_rootx(),
			self.start_y_phys / self.select_scaling - self.select_window.winfo_rooty(),
			current_x, current_y
		)

	def on_release(self, event):
		"""鼠标释放时确定选择区域"""
		self.is_selecting = False
		if not hasattr(self, 'rect'):
			self.select_window.destroy()
			return

		end_x_phys = int(event.x_root * self.select_scaling)
		end_y_phys = int(event.y_root * self.select_scaling)

		# 确定区域坐标（左上角到右下角）
		self.selected_region = (
			min(self.start_x_phys, end_x_phys),
			min(self.start_y_phys, end_y_phys),
			max(self.start_x_phys, end_x_phys),
			max(self.start_y_phys, end_y_phys)
		)

		# 验证区域有效性
		width = self.selected_region[2] - self.selected_region[0]
		height = self.selected_region[3] - self.selected_region[1]
		if width < 10 or height < 10:
			messagebox.showwarning("警告", "选择的区域太小，请重新选择（至少10x10像素）")
			self.selected_region = None
			self.select_window.destroy()
			return

		# 关闭选择窗口并更新状态
		self.select_window.destroy()
		self.start_btn.config(state=tk.NORMAL)
		self.status_label.config(
			text=f"已选择区域: 宽{width}px, 高{height}px | 准备就绪"
		)

	def start_capture(self):
		"""开始连续截图"""
		if not self.selected_region:
			messagebox.showwarning("警告", "请先选择截图区域")
			return

		# 获取保存路径
		self.save_path = self.path_entry.get().strip()
		if not self.save_path:
			messagebox.showwarning("警告", "请设置保存路径")
			return

		# 创建保存目录
		try:
			os.makedirs(self.save_path, exist_ok=True)
		except Exception as e:
			messagebox.showerror("错误", f"无法创建保存目录: {str(e)}")
			return

		# 更新状态和按钮
		self.running = True
		self.start_btn.config(state=tk.DISABLED)
		self.stop_btn.config(state=tk.NORMAL)
		self.select_btn.config(state=tk.DISABLED)
		self.status_label.config(text="开始截图...")

		# 启动截图线程
		self.capture_thread = threading.Thread(target=self.capture_images)
		self.capture_thread.daemon = True
		self.capture_thread.start()

	def stop_capture(self):
		"""停止截图"""
		self.running = False
		self.status_label.config(text="已停止")
		self.start_btn.config(state=tk.NORMAL)
		self.stop_btn.config(state=tk.DISABLED)
		self.select_btn.config(state=tk.NORMAL)

	def capture_images(self):
		"""连续截取10张图片并保存"""
		try:
			for i in range(1, self.capture_count + 1):
				if not self.running:
					break

				# 截取图片
				x1, y1, x2, y2 = self.selected_region
				screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))

				# 生成文件名（时间戳+序号）
				timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
				filename = f"screenshot_{timestamp}_{i:02d}.png"
				save_path = os.path.join(self.save_path, filename)

				# 保存图片
				screenshot.save(save_path)

				# 更新预览和状态
				self.update_preview(screenshot)
				self.root.after(0, lambda i=i: self.status_label.config(
					text=f"正在截图: {i}/{self.capture_count} 张"
				))

				# 每张间隔0.5秒（可调整）
				time.sleep(0.5)

			# 完成后更新状态
			if self.running:
				self.root.after(0, lambda: self.status_label.config(
					text=f"已完成！{self.capture_count}张图片保存至: {self.save_path}"
				))
			self.stop_capture()

		except Exception as e:
			self.root.after(0, lambda: self.status_label.config(
				text=f"截图失败: {str(e)}", fg="red"
			))
			self.stop_capture()

	def update_preview(self, image):
		"""更新预览窗口"""
		# 调整图片大小适应预览区域
		preview_width = self.preview_frame.winfo_width() or 640
		preview_height = self.preview_frame.winfo_height() or 480
		image.thumbnail((preview_width, preview_height))

		# 转换为Tkinter可用格式
		from PIL import ImageTk
		photo = ImageTk.PhotoImage(image)

		# 更新预览
		self.root.after(0, lambda p=photo: self.preview_label.config(image=p))
		self.preview_label.image = photo  # 保持引用

	def on_close(self):
		"""关闭窗口时清理资源"""
		self.running = False
		if hasattr(self, 'select_window') and self.select_window.winfo_exists():
			self.select_window.destroy()
		self.root.destroy()


if __name__ == "__main__":
	root = tk.Tk()
	app = ScreenCaptureApp(root)
	root.bind("<Escape>", lambda e: app.on_close())
	root.protocol("WM_DELETE_WINDOW", app.on_close)
	root.mainloop()