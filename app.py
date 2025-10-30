import tkinter as tk
from tkinter import messagebox, Label
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import ImageGrab, Image
import threading
import time
import ctypes
from ctypes import wintypes
import json
import os


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


# 加载模型（适配ResNet18）
def load_model(model_path='best_model.pth'):
	model = models.resnet18(pretrained=False)
	model.fc = nn.Linear(model.fc.in_features, 7)  # 7分类
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.to(device)
	model.eval()
	return model


# 图像预处理 - 与训练时保持一致
transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 预测函数
def predict_image(model, image):
	if not isinstance(image, Image.Image):
		raise ValueError("输入必须是PIL图像")

	image = transform(image).unsqueeze(0).to(device)

	with torch.no_grad():
		output = model(image)
		pred = output.argmax(1).item()

	return pred


class CountdownWindow:
	"""独立的倒计时窗口类"""

	def __init__(self, parent, region_name, x, y):
		self.parent = parent
		self.region_name = region_name
		self.seconds = 15
		self.active = True

		# 创建独立窗口
		self.window = tk.Toplevel(parent.root)
		self.window.overrideredirect(True)  # 无边框
		self.window.attributes("-topmost", True)  # 置顶
		self.window.attributes("-transparentcolor", "black")  # 黑色透明
		self.window.configure(bg='black')  # 背景设为黑色

		# 设置窗口位置
		self.window.geometry(f"+{x}+{y}")

		# 创建标签
		self.label = Label(
			self.window,
			text=f"{region_name}替身剩余{self.seconds}秒",
			font=("Arial", 16, "bold"),
			fg="red",
			bg="black"
		)
		self.label.pack(padx=5, pady=5)

		# 开始倒计时
		self.update()

	def update(self):
		"""更新倒计时"""
		if self.active and self.seconds > 0:
			self.seconds -= 1
			self.label.config(text=f"{self.region_name}替身剩余{self.seconds}秒")
			self.window.after(1000, self.update)
		else:
			self.active = False
			self.window.destroy()
			# 从父类的计时器列表中移除
			if self.region_name == "左侧":
				if self in self.parent.left_timers:
					self.parent.left_timers.remove(self)
			else:
				if self in self.parent.right_timers:
					self.parent.right_timers.remove(self)


class ScreenRecognitionApp:
	def __init__(self, root):
		self.root = root
		self.root.title("双区域屏幕实时识别")
		self.root.geometry("400x200")
		self.root.attributes("-topmost", True)  # 主窗口也置顶

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

		# 存储两个选择的区域
		self.regions = [None, None]  # 0: 左区域, 1: 右区域
		self.current_region_index = 0  # 当前正在选择的区域索引
		self.is_selecting = False
		self.start_x_phys = 0
		self.start_y_phys = 0

		# 左右区域分别的倒计时相关变量
		self.left_timers = []  # 左侧所有活动的倒计时
		self.last_left_pred = None

		self.right_timers = []  # 右侧所有活动的倒计时
		self.last_right_pred = None

		# 倒计时窗口位置
		self.left_window_x = 100
		self.left_window_y = 250
		self.right_window_x = 2000
		self.right_window_y = 250

		# 坐标文件路径
		self.regions_file = "regions.json"

		self.create_widgets()

		self.running = False
		self.capture_thread = None

		# 类别名称
		self.class_names = [f"类别{i}" for i in range(7)]

		# 尝试加载已保存的坐标
		self.load_regions()

	def load_regions(self):
		"""加载已保存的区域坐标"""
		if os.path.exists(self.regions_file):
			try:
				with open(self.regions_file, 'r') as f:
					saved_regions = json.load(f)

				if len(saved_regions) == 2:
					self.regions = saved_regions
					self.current_region_index = 2  # 标记为已选择两个区域

					# 更新界面显示
					left_region = self.regions[0]
					right_region = self.regions[1]
					left_width = left_region[2] - left_region[0]
					left_height = left_region[3] - left_region[1]
					right_width = right_region[2] - right_region[0]
					right_height = right_region[3] - right_region[1]

					self.left_label.config(text=f"左侧区域已选择\n宽{left_width}px, 高{left_height}px")
					self.right_label.config(text=f"右侧区域已选择\n宽{right_width}px, 高{right_height}px")
					self.status_label.config(text="已加载保存的区域坐标，可直接开始识别")
					self.start_btn.config(state=tk.NORMAL)

					print("已加载保存的区域坐标")
					return True
			except Exception as e:
				print(f"加载坐标文件失败: {e}")

		return False

	def save_regions(self):
		"""保存当前区域坐标到文件"""
		try:
			with open(self.regions_file, 'w') as f:
				json.dump(self.regions, f)
			print("区域坐标已保存")
			return True
		except Exception as e:
			print(f"保存坐标文件失败: {e}")
			return False

	def rearrange_left_windows(self):
		"""重新排列左侧倒计时窗口"""
		for i, timer in enumerate(self.left_timers):
			# 计算新的Y坐标 - 从底部开始向上排列
			new_y = self.left_window_y + (len(self.left_timers) - i - 1) * 40
			if hasattr(timer, 'window') and timer.window.winfo_exists():
				timer.window.geometry(f"+{self.left_window_x}+{new_y}")

	def rearrange_right_windows(self):
		"""重新排列右侧倒计时窗口"""
		for i, timer in enumerate(self.right_timers):
			# 计算新的Y坐标 - 从底部开始向上排列
			new_y = self.right_window_y + (len(self.right_timers) - i - 1) * 40
			if hasattr(timer, 'window') and timer.window.winfo_exists():
				timer.window.geometry(f"+{self.right_window_x}+{new_y}")

	def create_widgets(self):
		# 顶部控制区域
		top_frame = tk.Frame(self.root)
		top_frame.pack(pady=10, fill=tk.X, padx=10)

		self.select_btn = tk.Button(top_frame, text="选择区域", command=self.start_selection)
		self.select_btn.pack(side=tk.LEFT, padx=5)

		self.start_btn = tk.Button(top_frame, text="开始识别", command=self.start_recognition, state=tk.DISABLED)
		self.start_btn.pack(side=tk.LEFT, padx=5)

		self.stop_btn = tk.Button(top_frame, text="停止识别", command=self.stop_recognition, state=tk.DISABLED)
		self.stop_btn.pack(side=tk.LEFT, padx=5)

		self.reset_btn = tk.Button(top_frame, text="重置区域", command=self.reset_regions)
		self.reset_btn.pack(side=tk.LEFT, padx=5)

		self.status_label = Label(top_frame, text=self.status, fg="blue")
		self.status_label.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)

		# 结果显示区域
		results_frame = tk.Frame(self.root)
		results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

		self.left_label = Label(results_frame, text="左侧区域尚未选择", font=("Arial", 10))
		self.left_label.pack(anchor="w", pady=5)

		self.right_label = Label(results_frame, text="右侧区域尚未选择", font=("Arial", 10))
		self.right_label.pack(anchor="w", pady=5)

	def reset_regions(self):
		"""重置已选择的区域"""
		self.regions = [None, None]
		self.current_region_index = 0
		self.left_label.config(text="左侧区域尚未选择")
		self.right_label.config(text="右侧区域尚未选择")
		self.status_label.config(text="已重置区域，请重新选择")
		self.start_btn.config(state=tk.DISABLED)

		# 删除坐标文件
		if os.path.exists(self.regions_file):
			try:
				os.remove(self.regions_file)
				print("已删除保存的区域坐标文件")
			except Exception as e:
				print(f"删除坐标文件失败: {e}")

		# 清除所有倒计时窗口
		for timer in self.left_timers:
			timer.active = False
			if hasattr(timer, 'window') and timer.window.winfo_exists():
				timer.window.destroy()
		self.left_timers.clear()

		for timer in self.right_timers:
			timer.active = False
			if hasattr(timer, 'window') and timer.window.winfo_exists():
				timer.window.destroy()
		self.right_timers.clear()

		# 重置窗口位置
		self.left_window_y = 100
		self.right_window_y = 100

	def start_selection(self):
		# 检查是否已选择两个区域
		if self.current_region_index >= 2:
			messagebox.showinfo("提示", "已选择两个区域，请先重置再重新选择")
			return

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

		# 显示当前正在选择的区域
		region_text = "左侧区域" if self.current_region_index == 0 else "右侧区域"
		self.canvas.create_text(
			self.select_window.winfo_screenwidth() // 2,
			self.select_window.winfo_screenheight() // 2,
			text=f"拖动鼠标选择{region_text}，ESC取消",
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
		                   self.start_x_phys / self.select_scaling - self.select_window.winfo_rootx(),
		                   self.start_y_phys / self.select_scaling - self.select_window.winfo_rooty(),
		                   current_x, current_y)

	def on_release(self, event):
		self.is_selecting = False
		if not hasattr(self, 'rect'):
			self.select_window.destroy()
			return

		end_x_phys = int(event.x_root * self.select_scaling)
		end_y_phys = int(event.y_root * self.select_scaling)

		region = (
			min(self.start_x_phys, end_x_phys),
			min(self.start_y_phys, end_y_phys),
			max(self.start_x_phys, end_x_phys),
			max(self.start_y_phys, end_y_phys)
		)

		width = region[2] - region[0]
		height = region[3] - region[1]
		if width < 10 or height < 10:
			messagebox.showwarning("警告", "选择的区域太小，请重新选择（至少10x10像素）")
			self.select_window.destroy()
			return

		# 保存选择的区域
		self.regions[self.current_region_index] = region

		# 更新状态显示
		region_name = "左侧" if self.current_region_index == 0 else "右侧"
		status_text = f"{region_name}区域已选择: 宽{width}px, 高{height}px"

		# 更新对应区域的标签
		if self.current_region_index == 0:
			self.left_label.config(text=f"左侧区域已选择\n宽{width}px, 高{height}px")
		else:
			self.right_label.config(text=f"右侧区域已选择\n宽{width}px, 高{height}px")

		self.status_label.config(text=status_text)
		self.select_window.destroy()

		# 切换到下一个区域
		self.current_region_index += 1

		# 如果两个区域都已选择，启用开始按钮
		if all(region is not None for region in self.regions):
			self.start_btn.config(state=tk.NORMAL)
			self.status_label.config(text="两个区域都已选择，可以开始识别")
		else:
			# 提示选择下一个区域
			messagebox.showinfo("提示", f"请继续选择{'右侧' if self.current_region_index == 1 else '左侧'}区域")

	def start_left_countdown(self):
		"""开始左侧倒计时"""
		# 将所有左侧倒计时窗口向上移动40像素
		for timer in self.left_timers:
			if hasattr(timer, 'window') and timer.window.winfo_exists():
				# 获取当前窗口位置
				current_geometry = timer.window.geometry()
				# 解析当前位置字符串，例如"200x100+50+150"
				parts = current_geometry.split('+')
				x = parts[1]
				y = int(parts[2]) + 40  # Y坐标减少40，向上移动
				timer.window.geometry(f"+{x}+{y}")

		# 创建新的倒计时窗口在固定位置
		new_timer = CountdownWindow(self, "左侧", self.left_window_x, self.left_window_y)
		self.left_timers.append(new_timer)

	def start_right_countdown(self):
		"""开始右侧倒计时"""
		# 将所有右侧倒计时窗口向上移动40像素
		for timer in self.right_timers:
			if hasattr(timer, 'window') and timer.window.winfo_exists():
				current_geometry = timer.window.geometry()
				parts = current_geometry.split('+')
				x = parts[1]
				y = int(parts[2]) + 40  # Y坐标减少40，向上移动
				timer.window.geometry(f"+{x}+{y}")

		# 创建新的倒计时窗口在固定位置
		new_timer = CountdownWindow(self, "右侧", self.right_window_x, self.right_window_y)
		self.right_timers.append(new_timer)

	def check_and_start_countdown(self, left_pred, right_pred):
		"""检查是否需要开始倒计时，左右区域分别处理"""
		# 如果之前没有记录，初始化记录
		if self.last_left_pred is None:
			self.last_left_pred = left_pred
		if self.last_right_pred is None:
			self.last_right_pred = right_pred

		# 检查左侧区域是否需要开始倒计时
		if left_pred < self.last_left_pred:
			self.start_left_countdown()

		# 检查右侧区域是否需要开始倒计时
		if right_pred < self.last_right_pred:
			self.start_right_countdown()

		# 更新上一次的识别结果
		self.last_left_pred = left_pred
		self.last_right_pred = right_pred

	def start_recognition(self):
		if not all(region is not None for region in self.regions):
			messagebox.showwarning("警告", "请先选择两个区域")
			return
		if not self.model:
			messagebox.showerror("错误", "模型加载失败，无法进行识别")
			return

		# 保存区域坐标
		self.save_regions()

		self.running = True
		self.start_btn.config(state=tk.DISABLED)
		self.stop_btn.config(state=tk.NORMAL)
		self.select_btn.config(state=tk.DISABLED)
		self.reset_btn.config(state=tk.DISABLED)

		# 重置上一次的识别结果
		self.last_left_pred = None
		self.last_right_pred = None

		self.capture_thread = threading.Thread(target=self.capture_and_recognize)
		self.capture_thread.daemon = True
		self.capture_thread.start()

	def stop_recognition(self):
		self.running = False
		self.start_btn.config(state=tk.NORMAL)
		self.stop_btn.config(state=tk.DISABLED)
		self.select_btn.config(state=tk.NORMAL)
		self.reset_btn.config(state=tk.NORMAL)
		self.status_label.config(text="识别已停止")

	def capture_and_recognize(self):
		while self.running:
			try:
				# 处理左侧区域
				left_region = self.regions[0]
				left_screenshot = ImageGrab.grab(bbox=left_region)
				left_pred = predict_image(self.model, left_screenshot)
				left_result = f"左侧区域识别结果: {self.class_names[left_pred]} (ID: {left_pred})"

				# 处理右侧区域
				right_region = self.regions[1]
				right_screenshot = ImageGrab.grab(bbox=right_region)
				right_pred = predict_image(self.model, right_screenshot)
				right_result = f"右侧区域识别结果: {self.class_names[right_pred]} (ID: {right_pred})"

				# 检查是否需要开始倒计时
				self.root.after(0, self.check_and_start_countdown, left_pred, right_pred)

				# 更新UI显示
				self.root.after(0, lambda t=left_result: self.left_label.config(text=t, fg="green"))
				self.root.after(0, lambda t=right_result: self.right_label.config(text=t, fg="green"))
				self.root.after(0, lambda: self.status_label.config(text="识别中...", fg="blue"))

				time.sleep(1)  # 1秒识别一次
			except Exception as e:
				error_msg = f"识别出错: {str(e)}"
				self.root.after(0, lambda t=error_msg: self.status_label.config(text=t, fg="red"))
				time.sleep(1)

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