#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RECCE人脸伪造检测GUI界面

这个应用程序提供了一个图形界面，用于进行人脸伪造检测，
支持使用原始模式和同态加密模式进行检测。
"""
import matplotlib
import os
import sys
import torch
import numpy as np
import cv2
import time
import shutil
import threading
import queue
from pathlib import Path
from PIL import Image, ImageQt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar,
    QStatusBar, QComboBox, QGroupBox, QRadioButton, QButtonGroup,
    QSpinBox, QLineEdit, QFormLayout, QSplitter, QFrame, QCheckBox,
    QTabWidget, QListWidget, QAbstractItemView, QDialog, QPushButton,
    QDialogButtonBox, QScrollArea, QGridLayout, QToolButton, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QIcon, QFont, QImage

# 导入matplotlib用于绘制统计曲线
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# 确保可以导入项目模块
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入RECCE和同态加密模块
try:
    from model.network import Recce
    from model.common import freeze_weights
    from homomorphic.encryption import HomomorphicEncryption
    from model.homomorphic_feature_extractor import HomomorphicFeatureExtractor
    from albumentations import Compose, Normalize, Resize
    from albumentations.pytorch.transforms import ToTensorV2
    MODULES_IMPORTED = True
except ImportError as e:
    print(f"导入模块时出错: {e}")
    MODULES_IMPORTED = False


class ImageProcessor(QThread):
    """
    图像处理线程，用于在后台进行图像处理和预测
    """
    progress_update = pyqtSignal(int)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, image_path, model_path, use_encryption, 
                 device="cpu", image_size=299, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.model_path = model_path
        self.use_encryption = use_encryption
        self.device = device
        self.image_size = image_size
        self.feature_extractor = None
        
    def run(self):
        """执行图像处理和预测"""
        try:
            if not MODULES_IMPORTED:
                self.error_occurred.emit("无法导入必要的模块，请检查项目依赖是否正确安装")
                return
                
            # 加载模型
            self.progress_update.emit(10)
            device = torch.device(self.device)
            model = Recce(num_classes=1)
            
            # 加载模型权重
            try:
                if self.model_path.endswith(".bin"):
                    weights = torch.load(self.model_path, map_location="cpu")["model"]
                else:
                    weights = torch.load(self.model_path, map_location="cpu")
                model.load_state_dict(weights)
            except Exception as e:
                self.error_occurred.emit(f"加载模型失败: {str(e)}")
                return
                
            model = model.to(device)
            freeze_weights(model)
            model.eval()
            self.progress_update.emit(30)
            
            # 预处理图像
            try:
                img = cv2.imread(self.image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                compose = Compose([Resize(height=self.image_size, width=self.image_size),
                                  Normalize(mean=[0.5] * 3, std=[0.5] * 3),
                                  ToTensorV2()])
                img_tensor = compose(image=img)['image'].unsqueeze(0).to(device)
            except Exception as e:
                self.error_occurred.emit(f"预处理图像失败: {str(e)}")
                return
                
            self.progress_update.emit(50)
            
            # 检测结果
            results = {}
            results['image_path'] = self.image_path
            
            # 不使用加密进行预测
            if not self.use_encryption:
                try:
                    with torch.no_grad():
                        prediction = model(img_tensor)
                        prediction = torch.sigmoid(prediction).cpu().item()
                    results['prediction'] = prediction
                    results['is_fake'] = prediction >= 0.5
                    results['encryption_used'] = False
                except Exception as e:
                    self.error_occurred.emit(f"预测失败: {str(e)}")
                    return
            # 使用同态加密进行预测
            else:
                try:
                    # 初始化同态加密特征提取器
                    self.feature_extractor = HomomorphicFeatureExtractor()
                    
                    # 检查是否存在密钥，否则创建新密钥
                    keys_dir = Path("keys")
                    public_key_path = keys_dir / "public.key"
                    private_key_path = keys_dir / "private.key"
                    
                    if keys_dir.exists() and public_key_path.exists() and private_key_path.exists():
                        self.feature_extractor.load_keys(str(public_key_path), str(private_key_path))
                    else:
                        keys_dir.mkdir(exist_ok=True)
                        public_key, private_key = self.feature_extractor.initialize_keys()
                        self.feature_extractor.save_keys(str(public_key_path), str(private_key_path))
                    
                    # 提取特征
                    features = self.feature_extractor.extract_features(model, img_tensor)
                    self.progress_update.emit(70)
                    
                    # 加密特征
                    encrypted_features = self.feature_extractor.encrypt_features(features)
                    self.progress_update.emit(80)
                    
                    # 解密特征
                    decrypted_features = self.feature_extractor.decrypt_features(encrypted_features)
                    
                    # 简化处理：使用特征的平均值作为检测分数
                    prediction = float(np.mean(decrypted_features))
                    prediction = 1.0 / (1.0 + np.exp(-prediction))  # 使用sigmoid函数映射到0-1
                    
                    results['prediction'] = prediction
                    results['is_fake'] = prediction >= 0.5
                    results['encryption_used'] = True
                except Exception as e:
                    self.error_occurred.emit(f"加密预测失败: {str(e)}")
                    return
            
            self.progress_update.emit(100)
            self.result_ready.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(f"处理过程中发生错误: {str(e)}")


class ImageEncryptionThread(QThread):
    """图像加密线程，在后台处理图像加密操作"""
    progress_update = pyqtSignal(int)
    encryption_complete = pyqtSignal(object, str)  # 返回加密后的数据和临时文件路径
    error_occurred = pyqtSignal(str)
    
    def __init__(self, image_path, resize_to=(64, 64), parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.resize_to = resize_to
        
    def run(self):
        try:
            # 初始化加密模块
            self.progress_update.emit(10)
            encryption = HomomorphicEncryption()
            
            # 检查是否存在密钥，否则创建新密钥
            keys_dir = Path("keys")
            public_key_path = keys_dir / "public.key"
            private_key_path = keys_dir / "private.key"
            
            if keys_dir.exists() and public_key_path.exists() and private_key_path.exists():
                encryption.load_keypair(str(public_key_path), str(private_key_path))
            else:
                keys_dir.mkdir(exist_ok=True)
                public_key, private_key = encryption.generate_keypair()
                encryption.save_keypair(str(public_key_path), str(private_key_path))
            
            self.progress_update.emit(20)
            
            # 加载原始图像
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 调整大小以减少计算量
            img_small = cv2.resize(img, self.resize_to)
            
            # 将图像转换为平坦数组
            flat_img = img_small.flatten()
            
            self.progress_update.emit(30)
            
            # 分批处理以提高响应性
            batch_size = 1000  # 每批处理的像素数
            total_pixels = len(flat_img)
            encrypted_pixels = []
            
            for i in range(0, total_pixels, batch_size):
                batch = flat_img[i:min(i+batch_size, total_pixels)]
                encrypted_batch = encryption.encrypt(batch)
                encrypted_pixels.extend(encrypted_batch)
                
                # 更新进度
                progress = 30 + 60 * (i + batch_size) / total_pixels
                self.progress_update.emit(int(progress))
            
            # 创建临时文件保存加密数据
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            temp_file = temp_dir / f"encrypted_img_{int(time.time())}.npy"
            np.save(str(temp_file), encrypted_pixels)
            
            self.progress_update.emit(100)
            self.encryption_complete.emit(encrypted_pixels, str(temp_file))
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class ImageDecryptionThread(QThread):
    """图像解密线程，在后台处理图像解密操作"""
    progress_update = pyqtSignal(int)
    decryption_complete = pyqtSignal(np.ndarray, tuple)  # 返回解密后的图像数据和图像尺寸
    error_occurred = pyqtSignal(str)
    
    def __init__(self, encrypted_file_path, output_size=(64, 64), parent=None):
        super().__init__(parent)
        self.encrypted_file_path = encrypted_file_path
        self.output_size = output_size
        
    def run(self):
        try:
            # 初始化解密模块
            self.progress_update.emit(10)
            encryption = HomomorphicEncryption()
            
            # 检查密钥是否存在
            keys_dir = Path("keys")
            public_key_path = keys_dir / "public.key"
            private_key_path = keys_dir / "private.key"
            
            if not (keys_dir.exists() and public_key_path.exists() and private_key_path.exists()):
                raise Exception("找不到加密密钥，请确保keys目录中存在public.key和private.key文件")
            
            # 加载密钥
            encryption.load_keypair(str(public_key_path), str(private_key_path))
            self.progress_update.emit(20)
            
            # 加载加密数据
            try:
                encrypted_data = np.load(self.encrypted_file_path, allow_pickle=True)
            except Exception as e:
                raise Exception(f"无法加载加密文件: {str(e)}")
            
            self.progress_update.emit(30)
            
            # 分批解密数据以提高响应性
            batch_size = 1000  # 每批处理的像素数
            total_pixels = len(encrypted_data)
            decrypted_pixels = []
            
            for i in range(0, total_pixels, batch_size):
                batch = encrypted_data[i:min(i+batch_size, total_pixels)]
                decrypted_batch = encryption.decrypt(batch)
                decrypted_pixels.extend(decrypted_batch)
                
                # 更新进度
                progress = 30 + 60 * (i + batch_size) / total_pixels
                self.progress_update.emit(int(progress))
            
            # 将解密后的平坦数组重新整形为图像
            height, width = self.output_size
            channels = 3  # RGB图像
            
            # 确保解密的像素数与图像尺寸匹配
            expected_pixels = height * width * channels
            if len(decrypted_pixels) != expected_pixels:
                # 如果像素数不匹配，尝试调整
                if len(decrypted_pixels) > expected_pixels:
                    decrypted_pixels = decrypted_pixels[:expected_pixels]
                else:
                    # 填充额外的零像素
                    decrypted_pixels.extend([0] * (expected_pixels - len(decrypted_pixels)))
            
            # 将像素值转换为uint8类型
            decrypted_pixels = np.array(decrypted_pixels, dtype=np.uint8)
            
            # 重塑为3D图像数组
            decrypted_image = decrypted_pixels.reshape(height, width, channels)
            
            self.progress_update.emit(100)
            self.decryption_complete.emit(decrypted_image, self.output_size)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class BatchDetectionThread(QThread):
    """批量图像检测线程"""
    progress_update = pyqtSignal(int)  # 总体进度
    batch_progress = pyqtSignal(int, int, str)  # 当前数量, 总数, 当前处理文件
    batch_complete = pyqtSignal(list)  # 批量结果列表
    error_occurred = pyqtSignal(str)  # 错误信息
    
    def __init__(self, image_list, model_path, use_encryption, device="cpu", image_size=299, parent=None):
        super().__init__(parent)
        self.image_list = image_list
        self.model_path = model_path
        self.use_encryption = use_encryption
        self.device = device
        self.image_size = image_size
        
    def run(self):
        """执行批量检测"""
        try:
            # 检查是否存在图像
            if not self.image_list:
                raise Exception("图像列表为空")
                
            # 加载模型（只加载一次）
            self.progress_update.emit(5)
            
            if not MODULES_IMPORTED:
                self.error_occurred.emit("无法导入必要的模块，请检查项目依赖是否正确安装")
                return
                
            device = torch.device(self.device)
            model = Recce(num_classes=1)
            
            # 加载模型权重
            try:
                if self.model_path.endswith(".bin"):
                    weights = torch.load(self.model_path, map_location="cpu")["model"]
                else:
                    weights = torch.load(self.model_path, map_location="cpu")
                model.load_state_dict(weights)
            except Exception as e:
                self.error_occurred.emit(f"加载模型失败: {str(e)}")
                return
                
            model = model.to(device)
            freeze_weights(model)
            model.eval()
            
            self.progress_update.emit(10)
            
            # 初始化同态加密（如果需要）
            if self.use_encryption:
                feature_extractor = HomomorphicFeatureExtractor()
                
                # 检查密钥
                keys_dir = Path("keys")
                public_key_path = keys_dir / "public.key"
                private_key_path = keys_dir / "private.key"
                
                if keys_dir.exists() and public_key_path.exists() and private_key_path.exists():
                    feature_extractor.load_keys(str(public_key_path), str(private_key_path))
                else:
                    keys_dir.mkdir(exist_ok=True)
                    public_key, private_key = feature_extractor.initialize_keys()
                    feature_extractor.save_keys(str(public_key_path), str(private_key_path))
            
            # 图像预处理
            compose = Compose([Resize(height=self.image_size, width=self.image_size),
                              Normalize(mean=[0.5] * 3, std=[0.5] * 3),
                              ToTensorV2()])
            
            # 批量处理图像
            total_images = len(self.image_list)
            results = []
            
            for idx, img_path in enumerate(self.image_list):
                try:
                    # 更新进度
                    current = idx + 1
                    progress = 10 + 90 * (current / total_images)
                    self.progress_update.emit(int(progress))
                    self.batch_progress.emit(current, total_images, img_path)
                    
                    # 预处理图像
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = compose(image=img)['image'].unsqueeze(0).to(device)
                    
                    # 检测结果
                    result = {}
                    result['image_path'] = img_path
                    
                    # 不使用加密进行预测
                    if not self.use_encryption:
                        with torch.no_grad():
                            prediction = model(img_tensor)
                            prediction = torch.sigmoid(prediction).cpu().item()
                        result['prediction'] = prediction
                        result['is_fake'] = prediction >= 0.5
                        result['encryption_used'] = False
                    # 使用同态加密进行预测
                    else:
                        # 提取特征
                        features = feature_extractor.extract_features(model, img_tensor)
                        
                        # 加密特征
                        encrypted_features = feature_extractor.encrypt_features(features)
                        
                        # 解密特征
                        decrypted_features = feature_extractor.decrypt_features(encrypted_features)
                        
                        # 简化处理：使用特征的平均值作为检测分数
                        prediction = float(np.mean(decrypted_features))
                        prediction = 1.0 / (1.0 + np.exp(-prediction))  # 使用sigmoid函数映射到0-1
                        
                        result['prediction'] = prediction
                        result['is_fake'] = prediction >= 0.5
                        result['encryption_used'] = True
                    
                    results.append(result)
                    
                except Exception as e:
                    # 记录错误但继续处理其他图像
                    self.error_occurred.emit(f"处理图像 {img_path} 时出错: {str(e)}")
                    
                    # 添加错误结果
                    result = {
                        'image_path': img_path,
                        'error': str(e),
                        'prediction': 0,
                        'is_fake': False,
                        'encryption_used': self.use_encryption
                    }
                    results.append(result)
            
            # 完成所有处理
            self.progress_update.emit(100)
            self.batch_complete.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(f"批量处理过程中发生错误: {str(e)}")


class BatchProcessorThread(QThread):
    """批量处理图像线程"""
    progress_update = pyqtSignal(int, str)  # 进度值, 状态信息
    batch_complete = pyqtSignal(dict)  # 批处理结果
    error_occurred = pyqtSignal(str, str)  # 错误文件, 错误信息
    
    def __init__(self, 
                 file_list, 
                 output_dir, 
                 operation_type="encrypt", 
                 resize_to=(64, 64), 
                 parent=None):
        """
        初始化批处理线程
        
        Parameters:
        file_list (list): 待处理文件列表
        output_dir (str): 输出目录
        operation_type (str): 操作类型，"encrypt"或"decrypt"
        resize_to (tuple): 处理前调整图像大小
        """
        super().__init__(parent)
        self.file_list = file_list
        self.output_dir = Path(output_dir)
        self.operation_type = operation_type
        self.resize_to = resize_to
        
    def run(self):
        try:
            # 确保输出目录存在
            self.output_dir.mkdir(exist_ok=True)
            
            # 初始化加密模块
            encryption = HomomorphicEncryption()
            
            # 检查是否存在密钥，否则创建新密钥
            keys_dir = Path("keys")
            public_key_path = keys_dir / "public.key"
            private_key_path = keys_dir / "private.key"
            
            if not (keys_dir.exists() and public_key_path.exists() and private_key_path.exists()):
                if self.operation_type == "encrypt":
                    # 如果是加密操作，可以创建新密钥
                    keys_dir.mkdir(exist_ok=True)
                    public_key, private_key = encryption.generate_keypair()
                    encryption.save_keypair(str(public_key_path), str(private_key_path))
                    self.progress_update.emit(0, "已创建新密钥")
                else:
                    # 如果是解密操作，缺少密钥则报错
                    raise Exception("找不到加密密钥，请确保keys目录中存在public.key和private.key文件")
            
            # 加载密钥
            encryption.load_keypair(str(public_key_path), str(private_key_path))
            
            # 批量处理文件
            total_files = len(self.file_list)
            processed_files = 0
            success_files = 0
            failed_files = 0
            results = {
                "success": [],
                "failed": []
            }
            
            for file_path in self.file_list:
                try:
                    file_name = os.path.basename(file_path)
                    
                    # 更新进度和状态
                    progress = int((processed_files / total_files) * 100)
                    self.progress_update.emit(progress, f"正在处理: {file_name}")
                    
                    # 加密操作
                    if self.operation_type == "encrypt":
                        # 加载图像
                        img = cv2.imread(file_path)
                        if img is None:
                            raise Exception("无法加载图像文件")
                            
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # 调整大小以减少计算量
                        img_small = cv2.resize(img, self.resize_to)
                        
                        # 将图像转换为平坦数组
                        flat_img = img_small.flatten()
                        
                        # 加密图像
                        encrypted_pixels = encryption.encrypt(flat_img)
                        
                        # 保存加密后的数据
                        output_path = self.output_dir / f"{Path(file_name).stem}_encrypted.npy"
                        np.save(str(output_path), encrypted_pixels)
                        
                        # 记录成功
                        success_files += 1
                        results["success"].append({
                            "input": file_path,
                            "output": str(output_path)
                        })
                        
                    # 解密操作
                    elif self.operation_type == "decrypt":
                        # 加载加密数据
                        encrypted_data = np.load(file_path, allow_pickle=True)
                        
                        # 解密数据
                        decrypted_pixels = encryption.decrypt(encrypted_data)
                        
                        # 将解密后的平坦数组重新整形为图像
                        height, width = self.resize_to
                        channels = 3  # RGB图像
                        
                        # 确保解密的像素数与图像尺寸匹配
                        expected_pixels = height * width * channels
                        if len(decrypted_pixels) != expected_pixels:
                            # 如果像素数不匹配，尝试调整
                            if len(decrypted_pixels) > expected_pixels:
                                decrypted_pixels = decrypted_pixels[:expected_pixels]
                            else:
                                # 填充额外的零像素
                                decrypted_pixels.extend([0] * (expected_pixels - len(decrypted_pixels)))
                        
                        # 将像素值转换为uint8类型
                        decrypted_pixels = np.array(decrypted_pixels, dtype=np.uint8)
                        
                        # 重塑为3D图像数组
                        decrypted_image = decrypted_pixels.reshape(height, width, channels)
                        
                        # 保存解密后的图像
                        output_path = self.output_dir / f"{Path(file_name).stem}_decrypted.png"
                        cv2.imwrite(str(output_path), cv2.cvtColor(decrypted_image, cv2.COLOR_RGB2BGR))
                        
                        # 记录成功
                        success_files += 1
                        results["success"].append({
                            "input": file_path,
                            "output": str(output_path)
                        })
                    
                except Exception as e:
                    # 记录失败
                    failed_files += 1
                    results["failed"].append({
                        "input": file_path,
                        "error": str(e)
                    })
                    
                    # 发送错误信号
                    self.error_occurred.emit(file_path, str(e))
                
                # 更新进度
                processed_files += 1
                progress = int((processed_files / total_files) * 100)
                self.progress_update.emit(progress, f"已处理: {processed_files}/{total_files}")
            
            # 完成批处理
            self.progress_update.emit(100, "批处理完成")
            
            # 添加批处理汇总信息
            results["summary"] = {
                "total": total_files,
                "success": success_files,
                "failed": failed_files,
                "output_dir": str(self.output_dir)
            }
            
            # 发送完成信号
            self.batch_complete.emit(results)
            
        except Exception as e:
            self.error_occurred.emit("batch_process", f"批处理失败: {str(e)}")


class BatchProcessDialog(QDialog):
    """批量处理对话框"""
    
    def __init__(self, operation_type="encrypt", parent=None):
        """
        初始化批量处理对话框
        
        Parameters:
        operation_type (str): 操作类型，"encrypt"或"decrypt"
        """
        super().__init__(parent)
        self.operation_type = operation_type
        self.file_list = []
        self.output_dir = ""
        self.processor = None
        self.initUI()
        
    def initUI(self):
        """初始化界面"""
        op_name = "加密" if self.operation_type == "encrypt" else "解密"
        file_type = "图像" if self.operation_type == "encrypt" else "加密文件"
        
        self.setWindowTitle(f"批量{op_name}处理")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        layout = QVBoxLayout()
        
        # 文件列表
        list_group = QGroupBox(f"待处理{file_type}列表")
        list_layout = QVBoxLayout()
        
        self.file_listwidget = QListWidget()
        self.file_listwidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        list_layout.addWidget(self.file_listwidget)
        
        # 文件操作按钮
        file_buttons_layout = QHBoxLayout()
        
        self.add_files_btn = QPushButton(f"添加{file_type}")
        self.add_files_btn.clicked.connect(self.add_files)
        file_buttons_layout.addWidget(self.add_files_btn)
        
        self.add_dir_btn = QPushButton("添加文件夹")
        self.add_dir_btn.clicked.connect(self.add_directory)
        file_buttons_layout.addWidget(self.add_dir_btn)
        
        self.remove_files_btn = QPushButton("移除选中")
        self.remove_files_btn.clicked.connect(self.remove_selected_files)
        file_buttons_layout.addWidget(self.remove_files_btn)
        
        self.clear_files_btn = QPushButton("清空列表")
        self.clear_files_btn.clicked.connect(self.clear_file_list)
        file_buttons_layout.addWidget(self.clear_files_btn)
        
        list_layout.addLayout(file_buttons_layout)
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # 输出设置
        output_group = QGroupBox("输出设置")
        output_layout = QVBoxLayout()
        
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        output_dir_layout.addWidget(self.output_dir_edit)
        
        self.browse_output_btn = QPushButton("浏览...")
        self.browse_output_btn.clicked.connect(self.browse_output_directory)
        output_dir_layout.addWidget(self.browse_output_btn)
        
        output_layout.addLayout(output_dir_layout)
        
        # 调整大小设置
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("处理图像大小:"))
        
        self.width_spin = QSpinBox()
        self.width_spin.setRange(16, 256)
        self.width_spin.setValue(64)
        size_layout.addWidget(self.width_spin)
        
        size_layout.addWidget(QLabel("×"))
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(16, 256)
        self.height_spin.setValue(64)
        size_layout.addWidget(self.height_spin)
        
        size_layout.addWidget(QLabel("像素"))
        size_layout.addStretch()
        
        output_layout.addLayout(size_layout)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # 进度区域
        progress_group = QGroupBox("处理进度")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("就绪")
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # 操作按钮
        buttons_layout = QHBoxLayout()
        
        self.start_btn = QPushButton(f"开始{op_name}")
        self.start_btn.clicked.connect(self.start_processing)
        buttons_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)
        
    def add_files(self):
        """添加文件到处理列表"""
        options = QFileDialog.Options()
        
        if self.operation_type == "encrypt":
            files, _ = QFileDialog.getOpenFileNames(
                self, "选择图像文件", "", 
                "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)", 
                options=options
            )
        else:  # decrypt
            files, _ = QFileDialog.getOpenFileNames(
                self, "选择加密文件", "", 
                "加密文件 (*.npy);;所有文件 (*)", 
                options=options
            )
        
        if files:
            for file_path in files:
                if file_path not in self.file_list:
                    self.file_list.append(file_path)
                    self.file_listwidget.addItem(file_path)
            
            # 更新状态
            self.update_status()
    
    def add_directory(self):
        """添加目录中的所有符合条件的文件"""
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(
            self, "选择文件夹", "", options=options
        )
        
        if directory:
            # 根据操作类型确定文件扩展名
            if self.operation_type == "encrypt":
                extensions = [".png", ".jpg", ".jpeg", ".bmp"]
            else:  # decrypt
                extensions = [".npy"]
            
            # 遍历目录及子目录
            for root, _, files in os.walk(directory):
                for file in files:
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in extensions:
                        file_path = os.path.join(root, file)
                        if file_path not in self.file_list:
                            self.file_list.append(file_path)
                            self.file_listwidget.addItem(file_path)
            
            # 更新状态
            self.update_status()
    
    def remove_selected_files(self):
        """从列表中移除选中的文件"""
        selected_items = self.file_listwidget.selectedItems()
        
        for item in selected_items:
            row = self.file_listwidget.row(item)
            file_path = item.text()
            
            # 从列表中移除
            self.file_listwidget.takeItem(row)
            self.file_list.remove(file_path)
        
        # 更新状态
        self.update_status()
    
    def clear_file_list(self):
        """清空文件列表"""
        self.file_listwidget.clear()
        self.file_list.clear()
        
        # 更新状态
        self.update_status()
    
    def browse_output_directory(self):
        """选择输出目录"""
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(
            self, "选择输出目录", "", options=options
        )
        
        if directory:
            self.output_dir = directory
            self.output_dir_edit.setText(directory)
    
    def update_status(self):
        """更新状态信息"""
        count = len(self.file_list)
        op_name = "加密" if self.operation_type == "encrypt" else "解密"
        
        if count == 0:
            self.status_label.setText("就绪")
        else:
            self.status_label.setText(f"待{op_name}文件: {count}个")
    
    def update_progress(self, value, message):
        """更新进度条和状态信息"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def handle_processing_error(self, file_path, error_message):
        """处理批处理过程中的错误"""
        file_name = os.path.basename(file_path)
        self.status_label.setText(f"处理文件 {file_name} 时出错: {error_message}")
    
    def handle_batch_complete(self, results):
        """处理批处理完成事件"""
        summary = results.get("summary", {})
        total = summary.get("total", 0)
        success = summary.get("success", 0)
        failed = summary.get("failed", 0)
        output_dir = summary.get("output_dir", "")
        
        op_name = "加密" if self.operation_type == "encrypt" else "解密"
        
        # 显示结果消息
        if failed == 0:
            message = f"批量{op_name}已完成!\n\n成功处理 {success} 个文件\n\n输出目录: {output_dir}"
            QMessageBox.information(self, "批处理完成", message)
        else:
            message = (f"批量{op_name}已完成，但部分文件处理失败\n\n"
                      f"成功: {success} 个文件\n"
                      f"失败: {failed} 个文件\n\n"
                      f"输出目录: {output_dir}")
            QMessageBox.warning(self, "批处理部分完成", message)
        
        # 重置界面
        self.progress_bar.setValue(100)
        self.status_label.setText(f"批量{op_name}完成")
        self.processor = None
        
        # 默认接受对话框（关闭）
        self.accept()
    
    def start_processing(self):
        """开始批量处理"""
        # 检查文件列表
        if not self.file_list:
            QMessageBox.warning(self, "文件列表为空", "请先添加要处理的文件")
            return
        
        # 检查输出目录
        if not self.output_dir:
            QMessageBox.warning(self, "未设置输出目录", "请选择输出目录")
            return
        
        # 获取图像大小设置
        width = self.width_spin.value()
        height = self.height_spin.value()
        
        # 创建并启动处理线程
        self.processor = BatchProcessorThread(
            self.file_list, 
            self.output_dir, 
            self.operation_type, 
            resize_to=(width, height), 
            parent=self
        )
        
        # 连接信号
        self.processor.progress_update.connect(self.update_progress)
        self.processor.error_occurred.connect(self.handle_processing_error)
        self.processor.batch_complete.connect(self.handle_batch_complete)
        
        # 禁用界面元素
        self.add_files_btn.setEnabled(False)
        self.add_dir_btn.setEnabled(False)
        self.remove_files_btn.setEnabled(False)
        self.clear_files_btn.setEnabled(False)
        self.browse_output_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setText("中止处理")
        
        # 启动处理
        self.processor.start()
    
    def reject(self):
        """处理取消/关闭事件"""
        # 如果正在处理，询问是否终止
        if self.processor and self.processor.isRunning():
            reply = QMessageBox.question(
                self, "确认中止", 
                "批处理正在进行中，确定要中止吗？",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # 尝试终止线程
                self.processor.terminate()
                self.processor.wait()
                super().reject()
        else:
            super().reject()


class DecryptionSettingsDialog(QDialog):
    """解密设置对话框，允许用户配置解密参数"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """初始化对话框界面"""
        self.setWindowTitle("解密设置")
        self.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # 尺寸设置组
        size_group = QGroupBox("图像尺寸设置")
        size_layout = QFormLayout()
        
        # 宽度设置
        self.width_spin = QSpinBox()
        self.width_spin.setRange(16, 256)
        self.width_spin.setValue(64)
        self.width_spin.setSingleStep(8)
        size_layout.addRow("宽度:", self.width_spin)
        
        # 高度设置
        self.height_spin = QSpinBox()
        self.height_spin.setRange(16, 256)
        self.height_spin.setValue(64)
        self.height_spin.setSingleStep(8)
        size_layout.addRow("高度:", self.height_spin)
        
        # 保持宽高比选项
        self.maintain_ratio_check = QCheckBox("保持宽高比例")
        self.maintain_ratio_check.setChecked(True)
        self.maintain_ratio_check.stateChanged.connect(self.update_ratio)
        size_layout.addRow("", self.maintain_ratio_check)
        
        # 提示信息
        self.info_label = QLabel("注意: 尺寸应与加密时相同，否则可能导致解密失败或图像失真")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: red;")
        size_layout.addRow(self.info_label)
        
        size_group.setLayout(size_layout)
        layout.addWidget(size_group)
        
        # 按钮区域
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def update_ratio(self, state):
        """当保持宽高比选项状态改变时更新高度值"""
        if state == Qt.Checked:
            # 保持1:1的比例
            self.height_spin.setValue(self.width_spin.value())
            # 连接宽度变化信号
            self.width_spin.valueChanged.connect(self.sync_height)
        else:
            # 断开宽度变化信号
            try:
                self.width_spin.valueChanged.disconnect(self.sync_height)
            except:
                pass
    
    def sync_height(self, value):
        """同步高度值与宽度值"""
        if self.maintain_ratio_check.isChecked():
            self.height_spin.setValue(value)
    
    def get_size(self):
        """获取设置的尺寸"""
        return (self.width_spin.value(), self.height_spin.value())


class ImageGallery(QScrollArea):
    """
    图像浏览器组件，用于显示多张缩略图并允许用户选择
    """
    imageClicked = pyqtSignal(str)  # 图像被选中信号，传递图像路径
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        self.image_paths = []
        self.image_widgets = []
        self.selected_index = -1
        
    def initUI(self):
        """初始化UI组件"""
        # 设置滚动区域属性
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 创建内容容器
        content_widget = QWidget()
        self.setWidget(content_widget)
        
        # 使用网格布局
        self.grid_layout = QGridLayout(content_widget)
        self.grid_layout.setSpacing(10)
        
        # 设置每行显示的图像数量
        self.images_per_row = 3
    
    def clear(self):
        """清空图像浏览器"""
        # 移除所有图像小部件
        for widget in self.image_widgets:
            self.grid_layout.removeWidget(widget)
            widget.deleteLater()
        
        self.image_paths = []
        self.image_widgets = []
        self.selected_index = -1
    
    def add_images(self, image_paths):
        """添加多个图像到浏览器"""
        for path in image_paths:
            self.add_image(path)
    
    def add_image(self, image_path):
        """添加单个图像到浏览器"""
        if image_path in self.image_paths:
            return  # 避免重复添加
        
        # 添加到路径列表
        self.image_paths.append(image_path)
        
        # 创建图像容器小部件
        image_widget = QFrame()
        image_widget.setFixedSize(150, 170)  # 调整大小以适应图像和标题
        image_widget.setFrameShape(QFrame.Box)
        image_widget.setLineWidth(1)
        image_widget.setObjectName(f"image_frame_{len(self.image_paths) - 1}")
        
        # 设置边框样式
        image_widget.setStyleSheet("QFrame { border: 1px solid #CCCCCC; border-radius: 5px; }")
        
        # 创建布局
        layout = QVBoxLayout(image_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 添加图像标签
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setFixedSize(140, 120)
        layout.addWidget(img_label)
        
        # 加载并显示图像
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(140, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                img_label.setPixmap(pixmap)
            else:
                img_label.setText("图像无效")
        except Exception as e:
            img_label.setText("加载失败")
            print(f"加载图像失败: {str(e)}")
        
        # 添加图像名称标签
        name_label = QLabel(os.path.basename(image_path))
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setWordWrap(True)
        name_label.setFixedHeight(40)
        layout.addWidget(name_label)
        
        # 计算位置并添加到网格
        idx = len(self.image_paths) - 1
        row = idx // self.images_per_row
        col = idx % self.images_per_row
        self.grid_layout.addWidget(image_widget, row, col)
        
        # 添加到小部件列表
        self.image_widgets.append(image_widget)
        
        # 添加点击事件
        image_widget.mousePressEvent = lambda event, idx=idx: self.on_image_clicked(idx)
    
    def on_image_clicked(self, index):
        """图像被点击时的处理函数"""
        # 取消之前选中的图像的高亮
        if 0 <= self.selected_index < len(self.image_widgets):
            self.image_widgets[self.selected_index].setStyleSheet("QFrame { border: 1px solid #CCCCCC; border-radius: 5px; }")
        
        # 设置新选中的图像高亮
        self.selected_index = index
        self.image_widgets[index].setStyleSheet("QFrame { border: 2px solid #3874B6; border-radius: 5px; background-color: #EFF6FF; }")
        
        # 发送信号，传递被选中图像的路径
        self.imageClicked.emit(self.image_paths[index])
    
    def get_selected_image(self):
        """获取当前选中的图像路径"""
        if 0 <= self.selected_index < len(self.image_paths):
            return self.image_paths[self.selected_index]
        return None


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib画布用于嵌入PyQt界面"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        
        # 检查是否存在中文字体
        self.has_chinese_font = self.check_chinese_fonts()
        
        # 应用全局字体配置
        if self.has_chinese_font and self.chinese_font_name:
            # 全局设置所有matplotlib元素使用该字体
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['font.sans-serif'] = [self.chinese_font_name, 'DejaVu Sans', 'Arial']
            
            # 为了确保都能显示中文，直接设置全局字体
            font_props = matplotlib.font_manager.FontProperties(family=self.chinese_font_name)
            self.fig.suptitle('', fontproperties=font_props)  # 创建一个空标题以加载字体
        
        # 调整子图参数，增加边距
        self.fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
        
        # 使FigureCanvas成为焦点策略，使其能够接收键盘事件
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
    
    def check_chinese_fonts(self):
        """检查是否有可用的中文字体，并记录第一个可用的字体名称"""
        # 获取操作系统类型
        import platform
        system_type = platform.system()
        
        # 记录找到的第一个中文字体
        self.chinese_font_name = None
        
        # 打印系统中所有可用的字体（调试用）
        try:
            from matplotlib.font_manager import fontManager
            print("\n系统中可用的字体:")
            for font in sorted(set([f.name for f in fontManager.ttflist])):
                print(f"  - {font}")
            print("\n")
        except Exception as e:
            print(f"无法列出系统字体: {e}")
        
        # 根据操作系统选择字体列表
        if system_type == 'Windows':
            # Windows系统常见中文字体
            chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi', 
                            'Arial Unicode MS', 'Arial']
        elif system_type == 'Darwin':  # macOS
            chinese_fonts = ['STHeiti', 'Heiti TC', 'Heiti SC', 'PingFang TC', 'PingFang SC', 
                            'Hiragino Sans GB', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 
                            'AppleGothic', 'Arial']
        else:  # Linux或其他系统
            chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback',
                            'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Microsoft YaHei', 'SimHei',
                            'Arial Unicode MS', 'Arial']
        
        print(f"尝试以下中文字体: {', '.join(chinese_fonts)}")
        
        for font in chinese_fonts:
            try:
                font_path = matplotlib.font_manager.findfont(matplotlib.font_manager.FontProperties(family=font))
                if font_path:
                    print(f"可用中文字体: {font} (路径: {font_path})")
                    # 记录找到的第一个字体
                    self.chinese_font_name = font
                    # 全局设置该字体
                    plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial']
                    return True
            except Exception as e:
                print(f"尝试字体 {font} 时出错: {e}")
                continue
        
        # 如果没有找到中文字体，显示提示
        if parent := self.parent():
            app = QApplication.instance()
            if app and parent.isVisible():
                QMessageBox.information(
                    parent,
                    "字体提示",
                    "未找到中文字体，统计图表将使用英文显示。\n\n"
                    "如需显示中文，请安装中文字体，例如:\n"
                    "- Windows: 安装'微软雅黑'或'黑体'字体\n"
                    "- macOS: 安装'华文黑体'或'苹方'字体\n"
                    "- Linux: 安装'文泉驿'等中文字体"
                )
        
        print("未找到可用的中文字体，将使用默认英文字体")
        # 设置默认英文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        return False
        
    def clear_plot(self):
        """清空图表内容"""
        self.axes.clear()
        self.draw()


class RECCEApp(QMainWindow):
    """RECCE人脸伪造检测应用程序主窗口"""
    
    def __init__(self):
        super().__init__()
        self.processor = None
        self.encryption_thread = None
        self.decryption_thread = None
        self.batch_results = []  # 存储批量处理结果
        self.detection_results_history = []  # 存储所有检测结果的历史记录
        self.initUI()
        
    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle("RECCE人脸伪造检测")
        self.setMinimumSize(1200, 700)  # 增加窗口大小以适应新的组件
        
        # 创建中央部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 创建分割窗口
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧控制面板
        left_panel = QWidget()
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        
        # 模型设置组
        model_group = QGroupBox("模型设置")
        model_layout = QFormLayout()
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.browse_model_btn = QPushButton("浏览...")
        self.browse_model_btn.clicked.connect(self.browse_model)
        
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(self.browse_model_btn)
        
        model_layout.addRow("模型路径:", model_path_layout)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda:0", "cuda:1"])
        model_layout.addRow("计算设备:", self.device_combo)
        
        self.image_size_spin = QSpinBox()
        self.image_size_spin.setRange(64, 1024)
        self.image_size_spin.setValue(299)
        self.image_size_spin.setSingleStep(1)
        model_layout.addRow("图像大小:", self.image_size_spin)
        
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)
        
        # 加密设置组
        encryption_group = QGroupBox("加密设置")
        encryption_layout = QVBoxLayout()
        
        self.use_encryption_check = QCheckBox("使用同态加密")
        encryption_layout.addWidget(self.use_encryption_check)
        
        # 密钥操作按钮布局
        keys_layout = QHBoxLayout()
        
        self.generate_keys_btn = QPushButton("生成新密钥")
        self.generate_keys_btn.clicked.connect(self.generate_keys)
        keys_layout.addWidget(self.generate_keys_btn)
        
        self.export_keys_btn = QPushButton("导出密钥")
        self.export_keys_btn.clicked.connect(self.export_keys)
        keys_layout.addWidget(self.export_keys_btn)
        
        self.import_keys_btn = QPushButton("导入密钥")
        self.import_keys_btn.clicked.connect(self.import_keys)
        keys_layout.addWidget(self.import_keys_btn)
        
        encryption_layout.addLayout(keys_layout)
        
        encryption_group.setLayout(encryption_layout)
        left_layout.addWidget(encryption_group)
        
        # 图像控制组
        image_group = QGroupBox("图像控制")
        image_layout = QVBoxLayout()
        
        # 添加加载单个图像和批量加载图像按钮
        load_layout = QHBoxLayout()
        
        self.load_image_btn = QPushButton("加载图像")
        self.load_image_btn.clicked.connect(self.load_image)
        load_layout.addWidget(self.load_image_btn)
        
        self.load_images_btn = QPushButton("批量加载")
        self.load_images_btn.clicked.connect(self.load_multiple_images)
        load_layout.addWidget(self.load_images_btn)
        
        image_layout.addLayout(load_layout)
        
        self.detect_btn = QPushButton("开始检测")
        self.detect_btn.clicked.connect(self.start_detection)
        self.detect_btn.setEnabled(False)
        image_layout.addWidget(self.detect_btn)
        
        # 批量检测按钮
        self.batch_detect_btn = QPushButton("批量检测")
        self.batch_detect_btn.clicked.connect(self.start_batch_detection)
        self.batch_detect_btn.setEnabled(False)
        image_layout.addWidget(self.batch_detect_btn)
        
        # 添加清除结果按钮
        self.clear_results_btn = QPushButton("清除结果")
        self.clear_results_btn.clicked.connect(self.clear_results)
        image_layout.addWidget(self.clear_results_btn)
        
        # 加密/解密操作
        encrypt_layout = QHBoxLayout()
        
        self.encrypt_image_btn = QPushButton("加密图像")
        self.encrypt_image_btn.clicked.connect(self.encrypt_image)
        self.encrypt_image_btn.setEnabled(False)
        encrypt_layout.addWidget(self.encrypt_image_btn)
        
        self.decrypt_image_btn = QPushButton("解密图像")
        self.decrypt_image_btn.clicked.connect(self.decrypt_image)
        encrypt_layout.addWidget(self.decrypt_image_btn)
        
        image_layout.addLayout(encrypt_layout)
        
        # 批处理操作
        batch_layout = QHBoxLayout()
        
        self.batch_encrypt_btn = QPushButton("批量加密")
        self.batch_encrypt_btn.clicked.connect(self.batch_encrypt)
        batch_layout.addWidget(self.batch_encrypt_btn)
        
        self.batch_decrypt_btn = QPushButton("批量解密")
        self.batch_decrypt_btn.clicked.connect(self.batch_decrypt)
        batch_layout.addWidget(self.batch_decrypt_btn)
        
        image_layout.addLayout(batch_layout)
        
        image_group.setLayout(image_layout)
        left_layout.addWidget(image_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)
        
        # 填充空间
        left_layout.addStretch()
        
        # 中间面板 - 添加图像浏览器
        middle_panel = QWidget()
        middle_panel.setMinimumWidth(350)
        middle_layout = QVBoxLayout(middle_panel)
        
        # 添加图像浏览器标题
        browser_title = QLabel("图像浏览器")
        browser_title.setAlignment(Qt.AlignCenter)
        browser_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        middle_layout.addWidget(browser_title)
        
        # 添加图像浏览器
        self.image_gallery = ImageGallery()
        self.image_gallery.imageClicked.connect(self.on_gallery_image_clicked)
        middle_layout.addWidget(self.image_gallery)
        
        # 添加批量图像操作按钮
        gallery_buttons_layout = QHBoxLayout()
        
        self.clear_gallery_btn = QPushButton("清空图库")
        self.clear_gallery_btn.clicked.connect(self.clear_image_gallery)
        gallery_buttons_layout.addWidget(self.clear_gallery_btn)
        
        self.remove_selected_btn = QPushButton("移除选中")
        self.remove_selected_btn.setEnabled(False)  # 初始禁用
        # TODO: 添加移除选中图像功能
        gallery_buttons_layout.addWidget(self.remove_selected_btn)
        
        middle_layout.addLayout(gallery_buttons_layout)
        
        # 右侧显示面板
        right_panel = QTabWidget()
        
        # 图像显示选项卡
        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        
        self.image_label = QLabel("请加载图像...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setFrameShape(QFrame.Box)
        self.image_label.setFrameShadow(QFrame.Sunken)
        image_layout.addWidget(self.image_label)
        
        # 检测结果标签
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.result_label.setFont(font)
        image_layout.addWidget(self.result_label)
        
        right_panel.addTab(image_tab, "图像和结果")
        
        # 统计图表选项卡
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        # 添加图表切换按钮
        chart_buttons_layout = QHBoxLayout()
        
        self.show_pie_chart_btn = QPushButton("饼图统计")
        self.show_pie_chart_btn.clicked.connect(lambda: self.update_chart('pie'))
        chart_buttons_layout.addWidget(self.show_pie_chart_btn)
        
        self.show_bar_chart_btn = QPushButton("柱状图统计")
        self.show_bar_chart_btn.clicked.connect(lambda: self.update_chart('bar'))
        chart_buttons_layout.addWidget(self.show_bar_chart_btn)
        
        self.show_hist_chart_btn = QPushButton("直方图统计")
        self.show_hist_chart_btn.clicked.connect(lambda: self.update_chart('hist'))
        chart_buttons_layout.addWidget(self.show_hist_chart_btn)
        
        stats_layout.addLayout(chart_buttons_layout)
        
        # 添加Matplotlib画布
        self.chart_canvas = MatplotlibCanvas(stats_tab, width=5, height=4)
        stats_layout.addWidget(self.chart_canvas)
        
        # 统计信息标签
        self.stats_info_label = QLabel("尚无统计数据")
        self.stats_info_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.stats_info_label)
        
        right_panel.addTab(stats_tab, "统计图表")
        
        # 日志选项卡
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        
        self.log_text = QLabel("应用程序已启动...")
        self.log_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.log_text.setFrameShape(QFrame.Box)
        self.log_text.setFrameShadow(QFrame.Sunken)
        self.log_text.setWordWrap(True)
        log_layout.addWidget(self.log_text)
        
        right_panel.addTab(log_tab, "日志")
        
        # 添加左右面板到分割器
        splitter.addWidget(left_panel)
        splitter.addWidget(middle_panel)
        splitter.addWidget(right_panel)
        
        # 初始比例
        splitter.setSizes([300, 350, 550])
        
        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")
        
        # 初始化变量
        self.current_image_path = None
        self.loaded_images = []
    
    def load_multiple_images(self):
        """批量加载图像文件"""
        options = QFileDialog.Options()
        file_names, _ = QFileDialog.getOpenFileNames(
            self, "选择多个图像文件", "", 
            "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)", 
            options=options
        )
        
        if file_names:
            # 先清空当前图片
            self.image_gallery.clear()
            self.loaded_images = []
            
            # 添加图像到浏览器
            self.image_gallery.add_images(file_names)
            self.loaded_images = file_names
            
            # 更新状态
            self.log_message(f"已加载{len(file_names)}张图像")
            self.statusBar.showMessage(f"已加载{len(file_names)}张图像")
            
            # 启用批量检测按钮
            self.batch_detect_btn.setEnabled(len(file_names) > 0)
            
            # 如果有图像，则选择第一张
            if file_names:
                self.on_gallery_image_clicked(file_names[0])
    
    def on_gallery_image_clicked(self, image_path):
        """图像浏览器中的图像被点击时的回调函数"""
        try:
            # 加载并显示图像
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
                self.current_image_path = image_path
                self.detect_btn.setEnabled(True)
                self.encrypt_image_btn.setEnabled(True)
                self.remove_selected_btn.setEnabled(True)
                self.log_message(f"已选择图像: {image_path}")
                self.result_label.setText("")
            else:
                raise Exception("无法加载图像")
        except Exception as e:
            QMessageBox.warning(self, "图像加载错误", f"无法加载图像: {str(e)}")
            self.log_message(f"图像加载错误: {str(e)}")
    
    def clear_image_gallery(self):
        """清空图像浏览器"""
        if self.loaded_images:
            reply = QMessageBox.question(
                self, "确认操作", 
                "确定要清空图像浏览器吗？",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.image_gallery.clear()
                self.loaded_images = []
                self.current_image_path = None
                self.image_label.setPixmap(QPixmap())
                self.image_label.setText("请加载图像...")
                self.result_label.setText("")
                self.detect_btn.setEnabled(False)
                self.batch_detect_btn.setEnabled(False)
                self.encrypt_image_btn.setEnabled(False)
                self.remove_selected_btn.setEnabled(False)
                self.log_message("已清空图像浏览器")
                self.statusBar.showMessage("已清空图像浏览器")
    
    def browse_model(self):
        """浏览并选择模型文件"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", 
            "模型文件 (*.pth *.bin);;所有文件 (*)", 
            options=options
        )
        
        if file_name:
            self.model_path_edit.setText(file_name)
            self.log_message(f"已选择模型文件: {file_name}")
            self.statusBar.showMessage(f"已选择模型文件: {os.path.basename(file_name)}")
    
    def log_message(self, message):
        """记录日志信息"""
        # 获取当前时间
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 将带时间戳的消息添加到日志中
        log_entry = f"[{timestamp}] {message}"
        
        # 更新日志文本
        current_log = self.log_text.text()
        if current_log == "应用程序已启动...":
            # 如果是初始文本，则直接替换
            self.log_text.setText(log_entry)
        else:
            # 否则在现有日志上添加新行
            self.log_text.setText(f"{current_log}\n{log_entry}")
    
    def load_image(self):
        """加载单个图像文件"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", 
            "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)", 
            options=options
        )
        
        if file_name:
            try:
                # 加载并显示图像
                pixmap = QPixmap(file_name)
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_label.setPixmap(pixmap)
                    self.current_image_path = file_name
                    
                    # 更新图像浏览器
                    self.image_gallery.clear()
                    self.image_gallery.add_image(file_name)
                    self.loaded_images = [file_name]
                    
                    # 启用相关按钮
                    self.detect_btn.setEnabled(True)
                    self.encrypt_image_btn.setEnabled(True)
                    
                    # 更新日志和状态栏
                    self.log_message(f"已加载图像: {file_name}")
                    self.statusBar.showMessage(f"已加载图像: {os.path.basename(file_name)}")
                    
                    # 清空之前的检测结果
                    self.result_label.setText("")
                else:
                    raise Exception("无法加载图像")
            except Exception as e:
                QMessageBox.warning(self, "图像加载错误", f"无法加载图像: {str(e)}")
                self.log_message(f"图像加载错误: {str(e)}")
    
    def start_detection(self):
        """开始人脸伪造检测"""
        # 检查模型路径
        model_path = self.model_path_edit.text()
        if not model_path:
            QMessageBox.warning(self, "参数错误", "请先选择模型文件")
            return
        
        # 确保图像已加载
        if not self.current_image_path:
            QMessageBox.warning(self, "参数错误", "请先加载图像")
            return
        
        # 获取参数
        device = self.device_combo.currentText()
        image_size = self.image_size_spin.value()
        use_encryption = self.use_encryption_check.isChecked()
        
        # 禁用界面
        self.detect_btn.setEnabled(False)
        self.batch_detect_btn.setEnabled(False)
        self.load_image_btn.setEnabled(False)
        self.load_images_btn.setEnabled(False)
        self.encrypt_image_btn.setEnabled(False)
        self.decrypt_image_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # 记录开始信息
        detection_type = "同态加密" if use_encryption else "常规"
        self.log_message(f"开始{detection_type}检测，图像: {os.path.basename(self.current_image_path)}")
        self.statusBar.showMessage(f"{detection_type}检测中...")
        
        # 创建并启动处理线程
        self.processor = ImageProcessor(
            self.current_image_path, 
            model_path, 
            use_encryption, 
            device, 
            image_size
        )
        
        # 连接信号
        self.processor.progress_update.connect(self.update_progress)
        self.processor.result_ready.connect(self.handle_detection_result)
        self.processor.error_occurred.connect(self.handle_error)
        
        # 启动处理
        self.processor.start()
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def handle_detection_result(self, results):
        """处理检测结果"""
        # 更新UI
        prediction = results.get('prediction', 0)
        is_fake = results.get('is_fake', False)
        is_encrypted = results.get('encryption_used', False)
        
        # 更新结果标签
        if is_fake:
            self.result_label.setText(f"检测结果: 伪造图像 (概率: {prediction:.4f})")
            self.result_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.result_label.setText(f"检测结果: 真实图像 (概率: {prediction:.4f})")
            self.result_label.setStyleSheet("color: green; font-weight: bold;")
        
        # 更新状态和日志
        method = "同态加密" if is_encrypted else "常规"
        status = "伪造" if is_fake else "真实"
        self.log_message(f"检测完成 [{method}]: 图像被判定为 {status} (概率: {prediction:.4f})")
        self.statusBar.showMessage(f"检测完成: {status} (概率: {prediction:.4f})")
        
        # 重新启用UI
        self.detect_btn.setEnabled(True)
        self.batch_detect_btn.setEnabled(len(self.loaded_images) > 0)
        self.load_image_btn.setEnabled(True)
        self.load_images_btn.setEnabled(True)
        self.encrypt_image_btn.setEnabled(self.current_image_path is not None)
        self.decrypt_image_btn.setEnabled(True)
        
        # 添加到批量检测结果列表
        self.batch_results.append(results)
        
        # 添加到历史记录
        self.detection_results_history.append(results)
        
        # 更新统计图表
        self.update_statistics_chart()
    
    def handle_error(self, error_message):
        """处理处理过程中的错误"""
        QMessageBox.critical(self, "处理错误", error_message)
        self.log_message(f"处理错误: {error_message}")
        self.statusBar.showMessage("处理失败")
        
        # 重新启用UI
        self.detect_btn.setEnabled(self.current_image_path is not None)
        self.batch_detect_btn.setEnabled(len(self.loaded_images) > 0)
        self.load_image_btn.setEnabled(True)
        self.load_images_btn.setEnabled(True)
        self.encrypt_image_btn.setEnabled(self.current_image_path is not None)
        self.decrypt_image_btn.setEnabled(True)
    
    def encrypt_image(self):
        """加密当前图像"""
        # 检查是否已加载图像
        if not self.current_image_path:
            QMessageBox.warning(self, "参数错误", "请先加载图像")
            return
        
        # 禁用UI
        self.encrypt_image_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # 记录开始信息
        self.log_message(f"开始加密图像: {os.path.basename(self.current_image_path)}")
        self.statusBar.showMessage("加密中...")
        
        # 创建并启动加密线程
        self.encryption_thread = ImageEncryptionThread(
            self.current_image_path,
            resize_to=(64, 64)  # 默认尺寸
        )
        
        # 连接信号
        self.encryption_thread.progress_update.connect(self.update_progress)
        self.encryption_thread.encryption_complete.connect(self.handle_encryption_complete)
        self.encryption_thread.error_occurred.connect(self.handle_error)
        
        # 启动处理
        self.encryption_thread.start()
    
    def handle_encryption_complete(self, encrypted_data, temp_file_path):
        """处理图像加密完成事件"""
        # 更新状态和日志
        self.log_message(f"图像加密完成，保存至: {temp_file_path}")
        self.statusBar.showMessage(f"加密完成: {os.path.basename(temp_file_path)}")
        
        # 显示结果消息
        QMessageBox.information(
            self, 
            "加密完成", 
            f"图像加密成功！\n\n加密数据已保存至:\n{temp_file_path}"
        )
        
        # 重新启用UI
        self.encrypt_image_btn.setEnabled(True)
    
    def decrypt_image(self):
        """解密加密后的图像文件"""
        # 选择加密文件
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择加密图像文件", "", 
            "加密文件 (*.npy);;所有文件 (*)", 
            options=options
        )
        
        if not file_name:
            return
            
        # 获取解密参数
        settings_dialog = DecryptionSettingsDialog(self)
        if settings_dialog.exec_() != QDialog.Accepted:
            return
        
        output_size = settings_dialog.get_size()
        
        # 禁用UI
        self.decrypt_image_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # 记录开始信息
        self.log_message(f"开始解密图像: {os.path.basename(file_name)}")
        self.statusBar.showMessage("解密中...")
        
        # 创建并启动解密线程
        self.decryption_thread = ImageDecryptionThread(
            file_name,
            output_size=output_size
        )
        
        # 连接信号
        self.decryption_thread.progress_update.connect(self.update_progress)
        self.decryption_thread.decryption_complete.connect(self.handle_decryption_complete)
        self.decryption_thread.error_occurred.connect(self.handle_error)
        
        # 启动处理
        self.decryption_thread.start()
    
    def handle_decryption_complete(self, decrypted_image, image_size):
        """处理图像解密完成事件"""
        # 将numpy数组转换为QImage和QPixmap
        height, width = image_size
        bytes_per_line = 3 * width
        q_img = QImage(decrypted_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # 显示解密后的图像
        scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        
        # 保存解密后的图像
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        output_path = temp_dir / f"decrypted_img_{int(time.time())}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(decrypted_image, cv2.COLOR_RGB2BGR))
        
        # 更新状态和日志
        self.log_message(f"图像解密完成，保存至: {output_path}")
        self.statusBar.showMessage(f"解密完成: {os.path.basename(str(output_path))}")
        
        # 显示结果消息
        QMessageBox.information(
            self, 
            "解密完成", 
            f"图像解密成功！\n\n解密图像已保存至:\n{output_path}"
        )
        
        # 重新启用UI
        self.decrypt_image_btn.setEnabled(True)
    
    def generate_keys(self):
        """生成新的加密密钥对"""
        try:
            # 确认操作
            reply = QMessageBox.question(
                self, "确认操作", 
                "生成新的密钥对将覆盖现有密钥。继续操作？\n"
                "注意：如果您有使用旧密钥加密的数据，新密钥将无法解密这些数据。",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            # 初始化同态加密
            encryption = HomomorphicEncryption()
            
            # 创建密钥目录
            keys_dir = Path("keys")
            keys_dir.mkdir(exist_ok=True)
            
            # 生成新密钥
            public_key, private_key = encryption.generate_keypair()
            
            # 保存密钥
            public_key_path = keys_dir / "public.key"
            private_key_path = keys_dir / "private.key"
            encryption.save_keypair(str(public_key_path), str(private_key_path))
            
            # 更新状态和日志
            self.log_message(f"已生成新的密钥对，保存至: {keys_dir}")
            self.statusBar.showMessage("已生成新的密钥对")
            
            # 显示成功消息
            QMessageBox.information(
                self, 
                "密钥生成完成", 
                f"新的密钥对已生成并保存至:\n{keys_dir}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "密钥生成错误", f"生成密钥时发生错误: {str(e)}")
            self.log_message(f"密钥生成错误: {str(e)}")
    
    def export_keys(self):
        """导出当前密钥对"""
        try:
            # 检查密钥是否存在
            keys_dir = Path("keys")
            public_key_path = keys_dir / "public.key"
            private_key_path = keys_dir / "private.key"
            
            if not (keys_dir.exists() and public_key_path.exists() and private_key_path.exists()):
                QMessageBox.warning(
                    self, 
                    "密钥不存在", 
                    "找不到密钥文件，请先生成密钥。"
                )
                return
            
            # 选择导出目录
            options = QFileDialog.Options()
            export_dir = QFileDialog.getExistingDirectory(
                self, "选择导出目录", "", options=options
            )
            
            if not export_dir:
                return
            
            # 复制密钥文件到导出目录
            shutil.copy(public_key_path, os.path.join(export_dir, "public.key"))
            shutil.copy(private_key_path, os.path.join(export_dir, "private.key"))
            
            # 更新状态和日志
            self.log_message(f"已导出密钥对至: {export_dir}")
            self.statusBar.showMessage("密钥导出成功")
            
            # 显示成功消息
            QMessageBox.information(
                self, 
                "密钥导出完成", 
                f"密钥对已成功导出至:\n{export_dir}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "密钥导出错误", f"导出密钥时发生错误: {str(e)}")
            self.log_message(f"密钥导出错误: {str(e)}")
    
    def import_keys(self):
        """导入密钥对"""
        try:
            # 选择公钥文件
            options = QFileDialog.Options()
            public_key_file, _ = QFileDialog.getOpenFileName(
                self, "选择公钥文件", "", 
                "密钥文件 (*.key);;所有文件 (*)", 
                options=options
            )
            
            if not public_key_file:
                return
            
            # 选择私钥文件
            private_key_file, _ = QFileDialog.getOpenFileName(
                self, "选择私钥文件", "", 
                "密钥文件 (*.key);;所有文件 (*)", 
                options=options
            )
            
            if not private_key_file:
                return
            
            # 确认操作
            reply = QMessageBox.question(
                self, "确认操作", 
                "导入新的密钥对将覆盖现有密钥。继续操作？",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            # 创建密钥目录
            keys_dir = Path("keys")
            keys_dir.mkdir(exist_ok=True)
            
            # 复制密钥文件
            shutil.copy(public_key_file, keys_dir / "public.key")
            shutil.copy(private_key_file, keys_dir / "private.key")
            
            # 尝试加载密钥以验证有效性
            encryption = HomomorphicEncryption()
            encryption.load_keypair(str(keys_dir / "public.key"), str(keys_dir / "private.key"))
            
            # 更新状态和日志
            self.log_message("已成功导入密钥对")
            self.statusBar.showMessage("密钥导入成功")
            
            # 显示成功消息
            QMessageBox.information(
                self, 
                "密钥导入完成", 
                "密钥对已成功导入。"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "密钥导入错误", f"导入密钥时发生错误: {str(e)}")
            self.log_message(f"密钥导入错误: {str(e)}")
    
    def batch_encrypt(self):
        """打开批量加密对话框"""
        dialog = BatchProcessDialog(operation_type="encrypt", parent=self)
        dialog.exec_()
    
    def batch_decrypt(self):
        """打开批量解密对话框"""
        dialog = BatchProcessDialog(operation_type="decrypt", parent=self)
        dialog.exec_()
    
    def start_batch_detection(self):
        """批量执行人脸伪造检测"""
        # 检查模型路径
        model_path = self.model_path_edit.text()
        if not model_path:
            QMessageBox.warning(self, "参数错误", "请先选择模型文件")
            return
        
        # 确保有加载的图像
        if not self.loaded_images:
            QMessageBox.warning(self, "参数错误", "请先加载图像")
            return
        
        # 确认是否开始批量检测
        reply = QMessageBox.question(
            self, "确认批量检测", 
            f"是否对已加载的{len(self.loaded_images)}张图像进行批量检测？\n这可能需要一段时间。",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # 获取参数
        device = self.device_combo.currentText()
        image_size = self.image_size_spin.value()
        use_encryption = self.use_encryption_check.isChecked()
        
        # 禁用UI
        self.disable_ui_during_batch_process()
        
        # 清空之前的结果
        self.batch_results = []
        
        # 日志
        detection_type = "同态加密" if use_encryption else "常规"
        self.log_message(f"开始批量{detection_type}检测，共{len(self.loaded_images)}张图像...")
        self.statusBar.showMessage(f"批量检测中...")
        
        # 启动批量检测线程
        self.batch_processor = BatchDetectionThread(
            self.loaded_images, model_path, use_encryption, device, image_size)
        self.batch_processor.progress_update.connect(self.update_progress)
        self.batch_processor.batch_progress.connect(self.update_batch_status)
        self.batch_processor.batch_complete.connect(self.handle_batch_detection_complete)
        self.batch_processor.error_occurred.connect(self.handle_error)
        self.batch_processor.start()
    
    def disable_ui_during_batch_process(self):
        """在批处理过程中禁用UI元素"""
        self.detect_btn.setEnabled(False)
        self.batch_detect_btn.setEnabled(False)
        self.load_image_btn.setEnabled(False)
        self.load_images_btn.setEnabled(False)
        self.encrypt_image_btn.setEnabled(False)
        self.decrypt_image_btn.setEnabled(False)
        self.batch_encrypt_btn.setEnabled(False)
        self.batch_decrypt_btn.setEnabled(False)
        self.browse_model_btn.setEnabled(False)
        self.progress_bar.setValue(0)
    
    def enable_ui_after_batch_process(self):
        """批处理完成后恢复UI元素"""
        self.detect_btn.setEnabled(self.current_image_path is not None)
        self.batch_detect_btn.setEnabled(len(self.loaded_images) > 0)
        self.load_image_btn.setEnabled(True)
        self.load_images_btn.setEnabled(True)
        self.encrypt_image_btn.setEnabled(self.current_image_path is not None)
        self.decrypt_image_btn.setEnabled(True)
        self.batch_encrypt_btn.setEnabled(True)
        self.batch_decrypt_btn.setEnabled(True)
        self.browse_model_btn.setEnabled(True)
    
    def update_batch_status(self, current, total, path):
        """更新批量处理状态"""
        self.statusBar.showMessage(f"正在处理: {current}/{total} - {os.path.basename(path)}")
    
    def handle_batch_detection_complete(self, results):
        """处理批量检测完成事件"""
        self.batch_results = results
        
        # 将批量结果添加到历史记录
        self.detection_results_history.extend(results)
        
        # 统计伪造和真实图像数量
        fake_count = sum(1 for result in results if result.get('is_fake', False))
        real_count = len(results) - fake_count
        
        # 更新UI
        self.log_message(f"批量检测完成，共{len(results)}张图像，检测为伪造: {fake_count}张，检测为真实: {real_count}张")
        self.statusBar.showMessage(f"批量检测完成")
        
        # 显示统计图表
        self.update_statistics_chart()
        
        # 切换到统计图表选项卡
        tab_widget = self.findChild(QTabWidget)
        if tab_widget:
            tab_widget.setCurrentIndex(1)  # 切换到统计图表选项卡
        
        # 恢复UI
        self.enable_ui_after_batch_process()
    
    def update_chart(self, chart_type='pie'):
        """更新统计图表"""
        # 处理直方图类型 - 使用新的update_statistics_chart方法
        if chart_type == 'hist':
            self.update_statistics_chart()
            return
            
        # 清空当前图表
        self.chart_canvas.clear_plot()
        
        # 没有结果时返回
        if not self.batch_results:
            self.stats_info_label.setText("尚无统计数据，请先执行批量检测")
            self.chart_canvas.draw()
            return
        
        # 提取伪造概率
        probabilities = [result.get('prediction', 0) for result in self.batch_results]
        
        # 统计伪造和真实的数量
        fake_count = sum(1 for p in probabilities if p >= 0.5)
        real_count = len(probabilities) - fake_count
        
        # 使用画布中已检测到的字体状态，避免重复检测
        chinese_font_found = getattr(self.chart_canvas, 'has_chinese_font', False)
        
        # 设置全局字体和样式
        if chinese_font_found and hasattr(self.chart_canvas, 'chinese_font_name') and self.chart_canvas.chinese_font_name:
            # 明确设置检测到的中文字体
            font_name = self.chart_canvas.chinese_font_name
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial']
            print(f"使用中文字体: {font_name}")
        else:
            # 确保使用默认英文字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
                
        plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
        
        # 如果找不到中文字体，使用英文标题
        if chinese_font_found:
            main_title = 'RECCE人脸伪造检测结果统计'
            fake_label = '伪造图像'
            real_label = '真实图像'
            fake_count_text = f'伪造图像: {fake_count}张'
            real_count_text = f'真实图像: {real_count}张'
            y_axis_label = '图像数量'
            x_axis_label = '伪造概率值'
            threshold_label = '伪造阈值(0.5)'
            fake_area_label = '伪造区域'
            real_area_label = '真实区域'
            
            if chart_type == 'pie':
                chart_subtitle = '真假检测结果分布比例'
            elif chart_type == 'bar':
                chart_subtitle = '真假检测结果数量分布'
        else:
            # 英文替代
            main_title = 'RECCE Face Forgery Detection Results'
            fake_label = 'Fake'
            real_label = 'Real'
            fake_count_text = f'Fake: {fake_count}'
            real_count_text = f'Real: {real_count}'
            y_axis_label = 'Image Count'
            x_axis_label = 'Fake Probability'
            threshold_label = 'Threshold (0.5)'
            fake_area_label = 'Fake Area'
            real_area_label = 'Real Area'
            
            if chart_type == 'pie':
                chart_subtitle = 'Distribution Ratio'
            elif chart_type == 'bar':
                chart_subtitle = 'Count Distribution'
        
        if chart_type == 'pie':
            # 绘制饼图
            labels = [fake_label, real_label]
            sizes = [fake_count, real_count]
            colors = ['#FF6B6B', '#4ECDC4']
            explode = (0.1, 0)  # 突出伪造部分
            
            # 获取字体属性
            font_props = self.get_font_properties()
            
            # 添加阴影效果和百分比标签
            wedges, texts, autotexts = self.chart_canvas.axes.pie(
                sizes, 
                explode=explode, 
                labels=labels, 
                colors=colors,
                autopct='%1.1f%%', 
                shadow=True, 
                startangle=90,
                textprops={'fontsize': 12, 'weight': 'bold', 'fontproperties': font_props}
            )
            
            # 设置自动文本的颜色和字体大小
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(10)
                autotext.set_fontproperties(font_props)
                
            # 设置文本字体属性
            for text in texts:
                text.set_fontproperties(font_props)
            
            self.chart_canvas.axes.axis('equal')  # 确保饼图是圆的
            
            # 添加图例
            self.chart_canvas.axes.legend(
                labels=[fake_count_text, real_count_text],
                loc='best',
                fontsize=10,
                frameon=True,
                facecolor='#f8f9fa',
                prop=font_props
            )
            
        elif chart_type == 'bar':
            # 绘制柱状图
            categories = [fake_label, real_label]
            values = [fake_count, real_count]
            colors = ['#FF6B6B', '#4ECDC4']
            
            # 获取字体属性
            font_props = self.get_font_properties()
            
            bars = self.chart_canvas.axes.bar(
                categories, 
                values, 
                color=colors,
                width=0.6,
                edgecolor='black',
                linewidth=0.5
            )
            
            # 设置y轴标签
            self.chart_canvas.axes.set_ylabel(y_axis_label, fontsize=12, fontweight='bold', fontproperties=font_props)
            
            # 设置x轴刻度标签
            self.chart_canvas.axes.set_xticks(range(len(categories)))
            self.chart_canvas.axes.set_xticklabels(categories, fontsize=11, fontweight='bold', fontproperties=font_props)
            
            # 设置y轴范围，增加一些空间以便显示标签
            y_max = max(values) * 1.2
            self.chart_canvas.axes.set_ylim(0, y_max)
            
            # 添加网格线以便于阅读
            self.chart_canvas.axes.yaxis.grid(True, linestyle='--', alpha=0.7)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                self.chart_canvas.axes.annotate(
                    f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12,
                    fontweight='bold',
                    color='black',
                    fontproperties=font_props
                )
        
        # 设置图表标题和样式
        self.chart_canvas.axes.set_title(
            main_title + '\n' + chart_subtitle, 
            fontsize=14, 
            fontweight='bold',
            pad=15,
            fontproperties=self.get_font_properties()
        )
        
        # 添加图表边框
        for spine in self.chart_canvas.axes.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color('#555555')
        
        # 更新统计信息标签
        if chinese_font_found:
            self.stats_info_label.setText(f"批量检测结果：共{len(self.batch_results)}张图像，"
                                      f"检测为伪造：{fake_count}张 ({fake_count/len(self.batch_results)*100:.1f}%)，"
                                      f"检测为真实：{real_count}张 ({real_count/len(self.batch_results)*100:.1f}%)")
        else:
            self.stats_info_label.setText(f"Results: Total {len(self.batch_results)} images, "
                                      f"Fake: {fake_count} ({fake_count/len(self.batch_results)*100:.1f}%), "
                                      f"Real: {real_count} ({real_count/len(self.batch_results)*100:.1f}%)")
        
        # 刷新画布
        self.chart_canvas.fig.tight_layout()  # 自动调整布局
        self.chart_canvas.draw()

    def get_font_properties(self):
        """获取适合当前环境的字体属性"""
        if hasattr(self.chart_canvas, 'chinese_font_name') and self.chart_canvas.chinese_font_name:
            # 返回中文字体属性
            return matplotlib.font_manager.FontProperties(family=self.chart_canvas.chinese_font_name)
        else:
            # 返回默认字体属性
            return matplotlib.font_manager.FontProperties(family='DejaVu Sans')

    def clear_results(self):
        """清除所有结果显示、进度条和历史记录"""
        # 清除当前路径和批处理图像路径
        self.current_image_path = None
        
        # 清除图像显示和结果
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("请加载图像...")
        self.result_label.setText("")
        self.progress_bar.setValue(0)
        
        # 清除批量处理结果
        self.batch_results = []
        
        # 清除图像库显示
        self.image_gallery.clear()
        self.loaded_images = []
        
        # 清除图表
        self.chart_canvas.clear_plot()
        self.stats_info_label.setText("尚无统计数据，请先执行批量检测")
        
        # 禁用相关按钮
        self.detect_btn.setEnabled(False)
        self.batch_detect_btn.setEnabled(False)
        self.encrypt_image_btn.setEnabled(False)
        
        # 记录日志
        self.log_message("已清除所有结果")
        self.statusBar.showMessage("结果已清除")
        
        # 更新UI
        self.update_chart('pie')  # 重置图表
    
    def update_statistics_chart(self, results_data=None):
        """
        更新统计图表
        如果提供了results_data，则基于它更新；否则基于batch_results
        """
        # 确保画布和轴被正确初始化
        if not hasattr(self.chart_canvas, 'axes'):
            print("错误: 统计图表组件未正确初始化")
            return
            
        # 清空当前图表
        self.chart_canvas.clear_plot()
        
        # 决定使用哪个数据源
        data_source = results_data if results_data is not None else self.batch_results
        
        # 如果没有数据，显示提示并返回
        if not data_source:
            self.chart_canvas.axes.text(0.5, 0.5, "无数据可供统计",
                               horizontalalignment='center', verticalalignment='center',
                               transform=self.chart_canvas.axes.transAxes, fontsize=12)
            self.chart_canvas.axes.set_title("预测分数分布统计")
            self.stats_info_label.setText("尚无统计数据，请先执行批量检测")
            self.chart_canvas.draw()
            return
            
        # 提取有效的预测结果
        valid_results = [res for res in data_source if 'error' not in res or res['error'] is None]
        predictions = [res.get('prediction', 0) for res in valid_results if 'prediction' in res]
        
        if not predictions:
            self.chart_canvas.axes.text(0.5, 0.5, "无有效预测数据可供统计",
                               horizontalalignment='center', verticalalignment='center',
                               transform=self.chart_canvas.axes.transAxes, fontsize=12)
            self.chart_canvas.axes.set_title("预测分数分布统计")
            self.stats_info_label.setText("无有效预测数据")
            self.chart_canvas.draw()
            return
            
        # 绘制直方图
        n, bins, patches = self.chart_canvas.axes.hist(
            predictions, 
            bins=20, 
            color='deepskyblue', 
            edgecolor='black', 
            alpha=0.75,
            linewidth=0.5,
            rwidth=0.85
        )
        
        # 为每个bin着不同的颜色，伪造区域使用红色
        threshold_index = int(0.5 * 20)  # 0.5是阈值，20是bins数量
        for i, patch in enumerate(patches):
            bin_center = (bins[i] + bins[i+1]) / 2
            if bin_center >= 0.5:
                patch.set_facecolor('#FF6B6B')  # 伪造概率区域（>=0.5）
            else:
                patch.set_facecolor('#4ECDC4')  # 真实概率区域（<0.5）
        
        # 设置字体属性
        font_props = self.get_font_properties() if hasattr(self, 'get_font_properties') else None
        
        # 设置图表标题和轴标签
        if hasattr(self.chart_canvas, 'has_chinese_font') and self.chart_canvas.has_chinese_font:
            title = "预测分数分布统计"
            xlabel = "预测分数 (0 ≈ 真实, 1 ≈ 伪造)"
            ylabel = "图片数量"
            threshold_label = "伪造阈值(0.5)"
            real_area_label = "真实区域"
            fake_area_label = "伪造区域"
        else:
            title = "Prediction Score Distribution"
            xlabel = "Prediction Score (0 ≈ Real, 1 ≈ Fake)"
            ylabel = "Image Count"
            threshold_label = "Threshold (0.5)"
            real_area_label = "Real Area"
            fake_area_label = "Fake Area"
        
        self.chart_canvas.axes.set_title(title, fontproperties=font_props, fontsize=14, fontweight='bold')
        self.chart_canvas.axes.set_xlabel(xlabel, fontproperties=font_props, fontsize=12)
        self.chart_canvas.axes.set_ylabel(ylabel, fontproperties=font_props, fontsize=12)
        
        # 添加网格
        self.chart_canvas.axes.grid(True, linestyle=':', alpha=0.6)
        
        # 添加阈值线
        self.chart_canvas.axes.axvline(
            x=0.5, 
            color='red', 
            linestyle='--', 
            linewidth=2,
            label=threshold_label
        )
        
        # 添加图例
        if font_props:
            self.chart_canvas.axes.legend(
                [threshold_label, real_area_label, fake_area_label], 
                loc='upper center',
                fontsize=10,
                frameon=True,
                facecolor='#f8f9fa',
                prop=font_props
            )
        else:
            self.chart_canvas.axes.legend(
                [threshold_label, real_area_label, fake_area_label], 
                loc='upper center',
                fontsize=10,
                frameon=True,
                facecolor='#f8f9fa'
            )
        
        # 统计信息
        fake_count = sum(1 for p in predictions if p >= 0.5)
        real_count = len(predictions) - fake_count
        
        # 更新统计信息标签
        if hasattr(self.chart_canvas, 'has_chinese_font') and self.chart_canvas.has_chinese_font:
            self.stats_info_label.setText(f"统计结果：共{len(predictions)}张图像，"
                                      f"检测为伪造：{fake_count}张 ({fake_count/len(predictions)*100:.1f}%)，"
                                      f"检测为真实：{real_count}张 ({real_count/len(predictions)*100:.1f}%)")
        else:
            self.stats_info_label.setText(f"Results: Total {len(predictions)} images, "
                                      f"Fake: {fake_count} ({fake_count/len(predictions)*100:.1f}%), "
                                      f"Real: {real_count} ({real_count/len(predictions)*100:.1f}%)")
        
        # 更新图表
        self.chart_canvas.fig.tight_layout()
        self.chart_canvas.draw()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = RECCEApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 