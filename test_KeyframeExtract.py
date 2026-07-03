import math
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2

# ==========================================
# 方案一：基于里程计的空间抽帧（纯数学，0 延迟）
# ==========================================
class OdometryKeyframeFilter:
    def __init__(self, dist_threshold=0.5, angle_threshold_deg=15.0):
        self.dist_threshold = dist_threshold
        self.angle_threshold_rad = math.radians(angle_threshold_deg)
        
        # 记录上一个关键帧的位姿
        self.last_keyframe_pose = None 

    def update(self, current_x, current_y, current_yaw):
        """
        传入当前的 x, y 坐标和 yaw 朝向角 (弧度)
        返回: bool (是否应该作为新的关键帧)
        """
        if self.last_keyframe_pose is None:
            self.last_keyframe_pose = (current_x, current_y, current_yaw)
            return True # 第一帧永远是关键帧
        
        last_x, last_y, last_yaw = self.last_keyframe_pose
        
        # 计算平移距离
        dist = math.hypot(current_x - last_x, current_y - last_y)
        # 计算旋转差值 (处理跨越 180 度的跳变)
        angle_diff = abs(math.atan2(math.sin(current_yaw - last_yaw), 
                                    math.cos(current_yaw - last_yaw)))
        
        if dist >= self.dist_threshold or angle_diff >= self.angle_threshold_rad:
            self.last_keyframe_pose = (current_x, current_y, current_yaw)
            return True
            
        return False


# ==========================================
# 方案二：基于图像差异的 CV 抽帧（毫秒级，低 CPU 占用）
# ==========================================
class ORBKeyframeFilter:
    def __init__(self, match_ratio_threshold=0.35, max_features=500):
        """
        :param match_ratio_threshold: 匹配率阈值 (0.0~1.0)。
               比如设为 0.35，意味着如果当前帧能和上一个关键帧匹配上的特征点
               不足原来的 35%，说明视角发生了大改变，存为新关键帧。
        :param max_features: 每次提取的最大特征点数量 (500 是速度与精度的完美平衡)
        """
        self.match_ratio_threshold = match_ratio_threshold
        self.orb = cv2.ORB_create(nfeatures=max_features)
        
        # 使用汉明距离进行特征匹配，crossCheck=True 保证双向匹配，极其精准
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 记录上一个关键帧的特征描述子
        self.last_keyframe_des = None

    def update(self, current_bgr_image):
        """
        传入当前的 BGR 图像 (OpenCV 格式)
        返回: bool (是否应该作为新的关键帧)
        """
        # 转为灰度图提取特征
        gray = cv2.cvtColor(current_bgr_image, cv2.COLOR_BGR2GRAY)
        
        # 寻找关键点 (keypoints) 和 描述子 (descriptors)
        kp, des = self.orb.detectAndCompute(gray, None)
        
        # 极端边缘情况防崩溃：如果对着一面纯白墙，提取不到特征点
        if des is None or len(des) < 10:
            return False
            
        # 第一帧永远是关键帧
        if self.last_keyframe_des is None:
            self.last_keyframe_des = des
            return True
            
        # 让当前帧的特征点和上一个关键帧的特征点进行硬核匹配
        matches = self.bf.match(self.last_keyframe_des, des)
        
        # 计算匹配保留率：(成功匹配的特征点数量 / 上一帧总特征点数量)
        match_ratio = len(matches) / max(len(self.last_keyframe_des), 1)
        
        # 如果保留下来的特征点太少，说明场景已经切了！
        if match_ratio < self.match_ratio_threshold:
            # 更新“上一个关键帧”的特征为当前帧的特征
            self.last_keyframe_des = des
            return True
            
        return False


# ==========================================
# 方案三：轻量级 CLIP 语义波峰抽帧（需 GPU，带语义理解）
# ==========================================
# class SemanticCLIPKeyframeFilter:
#     # 【修改点 1】加上了 peak_delta 参数，默认值为 0.02
#     def __init__(self, target_states, threshold=0.25, peak_delta=0.02, device="cuda"):
#         self.device = device
#         print(f"🚀 Loading lightweight CLIP model on {self.device}...")
#         self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
#         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
#         inputs_text = self.processor(text=target_states, return_tensors="pt", padding=True).to(self.device)
#         with torch.no_grad():
#             self.text_features = self.model.get_text_features(**inputs_text)
#             self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            
#         self.states = target_states
#         self.threshold = threshold
#         self.peak_delta = peak_delta  # 保存到类属性
#         self.last_max_sim = 0.0

#     def update(self, current_pil_image):
#         inputs_image = self.processor(images=current_pil_image, return_tensors="pt").to(self.device)
#         with torch.no_grad():
#             image_features = self.model.get_image_features(**inputs_image)
#             image_features /= image_features.norm(dim=-1, keepdim=True)
            
#         similarities = (image_features @ self.text_features.T).squeeze(0)
#         max_sim_value, max_sim_idx = similarities.max(dim=0)
#         max_sim_value = max_sim_value.item()
#         best_state_text = self.states[max_sim_idx.item()]
        
#         is_keyframe = False
#         # 【修改点 2】使用你传进来的 peak_delta
#         if max_sim_value > self.threshold and max_sim_value > (self.last_max_sim + self.peak_delta):
#             is_keyframe = True
            
#         self.last_max_sim = max_sim_value
#         return is_keyframe, best_state_text

class CLIPClusterKeyframeFilter:
    def __init__(self, similarity_threshold=0.85, max_centers=100, device="cuda"):
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.max_centers = max_centers
        
        print(f"Loading CLIP Online Clustering on {self.device}...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 使用 Tensor 存储所有中心，便于利用 GPU 进行极致的矩阵并行计算
        self.cluster_centers = None 

    def extract_feature(self, pil_image):
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self.model.get_image_features(**inputs)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat  # 保持 2D 形状: [1, 512]

    def update(self, pil_image):
        feat = self.extract_feature(pil_image)

        if self.cluster_centers is None:
            self.cluster_centers = feat
            return True  # 记录第一个场景

        # 极致加速：用矩阵乘法一次性算出当前帧与所有历史中心的相似度！
        # feat 形状: [1, 512], cluster_centers.T 形状: [512, N]
        # sims 形状: [1, N]
        sims = torch.matmul(feat, self.cluster_centers.T)
        
        # 找到历史中最相似的那个场景的得分
        max_sim = sims.max().item()

        # 如果最相似的历史场景都不够像 (小于阈值)，说明到了新地方！
        if max_sim < self.similarity_threshold:
            # 限制总数量，防止 OOM (Out Of Memory)
            if self.cluster_centers.size(0) < self.max_centers:
                # 把新场景的特征拼接进记忆库
                self.cluster_centers = torch.cat([self.cluster_centers, feat], dim=0)
            return True  
        else:
            return False


class HybridORBOdomKeyframeFilter:
    def __init__(self, match_ratio_threshold=0.25, dist_threshold=2.0, angle_threshold_deg=45.0, max_features=500):
        # 初始化 ORB
        self.match_ratio_threshold = match_ratio_threshold
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last_keyframe_des = None

        # 初始化 里程计
        self.dist_threshold = dist_threshold
        self.angle_threshold_rad = math.radians(angle_threshold_deg)
        self.last_keyframe_pose = None

    def update(self, current_bgr_image, current_x, current_y, current_yaw):
        # 1. 提取当前帧图像特征
        gray = cv2.cvtColor(current_bgr_image, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        # 2. 首帧初始化
        if self.last_keyframe_pose is None:
            self.last_keyframe_pose = (current_x, current_y, current_yaw)
            self.last_keyframe_des = des if des is not None else []
            return True

        last_x, last_y, last_yaw = self.last_keyframe_pose
        dist = math.hypot(current_x - last_x, current_y - last_y)
        rad_diff = abs(math.atan2(math.sin(current_yaw - last_yaw), math.cos(current_yaw - last_yaw)))
        angle_diff = math.degrees(rad_diff)
        print(angle_diff)

        is_keyframe = False

        # 条件 A: 里程计强制触发 (超过 2m 或 45度)
        if dist >= self.dist_threshold or angle_diff >= self.angle_threshold_rad:
            is_keyframe = True

        # 条件 B: ORB 视觉差异触发
        if not is_keyframe:
            if des is not None and len(des) >= 10 and self.last_keyframe_des is not None and len(self.last_keyframe_des) > 0:
                matches = self.bf.match(self.last_keyframe_des, des)
                match_ratio = len(matches) / max(len(self.last_keyframe_des), 1)
                if match_ratio < self.match_ratio_threshold:
                    is_keyframe = True

        # 如果被判定为关键帧，同时更新 位姿记忆 和 图像特征记忆
        if is_keyframe:
            self.last_keyframe_pose = (current_x, current_y, current_yaw)
            self.last_keyframe_des = des if des is not None else []
            return True

        return False


if __name__ == "__main__":
    print("=== 开始测试 keyframe_filters 模块 ===")
    
    # 1. 测试里程计滤波器
    odom_filter = OdometryKeyframeFilter(dist_threshold=0.5)
    print("Odom 测试1 (首帧):", odom_filter.update(0.0, 0.0, 0.0))  # 应该为 True
    print("Odom 测试2 (微动):", odom_filter.update(0.1, 0.0, 0.0))  # 应该为 False
    print("Odom 测试3 (越界):", odom_filter.update(0.6, 0.0, 0.0))  # 应该为 True
    
    # 2. 测试 CV 滤波器
    cv_filter = DiffKeyframeFilter()
    # cv_filter = ORBKeyframeFilter(match_ratio_threshold=0.35)
    dummy_img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_img2 = np.ones((480, 640, 3), dtype=np.uint8) * 255
    print("CV 测试1 (首帧):", cv_filter.update(dummy_img1)) # 应该为 True
    print("CV 测试2 (相同):", cv_filter.update(dummy_img1)) # 应该为 False
    print("CV 测试3 (剧变):", cv_filter.update(dummy_img2)) # 应该为 True
    
    print("\n✅ 基础模块加载成功！可以安全 import 到主程序中了。")
    # CLIP 模块由于需要下载模型，默认在此不自动测试。