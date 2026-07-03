import cv2
import os
import json
from PIL import Image
from test_KeyframeExtract import ORBKeyframeFilter, CLIPClusterKeyframeFilter, OdometryKeyframeFilter, HybridORBOdomKeyframeFilter

video_path = '/mnt/pfs/3zpd5q/code/RynnBrain/AKS/datasets/Test/my_ros_video_final.mp4'
json_path = '/mnt/pfs/3zpd5q/code/RynnBrain/AKS/datasets/Test/frame_data.json' 

output_dir_cv = './filter_results/cv_orb/'
output_dir_clip = './filter_results/clip_cluster/'
output_dir_odom = './filter_results/odom/'
output_dir_hybrid = './filter_results/hybrid_orb_odom/'

sample_interval = 5 

def run_test():
    if not os.path.exists(video_path):
        print(f"找不到视频文件: {video_path}")
        return
        
    if not os.path.exists(json_path):
        print(f"找不到里程计数据文件: {json_path}")
        return

    os.makedirs(output_dir_cv, exist_ok=True)
    os.makedirs(output_dir_clip, exist_ok=True)
    os.makedirs(output_dir_odom, exist_ok=True)
    os.makedirs(output_dir_hybrid, exist_ok=True)

    print("加载里程计数据...")
    with open(json_path, 'r') as f:
        frame_data_map = json.load(f)

    print("初始化 里程计过滤器 (移动 5m 或 旋转 45度)...")
    odom_filter = OdometryKeyframeFilter(dist_threshold=5.0, angle_threshold_deg=45.0)

    print("初始化 ORB 图像特征过滤器 (阈值 0.25)...")
    cv_filter = ORBKeyframeFilter(match_ratio_threshold=0.25)
    
    print("初始化 混合过滤器 (ORB 0.25 兜底: 5m / 45度)...")
    hybrid_filter = HybridORBOdomKeyframeFilter(
        match_ratio_threshold=0.25, 
        dist_threshold=5.0, 
        angle_threshold_deg=45.0
    )

    print("初始化 CLIP 聚类过滤器 (相似度 0.85)...")
    clip_filter = CLIPClusterKeyframeFilter(
        similarity_threshold=0.85,   
        device="cuda"
    )

    print(f"开始读取视频: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    frame_idx = 0
    cv_saved = 0
    clip_saved = 0
    odom_saved = 0
    hybrid_saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue
            
        if frame_idx % 100 == 0:
            print(f"正在处理第 {frame_idx} 帧...")

        # 尝试获取当前帧的里程计数据，如果找不到就跳过依赖里程计的过滤器
        str_idx = str(frame_idx)
        has_odom = str_idx in frame_data_map
        if has_odom:
            x, y, yaw = frame_data_map[str_idx]["odom"]

        # --------------------------------------------------
        # 1. 测试 里程计过滤器
        # --------------------------------------------------
        if has_odom and odom_filter.update(x, y, yaw):
            img_name = os.path.join(output_dir_odom, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(img_name, frame)
            odom_saved += 1

        # --------------------------------------------------
        # 2. 测试 CV ORB 过滤器
        # --------------------------------------------------
        if cv_filter.update(frame):
            img_name = os.path.join(output_dir_cv, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(img_name, frame)
            cv_saved += 1
            
        # --------------------------------------------------
        # 3. 测试 混合(ORB+Odom) 过滤器
        # --------------------------------------------------
        if has_odom and hybrid_filter.update(frame, x, y, yaw):
            img_name = os.path.join(output_dir_hybrid, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(img_name, frame)
            hybrid_saved += 1

        # --------------------------------------------------
        # 4. 测试 CLIP 聚类过滤器
        # --------------------------------------------------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        if clip_filter.update(pil_img):
            img_name = os.path.join(output_dir_clip, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(img_name, frame)
            clip_saved += 1

        frame_idx += 1

    cap.release()
    print("\n视频处理完成！")
    print(f"里程计过滤器 提取了: {odom_saved} 张关键帧")
    print(f"ORB 过滤器 提取了: {cv_saved} 张关键帧")
    print(f"混合(ORB+Odom)过滤器 提取了: {hybrid_saved} 张关键帧")
    print(f"CLIP 聚类过滤器 提取了: {clip_saved} 张关键帧")

if __name__ == '__main__':
    run_test()