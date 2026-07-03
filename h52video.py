import h5py
import cv2
import numpy as np
import json
import os
import time

h5_file = '/mnt/pfs/3zpd5q/code/eval/DexVLA/c_floor_1/datasets/episode_006.h5'
output_dir = '/mnt/pfs/3zpd5q/code/RynnBrain/AKS/selected_frames/custom_ros/blip/episode_006'
output_avi = '/mnt/pfs/3zpd5q/code/RynnBrain/AKS/datasets/Test/episode_006/my_h5_video_temp.avi'
output_mp4 = '/mnt/pfs/3zpd5q/code/RynnBrain/AKS/datasets/Test/episode_006/my_h5_video_final.mp4'
output_json = '/mnt/pfs/3zpd5q/code/RynnBrain/AKS/datasets/Test/episode_006/frame_data.json'
fps = 30.0

def extract_video_images_and_odom():
    if not os.path.exists(h5_file):
        print("Error: h5 file not found:", h5_file)
        return

    os.makedirs(output_dir, exist_ok=True)
    frame_data_map = {}
    video_writer = None

    print("Opening h5 file:", h5_file)
    with h5py.File(h5_file, 'r') as f:
        positions = f['position'][:]
        rgbs = f['rgb'][:]

        total_frames = len(positions)
        print(f"Total frames to extract: {total_frames}")

        for i in range(total_frames):
            # 1. 获取 Odom 数据 [x, y, z, yaw]，保留 x, y, yaw
            x, y, z, yaw = positions[i]
            latest_odom = [float(x), float(y), float(yaw)]

            # 2. 存入 JSON 字典，用帧率模拟生成时间戳
            frame_data_map[i] = {
                "timestamp": float(i) / fps, 
                "odom": latest_odom
            }

            # 3. 处理图像
            img_array = rgbs[i]
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            if video_writer is None:
                height, width, _ = img_bgr.shape
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(output_avi, fourcc, fps, (width, height))
                print(f"Video resolution: {width}x{height}, initializing writer...")

            # 保存为单张图片
            img_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(img_path, img_bgr)
            
            # 写入视频流
            video_writer.write(img_bgr)

            if i % 50 == 0 and i > 0:
                print(f"Processed {i} frames...")

    if video_writer is not None:
        video_writer.release()

    # 4. 写入 JSON 文件
    with open(output_json, 'w') as jf:
        json.dump(frame_data_map, jf, indent=4)

    print(f"Extraction complete. Total frames saved: {total_frames}")
    print(f"Images saved to directory: {output_dir}/")
    print(f"Frame data JSON saved to: {output_json}")

    # 5. 转换视频格式
    print("Converting to MP4 format...")
    if os.path.exists(output_mp4):
        os.remove(output_mp4)

    ffmpeg_cmd = f"ffmpeg -i {output_avi} -c:v libx264 -crf 23 -c:a aac {output_mp4} -loglevel warning"
    ret_code = os.system(ffmpeg_cmd)

    if ret_code == 0:
        print(f"Conversion complete. Final video: {output_mp4}")
        os.remove(output_avi) 
    else:
        print("Warning: format conversion failed.")

if __name__ == '__main__':
    start_time = time.time()
    extract_video_images_and_odom()
    print(f"Time elapsed: {time.time() - start_time:.4f} seconds")