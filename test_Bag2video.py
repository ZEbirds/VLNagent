import rosbag
import cv2
import numpy as np
import json
import os
import time
import math
from tf.transformations import euler_from_quaternion

bag_file = '/mnt/pfs/3zpd5q/code/RynnBrain/AKS/datasets/Test/test-data.bag'            
image_topic = '/zj_humanoid/sensor/realsense_head/color/image_raw/compressed' 
odom_topic = '/Odometry' 

output_avi = 'my_ros_video_temp1.avi'     
output_mp4 = 'my_ros_video_final.mp4'    
output_json = 'frame_data.json'
fps = 30.0                            

def extract_video_and_odom():
    if not os.path.exists(bag_file):
        print("Error: bag file not found:", bag_file)
        return

    print("Opening bag file:", bag_file)
    bag = rosbag.Bag(bag_file, 'r')
    
    video_writer = None
    frame_idx = 0
    frame_data_map = {} 
    
    # 状态变量
    latest_odom = None 
    pending_frames = [] # 存放等待真实 odom 的幽灵帧的索引
    first_odom_received = False

    print("Reading image and odometry topics...")
    
    try:
        for topic, msg, t in bag.read_messages(topics=[image_topic, odom_topic]):
            
            if topic == odom_topic:
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                
                orientation_q = msg.pose.pose.orientation
                q = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                _, _, yaw = euler_from_quaternion(q)
                
                latest_odom = [x, y, yaw]
                
                # 如果这是收到的第一个真实 Odom，赶紧去救活前面的幽灵帧
                if not first_odom_received:
                    first_odom_received = True
                    for pending_idx in pending_frames:
                        frame_data_map[pending_idx]["odom"] = latest_odom
                    # 清空等待列表
                    pending_frames.clear()
                continue

            if topic == image_topic:
                try:
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if cv_image is None:
                        continue
                except Exception as e:
                    print("Image decode error:", e)
                    continue
                
                if video_writer is None:
                    height, width, _ = cv_image.shape
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(output_avi, fourcc, fps, (width, height))
                    print(f"Video resolution: {width}x{height}, initializing writer...")

                video_writer.write(cv_image)
                
                # 先把时间戳存下。如果是幽灵帧，Odom 设为 None 并加入等待列表
                frame_data_map[frame_idx] = {
                    "timestamp": t.to_sec(),
                    "odom": latest_odom 
                }
                
                if not first_odom_received:
                    pending_frames.append(frame_idx)
                
                if frame_idx % 500 == 0 and frame_idx > 0:
                    print(f"Processed {frame_idx} frames...")
                    
                frame_idx += 1

    finally:
        if video_writer is not None:
            video_writer.release()
        bag.close()

    with open(output_json, 'w') as f:
        json.dump(frame_data_map, f, indent=4)

    print(f"Extraction complete. Total frames: {frame_idx}")
    print(f"Frame data saved to: {output_json}")

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
    extract_video_and_odom()
    print(f"Time elapsed: {time.time() - start_time:.4f} seconds")