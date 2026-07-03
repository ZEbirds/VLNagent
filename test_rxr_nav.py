from rynn_planning import RynnBrain_Nav, RynnBrain_Planning
import os
import json
from test_KeyframeExtract import HybridORBOdomKeyframeFilter
import cv2

if __name__ == "__main__":
    model_path = "models/Qwen3-VL-32B-Instruct"
    # rynn_model = RynnBrain_Nav(model_path)
    rynn_model = RynnBrain_Planning(model_path=None)
    rynn_model.reset_history()
    
    #/mnt/pfs/3zpd5q/code/eval/DexVLA/decision/datasets/my_h5_video_final.mp4
    # image_folder = "cookbooks/assets/elevator"
    video_path = '/mnt/pfs/3zpd5q/code/RynnBrain/AKS/datasets/Test/my_ros_video_final.mp4'
    json_path = '/mnt/pfs/3zpd5q/code/RynnBrain/AKS/datasets/Test/frame_data.json'
    temp_img_dir = "visual_result/frame_data1/extracted_keyframes"
    os.makedirs(temp_img_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        frame_data_map = json.load(f)

    hybrid_filter = HybridORBOdomKeyframeFilter(
        match_ratio_threshold=0.25, 
        dist_threshold=5.0, 
        angle_threshold_deg=45.0
    )

    cap = cv2.VideoCapture(video_path)
    
    frame_idx = 0
    sample_interval = 5 

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue

        str_idx = str(frame_idx)
        if str_idx in frame_data_map:
            x, y, yaw = frame_data_map[str_idx]["odom"]
            is_keyframe = hybrid_filter.update(frame, x, y, yaw)
            if is_keyframe:
                filename = f"frame_{frame_idx:04d}.jpg"
                orig_path = os.path.join(temp_img_dir, filename)
                cv2.imwrite(orig_path, frame)

                rynn_8b = rynn_model.inference_split_with_history(
                    ['Pass by corridor','Go toward toilet','Turn left At the wall corner of the hallway intersection', 'Enter the gate', 'Pass by the office'],
                    # ['go forward through carteen', 'pass by vending machines', 'enter the lobby', 'turn right when see AD billboard', 'go forward through the turnstiles' , 'walk into the elevator' ],
                    # instruction_steps = [
                    #     "Leave the starting lounge area near the dark shower cabin", 
                    #     "Go straight down the corridor passing the yellow '3 Home Zone' sign on the left", 
                    #     "Pass between the white exhibition desks featuring robot arms", 
                    #     "Approach the illuminated 'HONOR' shelf and step onto the circular floor pattern", 
                    #     "Turn left to enter the open wooden doorway situated beside the display shelf when see the door"
                    # ],
                    current_image_path = orig_path,
                    save_dir = f"visual_result/frame_data1/{filename}",
                )
                print(f"########################### Prediction for {filename}:\n{rynn_8b}\n")

        frame_idx += 1

    cap.release()