from rynn_planning import RynnBrain_Planning
import os
import time
from typing import List, Dict, Any, Optional, Union
import shutil

def clear_directory(path: str):
    if not os.path.exists(path):
        return  # 不存在直接返回

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)      # 删除文件或软链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除子目录
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
class VLMEnsembleAgent:
    def __init__(
        self,
        thinking_vlm: RynnBrain_Planning,
        fast_vlm: RynnBrain_Planning,
        rynnbrain_vlm: RynnBrain_Planning,
    ):
        self.thinking_vlm = thinking_vlm
        self.fast_vlm = fast_vlm
        self.pixel_vlm = rynnbrain_vlm

        self.target_box_img = None
        self.target_pixel_img = None 
        self.target_desc = None

        

        self.search_num = 10
        self.exec_step = 0

        self.target_img_path = 'visual_for_agent/target_box'
        os.makedirs(self.target_img_path, exist_ok=True)
        self.exec_pixel_img_path = 'visual_for_agent/exec_pixel'
        os.makedirs(self.exec_pixel_img_path, exist_ok=True)

    def reset_object(self):
        clear_directory(self.target_img_path)
        clear_directory(self.exec_pixel_img_path)
        self.target_box_img = None
        self.target_pixel_img = None 
        self.target_desc = None     
        self.exec_step = 0
    def reset_split(self):
        self.fast_vlm.reset_history()

    def check_object(self,target_desc: str, image: str , save_dir: str, use_thinking: bool = False, temperature: float = 0.7):
        instruction = f'Do you clearly see {target_desc}? Only answer yes or no'
        if use_thinking:
            output = self.thinking_vlm.inference(
                instruction,
                [image],
                save_dir = save_dir, 
                task="find",
                plot=True,
                do_sample=False,
                temperature=temperature
            )
            # output = self.extract_answer(output)
        else:
            output = self.fast_vlm.inference(
                instruction,
                [image],
                save_dir = save_dir, 
                task="find",
                plot=True,
                do_sample=False,
                temperature=temperature
            )
        if 'YES' in output.upper():
            return True
        else:
            return False

    def move2target(self, target_desc: str, image: str):
        if self.target_desc is None:
            for i in range(self.search_num):
                target_found = self.check_object(target_desc, image,save_dir = self.target_pixel_img, use_thinking=True, temperature=0.3)
                if target_found:
                    self.target_box_img = self.target_img_path + '/' + 'target_box.png'
                    box = self.rynnbrain_vlm.inference(
                        instruction = f'Where is the {target_desc}',
                        image = [orig_path],
                        save_dir = self.target_box_img , 
                        # task="affordance_location",
                        task = "object_location",
                        plot=True,
                        do_sample=False
                    )
                    break
        
        else:
            instruction = (
            f"Identify the target: {self.target_desc}. "
            f"The target is labeled with a bounding box in frame 1. "
            f"Find the same target in frame 2.")

            pixel = self.rynnbrain_vlm.inference(
                instruction = instruction,
                image = [self.target_box_img, image],
                save_dir = self.exec_pixel_img_path + '/' + f'exec_{self.exec_step}.png', 
                task="affordance_location",
                plot=True,
                do_sample=False
            )
        
        self.exec_step+=1


    def judge_current_instruction(self, instruction_steps, image):
        self.fast_vlm.inference_split_with_history(
            instruction_steps,
            image,
            save_dir = f"visual_result/rxr2/{filename}",
            )
    def extract_target_from_instruction(instruction):
        target = None
        return target
if __name__ == "__main__":
    model_path = "models/RynnBrain-30B-A3B"
    model_path = "models/Qwen3-VL-32B-Thinking"
    rynn_model = RynnBrain_Planning(model_path)
    image_path = "cookbooks/assets/runway/toright.png"


    # instruction = "Move forward alone the middle lane."
    # # instruction = "Go to the rightside on the red lanes."
    # instruction = "Are you on the right lane?"
    # rynn_8b = rynn_model.inference(
    #     instruction,
    #     [image_path],
    #     save_dir = 'visual_result/move_traj.png', 
    #     task="ocr",
    #     plot=True,
    #     do_sample=False
    # )
    # instruction = "Stand on the rightmost red track."
    
    # rynn_8b = rynn_model.inference(
    #     instruction,
    #     [image_path],
    #     save_dir = 'visual_result/to_right_afford.png', 
    #     # task="area_location",
    #     task="affordance_location",
    #     plot=True,
    #     do_sample=False
    # )
    # instruction = "Find one location at the middle of the rightmost lane of these red tracks."
    # rynn_8b = rynn_model.inference(
    #     instruction,
    #     [image_path],
    #     save_dir = 'visual_result/to_right_area.png', 
    #     task="area_location",
    #     plot=True,
    #     do_sample=False
    # )
    
    image_folder = "cookbooks/assets/Object"
    # instruction = "Which lane are you standing on? Left, Middle or Right?"
    
    for filename in sorted(os.listdir(image_folder)):
        # 只处理常见图片格式
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            orig_path = os.path.join(image_folder, filename)
            print(f"Processing {filename} ...")
            
            # 调用模型
            
            # instruction = "Does the human in first image appear in the second image?"
            instruction = "Does you see the black bag on the white wooden cabinet? Only output yes or no"
            t1 = time.time()
            rynn_8b = rynn_model.inference(
                instruction,
                # ['cookbooks/assets/follow/rgb_raw_000009.png' , orig_path],
                [orig_path],
                save_dir = f"visual_result/Object/{filename}", 
                task="ocr",
                plot=True,
                do_sample=False
            )
            t2 = time.time()
            print(f"time: {t2-t1}")
            t1 = time.time()
            if 'yes' in rynn_8b:
                instruction = "Go to the black bag on the white wooden cabinet."
                rynn_8b = rynn_model.inference(
                    instruction,
                    [orig_path],
                    save_dir = f"visual_result/Object/{filename}", 
                    task="affordance_location",
                    plot=True,
                    do_sample=False
                )

            t2 = time.time()
            # print(f"time: {t2-t1}")
            print(f"Prediction for {filename}:\n{rynn_8b}\n")
