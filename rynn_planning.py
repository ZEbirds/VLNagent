import os
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Image as IPyImage
import io
import re
import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
from transformers import AutoModelForImageTextToText, AutoProcessor
from pathlib import Path
from visual_rynn import PointVisual, BoxVisual
import json
import re
import base64
from io import BytesIO
def encode_image(image):
    """
    支持输入：
    - str: 图片路径
    - np.ndarray: 图像数组
    - PIL.Image: 图像对象
    """
    
    # ✅ 如果是路径
    if isinstance(image, str):
        if not os.path.exists(image):
            raise ValueError(f"Image path does not exist: {image}")
        image = Image.open(image).convert("RGB")
    
    # ✅ 如果是 numpy
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # ✅ 如果是 PIL.Image
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    # 转为 base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    
    encoded_image = base64.b64encode(img_bytes).decode("utf-8")
    return encoded_image

def extract_step(text):
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    
    try:
        data = json.loads(match.group())
        return data.get("step")
    except:
        return None
def extract_answer(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[-1].strip()
    return text.strip()

class RynnBrain_Planning:

    def __init__(self, model_path, display=False):
        self.point_visual = PointVisual(display)
        self.box_visual = BoxVisual(display)
        self.model = None
        if model_path:
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype="auto", device_map="auto")
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            import os
            os.environ["NO_PROXY"] = "localhost,127.0.0.1"
            from openai import OpenAI
            self.client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

        self.action_list = []
        self.rgb_list = []
        self.locked_step = 0

    def reset_history(self):
        self.action_list = []
        self.rgb_list = []
        self.locked_step = 0

    def generate(self, messages, do_sample=True, temperature=0.7, max_new_tokens= 8192):
        if self.model:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Inference: Generation of the output
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        else:
            response = self.client.chat.completions.create(
                model="Qwen3-VL-32B-Instruct",
                messages=messages,
                max_tokens=30720,
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                # extra_body={
                #     "top_k": 20,
                # }, 
            )

            output_text = response.choices[0].message.content
        return output_text
        
    def inference(self, instruction: str, image: Union[list, str], save_dir: str , task="general", 
                 plot=False,do_sample=True, temperature=0.7):
        
        max_new_tokens = 128
        if isinstance(image, str):
            image = [image]

        assert task in ["general", "object_location", "ocr", "area_location", "affordance_location", "trajectory", "find"], \
            f"Invalid task type: {task}. Supported tasks are 'ocr', 'area_location', 'affordance_location', 'trajectory', 'object_location'"

        if task == "area_location":
            print("area_location task detected. Adding area_location prompt.")
            format_prompt = "Express the coordinates as a tuple sequence in the format <area> (x1, y1), (x2, y2), ... </area> with all coordinate values normalized to the standardized pixel coordinate system spanning 0 to 1000."
            # format_prompt = "Express the single point coordinates in the format <area> (x, y) </area> with all coordinate values normalized to the standardized pixel coordinate system spanning 0 to 1000."
            text = f"{instruction}\n{format_prompt}"
            image_paths = [image[0]]
        elif task == "find":
            print("Find task detected. Adding find prompt.")
            format_prompt = f"""If you clearly see the target, answer 'yes' and identify the target point and provide its normalized pixel coordinates. The x and y values must be between 0 and 1000. Output format: yes (x, y).
                                Else answer no.
                            """
            text = f"{instruction}\n{format_prompt}"
            image_paths = image
            max_new_tokens = 2048
        elif task == "object_location":
            print("object_location task detected. Adding object_location prompt.")
            # format_prompt = "Output the bounding box in the format <object> <frame n>: ...; (x1,y1), (x2,y2) </object>, where x,y are in the range of (0,1000), n is the chosen frame index."
            format_prompt = "Output the bounding box in the format <object> <frame n>: ...; (x1,y1), (x2,y2) </object>, where x,y are in the range of (0,1000), n is the chosen frame index start from 0."
            text = f"{instruction}\n{format_prompt}"
            image_paths = image
        elif task == "affordance_location":
            print("affordance_location task detected. Adding affordance_location prompt.")
            format_prompt = "Identify one affordance point and provide its normalized pixel coordinates. The x and y values must be between 0 and 1000. Output format: <affordance> (x, y) </affordance>."
            text = f"{instruction}\n{format_prompt}"
            image_paths = [image[0]]
        elif task == "trajectory":
            print("trajectory task detected. Adding trajectory prompt.")
            if len(image)>1:
                format_prompt = "First predict the frame containing the trajectory start point, then output up to 10 key trajectory points as a list of tuples in the format: <trajectory> <frame n>: ...; (x1, y1), (x2, y2), .... </trajectory> All coordinates must be normalized between 0 and 1000."
                image_paths = image
            else:
                format_prompt = "Predict a trajectory comprising up to 10 key points. Return coordinates in the format <trajectory> (x1, y1), (x2, y2), ... </trajectory> with all values normalized to the [0, 1000] range."
                image_paths = [image[0]]
            text = f"{instruction}\n{format_prompt}"
            

        elif task == "ocr" or task == "general":
            print(f"{task} task detected. Adding ocr prompt.")
            text = f"{instruction}"
            image_paths = image
            max_new_tokens = 2048

        
        # print(f"\n{'='*20} INPUT {'='*20}\n{text}\n{'='*47}\n")
        content = []
        if self.model:
        
            
            for p in image_paths:
                content.append({"type": "image", "image": str(p)})
            content.append({"type": "text", "text": text})
        else:

            for p in image_paths:
                img_b64=encode_image(str(p))
                content.append({"type": "image_url", 
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                                })
            content.append({"type": "text", "text": text})
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        output_text = self.generate(messages=messages,do_sample=do_sample,temperature=temperature,max_new_tokens=max_new_tokens)
        print(output_text)



        if plot:
            try:
                if task == "area_location":
                    points = self.point_visual.parse_points(output_text)
                    image = Image.open(image_paths[0]).convert("RGB")
                    image = image.resize((int(1080*image.size[0]/image.size[1]),1080))
                    w, h = image.size
                    points_raw = self.point_visual.convert_points_to_raw(points, w, h)
                    # print(frame_id, points, w, h, points_raw)
                    self.point_visual.draw_points_on_image(image, points_raw, save_dir=save_dir ,color="red", point_radius=6, width=4, show_width=600, text=instruction)
                elif task == "find":
                    output_text = extract_answer(output_text)
                    if 'YES' in output_text.upper():
                        points = self.point_visual.parse_points(output_text)
                        image = Image.open(image_paths[0]).convert("RGB")
                        image = image.resize((int(1080*image.size[0]/image.size[1]),1080))
                        w, h = image.size
                        points_raw = self.point_visual.convert_points_to_raw(points, w, h)
                        # print(frame_id, points, w, h, points_raw)
                        self.point_visual.draw_points_on_image(image, points_raw, save_dir=save_dir ,color="red", point_radius=6, width=4, show_width=600, text=instruction)
                        return 'YES'
                    else:
                        return 'NO'

                elif task == "object_location":
                    frame_id, bbox = self.box_visual.parse_frame_id_and_bbox(output_text)
                    if frame_id is None or frame_id >= len(image_paths):
                        return 'NO'
                    img_path = image_paths[frame_id]
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((int(1080*img.size[0]/img.size[1]),1080))
                    w, h = img.size
                    bbox_raw = self.box_visual.convert_bbox_to_raw(bbox, w, h)
                    self.box_visual.draw_bbox_on_image(img, bbox_raw, save_dir=save_dir,color="red", width=4, show_width=600, text=None)
                elif task == "affordance_location":
                    points = self.point_visual.parse_points(output_text)
                    image = Image.open(image_paths[0]).convert("RGB")
                    image = image.resize((int(1080*image.size[0]/image.size[1]),1080))
                    w, h = image.size
                    points_raw = self.point_visual.convert_points_to_raw(points, w, h)
                    # print(frame_id, points, w, h, points_raw)
                    self.point_visual.draw_points_on_image(image, points_raw, save_dir=save_dir , color="red", point_radius=6, width=4, show_width=600, text=instruction)
                elif task == "trajectory":
                    if len(image_paths)>1:
                        frame_id, points = self.point_visual.parse_frame_id_and_points(output_text)
                        img_path = image_paths[frame_id]
                        img = Image.open(img_path).convert("RGB")
                        img = img.resize((int(1080*img.size[0]/img.size[1]),1080))
                        w, h = img.size
                        points_raw = self.point_visual.convert_points_to_raw(points, w, h)
                        # print(frame_id, points, w, h, points_raw)
                    else:
                        points = self.point_visual.parse_points(output_text)
                        img = Image.open(image_paths[0]).convert("RGB")
                        img = img.resize((int(1080*img.size[0]/img.size[1]),1080))
                        w, h = img.size
                        points_raw = self.point_visual.convert_points_to_raw(points, w, h)
                        # print(frame_id, points, w, h, points_raw)
                    self.point_visual.draw_trajectory_on_image(img, points_raw, save_dir=save_dir , color="red", point_radius=6, width=4, show_width=600, text=instruction)
                elif task == "ocr" or task == "general":
                    # print(f"{task} result {output_text}")
                    return output_text
            except Exception as e:
                print(f"No result when plot: Error {e}")
        return output_text
    
    # 判断当前处于
    # def inference_split_with_history(self, instruction_steps: Union[list, str], current_image_path: str,  save_dir: str, do_sample=True, temperature=0.7, max_new_tokens= 8192):
    #     messages = []
    #     user_content = [] 
    #     recent_history_size = 5
    #     history_frames = []
    #     if len(self.rgb_list) > 0:
    #         start_idx = max(0, len(self.rgb_list) - recent_history_size)
    #         history_frames = list(range(start_idx, len(self.rgb_list)))

    #     if history_frames:
    #         user_content.append({"type": "text", "text":  " These are your historical observations: "})
    #         for index in history_frames:
    #             if self.model:
    #                 user_content.append({"type": "image", "image": self.rgb_list[index]})
    #             else:
    #                 img_b64 = encode_image(self.rgb_list[index])
    #                 user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

    #     user_content.append({"type": "text", "text": " you can see "})
    #     if self.model:
    #         user_content.append({"type": "image", "image": current_image_path})
    #     else:
    #         img_b64=encode_image(current_image_path)
    #         user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

    #     messages.append({
    #         "role": "user",
    #         "content": user_content
    #     })

    #     # instruction = f"""
    #     # Steps:

    #     # {chr(10).join([f"[{i}] {s}" for i, s in enumerate(instruction_steps)])}

    #     # You are an autonomous navigation assistant tagging sparse keyframes.

    #     # CRITICAL RULES - YOU MUST OBEY:
    #     # 1. NO FLICKERING (Solve 1-1-2-1-1 error): Navigation is continuous. Do NOT randomly jump to a different step for a single frame just because the angle changed. You must have UNDENIABLE visual evidence to choose a new step.
    #     # 2. SPECIFIC LANDMARKS > GENERAL AREAS: If the robot is inside a general area (e.g., 'enter the lobby' or 'canteen') and you clearly see a specific local target (e.g., 'reception desk', 'vending machines', 'turnstiles'), you MUST select the specific target's step. Do NOT select the general area if the specific landmark is visible!

    #     # Given the visual input, determine:
    #     # 1. Which step is currently being executed?
    #     # 2. Whether it is: in progress / completed?

    #     # First, analyze the visual evidence step-by-step inside <think> tags. Ask yourself: "What specific landmarks are visible? Am I just in a general area, or am I at a specific target?"
        
    #     # Return exactly this JSON format after thinking:
    #     # {{
    #     # "step": int,
    #     # "status": "in_progress" | "completed"
    #     # }}
    #     # """

    #     # 获取上一帧的记忆索引
    #     current_memory = getattr(self, 'last_step_idx', 0)

    #     instruction = f"""
    #     Navigation Sequence (TOPOLOGICAL ORDER):
    #     {chr(10).join([f"[{i}] {s}" for i, s in enumerate(instruction_steps)])}

    #     PREVIOUS STATE:
    #     Last frame, the robot was at Step [{current_memory}].

    #     SPARSE KEYFRAME RULES (CRITICAL FOR ACCURACY):
    #     1. EXIT CONDITIONS (When is a step done?): An action (like 'pass by vending machines') is considered DEFINITIVELY EXITED the moment the target is no longer the main subject, and a new environment or landmark dominates the view. Do not hold onto the past.
    #     2. ALLOW FAST-FORWARDING (Skipping steps is normal): Keyframes are taken seconds apart. The robot may visually skip intermediate steps. If you were at Step [1], but you suddenly see the clear visual trigger for Step [3] (e.g., the AD billboard), you MUST skip directly to Step [3]. 
    #     3. TRIGGER DOMINANCE: Always tag the HIGHEST-index step whose visual trigger is currently present. If you see the AD billboard, you are at the AD billboard step. Period.

    #     Task: Analyze the [Current Frame]. Which step is the robot executing RIGHT NOW?

    #     First, conduct a State Verification inside <think> tags:
    #     - Q1: What dominant area or specific landmark do I see RIGHT NOW in this frame?
    #     - Q2: Does this match the current step [{current_memory}], or a FUTURE step in the sequence?
    #     - Q3: If I see a future landmark (like an AD billboard or lobby), I must accept that previous steps are completed and fast-forward to this new step immediately.

    #     Return exactly this JSON format after thinking:
    #     {{
    #     "step": int,
    #     "status": "in_progress" | "completed"
    #     }}
    #     """
        
    #     user_content_prompt = [{"type": "text", "text": instruction}]
    #     messages.append({
    #         "role": "user",
    #         "content": user_content_prompt
    #     })

    #     output_text = self.generate(messages=messages,do_sample=do_sample,temperature=temperature,max_new_tokens=max_new_tokens)
    #     print(output_text)

    #     output_text_clean = extract_answer(output_text)
    #     img = Image.open(current_image_path).convert("RGB")
    #     img = img.resize((int(1080*img.size[0]/img.size[1]),1080))

    #     step_idx = extract_step(output_text_clean)
        
    #     if step_idx is not None and step_idx < len(instruction_steps):
    #         self.point_visual.draw_text_on_image(img, save_dir, text = instruction_steps[step_idx])
    #     else:
    #         self.point_visual.draw_text_on_image(img, save_dir, text = "Unknown Step")

    #     self.rgb_list.append(current_image_path)

    #     return output_text_clean
    
    def inference_split_with_history(self, instruction_steps: Union[list, str], current_image_path: str,  save_dir: str, do_sample=True, temperature=0.7, max_new_tokens= 8192):
        messages = []
        user_content = [] 
        recent_history_size = 5
        history_frames = []
        if len(self.rgb_list) > 0:
            start_idx = max(0, len(self.rgb_list) - recent_history_size)
            history_frames = list(range(start_idx, len(self.rgb_list)))

        if history_frames:
            user_content.append({"type": "text", "text":  " These are your historical observations: "})
            for index in history_frames:
                if self.model:
                    user_content.append({"type": "image", "image": self.rgb_list[index]})
                else:
                    img_b64 = encode_image(self.rgb_list[index])
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

        user_content.append({"type": "text", "text": " you can see "})
        if self.model:
            user_content.append({"type": "image", "image": current_image_path})
        else:
            img_b64=encode_image(current_image_path)
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

        messages.append({
            "role": "user",
            "content": user_content
        })

        instruction = f"""
        Steps:

        {chr(10).join([f"[{i}] {s}" for i, s in enumerate(instruction_steps[self.locked_step:], start=self.locked_step)])}

        You are an autonomous navigation assistant tagging sparse keyframes.

        CRITICAL RULES - YOU MUST OBEY:
        1. NO FLICKERING (Solve 1-1-2-1-1 error): Navigation is continuous. Do NOT randomly jump to a different step for a single frame just because the angle changed. You must have UNDENIABLE visual evidence to choose a new step.
        2. SPECIFIC LANDMARKS > GENERAL AREAS: If the robot is inside a general area (e.g., 'enter the lobby' or 'canteen') and you clearly see a specific local target (e.g., 'reception desk', 'vending machines', 'turnstiles'), you MUST select the specific target's step. Do NOT select the general area if the specific landmark is visible!
        3. VERB & ACTION LIFECYCLE (CRITICAL): Pay close attention to the VERB in the step (e.g., 'Enter', 'Pass by', 'Turn', 'Face'). You MUST compare the Current Frame with the Historical Observations to judge the motion trend. Is the robot actively performing this verb? Has the action just started, is it ongoing, or is it fully completed?
        
        Given the visual input, determine:
        1. Which step is the agent needed to execute?
        2. Whether it is: in progress / completed?

        First, analyze the visual evidence step-by-step inside <think> tags. Ask yourself: "What specific landmarks are visible? Am I just in a general area, or am I at a specific target?"
        
        Return exactly this JSON format after thinking:
        {{
        "step": int,
        "status": "in_progress" | "completed"
        }}
        """

        # 限制选择最近的地标
        # instruction = f"""
        # Steps:

        # {chr(10).join([f"[{i}] {s}" for i, s in enumerate(instruction_steps[self.locked_step:], start=self.locked_step)])}

        # You are an autonomous navigation assistant tagging sparse keyframes.

        # CRITICAL RULES - YOU MUST OBEY:
        # 1. NO FLICKERING: Navigation is continuous. Do NOT randomly jump to a different step for a single frame.
        # 2. NEAREST LANDMARK PRIORITY (CRITICAL): You MUST prioritize the LARGEST and CLOSEST landmark in your current field of view. 
        #    - Do NOT be distracted by distant signs or objects.
        #    - If you see a distant target (e.g., the 'HONOR' shelf) but there are closer objects in the foreground (e.g., white desks, robot arms) taking up more screen space, you MUST select the step associated with the CLOSEST objects.
        #    - Only transition to a distant target's step when it becomes the nearest, most dominant object in the foreground.

        # Given the visual input and history, determine:
        # 1. Which step is currently being executed?
        # 2. Whether it is: in progress / completed?

        # First, analyze the visual evidence inside <think> tags. You MUST explicitly answer these 3 questions:
        # - Q1: What are ALL the landmarks currently visible in the frame (both foreground and background)?
        # - Q2: Out of these landmarks, which one is the CLOSEST to the camera and takes up the most space?
        # - Q3: Based STRICTLY on the CLOSEST landmark from Q2, which step matches? Do NOT match based on background objects!
        
        # Return exactly this JSON format after thinking:
        # {{
        # "step": int,
        # "status": "in_progress" | "completed"
        # }}
        # """
        
        user_content_prompt = [{"type": "text", "text": instruction}]
        messages.append({
            "role": "user",
            "content": user_content_prompt
        })

        output_text = self.generate(messages=messages,do_sample=do_sample,temperature=temperature,max_new_tokens=max_new_tokens)
        print(output_text)

        output_text_clean = extract_answer(output_text)
        img = Image.open(current_image_path).convert("RGB")
        img = img.resize((int(1080*img.size[0]/img.size[1]),1080))

        step_idx = extract_step(output_text_clean)
        
        if step_idx is not None and step_idx < len(instruction_steps):
            # 不管大模型输出什么，绝对不允许后退！
            if step_idx >= self.locked_step:
                self.locked_step = step_idx  # 允许前进，更新最高记录
            else:
                # 拦截：如果模型退回了以前的步骤（比如 3 退回 2），强行拉回最高记录
                print(f"拦截到回退：模型预测 {step_idx}，强行锁定在 {self.locked_step}")
                step_idx = self.locked_step

            self.point_visual.draw_text_on_image(img, save_dir, text = instruction_steps[step_idx])
        else:
            self.point_visual.draw_text_on_image(img, save_dir, text = "Unknown Step")

        self.rgb_list.append(current_image_path)

        return output_text_clean

class RynnBrain_Nav:

    def __init__(self, model_path):
        self.point_visual = PointVisual()
        self.box_visual = BoxVisual()
        if model_path:
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype="auto", device_map="auto")
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            import os
            os.environ["NO_PROXY"] = "localhost,127.0.0.1"
            from openai import OpenAI
            self.client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

        self.action_list = []
        self.rgb_list = []

    def reset(self):
        self.action_list = []
        self.rgb_list = []

    def inference_with_frame(self, instruction: str, image_path: str, save_dir: str):
        #Define the instruction
        # instruction = "exit the bedroom and turn left . walk down the hallway and enter the bedroom on the right . wait near the bed .messages"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"You are an autonomous navigation assistant. Your task is to {instruction}. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."},
                    {"type": "text", "text": "you can see "},
                    {
                        "type": "image",
                        "image": image_path,
                    },
                ],
            }
        ]



        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # print(output_text)

        img = Image.open(image_path).convert("RGB")
        img = img.resize((int(1080*img.size[0]/img.size[1]),1080))
        self.point_visual.draw_text_on_image(img, save_dir, text = output_text)
        return output_text

    def inference_with_history(self, instruction: str, current_image_path: str,  save_dir: str):
        conversation_list = []
        user_content = [] 

        curr_rgb = Image.open(current_image_path).convert("RGB")
        num_future_steps = 4
        history_frames = [i for i in range(0, max(0,len(self.rgb_list)-32), max(1, (len(self.rgb_list)-32)//8))]

        sample_frames = [i for i in range(max(0, len(self.rgb_list)-32), len(self.rgb_list), 4)]

        prompt = f"You are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."



        prompt_text = prompt.replace('<instruction>', instruction)
        if history_frames:
            prompt_text += " These are your historical observations: "

        user_content.append({"type": "text", "text": prompt_text})



        for index in history_frames:
            width, height = self.rgb_list[index].size
            user_content.append({"type": "image", "image": self.rgb_list[index], "width": width, "height": height})

        conversation_list.append({
            "role": "user",
            "content": user_content
        })



        for index in sample_frames:
            user_content = []
            assistant_content = []

            user_content.append({"type": "text", "text": 'You can see'})
            width, height = self.rgb_list[index].size
            user_content.append({"type": "image", "image": self.rgb_list[index], "width": width, "height": height})
            

            step_actions = self.action_list[index:index+num_future_steps]

            conversation_list.append({
            "role": "user",
            "content": user_content
            })


            assistant_content.append({"type": "text", "text": step_actions})

            conversation_list.append({
            "role": "assistant",
            "content": assistant_content
            })

        width, height = curr_rgb.size

        user_content = []
        user_content.append({"type": "text", "text": " you can see "})
        user_content.append({"type": "image", "image": curr_rgb, "width": width, "height": height})

        conversation_list.append({
            "role": "user",
            "content": user_content
        })

        inputs = self.processor.apply_chat_template(
            conversation_list,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(output_text)

        img = Image.open(current_image_path).convert("RGB")
        img = img.resize((int(1080*img.size[0]/img.size[1]),1080))
        self.point_visual.draw_text_on_image(img, save_dir, text = output_text)

        for each in output_text:    
            self.action_list.append(each)
        self.rgb_list.append(curr_rgb)


    def inference_split_with_history(self, instruction_steps: Union[list, str], current_image_path: str,  save_dir: str):
        conversation_list = []
        user_content = [] 

        curr_rgb = Image.open(current_image_path).convert("RGB")
        num_future_steps = 4
        history_frames = [i for i in range(0, max(0,len(self.rgb_list)-32), max(1, (len(self.rgb_list)-32)//8))]
        history_frames = []
        sample_frames = [i for i in range(max(0, len(self.rgb_list)-32), len(self.rgb_list), 4)]


        # prompt_text = prompt.replace('<instruction>', instruction)
        if history_frames:
            # prompt_text += " These are your historical observations: "
            user_content.append({"type": "text", "text":  " These are your historical observations: "})

        for index in history_frames:
            width, height = self.rgb_list[index].size
            user_content.append({"type": "image", "image": self.rgb_list[index], "width": width, "height": height})

        conversation_list.append({
            "role": "user",
            "content": user_content
        })

        width, height = curr_rgb.size

        user_content = []
        user_content.append({"type": "text", "text": " you can see "})
        user_content.append({"type": "image", "image": curr_rgb, "width": width, "height": height})

        conversation_list.append({
            "role": "user",
            "content": user_content
        })

        instruction = f"""
        Steps:

        {chr(10).join([f"[{i}] {s}" for i, s in enumerate(instruction_steps)])}

        You are an autonomous navigation assistant.
        Given the visual input, determine:
        1. Which step is currently being executed
        2. Whether it is: in progress / completed

        Return JSON:
        {{
        "step": int,
        "status": "in_progress" | "completed"
        }}
        """
        user_content = []
        user_content.append({"type": "text", "text": instruction})
        conversation_list.append({
            "role": "user",
            "content": user_content
        })

        inputs = self.processor.apply_chat_template(
            conversation_list,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=81920,
                                            temperature=0.7,
                                            top_p=0.8,)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(output_text)

        output_text = extract_answer(output_text)
        img = Image.open(current_image_path).convert("RGB")
        img = img.resize((int(1080*img.size[0]/img.size[1]),1080))

        step_idx = extract_step(output_text)
        self.point_visual.draw_text_on_image(img, save_dir, text = instruction_steps[step_idx])

        # for each in output_text:    
        #     self.action_list.append(each)
        self.rgb_list.append(curr_rgb)

        return output_text  