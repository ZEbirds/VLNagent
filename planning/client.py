# local_ui.py (Final Complete Version v9.1: Ensure Background Tasks Start, Include All Functions)

import gradio as gr
import requests
import os
import mimetypes
import glob
from typing import List, Tuple
import cv2
import numpy as np
import datetime
import atexit
import threading
import time
import tkinter as tk
from tkinter import filedialog
import json
from deep_translator import GoogleTranslator
import argparse

# --- Configuration ---
LLM_TYPE_OPTIONS = ['rynnbrain_planning']

# translator = Translator()
# --- Camera and Global State Management ---
try:
    cap = cv2.VideoCapture(1)  # camera id
    if not cap.isOpened(): raise IOError("Cannot open camera")
    CAMERA_AVAILABLE = True
except Exception as e:
    print(f"Camera initialization failed: {e}"); cap = None; CAMERA_AVAILABLE = False

last_frame = None
is_saving = False
folder_to_save = ""
state_lock = threading.Lock()
THREAD_STARTED = False  # Flag to ensure thread starts only once
conversation = []  # Global: Save each input image and model return results (streaming updates)
save_start_time = None  # Record the time point when saving starts


from PIL import Image, ImageDraw, ImageFont
import json
import re
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
# --- Drawing Function 1: Draw a point on a PIL Image object ---

def denormalize_points(point, width, height):
    """Convert coordinates from [0, 1000] range to pixel coordinates"""
    # for point in points_norm:
    x_norm, y_norm = point[0], point[1]

    x = round(x_norm / 1000 * width)
    y = round(y_norm / 1000 * height)
    point_list = [x, y]
    return point_list

def clean_text(outputs_str: str):
    """
    Extract coordinates from string and clean label content
    """
    # Extract coordinates
    
    # Clean string, keep only labels
    cleaned_str = re.sub(r'<affordance>.*?</affordance>', '<affordance></affordance>', outputs_str, flags=re.DOTALL)
    cleaned_str = re.sub(r'<object>.*?</object>', '<object></object>', cleaned_str, flags=re.DOTALL)
    cleaned_str = re.sub(r'<area>.*?</area>', '<area></area>', cleaned_str, flags=re.DOTALL)
    
    return cleaned_str
def parse_complex_response(response_str: str) -> dict:
    
    outputs_str = response_str

    result = {}

    affordance_pattern = re.compile(r"<affordance>.*?\((\d+),\s*(\d+)\).*?</affordance>", re.DOTALL)
    
    # Pattern 2: Extract two coordinates (x1, y1), (x2, y2) inside <object> tags
    object_pattern = re.compile(r"<object>.*?\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\).*?</object>", re.DOTALL)
    area_pattern = re.compile(r"<area>.*?\((\d+),\s*(\d+)\).*?</area>", re.DOTALL)

    # --- 4. Perform matching and extraction ---
    affordance_match = affordance_pattern.search(outputs_str)
    object_match = object_pattern.search(outputs_str)
    area_match = area_pattern.search(outputs_str)


    if affordance_match:
        affordance_point = [int(affordance_match.group(1)), int(affordance_match.group(2))]
        result['affordance'] = affordance_point
    if object_match:
        object_points = [
          [int(object_match.group(1)), int(object_match.group(2))],
          [int(object_match.group(3)), int(object_match.group(4))]
        ]
        result['object'] = object_points
    if area_match:
        area_points = [int(area_match.group(1)), int(area_match.group(2))]
        result['area'] = area_points

    
    return result


def draw_point_on_image(
    image: Image.Image,
    point,
    point_color: str = 'red',
    point_radius: int = 10
) -> Image.Image:
    """
    Draw a point on the given PIL Image object and return a new Image object with the mark.

    Parameters:
    - image (Image.Image): Input Pillow Image object.
    - point (tuple): Point coordinates in format (x, y).
    - point_color (str): Point color, e.g., 'red', 'green', '#FF0000'.
    - point_radius (int): Point radius size (pixels).

    Returns:
    - Image.Image: A new Image object with the drawn point.
    """
    # 1. Create a copy to avoid modifying the original image
    #    and ensure the image is in RGB format for drawing
    output_image = image.copy().convert("RGB")
    
    # 2. Create a drawable object
    draw = ImageDraw.Draw(output_image)
    
    # 3. Calculate the bounding box for the point to draw a circle
    x, y = point
    # ellipse method needs a bounding box: [x0, y0, x1, y1]
    point_bbox = [x - point_radius, y - point_radius, x + point_radius, y + point_radius]
    
    # 4. Draw a solid circular point on the image copy
    draw.ellipse(point_bbox, fill=point_color, outline=point_color)
    
    # 5. Return the modified image object
    return output_image

# --- Drawing Function 2: Draw a bounding box on a PIL Image object ---

def draw_bbox_on_image(
    image: Image.Image,
    top_left,
    bottom_right,
    box_color: str = 'lime',
    box_width: int = 5,
    label: str = None,
    label_color: str = 'white'
) -> Image.Image:
    """
    Draw a bounding box on the given PIL Image object and return a new Image object with the mark.

    Parameters:
    - image (Image.Image): Input Pillow Image object.
    - top_left (tuple): Coordinates of the top-left corner of the bounding box, format (x_min, y_min).
    - bottom_right (tuple): Coordinates of the bottom-right corner of the bounding box, format (x_max, y_max).
    - box_color (str): Color of the bounding box.
    - box_width (int): Line width of the bounding box.
    - label (str, optional): Label text to display next to the bounding box.
    - label_color (str): Color of the label text.

    Returns:
    - Image.Image: A new Image object with the drawn bounding box.
    """
    # 1. Create a copy to avoid modifying the original image and ensure RGB format
    output_image = image.copy().convert("RGB")
    
    # 2. Create a drawable object
    draw = ImageDraw.Draw(output_image)
    
    # 3. Prepare bounding box coordinates
    x_min, y_min = top_left
    x_max, y_max = bottom_right
    bbox = [x_min, y_min, x_max, y_max]
    
    # 4. Draw rectangle on image copy
    draw.rectangle(bbox, outline=box_color, width=box_width)
    
    # 5. (Optional) If there is a label, draw the label and background
    if label:
        try:
            # Try to load a font, use default if failed
            font = ImageFont.truetype("arial.ttf", size=max(15, int((x_max - x_min) / 10)))
        except IOError:
            font = ImageFont.load_default()

        # Get dimensions of the label text
        # Use textlength instead of textbbox for older Pillow versions, but textbbox is more precise
        if hasattr(draw, 'textbbox'):
             text_bbox = draw.textbbox((x_min, y_min), label, font=font)
             text_width = text_bbox[2] - text_bbox[0]
             text_height = text_bbox[3] - text_bbox[1]
        else: # Compatible with older Pillow versions
             text_width, text_height = draw.textlength(label, font=font)

        # Create a rectangle as background for the label
        # Ensure the label background doesn't go above the image
        label_y0 = max(0, y_min - text_height - 5)
        label_background = [x_min, label_y0, x_min + text_width + 4, y_min]
        draw.rectangle(label_background, fill=box_color)
        
        # Draw text on the background
        draw.text((x_min + 2, label_y0), label, fill=label_color, font=font)

    # 6. Return the modified image object
    return output_image

# --- Example of how to use these two functions ---

def _pick_two_frames(image_paths: List[str]):
    if not image_paths:
        return None, None

    last_path = image_paths[-1]
    mid_path = image_paths[len(image_paths)//2]

    with state_lock:
        sst = save_start_time

    if sst is not None:
        after = []
        for p in image_paths:
            try:
                if os.path.getmtime(p) >= sst:
                    after.append(p)
            except Exception:
                pass
        if after:
            mid_path = after[len(after)//2]
            last_path = after[-1]

    return mid_path, last_path




def release_camera():
    """Automatically release camera when program exits"""
    if cap:
        cap.release()
        print("Camera released.")
atexit.register(release_camera)

# --- Core Functions ---

def stream_video():
    """Capture frames from camera and update global variables"""
    global last_frame
    if not CAMERA_AVAILABLE:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    ret, frame = cap.read()
    if ret:
        with state_lock:
            last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    with state_lock:
        return last_frame

def toggle_saving():
    """Toggle start/stop saving status"""
    global is_saving, save_start_time
    with state_lock:
        is_saving = not is_saving
        if is_saving:
            save_start_time = time.time()
        else:
            save_start_time = None
    return gr.update(value="Stop Saving" if is_saving else "Start Saving")


def _internal_scan(folder_path):
    """Scan for image files in the specified folder"""
    if not folder_path or not os.path.isdir(folder_path):
        return None
    image_paths = [p for ext in ["*.jpg","*.jpeg","*.png","*.bmp","*.webp"] for p in glob.glob(os.path.join(folder_path, ext))]
    image_paths.sort()
    if not image_paths:
        gr.Warning("Folder is empty or contains no supported image formats.")
        return None
    return image_paths

def select_folder_path():
    """Pop up folder selection dialog and return path - Thread-safe version"""
    import tkinter as tk
    from tkinter import filedialog
    import threading
    
    # Check if running in main thread
    if threading.current_thread() is threading.main_thread:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title="Select a base directory (can be manually modified later)")
        root.destroy()
        return path
    else:
        # If called from background thread, return empty string and notify user
        print("Warning: File selection can only run in main thread, please enter path directly")
        return ""

def handle_path_change(path):
    """When path textbox content changes, update global variable and control button status"""
    global folder_to_save
    with state_lock:
        folder_to_save = path
    if path:
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=False)

def refresh_preview(path):
    """Refresh image preview"""
    if not path or not os.path.isdir(path):
        gr.Warning("Invalid path or does not exist, cannot preview.")
        return None
    return _internal_scan(path)

def background_saver_thread_logic():
    """Background thread execution loop logic, responsible for periodically saving images"""
    while True:
        time.sleep(2)
        with state_lock:
            should_save, frame, save_dir = is_saving, last_frame, folder_to_save
        
        #print(f"Background thread check: is_saving={should_save}, save_dir='{save_dir}', frame_exists={frame is not None}")

        if should_save and frame is not None and save_dir:
            try:
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"capture_{timestamp}.jpg"
                save_path = os.path.join(save_dir, filename)
                frame_to_save = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, frame_to_save)
                print(f"✅ Background save successful: {filename}")
            except Exception as e:
                print(f"❌ Error saving image in background: {e}")

def start_background_thread():
    """Called by Gradio's load event, ensures thread starts only once"""
    global THREAD_STARTED
    if not THREAD_STARTED:
        print("Gradio `load` event triggered: Starting background image saving thread...")
        thread = threading.Thread(target=background_saver_thread_logic, daemon=True)
        thread.start()
        THREAD_STARTED = True
        print("Background image saving thread started.")

def save_result_to_jsonl(folder_path, result_data, prompt, llm_type, image_count):
    """Append single inference result to log file in JSONL format"""
    log_file_path = os.path.join(folder_path, "inference_results.jsonl")
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": llm_type,
        "prompt": prompt,
        "image_count": image_count,
        "result": result_data
    }
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        print(f"Result successfully logged to: {log_file_path}")
    except Exception as e:
        print(f"Error writing result to JSONL file: {e}")
        gr.Warning(f"Cannot write to log file: {e}")

def trigger_load_model(llm_type: str, server_url: str):
    """Trigger model loading request to FastAPI server"""
    if not llm_type:
        raise gr.Error("Please select a model!")
    
    LOAD_URL = f"{server_url}/loading/"
    data = {'llm_type': llm_type}
    try:
        gr.Info(f"Sending model loading request to server, please wait...")
        response = requests.post(LOAD_URL, data=data, timeout=1000)
        
        if response.status_code == 200:
            gr.Info("Model loaded successfully!")
            return f"Model '{llm_type}' loaded successfully!"
        else:
            error_detail = "Cannot parse error response"
            try:
                error_detail = response.json().get('detail', response.text)
            except json.JSONDecodeError:
                error_detail = response.text
            raise gr.Error(f"Server error (status code {response.status_code}): {error_detail}")
        
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"Network connection error: {e}")

def trigger_inference(llm_type: str, prompt: str, folder_path: str, server_url: str):
    """Trigger inference request to FastAPI server"""
    global conversation

    if not llm_type: raise gr.Error("Please select a model!")
    if not prompt: raise gr.Error("Please enter a prompt!")
    if not folder_path or not os.path.isdir(folder_path):
        raise gr.Error("Invalid path or does not exist, cannot perform inference.")

    INFER_URL = f"{server_url}/inference/"
    gr.Info("Scanning folder to get latest file list...")
    print('folder_path: ', folder_path)
    image_paths = [p for ext in ["*.jpg","*.jpeg","*.png","*.bmp","*.webp"] for p in glob.glob(os.path.join(folder_path, ext))]
    image_paths.sort()
    print(image_paths)
    if not image_paths:
        raise gr.Error("No images found in specified folder, cannot execute inference.")
    # Select two frames: middle frame after start saving + current last frame
    mid_path, last_path = _pick_two_frames(image_paths)
    if mid_path is None or last_path is None:
        raise gr.Error("Cannot select two frames required for inference.")
    print('mid_path: ', mid_path)
    print('last_path: ', last_path)
    if len(conversation)==0:
        user_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": last_path},
            ]
        }
        conversation.append(user_msg)
        cur_paths = [last_path]
    else:
        user_msg = {
            "role": "user",
            "content": [
                {"type": "image", "image": mid_path},
                {"type": "image", "image": last_path},
            ]
        }
        conversation.append(user_msg)
        cur_paths = [mid_path, last_path]

    # === Send: all historical round images in conversation (including this round) ===
    conv_image_paths = []
    for msg in conversation:
        if msg.get("role") != "user":
            continue
        for item in msg.get("content", []):
            if item.get("type") == "image":
                p = item.get("image")
                conv_image_paths.append(p)

    image_paths = conv_image_paths  # Override original image_paths (no longer using [-30:])

    print('input conversation:', conversation)
    

    data = {'llm_type': llm_type, 'conversation': json.dumps(conversation, ensure_ascii=False)}
    files_to_send, files_to_close = [], []
    # image_paths = image_paths[-30:]
    for path in image_paths:
        filename = os.path.basename(path)
        content_type, _ = mimetypes.guess_type(filename)
        if content_type is None: content_type = 'application/octet-stream'
        f = open(path, 'rb'); files_to_close.append(f)
        files_to_send.append(('video_files', (filename, f, content_type)))

    try:
        gr.Info(f"Sending {len(files_to_send)} files to server, please wait...")
        response = requests.post(INFER_URL, data=data, files=files_to_send, timeout=300)
        
        if response.status_code == 200:
            result = response.json().get('result', 'Cannot parse result')
            if isinstance(result, str):
                # If result is JSON string, need to parse to dictionary
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    raise gr.Error(f"Cannot parse server returned result: {result}")

            gr.Info("Inference successful!")
            print(result)
            assist_msg = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": result['outputs']}
                ]
            }
            conversation.append(assist_msg)
            frame_idx = result['frame_idx']
            result_string = result['outputs']

            response_data = {
                "frame_idx": frame_idx,
                "outputs": result_string
            }
            
            save_result_to_jsonl(folder_path, result_string, prompt, llm_type, len(image_paths))
            print(clean_text(result_string))
            

            if result_string == 'Done.':
                return "Done.", None
            
            final_result = clean_text(result_string) # + "\n" + result_cn

            # try:
            visualization_img = visualize_grounding(response_data, cur_paths)
            return final_result, visualization_img  # Return result and visualization image
            # except Exception as e:
            #     print(f"Error during visualization: {e}")
            #     return final_result, None
            
        else:
            error_detail = "Cannot parse error response"
            try:
                error_detail = response.json().get('detail', response.text)
            except json.JSONDecodeError:
                error_detail = response.text
            raise gr.Error(f"Server error (status code {response.status_code}): {error_detail}")
        
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"Network connection error: {e}")
    finally:
        for f in files_to_close:
            f.close()
            
def add_count(count):
	new_count = count + 1
	return new_count, new_count

def min_count(count):
	new_count = count - 1
	return new_count, new_count
	
def clear_count():
	return 0, 0

def reset_conv():
    """Reset conversation history"""
    global conversation
    conversation.clear()  # Clear list instead of reassigning, keeping reference
    print("Conversation history cleared")
def visualize_grounding(response, image_paths):
    """
    Process single data item: parse, draw and save.
    This function runs independently in each thread.
    """

    # Parse response to get coordinates and frame index
    response_text = response['outputs']
    result = parse_complex_response(response_text)
    frame_idx = response['frame_idx']
    
    image_path = image_paths[frame_idx]
    original_image = Image.open(image_path)
    w, h = original_image.size

    final_image = original_image.copy() # Use copy for drawing
    
    # Chain drawing
    if "affordance" in result:
        points = denormalize_points(result['affordance'], w, h)
        final_image = draw_point_on_image(final_image, points)
    
    if "object" in result:
        points0 = denormalize_points(result['object'][0], w, h)
        points1 = denormalize_points(result['object'][1], w, h)
        final_image = draw_bbox_on_image(final_image, points0, points1, label="object")
        
    if "area" in result:
        points = denormalize_points(result['area'], w, h)
        final_image = draw_point_on_image(final_image, points, point_color='blue')
    
    return final_image
        
	

def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Smart Analysis Client")
    parser.add_argument(
        "--server_url",
        type=str,
        required=True,
        help="URL of the FastAPI server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio interface on"
    )
    return parser.parse_args()

# --- Build UI Interface ---
def create_demo(server_url):
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎬 LLM Smart Analysis Control Panel")
        gr.Markdown("""
        **Workflow:**
        1.  **Step 1**: Specify target path in the lower left input field.
        2.  After entering the path, the "Start Saving" button will be activated.
        3.  **Collection**: Click "Start Saving". If the path doesn't exist, the program will **automatically create the folder for you**.
        4.  **Preview**: Click "Refresh Preview", the right side will display images in the target folder.
        5.  **Inference**: Enter prompt, select model, then click "Submit to Server for Inference".
        6.  **Record**: Inference results will be automatically saved to the `inference_results.jsonl` file in the target folder.
        """)
        counter_state = gr.State(value=0)
        
        
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Camera Control")
                camera_feed = gr.Image(label="Live Feed", streaming=True, source="webcam")  # Modified code
                save_btn = gr.Button("Start Saving", interactive=False)
                
                gr.Markdown("### 2. Files and Inference")
                with gr.Group():
                    folder_path_input = gr.Textbox(
                        label="Image Save/Analysis Folder Path",
                        interactive=True,
                        placeholder="Please enter or select a path here..."
                    )
                    with gr.Row():
                        select_folder_btn = gr.Button("📂 Select Base Directory")
                        refresh_preview_btn = gr.Button("🔄 Refresh Preview")
                
                with gr.Group():
                    llm_type_input = gr.Dropdown(LLM_TYPE_OPTIONS, label="Model Selection")
                    load_model_btn = gr.Button("Load Model to Server", variant="primary")
                    prompt_input = gr.Textbox(label="Enter TASK", lines=3)
                    submit_btn = gr.Button("Submit to Server for Inference", variant="primary")
                    
                    gr.Markdown("#### Inference Count Statistics")
                    with gr.Row():
                          inference_number = gr.Number(value=0, label="Current Inference Count", interactive=False)
                          add_button = gr.Button("Inference +1")
                          min_button = gr.Button("Inference -1")
                          clear_button = gr.Button("Reset")
                    
                    

            with gr.Column(scale=2):
                gallery_output = gr.Gallery(label="Image Preview", show_label=True, elem_id="gallery", columns=5, rows=3, height=500, object_fit="cover")
                result_output = gr.Textbox(label="Server Response", lines=10, interactive=False)
                visualization_output = gr.Image(label="Visualization Result", interactive=False)
                clear_conv_button = gr.Button("End Conversation, Clear History", variant="stop")
                

        # --- Event Binding ---
        # Start video stream (recurring task)
        demo.load(stream_video, None, camera_feed, every=0.1 if CAMERA_AVAILABLE else 99999)
        # Start background saving thread (one-time task)
        demo.load(start_background_thread, None, None)

        # Other button events
        save_btn.click(toggle_saving, None, save_btn)
        select_folder_btn.click(select_folder_path, None, folder_path_input)
        folder_path_input.change(handle_path_change, inputs=[folder_path_input], outputs=[save_btn])
        refresh_preview_btn.click(refresh_preview, inputs=[folder_path_input], outputs=[gallery_output])
        submit_btn.click(
            lambda llm_type, prompt, folder_path: trigger_inference(llm_type, prompt, folder_path, server_url), 
            inputs=[llm_type_input, prompt_input, folder_path_input], 
            outputs=[result_output, visualization_output]  # Update output components
        ).then(
            add_count, 
            inputs=[counter_state], 
            outputs=[counter_state, inference_number]
        )
        load_model_btn.click(
            lambda llm_type: trigger_load_model(llm_type, server_url), 
            inputs=[llm_type_input]
        )
        add_button.click(add_count, inputs=[counter_state], outputs=[counter_state, inference_number])
        min_button.click(min_count, inputs=[counter_state], outputs=[counter_state, inference_number])
        clear_button.click(clear_count, inputs=None, outputs=[counter_state, inference_number])
        clear_conv_button.click(reset_conv, None, None)
    
    return demo

# --- Launch ---
if __name__ == "__main__":
    args = parse_arguments()
    demo = create_demo(args.server_url)
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, show_error=True)