import os
# ROOT_PATH="" # set the path to root dir
# assert ROOT_PATH != "", "Please set ROOT_PATH to the cookbooks directory."
# os.chdir(ROOT_PATH)

import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Image as IPyImage
import io
import re
import cv2
import numpy as np
from typing import List, Tuple, Optional
def split_at_middle_space(text, first_line_ratio=0.51):
    text = text.strip()
    space_positions = [i for i, char in enumerate(text) if char == ' ']
    
    if not space_positions:  
        return text
    
    target_pos = int(len(text) * first_line_ratio)
    
    split_pos = min(space_positions, key=lambda x: abs(x - target_pos))
    
    first_line = text[:split_pos].strip()
    second_line = text[split_pos:].strip()
    
    return f"{first_line}\n{second_line}"

def add_frame_id(conversation):
  for message in conversation:
    if message["role"] == "user":
        image_idx = 0
        new_contents = []
        for i, content in enumerate(message["content"]):
            if content["type"] == "image":
                    new_contents.append({"type": "text", "text": f"<frame {image_idx}>: "})
                    image_idx += 1
            new_contents.append(content)
        message["content"] = new_contents
  return conversation

def show_images_grid(
    img_dir,
    n=8,
    cols=4,
    figsize_per_cell=(4, 4),
    exts=("png", "jpg", "jpeg", "bmp", "webp"),
    sort=True,
    show_title=True,
    keep_axis=True,
):
    patterns = [os.path.join(img_dir, f"*.{e}") for e in exts]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat))
    if sort:
        paths = sorted(paths)

    paths = paths[:n]
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found in {img_dir} with extensions {exts}")

    rows = (len(paths) + cols - 1) // cols
    fig_w = figsize_per_cell[0] * cols
    fig_h = figsize_per_cell[1] * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

    for i, ax in enumerate(axes.flat):
        if i < len(paths):
            p = paths[i]
            img = Image.open(p)
            ax.imshow(img)

            if show_title:
                ax.set_title(os.path.basename(p), fontsize=10)

            if keep_axis:
                # ax.set_xlabel("x (px)")
                # ax.set_ylabel("y (px)")
                ax.tick_params(labelsize=8)
            else:
                ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()
class PointVisual:

    def __init__(self, display: bool):
        self.display = display
        return
    

    def convert_points_to_raw(self, points_norm, width, height):
        point_list = []
        for point in points_norm:
            x_norm, y_norm = point[0], point[1]
        
            x = round(x_norm / 1000 * width)
            y = round(y_norm / 1000 * height)
            point_list.append((x, y))
        return point_list
    

    def parse_frame_id_and_points(self, output_text: str) -> Tuple[Optional[int], List[Tuple[int, int]]]:
        try:
            frame_match = re.search(r'frame (\d+)', output_text)
            if not frame_match:
                return None, []
            
            frame_id = int(frame_match.group(1))
            
            coord_pattern = r'\((\d+),\s*(\d+)\)'
            coord_matches = re.findall(coord_pattern, output_text)
            
            points = []
            for x_str, y_str in coord_matches:
                try:
                    x = int(x_str)
                    y = int(y_str)
                    points.append((x, y))
                except ValueError:
                    continue
            
            return frame_id, points
            
        except Exception as e:
            return None, []

    def parse_points(self, output_text: str) -> Tuple[Optional[int], List[Tuple[int, int]]]:
        try:
            coord_pattern = r'\((\d+),\s*(\d+)\)'
            coord_matches = re.findall(coord_pattern, output_text)
            
            points = []
            for x_str, y_str in coord_matches:
                try:
                    x = int(x_str)
                    y = int(y_str)
                    points.append((x, y))
                except ValueError:
                    continue
            
            return points
            
        except Exception as e:
            return []

    def draw_points_on_image(self, img, points, save_dir ,color="red", point_radius=6, width=4, show_width=400, text=None):
        # img = Image.open(img_path).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)

        for point in points:
            x, y = point
            draw.ellipse([x-point_radius, y-point_radius, x+point_radius, y+point_radius], 
                        outline=color, width=width, fill=color)
        
        if text:
            if isinstance(text, list):
                text = "\n".join(text)
            text = split_at_middle_space(text)
            
            if points:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                points_bbox_width_shift = 0.1*w
                points_bbox_width = max_x - min_x + points_bbox_width_shift 
                text_length = len(text)
                min_fontsize = 10
                max_fontsize = 40
                
                if points_bbox_width > 0:
                    fontsize = max(min_fontsize, min(max_fontsize, int(points_bbox_width / max(1, text_length * 0.07))))
                else:
                    fontsize = 12  
            else:
                min_x, min_y = 10, 10  
                fontsize = 12
            # print(fontsize, text_length, points_bbox_width)
            font = ImageFont.truetype("cookbooks/assets/arial.ttf", fontsize)
            
            try:
                bbox_text = draw.textbbox((0, 0), text, font)
            except:
                bbox_text = draw.textsize(text, font)
                bbox_text = (0, 0, bbox_text[0], bbox_text[1])
            
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            padding = 10
            text_x = max(min(max_x + 3 * padding, w - text_width - padding), padding)
            text_y = max(min(min_y - text_height - padding, h - text_height - padding), padding)
            
            if text_y < padding:
                text_y = min(max_y + padding, h - text_height - padding)
            
            background_rect = [
                text_x - padding, text_y - padding,
                text_x + text_width + padding, text_y + text_height + 3 * padding
            ]
            draw.rectangle(background_rect, fill=color)
            
            draw.text((text_x, text_y), text, fill="white", font=font)
        img.save(save_dir)
        if self.display:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            display(IPyImage(data=buf.getvalue(), width=show_width))
        

    def draw_trajectory_on_image(self, img, points, save_dir, color="red", point_radius=6, width=4, show_width=400, text=None):
        # img = Image.open(img_path).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)

        if len(points) >= 2:
            draw.line(points, fill=color, width=width)
            
            if len(points) >= 2:
                x_end, y_end = points[-1]  
                x_prev, y_prev = points[-2]  
                
                arrow_size = width * 3
                
                dx = x_end - x_prev
                dy = y_end - y_prev
                length = (dx*dx + dy*dy) ** 0.5
                
                if length > 0:
                    dx /= length
                    dy /= length
                    
                    perp_dx = -dy
                    perp_dy = dx
                    
                    left_x = x_end - dx*arrow_size + perp_dx*arrow_size*0.5
                    left_y = y_end - dy*arrow_size + perp_dy*arrow_size*0.5
                    
                    right_x = x_end - dx*arrow_size - perp_dx*arrow_size*0.5
                    right_y = y_end - dy*arrow_size - perp_dy*arrow_size*0.5
                    
                    draw.polygon([(x_end, y_end), (left_x, left_y), (right_x, right_y)], 
                                fill=color, outline=color)
        
        if text:
            if isinstance(text, list):
                text = "\n".join(text)
            text = split_at_middle_space(text)
            
            if points:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                points_bbox_width_shift = 0.1*w
                points_bbox_width = max_x - min_x + points_bbox_width_shift 
                text_length = len(text)
                min_fontsize = 10
                max_fontsize = 40
                
                if points_bbox_width > 0:
                    fontsize = max(min_fontsize, min(max_fontsize, int(points_bbox_width / max(1, text_length * 0.07))))
                else:
                    fontsize = 12  
            else:
                min_x, min_y = 10, 10  
                fontsize = 12
            # print(fontsize, text_length, points_bbox_width)
            font = ImageFont.truetype("cookbooks/assets/arial.ttf", fontsize)
            
            try:
                bbox_text = draw.textbbox((0, 0), text, font)
            except:
                bbox_text = draw.textsize(text, font)
                bbox_text = (0, 0, bbox_text[0], bbox_text[1])
            
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            padding = 10
            text_x = max(min(max_x + 3 * padding, w - text_width - padding), padding)
            text_y = max(min(min_y - text_height - 5 * padding, h - text_height - padding), padding)
            
            if text_y < padding:
                text_y = min(max_y + padding, h - text_height - padding)
            
            background_rect = [
                text_x - padding, text_y - padding,
                text_x + text_width + padding, text_y + text_height + 3 * padding
            ]
            draw.rectangle(background_rect, fill=color)
            
            draw.text((text_x, text_y), text, fill="white", font=font)
        

        img.save(save_dir)
        if self.display:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            display(IPyImage(data=buf.getvalue(), width=show_width))


    def draw_text_on_image(self, img, save_dir ,color="red", text=None, width=4, show_width=400):
        # img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        
        if text:
            if isinstance(text, list):
                text = "\n".join(text)
            text = split_at_middle_space(text)
            

            min_x, min_y = 10, 10  
            fontsize = 36
            # print(fontsize, text_length, points_bbox_width)
            font = ImageFont.truetype("cookbooks/assets/arial.ttf", fontsize)
            
            try:
                bbox_text = draw.textbbox((0, 0), text, font)
            except:
                bbox_text = draw.textsize(text, font)
                bbox_text = (0, 0, bbox_text[0], bbox_text[1])

            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            padding = 10
            text_x = padding
            text_y = padding    
            background_rect = [
                text_x - padding, text_y - padding,
                text_x + text_width + padding, text_y + text_height + 3 * padding
            ]
            draw.rectangle(background_rect, fill=color)

            draw.text((text_x, text_y), text, fill="white", font=font)
        else:
            print("No text")
            return
        img.save(save_dir)

        if self.display:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            display(IPyImage(data=buf.getvalue(), width=show_width))

class BoxVisual:

    def __init__(self, display: bool):
        self.display = display
        return
    

    def convert_bbox_to_raw(self, bbox, w, h):
        bbox_norm = [max(0, min(1000, v)) for v in bbox]
        bbox_raw = [
            int(round(bbox_norm[0] / 1000 * (w-1))),
            int(round(bbox_norm[1] / 1000 * (h-1))),
            int(round(bbox_norm[2] / 1000 * (w-1))),
            int(round(bbox_norm[3] / 1000 * (h-1))),
        ]
        return bbox_raw


    def draw_bbox_on_image(self, img, bbox,save_dir, color="red", width=4, show_width=400, text=None):
        # img = Image.open(img_path).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)

        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        if text:
            if isinstance(text, list):
                text = "\n".join(text)
            text = split_at_middle_space(text)
            bbox_pixels =  x2 - x1
            text_length = len(text)
            min_fontsize = 6
            max_fontsize = 40
            fontsize = max(min_fontsize, min(max_fontsize, int(bbox_pixels / max(1, text_length * 0.3))))
            font = ImageFont.truetype("cookbooks/assets/arial.ttf", fontsize)
            
            try:
                bbox_text = draw.textbbox((0, 0), text, font)
            except:
                bbox_text = draw.textsize(text, font)
                bbox_text = (0, 0, bbox_text[0], bbox_text[1])
            
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            y_shift = 0
            padding = 5
            text_x = max(min(x1+padding, 0.8*w),padding)
            text_y = max(min(y1+y_shift+padding, 0.95*h),padding)
            
            # print(bbox_pixels, text_length, fontsize)

            background_rect = [
                text_x - padding, text_y - padding,
                text_x + text_width + padding, text_y + text_height + 3 * padding
            ]
            draw.rectangle(background_rect, fill=color)
            
            draw.text((text_x, text_y), text, fill="white", font=font)

        img.save(save_dir)
        if self.display:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            display(IPyImage(data=buf.getvalue(), width=show_width))

    def parse_frame_id_and_bbox(self, output_text):
        try:
            frame_match = re.search(r'frame (\d+)', output_text)
            if frame_match:
                frame_id = int(frame_match.group(1))
                
                coord_match = re.search(r'\((\d+),\s*(\d+)\)\,\s*\((\d+),\s*(\d+)\)', output_text)
                if coord_match:
                    bbox = (
                        int(coord_match.group(1)),
                        int(coord_match.group(2)),
                        int(coord_match.group(3)),
                        int(coord_match.group(4))
                    )
                    return frame_id, bbox
        except:
            pass
        return None, None

    def parse_bbox(self, output_text):
        try:    
            coord_match = re.search(r'\((\d+),\s*(\d+)\)\,\s*\((\d+),\s*(\d+)\)', output_text)
            if coord_match:
                bbox = (
                    int(coord_match.group(1)),
                    int(coord_match.group(2)),
                    int(coord_match.group(3)),
                    int(coord_match.group(4))
                )
                return bbox
        except:
            pass
        return None

