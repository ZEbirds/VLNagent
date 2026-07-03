from rynn_planning import RynnBrain_Nav, RynnBrain_Planning
import os

if __name__ == "__main__":
    # 1. 初始化模型
    model_path = "models/Qwen3-VL-32B-Instruct"
    rynn_model = RynnBrain_Planning(model_path=None)
    rynn_model.reset_history()
    
    # 2. 设置输入和输出文件夹路径
    input_img_dir = "/mnt/pfs/3zpd5q/code/RynnBrain/visual_result/episode_000/extracted_keyframes"
    output_save_dir = "visual_result/frame_data1"
    os.makedirs(output_save_dir, exist_ok=True)

    # 3. 准备导航指令
    instruction_steps = [
        "Leave the starting lounge area near the dark shower cabin", 
        "Go straight down the corridor passing the yellow '3 Home Zone' sign on the left", 
        "Pass between the white exhibition desks featuring robot arms", 
        "Go toward the illuminated 'HONOR' shelf if you don't see the door on its left side", 
        "Go through the wooden door when you see it"
    ]

    # 4. 获取文件夹中所有的图片，并按文件名排序（确保按时间顺序输入）
    if not os.path.exists(input_img_dir):
        print(f"错误：找不到文件夹 {input_img_dir}")
        exit()

    image_files = [f for f in os.listdir(input_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort()  # 这一步极其重要！保证 frame_0001, frame_0005 是按顺序喂给模型的

    print(f"共找到 {len(image_files)} 张关键帧，开始打标...")

    # 5. 遍历图片进行打标
    for filename in image_files:
        orig_path = os.path.join(input_img_dir, filename)
        save_plot_path = os.path.join(output_save_dir, filename)

        # 统一使用关键字参数 (key=value) 防止报错
        rynn_8b = rynn_model.inference_split_with_history(
            instruction_steps=instruction_steps,
            current_image_path=orig_path,
            save_dir=save_plot_path
        )
        
        print(f"########################### Prediction for {filename}:\n{rynn_8b}\n")

    print("所有帧标注完成！")