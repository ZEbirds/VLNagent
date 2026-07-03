from rynn_planning import RynnBrain_Planning
import os
import time
if __name__ == "__main__":
    model_path = "models/RynnBrain-30B-A3B"
    # model_path = "models/Qwen3-VL-32B-Instruct"
    rynn_model = RynnBrain_Planning(model_path=None)
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
    
    image_folder = "cookbooks/assets/elevator"
    # instruction = "Which lane are you standing on? Left, Middle or Right?"
    count = 0
    for filename in sorted(os.listdir(image_folder)):
        if count ==0 :
            count+=1
            continue
        # 只处理常见图片格式
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            orig_path = os.path.join(image_folder, filename)
            print(f"Processing {filename} ...")
            
            # 调用模型
            
            # instruction = "Does the human in first image appear in the second image?"
            # instruction = "Where is the nearest elevator?"
            instruction = (
            # f"Identify the target: the turnstiles with most space. (If the target has a gate, the target should be the center space of it for walking)"
            f"Identify the target: the person with white coat and black pants"
            f"The target is labelled with a bounding box in frame 1. "
            f"Find the same target in frame 0 (not only the type but also the position if there are multiple of them, especially their relative spatial relationship).")

            # instruction = (
            # f"Identify the target: the elevator. "
            # f"The target is labelled with a bounding box in the image. "
            # f"I need you to describe the semantic instruction for a navigation robot to move to the target. If there is any obvious landmark or symbol that is helpful for locating, use it.")
            # instruction = "Where is the turnstiles with most space. (If the target has a gate, the target should be the center space of it for walking)?"

            # instruction = "Where is the person with white coat and black pants"
            instruction = f"""Find the center of the passable walking area between two ticket gates in frame 0. the frame 1 show the example
                The point must:
                - be on the floor
                - be in empty space
                - be exactly between two gate barriers

                The point must NOT:
                - be on the gate structure
                - be on metal or glass
            """

            t1 = time.time()
            rynn_8b = rynn_model.inference(
                instruction,
                [orig_path, 'cookbooks/assets/elevator/example.png'],
                # [orig_path, 'visual_result/follow/box_rgb_raw_000007.png'],
                save_dir = f"visual_result/gate/target_{filename}", 
                task="affordance_location",
                plot=True,
                do_sample=False
            )
            t2 = time.time()
            print(f"time: {t2-t1}")
            count+=1
            # t1 = time.time()
            # if 'yes' in rynn_8b:
            #     instruction = f'Identify the target(labelled with bounding box in frame 1) in frame 2'
            #     rynn_8b = rynn_model.inference(
            #         instruction,
            #         [ , orig_path],
            #         save_dir = f"visual_result/Object/{filename}", 
            #         task="affordance_location",
            #         # task = "object_location",
            #         plot=True,
            #         do_sample=False
            #     )

            # t2 = time.time()
            # print(f"time: {t2-t1}")
            print(f"Prediction for {filename}:\n{rynn_8b}\n")
            # break
