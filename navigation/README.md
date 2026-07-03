## 🛠 Setup

We test under the following environment:
* Python 3.9
* Pytorch 2.7.0
* CUDA Version 12.8


1. **Preparing  a conda env with `Python3.9` & Install habitat-sim and habitat-lab**
    ```bash
    conda create -n RynnBrain-Nav python=3.9
    conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat
    git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
    cd habitat-lab
    pip install -e habitat-lab  # install habitat_lab
    pip install -e habitat-baselines # install habitat_baselines
    ```

2. **Clone this repository**
    ```bash
    git clone 'code'
    cd RynnBrain/navigation
    ```


## 📁 Data Preparation
To get started, you need to prepare three types of data:

1. **Scene Datasets**  
   - For **R2R**, **RxR** and **EnvDrop**: Download the MP3D scenes from the [official project page](https://niessner.github.io/Matterport/), and place them under `data/scene_datasets/mp3d/`.
   - For **ScaleVLN**: Download the HM3D scenes from the [official github page](https://github.com/matterport/habitat-matterport-3dresearch), and place the `train` split under `data/scene_datasets/hm3d/`

2. **VLN-CE Episodes**  
   Download the VLN-CE episodes:
   - [r2r](https://drive.google.com/file/d/18DCrNcpxESnps1IbXVjXSbGLDzcSOqzD/view) (Rename `R2R_VLNCE_v1/` -> `r2r/`)
   - [rxr](https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view) (Rename `RxR_VLNCE_v0/` -> `rxr/`)
   - [envdrop](https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view) (Rename `R2R_VLNCE_v1-3_preprocessed/envdrop/` -> `envdrop/`)
   - [scalevln](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data/blob/main/ScaleVLN/scalevln_subset_150k.json.gz) (This is a subset of the ScaleVLN dataset, converted to the VLN-CE format. For the original dataset, please refer to the [official repository](https://github.com/wz0919/ScaleVLN).)
  
   Extract them into the `data/datasets/` directory.

3. **Collected Trajectory Data**  
  We provide pre-collected observation-action trajectory data for training. These trajectories were collected using the **training episodes** from **R2R** and **RxR** under the Matterport3D environment. For the **EnvDrop** and **ScaleVLN** subset, please refer to [here](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data/blob/main/README.md) for instructions on how to collect it yourself.
  Download the observation-action trajectory data from [Hugging Face](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data), and extract it to `data/trajectory_data/`.

<!-- 4. **Co-training Data Preparation**

    Download the respective datasets from their official sources and place them in the `data/co-training_data/`.

    - LLaVA-Video-178K: Available on Hugging Face at [lmms-lab/LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K).

    - ScanNet: 

      - The main dataset can be downloaded from the [official GitHub repository](https://github.com/ScanNet/ScanNet).

      - Download the annotation files `scanqa_annotations.json` and `sqa3d_annotations.json` from [here](https://huggingface.co/datasets/chchnii/StreamVLN-ScanQA-SQA3D-Data). These files are subsets of the [LLaVA-3D-DATA](https://huggingface.co/datasets/ChaimZhu/LLaVA-3D-Data).


    - MMC4-core: Available via the [official GitHub repository](https://github.com/allenai/mmc4). -->

Your final folder structure should look like this:

```bash
data/
├── datasets/
│   ├── r2r/
│   │   ├── train/
│   │   ├── val_seen/
│   │   │   └── val_seen.json.gz
│   │   └── val_unseen/
│   │       └── val_unseen.json.gz
│   ├── rxr/
│   │   ├── train/
│   │   ├── val_seen/
│   │   │   ├── val_seen_guide.json.gz
│   │   │   └── ...
│   │   └── val_unseen/
│   │       ├── val_unseen_guide.json.gz
│   │       └── ...
│   ├── envdrop/
│   │   ├── envdrop.json.gz
│   │   └── ...
│   └── scalevln/
│       └── scalevln_subset_150k.json.gz
├── scene_datasets/
│   └── hm3d/
│       ├── 00000-kfPV7w3FaU5/
│       ├── 00001-UVdNNRcVyV1/
│       └── ...
│   └── mp3d/
│       ├── 17DRP5sb8fy/
│       ├── 1LXtFkjw3qL/
│       └── ...
├── trajectory_data/
│   ├── R2R/
│   │   ├── images/
│   │   └── annotations.json
│   ├── RxR/
│   │   ├── images/
│   │   └── annotations.json
│   ├── EnvDrop/
│   │   ├── images/
│   │   └── annotations.json
│   └── ScaleVLN/
│       ├── images/
│       └── annotations.json
├── dagger_data/
│   ├── R2R/
│   │   ├── images/
│   │   └── annotations.json
│   ├── RxR/
│   │   ├── images/
│   │   └── annotations.json
│   └── EnvDrop/
│       ├── images/
│       └── annotations.json
```

4. **Convert To RynnBrain Format** 

* The format of the input data:
```json
[
  {
    "id": 0,
    "video": "images/test",
    "instructions": [
      "Your task is to walk past the bed and into the  bedroom . walk past the bed and stop in the doorway ."
    ],
    "actions": [
      -1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      3
    ]
  }
]
```

* The format of the target data
```json
    {   "id": "0", 
        "data_source": "", 
        "conversations": [
            {"role": "user", 
            "content": 
                [
                    {"type": "text", "text": "You are an autonomous navigation assistant. Your task is to walk past the bed and into the  bedroom . walk past the bed and stop in the doorway . Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←), TURN RIGHT (→), MOVE FORWARD (↑), or STOP."}, 
                    {"type": "text", "text": " This is your current view: "}, 
                    {"type": "image", "image": "image0.jpg"}
                ]},
            {"role": "assistant", 
            "content": 
                [
                    {"type": "text", "text": "←←←←"}
                ]
            },
            {"role": "user", 
            "content": 
                [
                    {"type": "text", "text": " you can spot "}, 
                    {"type": "image", "image": "image4.jpg"}
                ]
            },
            {"role": "assistant", 
            "content": 
                [
                    {"type": "text", "text": "←←←→"}
                ]
            } 
        ]
    }
```

Use the following script to convert the original StreamVLN data to our required format:

```bash
    bash scripts/rynnvln_trajectory_generation.sh
```

## 🚀 Training

First, prepare your data according to the specified format. Then, execute the provided training scripts to start training.




## 💻 Inference
Follow the visual language navigation cookbook to implement the inference [vln_inference](../cookbooks/11_visual_language_navigation.ipynb).


## 🤖 Evaluation


### Standard Model Evaluation

To perform multi-GPU evaluation of 2B, 4B and 8B model, just run

```bash
sh scripts/rynnvln_eval.sh
```

### Mixture-of-Experts (MoE) Model Evaluation

To perform multi-GPU evluation of MOE model, just run

```bash
sh scripts/rynnvln_moe_eval.sh
```

### Dagger Collection
To perform multi-GPU dagger data collection, just run
```bash
sh scripts/rynnvln_dagger.sh
```

## 👏 Acknowledgements

Thanks to the great work of [StreamVLN](https://github.com/InternRobotics/StreamVLN).
