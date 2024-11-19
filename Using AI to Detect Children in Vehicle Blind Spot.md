# **OverView of the Project**

+ **Opening background information** 

In recent advancements in traffic safety technologies, AI-based detection systems to prevent accidents in vehicle blind spots have gained significant attention. Particularly, the growing prevalence of SUVs and large trucks has expanded rear blind spots, leading to frequent accidents where children are entirely obscured from the driver's view. To address this issue, real-time object detection technology powered by AI can be crucial in effectively identifying children and preventing accidents in vehicle blind spots.

+ **General description of the current project**

This project aims to develop a child detection system for vehicle blind spots using deep learning technology and YOLOv5. Specifically, it seeks to analyze the vehicle's surrounding environment in real time through the YOLOv5 model, detecting children in rear blind spots and alerting drivers to prevent accidents.

+ **Proposed idea for enhancements to the project**

**1. Real-time detection performance of YOLOv5**
YOLOv5 is a lightweight and efficient model optimized for real-time detection. In particular, it excels in detecting small objects, such as children in vehicle blind spots, ensuring quick and accurate detection in critical scenarios.

**2. Enhanced safety and applicability**
This system combines the vehicle's rear camera with real-time detection to provide immediate warnings, effectively preventing accidents. With simple settings and high flexibility, it can be easily applied to various vehicle models, contributing to improved overall traffic safety.

+ **Value and Significance of this Project**

**Combined with Traffic Safety and Technology Innovation**

Solving the problem of vehicle blind spots goes beyond social responsibility for protecting children and pedestrians; it contributes to enhancing the safety of future transportation systems. An AI-based sensing system not only addresses rear blind spot issues but also provides innovative solutions to improve overall vehicle safety. Through this, it plays an essential role in creating safer road environments and building sustainable traffic systems.


+ **Current limitations**

**Small Objects and Lack of Data**
Small objects, such as children, can be challenging to detect due to factors like lighting changes, movement, and complex environments. Moreover, if the data is biased or insufficient for certain conditions, the model may struggle to generalize effectively in real-world scenarios. A lack of data suited to diverse weather and road conditions further limits the model's ability to accurately recognize small objects. This can result in reduced detection performance and diminished effectiveness of the accident prevention system.

# **Image acquisition method**

I downloaded the dataset 'Vehicle Pedestrian_Video' from AI-Hub and obtained the video files.

**AI-Hub:**  https://www.aihub.or.kr/

**Drive** https://drive.google.com/drive/folders/1DJ1xIcWginzH-tG0hDcbvWzMc4a4Nqvw?usp=sharing

# **Learning Data Extraction and Learning Annotation**

To learn from YOLOv5 with 640 resolution images, we first made the images into 640 x 640 resolution images.

## **Video resolution adjustment**

**Purpose: Resizing a video**
[\[link\]](https://online-video-cutter.com/ko/resize-video)
![video_resize](https://github.com/user-attachments/assets/978aeb83-86c1-456b-b6b7-2316a4274d7e)

I used DarkLabel to extract frames from a 640 x 640 resolution video or to annotate them.
[\[DarkLabel\]](https://drive.google.com/drive/folders/1oZleCjucI_-86zFxZa8O0o3Jebxqi6ZR?usp=sharing)


Extract the files, open the DarkLabel.yaml file, and add the classes and format for the content to be extracted.

~~~
det_person: ["person"]

format9:    # darknet yolo (predefined format]
fixed_filetype: 1                 # if specified as true, save setting isn't changeable in GUI
data_fmt: [classid, ncx, ncy, nw, nh]
gt_file_ext: "txt"                 # if not specified, default setting is used
gt_merged: 0                    # if not specified, default setting is used
delimiter: " "                     # if not spedified, default delimiter(',') is used
classes_set: "det_person"     # if not specified, default setting is used
name: "person"           # if not specified, "[fmt%d] $data_fmt" is used as default format name

~~~
Class and format code

![DarkLabel](https://github.com/user-attachments/assets/ac4e98a8-5e1c-4d3e-ad77-daa942211f57)

**How to Extract Frames from a Video**

1. Click "Open Video" and select the video you want to extract frames from.

![DarkLabel3](https://github.com/user-attachments/assets/4771e4e3-a25d-405b-8206-0a0599ab4cbc)

2. Create a folder to save the images, and extract frames from the video into that folder.

![DarkLabel4](https://github.com/user-attachments/assets/ce4ef8aa-32f7-4649-96e4-d69b6f2d5023)

3. Check Box + Label and label the objects as "person."

![DarkLabel5](https://github.com/user-attachments/assets/6741b675-7284-48e8-843f-d0066b67c13c)

>[!WARNING]
> If you extract images first and try to label them without turning off 'Labeled Frame Only,' you may encounter an error.

## **Learning process**

+ **Setting the Path**

First, set the path to the folder containing the training files and verify it.
![prompt1](https://github.com/user-attachments/assets/435e1a8a-2ec0-4d5e-a65d-9dd2a0bdb9d3)

+ **Install YOLOv5**


After cloning YOLOv5 from GitHub, install the necessary packages for YOLOv5 training.

![prompt2](https://github.com/user-attachments/assets/741b750e-3501-4847-81fb-fbe42db2aa29)
![prompt3](https://github.com/user-attachments/assets/3699dc2d-f7d0-4899-ac08-7fc760cfc86d)

Create a folder in the specified path to store the training data.
~~~
mkdir -p Train/labels
mkdir -p Train/images
mkdir -p Val/labels
mkdir -p Val/images
~~~

Use the create_validation_set.py script to generate validation data, and then use the start_train.py script to train the model with the prepared data.

[create_validation_set.py]
~~~
import os
import shutil
from sklearn.model_selection import train_test_split

def create_validation_set(train_path, val_path, split_ratio=0.3):
    """
    Move a portion of the train data to validation
    """
    # Create necessary directories
    os.makedirs(os.path.join(val_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_path, 'labels'), exist_ok=True)

    # Get the list of train images
    train_images = os.listdir(os.path.join(train_path, 'images'))
    train_images = [f for f in train_images if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Split into train/validation
    _, val_images = train_test_split(train_images,
                                   test_size=split_ratio,
                                   random_state=42)

    # Copy files to the validation folder
    for image_file in val_images:
        # Copy image
        src_image = os.path.join(train_path, 'images', image_file)
        dst_image = os.path.join(val_path, 'images', image_file)
        shutil.copy2(src_image, dst_image)

        # Copy label file
        label_file = os.path.splitext(image_file)[0] + '.txt'
        src_label = os.path.join(train_path, 'labels', label_file)
        dst_label = os.path.join(val_path, 'labels', label_file)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)

    print(f"Created validation set with {len(val_images)} images")

# Run
train_path = 'D:\\AI\\yolov5\\Train'
val_path = 'D:\\AI\\yolov5\\Val'

create_validation_set(train_path, val_path)
~~~

[start_train.py]

~~~
python train.py --img 640 --batch 8 --epochs 200 --data D:\AI\yolov5\data.yaml --weights yolov5n.pt --cache
~~~

## **Validation and Results**

Use the start_detect.py script to perform validation.

[start_detect.py]

~~~
python detect.py --weights D:\AI\yolov5\runs\train\exp2\weights\best.pt --img 640 --conf 0.25 --source D:\video\verification\vmix.mp4
~~~

### **Result**

**[train_batch]**
![batchs](https://github.com/user-attachments/assets/6a9ee361-50cf-4780-a1c3-e33635d53fba)

**[curve]**
![curve](https://github.com/user-attachments/assets/49d36b1b-f28b-46c7-b873-a7be9570bce0)
![label](https://github.com/user-attachments/assets/7afb801f-3ea4-4a29-813d-141e9bfe46f0)
![results](https://github.com/user-attachments/assets/8a9dd660-1eef-488d-84ad-956a8c38f013)

**[Origin Video](https://drive.google.com/drive/folders/1g6iuVxCOrKzsLdIAehB_c90aZaQdf2Mf?usp=sharing)**

**[Validated Video](https://drive.google.com/drive/folders/16TRbOfAs2ROrzW8fowCZw3WN1Be-mQrt?usp=sharing)**

**Thank you for reviewing this report.** 
