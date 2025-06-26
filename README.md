# Project description





### The yolo11s model was used:


<img src="https://github.com/user-attachments/assets/3a6add21-93d1-4eb8-aa7b-8534bcc37b0e" width="300" height="250" />

#### The model was exported using the following command:

```bash
yolo export model=_weights/yolo11s.pt format=onnx imgsz=640 nms=True conf=0.25 iou=0.45 

Move-Item -Path "_weights\yolo11s.onnx" -Destination "onnx_models\yolo11s_balanced.onnx"
```

### To reduce the Docker image size, some modules were simplified or replaced with lighter alternatives.
##### Examples:
+ opencv-python was replaced with opencv-python-headless

+ Two useful functions from the large ultralytics.utils package were reimplemented directly in the application

+ During development, some functions from the PIL module were replaced with equivalents from cv2, and the PIL package was removed

+ A slim version of Python was used in the Docker image

+ ...

## Design Decisions:

The letterbox function in preprocessing_utils significantly improves model confidence.
However, if the input images are already proportional to the exported model’s height and width, using this function increases the Docker image size.
Therefore, if the project is going to be released, the decision to use letterbox should be made based on the specific use case.

### Benefits of letterbox with an example:
#### An image passed to the model after letterbox processing looks like this.There’s no distortion while resizing, so the model’s decision is not negatively affected.
![1010](https://github.com/user-attachments/assets/6a17f9e9-6e83-4258-b596-cc042c739bd6)

#### Example of a directly resized image without letterbox:
<img width="640" alt="resized" src="https://github.com/user-attachments/assets/325d7548-ae93-45b5-a4f5-86ef35717c4d" />



### ...
