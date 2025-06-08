import streamlit as st
import numpy as np
import torch
import pydicom
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from pydicom.pixel_data_handlers.util import apply_voi_lut
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DetectionModel
import time

# Page Configuration
st.set_page_config(page_title="YOLOv10 DICOM Predictor", layout="wide")
st.title("🩻 YOLOv10 DICOM Predictor")
st.markdown("Upload a `.dcm` DICOM file to detect abnormalities using a YOLOv10 model.")

# File Uploader
uploaded_file = st.file_uploader("📤 Upload DICOM file", type=["dcm"])

if uploaded_file is not None:
    progress = st.progress(0, text="🔄 Starting...")

    # Step 1: Load DICOM image
    progress.progress(10, "📥 Reading DICOM file...")
    #dicom = pydicom.dcmread(BytesIO(uploaded_file.read()))

    ##########################################################################
    import os

    os.chdir(r'F:/project')

    folder_path = './yolov10_drive/predictions'  # Change this to your target folder

    # Delete all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    ######
    import random
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    import os
    import io
    import numpy as np
    import cv2
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    from PIL import Image

    from pydicom.uid import ExplicitVRLittleEndian
    dicom = pydicom.dcmread(BytesIO(uploaded_file.read()))
    pixel_array = apply_voi_lut(dicom.pixel_array, dicom)

    # Step 2: Handle MONOCHROME1 and normalize
    progress.progress(25, "🧪 Preprocessing image...")
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = np.max(pixel_array) - pixel_array
    if dicom.PhotometricInterpretation == "MONOCHROME2":
        pixel_array = np.max(pixel_array) - pixel_array
    # Normalize pixel values to 0-255
    img=pixel_array
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img * 255).astype(np.uint8)
    TARGET_SIZE=1024
            # Convert to 3-channel RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h0, w0 = img.shape[:2]

            # Resize
    img_resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))

            # Save as PNG
    output_path = os.path.join("./images/input.png")
    cv2.imwrite(output_path, img_resized)



    #####
    import torch
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.tasks import DetectionModel

    # Allow YOLO custom modules (required by PyTorch 2.6+)
    torch.serialization.add_safe_globals([
        Conv,
        DetectionModel
    ])

    # Import the YOLO model class
    #from yolov10.ultralytics.models.yolo import YOLO
    import ultralytics
    from ultralytics import YOLO
    # Load model
    model = YOLO("./my_yv10_5m/weights/best.pt")
    # Run inference
    results = model.predict(
        source="./images/input.png",
        imgsz=640,
        conf=0.1,
        iou=0.45,
        save=True,
        project="yolov10_drive",
        name="predictions",
        exist_ok=True
    )

    # Process predictions
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls.item())             # class index
            score = float(box.conf.item())           # confidence score
            label = result.names[cls_id]             # class name

            xyxy = box.xyxy[0].tolist()              # bounding box [x1, y1, x2, y2]

            print(f"Label: {label}, Confidence: {score:.2f}, BBox: {xyxy}")
    # print maximum confidence label and score
    # Process predictions
    for result in results:
        max_conf = -1
        max_label = None
        max_xyxy = None

        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls.item())
            score = float(box.conf.item())
            label = result.names[cls_id]
            xyxy = box.xyxy[0].tolist()

            if score > max_conf:
                max_conf = score
                max_label = label
                max_xyxy = xyxy

        if max_label is not None:
            print('')

            #print(f"Max Label: {max_label}, Confidence: {max_conf:.2f}, BBox: {max_xyxy}")

    ######## display the predicted image

    # Load image
    img = cv2.imread("./yolov10_drive/predictions/input.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display
    plt.imshow(img)
    plt.title("Predicted Image")
    plt.axis("off")
    #plt.show()
    ############### feature extraction
    import feature_extraction
    from feature_extraction import fec
    fec()
    #################3 ML model prediction
    import ens_modelling_5m_test
    from ens_modelling_5m_test import ens
    Final_prediction,predicted_value1,probabilities=ens()

    Risk_level="Low"
    if  predicted_value1!=2:
        Risk_level = "High"
    imp=""
    if  predicted_value1!=0:
        imp=" The Patient may have COPD"
    if  predicted_value1!=1:
        imp=" The Patient may have Lung Cancer"
    if  predicted_value1!=2:
        imp=" The Patient may have Normal"
    if  predicted_value1!=3:
        imp=" The Patient may have TB"
    if  predicted_value1!=4:
        imp=" The Patient may have Silicosis"


    #### change bounding box as per ML
    # Load and resize input image
    input_path = "./images/input.png"
    image = cv2.imread(input_path)
    resized_image = image#cv2.resize(image, (640, 640))

    # Define fixed label and prepare colors
    label = str(Final_prediction)
    color_list = []

    # Draw bounding boxes
    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            score = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)

            # Generate a random color for this box
            color = tuple(random.randint(0, 255) for _ in range(3))
            color_list.append(color)

            # Draw the box
            cv2.rectangle(resized_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(resized_image, f"{label} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save image
    cv2.imwrite("output_with_boxes_colored.jpg", resized_image)
    print("Saved image with colorful boxes and label  → output_with_boxes_colored.jpg")


    """ predictions yoloV10 """
    Predicted_class_DL=max_label     # class
    max_confidence_DL=f"{max_conf:.2f}"   # confidence score
    """ predictions ML"""
    Predicted_class_ML=str(Final_prediction) # class
    max_confidence_ML=f"{probabilities:.2f}"# confidence score
    Risk_level=Risk_level     # rish level
    impression=imp    ### impression text
    """ Patient information"""
    Patient_ID=uploaded_file.name
    Patient_Name="NA"
    Patient_Age="NA"
    Patient_Sex="NA"
    
    ##########################################################

    st.image(resized_image, caption="🖼️ Annotated Image", use_container_width=True)
