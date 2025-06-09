## 🫁 PulmoVista v1.0, AI-Powered Medical Imaging Diagnostics for Lung Conditions

## 🚀 Overview

PulmoVista is a Streamlit web application deployed on Streamlit Cloud that detects lung abnormalities from DICOM images using 
a YOLOv10 trained model and Machine learning ensemble models for feature engineering and classification on medical imaging data.

## 📁 Project Structure

├── Single_page_file_pil.py &nbsp;&nbsp;&nbsp;&nbsp;    # main script Streamlit app

├── yolo_5cl.py   &nbsp;&nbsp;&nbsp;&nbsp; # Yolov10 model for 5 class detection

├──best.py   &nbsp;&nbsp;&nbsp;&nbsp; # Best model for yolo10

├── input_feature_unlabeled.csv    &nbsp;&nbsp;&nbsp;&nbsp; # contains features extracted from the yolov10 model

├──feature_extraction.py    &nbsp;&nbsp;&nbsp;&nbsp; # script for extracting best features from Deep Learning model using various ML feature selection algorithms

├── selected_models/    &nbsp;&nbsp;&nbsp;&nbsp; # This folder contains all the machine learning models used for ensembling

├── ens_modelling_5m_test.py    &nbsp;&nbsp;&nbsp;&nbsp; # script for ensembling machine learning models

├── stacked_ensemble_model_ML_5m_cl_F.pkl    &nbsp;&nbsp;&nbsp;&nbsp; # Combined model of Yolov10 + ensembled ML models

├── selected_models/    &nbsp;&nbsp;&nbsp;&nbsp; # This folder contains sample images for testing

├── requirements.txt    &nbsp;&nbsp;&nbsp;&nbsp; # Required packages

└── README.md    &nbsp;&nbsp;&nbsp;&nbsp;# Project documentation


## 🔧 Installation

1. Clone the repository to download and view the project files
   
   git clone https://github.com/diagno-intelligent/Diagnostics_Diagno_Intelliegent_Systems.git

## 💻 Usage
2. Click the Streamlit link given below to run the project

https://diagnosticsdiagnointelliegentsystems-m7lwvt4hxbsnhlsnzrgrxn.streamlit.app/

3. Steps for Analysis
   
   a) click browse files to upload Dicom Xray image
   
   b) Once image loaded, click 'Analyze Image' for image analysis and prediction
   
   c) The progress bar in the 'Analysis Image" section indicates the status of image processing
   
   d) Once the status is "finalizing the results", the Xray image with predicted class will be displayed
   
   e) 'View Report' displays the report 
   
   f) Download button displays a "Download Report button" which when clicked downloads the report
   
   e) 'New Analysis' clears the output section and new image analysis can be done again
   

## Acknowledgments

Telangana Challenge Team for providing DICOM Chest Xray image dataset annotated with bounding boxes for 5 lung conditions
 
Open-source contributors - Ultralytics YOLO



