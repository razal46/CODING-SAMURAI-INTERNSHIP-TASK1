# Facial Expression Recognition Web Application

## **1. Project Overview**
This project focuses on developing an AI-powered **Facial Expression Recognition Web Application** using **Deep Learning** and **Streamlit**. The application classifies human emotions based on facial images using a **pretrained MobileNet model**. The goal is to create a user-friendly interface where users can upload an image, and the system will predict the corresponding facial expression.

## **2. Technologies Used**
- **Programming Language:** Python
- **Libraries & Frameworks:**
  - TensorFlow/Keras (Deep Learning Model)
  - OpenCV (Image Processing)
  - NumPy, Pandas, Matplotlib (Data Handling & Visualization)
  - Streamlit (Web UI Framework)
- **Tools:**
  - VS Code (Development Environment)
  - Git & GitHub (Version Control)

## **3. Dataset**
The dataset used for training consists of facial expression images categorized into seven emotions:
- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**
- **Surprise**

## **4. Model Development**
### **Step 1: Data Preprocessing**
- Loaded images and converted them to **grayscale**.
- Resized images to **48x48** pixels for consistency.
- Applied **data augmentation** techniques like flipping and rotation.
- Normalized pixel values to **[0,1] range** to improve training stability.
- Converted labels into categorical format using **one-hot encoding**.

### **Step 2: Model Selection & Training**
- Used **MobileNetV2** as the feature extractor.
- Added **fully connected Dense layers** and a **Softmax activation function** for classification.
- Configured the model with **Categorical Crossentropy Loss** and **Adam Optimizer**.
- Trained the model using a batch size of **32** and **20 epochs**.
- Achieved a test accuracy of **40.96%**, indicating room for improvement with additional data and hyperparameter tuning.

## **5. Web Application Development**
### **Step 1: Setting Up Streamlit**
- Installed required libraries using:
  ```bash
  pip install streamlit opencv-python-headless tensorflow numpy matplotlib
  ```
- Created `app.py` to load the trained model, process user-uploaded images, and make predictions.

### **Step 2: User Interface (UI) Design**
- Developed a clean UI using **Streamlit**.
- Allows users to **upload an image** for emotion prediction.
- Displays the **predicted emotion label** along with the image.
- Used **Matplotlib and OpenCV** for image visualization.

### **Step 3: Image Processing & Prediction**
- Reads uploaded image using **OpenCV**.
- Resizes image to **48x48** and normalizes pixel values.
- Converts the image into a **batch format** to feed into the model.
- Model predicts the emotion, and the result is displayed on the UI.

## **6. Deployment**
### **Step 1: Hosting on GitHub**
- Uploaded all project files to a **GitHub repository**.

### **Step 2: Running Locally**
- Clone the repository and navigate to the project folder:
  ```bash
  git clone https://github.com/your-username/facial-expression-recognition.git
  cd facial-expression-recognition
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Run the application:
  ```bash
  streamlit run app.py
  ```

### **Step 3: Future Deployment Options**
- Deploy on **Streamlit Sharing** for online accessibility.
- Host using **Hugging Face Spaces** or **Render** for wider distribution.

## **7. Project Folder Structure**
```
📂 facial_expression_recognition_project
 ├── 📂 dataset (if applicable)
 ├── 📜 app.py  # Main Streamlit application
 ├── 📜 model.py  # Model training script
 ├── 📜 requirements.txt  # Dependencies
 ├── 📜 emotion_recognition_mobilenet.keras  # Trained model
 ├── 📜 README.md  # Documentation
```

## **8. Future Improvements**
- Train on a **larger dataset** for higher accuracy.
- Optimize the model with **hyperparameter tuning**.
- Implement **real-time emotion detection** using webcam input.
- Improve UI/UX with better **design elements and responsiveness**.
- Deploy the application on **cloud platforms** for accessibility.

## **9. Conclusion**
This project successfully implements an **AI-based emotion detection system** using deep learning. The integration of **Streamlit** provides a user-friendly interface for interacting with the model. Future enhancements will focus on **improving accuracy**, **adding real-time detection**, and **deploying the app for wider use**.

