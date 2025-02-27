# **Distracted Driver Detection**

## 🚗 Overview
This project focuses on detecting distracted driving behaviors using machine learning and deep learning techniques. The primary objective is to enhance road safety by identifying risky driver activities such as:

- Texting while driving 📱
- Talking on the phone 📞
- Drinking or eating 🍔🥤
- Interacting with passengers 🗣️
- Adjusting the radio 🎵
- Other distracted behaviors 🚦

## 📂 Dataset
- The dataset for this project is sourced from **Kaggle**.
- It consists of **labeled images of drivers** engaged in different activities while driving.
- The dataset is categorized into **10 distinct classes**, each representing a unique driving behavior.

## 🏗️ Model Architecture & Approach
This project employs a combination of **deep learning** and **machine learning** techniques:

### 🔍 Feature Extraction
- **Convolutional Neural Network (CNN)**: Used to extract meaningful visual features from images.
- **Principal Component Analysis (PCA)**: Applied for dimensionality reduction to improve computational efficiency.

### 📊 Classification Algorithms
- **Support Vector Machine (SVM)** ✔️ (Best performer)
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**

Each classifier was evaluated **with and without PCA** to determine its impact on performance.

## ⚙️ Training Process
1. **Preprocessing**:
   - Images resized to **100x100 pixels** 📏
   - Pixel values normalized 🖼️
2. **Feature Extraction**:
   - CNN model used to extract deep features 🧠
3. **Classification**:
   - SVM, KNN, and Random Forest models trained on extracted features 🔎
4. **PCA Implementation**:
   - Dimensionality reduction applied, followed by re-training ⚡
5. **Performance Evaluation**:
   - Metrics such as **accuracy, confusion matrix, and classification report** used 📊

## 🏆 Results & Analysis
| Model | Without PCA | With PCA |
|--------|-------------|------------|
| **SVM** | ⭐ High Accuracy | ✅ Best Performance |
| **KNN** | Moderate | Slight Improvement |
| **Random Forest** | Moderate | No Significant Change |

- **SVM with PCA** provided the best results, balancing accuracy and computational efficiency.
- **KNN & Random Forest** performed reasonably well but were outperformed by SVM.
- **PCA** enhanced computational efficiency while maintaining high accuracy.

## 🚀 Future Enhancements
🔹 **Fine-tune CNN** for better feature extraction 🛠️  
🔹 **Experiment with deep learning models** like ResNet or MobileNet 📡  
🔹 **Implement real-time detection** on edge devices 🎥  
🔹 **Expand dataset diversity** for better generalization 🌍  

## 🏁 Conclusion
This project demonstrates the **effectiveness of combining CNN-based feature extraction with classical machine learning algorithms** for detecting distracted driving behaviors. The **SVM classifier with PCA** emerged as the best performer, achieving the highest accuracy while improving computational efficiency.

## 🙌 Acknowledgments
A special thanks to **Kaggle** for providing the dataset used in this research. 🤝

