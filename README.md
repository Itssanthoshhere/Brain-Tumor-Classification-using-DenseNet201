# 🧠 Brain Tumor MRI Classification Using DenseNet201: A Deep Learning Approach 🚀

---

## 🌟 Overview

This project implements a deep learning model using the DenseNet201 architecture to classify brain tumors from MRI images. The model is trained on a labeled dataset that includes four classes: **no tumor**, **pituitary tumor**, **meningioma tumor**, and **glioma tumor**. The project involves data preprocessing, augmentation, and training a convolutional neural network to achieve high classification accuracy.

What is a brain tumor?
A brain tumor is a collection, or mass, of abnormal cells in your brain. Your skull, which encloses your brain, is very rigid. Any growth inside such a restricted space can cause problems. Brain tumors can be cancerous (malignant) or noncancerous (benign). When benign or malignant tumors grow, they can cause the pressure inside your skull to increase. This can cause brain damage, and it can be life-threatening.



## 📚 Table of Contents

- [📦 Libraries Used](#libraries-used)
- [🗂 Dataset](#dataset)
- [📝 Steps](#steps)
- [🏗 Model Architecture](#model-architecture)
- [📊 Results](#results)
- [💡 Acknowledgements](#acknowledgements)
- [🔍 Check out the code](#check-out-the-code)

## 📦 Libraries Used

- **🔢 NumPy**: Numerical operations and handling arrays
- **📊 Pandas**: Data manipulation and analysis
- **📈 Matplotlib**: Data visualization
- **🤖 TensorFlow**: Deep learning framework
- **🧬 Keras**: High-level API for TensorFlow
- **🔍 Scikit-Learn**: Model evaluation and data splitting

## 🗂 Dataset

We’re working with the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) – a treasure of brain tumor MRI dataset! 

The dataset used in this project contains MRI images categorized into four classes:
1. **🧠 No Tumor**
2. **🧠 Pituitary Tumor**
3. **🧠 Meningioma Tumor**
4. **🧠 Glioma Tumor**

### 📁 File Paths
- **📂 Training Path**: `D:/Skill VERTEX/Artificial Intelligence Program/Datasets/Brain Tumor/Training`
- **📂 Testing Path**: `D:/Skill VERTEX/Artificial Intelligence Program/Datasets/Brain Tumor/Testing`

The images are preprocessed and augmented before being fed into the model for training.

## 📝 Steps

1. **📥 Import Necessary Libraries**: Load all the required libraries for data processing, visualization, and deep learning.
2. **📂 Define File Paths**: Specify the paths for training and testing datasets.
3. **🗃 Initialize Data Storage**: Create empty lists to store filenames and data.
4. **🖼 Load and Preprocess Data**: Load images, normalize, and preprocess them into training and testing sets.
5. **🚀 Model Training**: Use DenseNet201 pre-trained on ImageNet as the base model, add custom layers, and train the model using data augmentation.
6. **🧪 Model Evaluation**: Evaluate the model using validation data and generate classification reports.
7. **📉 Results Visualization**: Plot training and validation accuracy/loss over epochs.
8. **🔍 Test the Model on a Sample Image**: Load a sample image, predict its class, and visualize the result.

## 🏗 Model Architecture

- **🧠 Base Model**: DenseNet201 (pre-trained on ImageNet)
- **🔧 Custom Layers**:
  - GlobalAveragePooling2D
  - Dense layer with 128 units and ReLU activation
  - Output layer with 4 units (softmax activation)

The model is compiled with the Adam optimizer and categorical crossentropy loss, and is trained for 30 epochs.

## 📊 Results

The model is evaluated on the testing dataset using accuracy, confusion matrix, and classification reports. The training and validation accuracy/loss are visualized to understand the model's performance.

## 💡 Acknowledgements

- **📄 Dataset**: The brain tumor MRI images were sourced from a labeled dataset designed for classification tasks.
- **🛠 Libraries**: The project utilizes powerful libraries like TensorFlow, Keras, Pandas, NumPy, and Matplotlib.
- **💭 Inspiration**: This project is inspired by various medical imaging and deep learning research efforts aimed at improving diagnostic accuracy.

## 🔍 Check out the code

Explore the complete code and all project details on my GitHub: [Brain Tumor MRI Classification Project](https://github.com/Itssanthoshhere/Brain-Tumor-MRI-Classification-using-DenseNet201). 

## 👨‍💻 Author

- **Santhosh VS** - [LinkedIn Profile](https://www.linkedin.com/in/thesanthoshvs/)

## 📧 Contact

For any questions or feedback, feel free to reach out at [santhosh02vs@gmail.com](mailto:santhosh02vs@gmail.com).

---
