# food101-image-classification
Food-101 Transfer Learning project using EfficientNet-B0, PyTorch, and Streamlit (with camera support).

# 🍽️ **Food-101 Image Classification using Transfer Learning**

### *EfficientNet-B0 · PyTorch · Streamlit App with Camera Support*

---

## 📌 **Project Overview**

This project implements **fine-grained food classification** using the **Food-101 dataset** and **Transfer Learning with EfficientNet-B0**.
The system classifies images into **101 food categories** such as pizza, sushi, nachos, ramen, etc.

The project includes:

* Complete training pipeline
* EDA and visualization
* Model evaluation (Accuracy, Macro F1-score, Confusion Matrix)
* Saved model checkpoint
* A modern **Streamlit web app** with:

  * Image upload
  * **Camera input support**
  * Top-5 predictions
  * Glassmorphic UI

---

## 📂 **Project Structure**

```
food101_project/
│── app.py                     # Streamlit application
│── train_food101.ipynb        # Training notebook
│── checkpoints/
│     └── best_model.pth       # Saved best model
│── data/
│     └── food-101/
│           ├── images/        # All images
│           └── meta/          # train/test split, class names
│── reports/
│     └── confusion_matrix.png # Evaluation visualization
│── requirements.txt
│── README.md
```

---

## 📊 **Model Performance**

| Metric                  | Value      |
| ----------------------- | ---------- |
| **Train Accuracy**      | **89.58%** |
| **Validation Accuracy** | **74.34%** |
| **Macro F1-Score**      | **0.7416** |

The best model is saved at:

```
checkpoints/best_model.pth
```

A confusion matrix is saved at:

```
reports/confusion_matrix.png
```

---

## 🧠 **Model Details**

* Architecture: **EfficientNet-B0**
* Framework: **PyTorch**
* Pretrained: Yes (on ImageNet)
* Final layer replaced with:
  `nn.Linear(1280, 101)`
* Optimizer: **AdamW**
* Loss: **CrossEntropyLoss**
* Image Size: **224 × 224**
* Augmentations:

  * Horizontal Flip
  * Color Jitter
  * Normalization

---

## 🔧 **Installation & Setup**

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd food101_project
```

### 2️⃣ Create and Activate Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Download the Food-101 Dataset

Download from official link:

[https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

Extract and place inside:

```
data/food-101/
```

Ensure the structure matches:

```
data/food-101/images/
data/food-101/meta/train.txt
data/food-101/meta/test.txt
data/food-101/meta/classes.txt
```

---

## 🏋️ **Training the Model**

Use the provided Jupyter Notebook:

```bash
train_food101.ipynb
```

This notebook performs:

* EDA
* Data loading
* Augmentations
* Transfer learning
* Validation metrics
* Model checkpoint saving

---

## 🌐 **Running the Streamlit App**

Start the web app with:

```bash
streamlit run app.py
```

### Features:

* 📤 Upload an image
* 📷 Use your webcam/camera
* 🍛 Predict the food dish
* 📈 See top-5 predictions with confidence
* 💎 Modern glassmorphic UI

---

## 📸 **Demo Workflow (What the App Does)**

1. User uploads or captures an image
2. Image is resized and normalized
3. EfficientNet-B0 performs inference
4. App shows:

   * Predicted dish
   * Confidence score
   * Top-5 predictions
   * Original image preview

---

## 🧪 **Evaluation Tools**

* Macro F1-score
* Confusion Matrix
* Accuracy
* Class-wise predictions

---

## 🚀 **Future Improvements**

* Add **Grad-CAM** visualization to show model attention
* Train on the **full Food-101 dataset**
* Try models like **EfficientNet-B3** or **Vision Transformers**
* Add mobile support
* Deploy on **Streamlit Cloud** or **HuggingFace Spaces**
* Add calorie estimation

---

## ✨ **Author**

Your Name
Computer Vision Final Project
Food-101 Transfer Learning

---



👉 **“Give me requirements.txt”**
