# ChoufMoney: Algerian Currency Recognition Using CNNs

ChoufMoney is a local computer vision prototype designed to accurately recognize and classify Algerian currency—including both paper banknotes (500 DA, 1000 DA, 2000 DA) and solid metallic coins (50 DA, 100 DA, 200 DA). 

Built using **TensorFlow** and **Keras** on Linux (Fedora), this project transitions from traditional machine learning pixel-matching constraints to deep feature extraction using a Custom Convolutional Neural Network (CNN).

## 🚀 Key Features
* **Multi-Format Classification:** Robustly handles both textured 3D solid coins (heads/tails) and flat 2D paper bills (front/back).
* **On-the-Fly Data Augmentation:** Utilizes random rotations, zooms, and brightness shifts within the pipeline to mitigate dataset bias and prevent overfitting.
* **Instant Interactive Inference:** Features an optimized terminal loop script that loads the heavy `.keras` network weights into memory *once*, enabling sequential, near-instantaneous image classification (sub-150ms execution runtime).

## 🛠️ Tech Stack & Dependencies
* **OS:** Fedora Linux
* **Runtime:** Python 3.12 (Virtual Environment)
* **Core Frameworks:** TensorFlow / Keras
* **Libraries:** NumPy, Pillow (PIL), Matplotlib

## 📂 Dataset Architecture
The model trains on a balanced, local dataset structured into categorical directories:
dataset/
├── 50/     # 50 DA Coins
├── 100/    # 100 DA Coins
├── 200/    # 200 DA Coins
├── 500/    # 500 DA Bills
├── 1000/   # 1000 DA Bills (Scraped & Augmented)
└── 2000/   # 2000 DA Bills
