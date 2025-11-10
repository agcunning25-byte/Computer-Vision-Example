# HelmNet: Deep Learning for Workplace Safety Helmet Detection

## Project Overview & Business Problem

This project develops and evaluates a deep learning model, "HelmNet," to enhance worker safety by automatically detecting the presence of safety helmets in images.

**Business Context:** In hazardous environments like construction sites and industrial plants, non-compliance with safety helmet regulations is a major risk, leading to serious injuries or fatalities. Manual safety monitoring is often inefficient and prone to error, especially in large-scale operations.

**Objective:** The goal is to develop and deploy a robust image classification model for **SafeGuard Corp** that can accurately classify images into two categories: **With Helmet** or **Without Helmet**. This automated system aims to improve safety compliance, reduce accidents, and enhance monitoring efficiency.

## Dataset

The analysis was performed on a dataset of **631 images** (200x200 pixels, RGB) split into two balanced classes:
* **With Helmet:** 311 images
* **Without Helmet:** 320 images

The dataset features variations in lighting, angles, and worker activities (standing, using tools, etc.) to simulate real-world conditions.

## Machine Learning Workflow & Skills

This project demonstrates an end-to-end deep learning pipeline, from data preprocessing to model comparison and critical analysis.

### 1. Data Preprocessing
* **Data Splitting:** The 631 images were strategically divided into a **65% Training Set (410 images)**, a **17% Validation Set (110 images)**, and a **17% Test Set (111 images)** using `train_test_split` with stratification to maintain class balance.
* **Normalization:** All image pixel values were scaled from the 0-255 range to a 0-1 range by dividing by 255.0, a standard practice for neural networks.

### 2. Model Development & Comparative Analysis
To identify the most effective and efficient model, four different architectures were built and evaluated:

* **Model 1: Simple CNN (Baseline)**
    * A foundational CNN was built from scratch using `Sequential`, `Conv2D`, `MaxPooling2D`, and `Dense` layers.
    * This model established a baseline performance of **99.09% validation accuracy**.

* **Model 2: Transfer Learning (VGG-16 Base)**
    * Implemented transfer learning by using the pre-trained **VGG-16** model as a convolutional base.
    * All VGG-16 layers were frozen (`layer.trainable = False`) so only the final `Dense` output layer was trained.
    * This model achieved **100% validation accuracy**, demonstrating the power of pre-trained features even with minimal trainable parameters (~18k).

* **Model 3: Fine-Tuning (VGG-16 + FFNN Head)**
    * This model improved upon the VGG-16 base by adding a custom Feed-Forward Neural Network (FFNN) head, consisting of `Dense(16)`, `Dropout(0.5)`, and `Dense(8)` layers.
    * The `Dropout` layer was added to prevent overfitting, a key consideration for small datasets.
    * This model also achieved **100% validation accuracy**.

* **Model 4: Data Augmentation**
    * To further improve robustness and prevent overfitting, Model 3 was trained using `ImageDataGenerator` for on-the-fly data augmentation.
    * Augmentations included `horizontal_flip`, `height_shift`, `width_shift`, `rotation`, `shear`, and `zoom`.
    * This model also achieved **100% validation accuracy**, showing strong generalization.

## Model Performance

**Accuracy** was chosen as the primary evaluation metric given the balanced dataset. All models performed exceptionally well, with the VGG-16-based models achieving perfect accuracy on the validation set.

| Model | Train Accuracy | Validation Accuracy | Train-Val Difference |
| :--- | :--- | :--- | :--- |
| 1. Simple CNN | 98.54% | 99.09% | -0.0055 |
| 2. VGG-16 (Base) | 100.00% | 100.00% | 0.0 |
| 3. **VGG-16 + FFNN (Selected)** | **100.00%** | **100.00%** | **0.0** |
| 4. VGG-16 + FFNN + Augment | 92.93% | 100.00% | -0.0707 |

**Model 3 (VGG-16 + FFNN)** was selected as the final model. It was then evaluated on the unseen **Test Set**, achieving a perfect **100% accuracy**.


## Critical Analysis & Recommendations

While the 100% test accuracy is an outstanding result, a critical review of the data revealed a potential source of bias:
* **Data Bias:** The "With Helmet" images primarily show workers in industrial settings, while the "Without Helmet" images are mostly from recreational settings. The model may have learned to distinguish *settings* (industrial vs. recreational) rather than exclusively focusing on the *helmet*.

This insight is crucial for real-world deployment.

### Recommendations for SafeGuard Corp
1.  **Deploy Model 3:** Deploy the VGG-16 + FFNN model for initial use, as it is highly accurate on the available data.
2.  **Urgent Data Collection:** Before full reliance, gather a more representative test set, specifically images of **workers in industrial settings *without* helmets**. This is critical to confirm the model is generalizing correctly.
3.  **Retrain with New Data:** Retrain the model with this new, more diverse dataset to ensure it is robust against setting-related bias.
4.  **Maintain Human Oversight:** AI should be used as an assistive tool, not a complete replacement. Randomized human safety checks should continue to ensure worker safety against scenarios the AI may not cover.

## Technologies Used

* **Python**
* **TensorFlow & Keras** (for building, training, and evaluating all neural networks)
* **Scikit-learn** (for data splitting and performance metrics)
* **Pandas & NumPy** (for data loading and manipulation)
* **Matplotlib & Seaborn** (for data visualization and plotting accuracy)
* **OpenCV (cv2)** (for image processing)
