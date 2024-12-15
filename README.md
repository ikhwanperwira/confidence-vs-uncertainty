# Explainable AI: Confidence and Uncertainty

Welcome to the repository for the project **Explainable AI: Confidence and Uncertainty Based on Minimax and Relative Uniformity of Softmax Output**. This repository includes the code, resources, and documentation for analyzing the confidence and uncertainty of machine learning models using Softmax probabilities.

## Table of Contents

- [Overview](#overview)
- [Visualizations](#visualizations)
- [References](#references)

---

## Overview

This project explores:

- The limitations of using `argmax` in evaluating model confidence.
- New approaches for measuring uncertainty, such as **minimax** and **relative uniformity**.
- Methods for improving Explainable AI (XAI) by visualizing confidence and uncertainty.

Key highlights include:
- Evaluation on the MNIST dataset with experiments involving rotated samples.
- A focus on increasing interpretability of predictions.

---

## Visualizations

### MNIST Examples

![MNIST Rotation](https://github.com/user-attachments/assets/114720ec-bb63-453e-9fb8-dfe2f866b950)
*Comparison of original and rotated MNIST digits.*

### Confidence and Uncertainty Clusters

- **Digit 0**
  
  ![digit0](https://github.com/user-attachments/assets/d49fbd12-0a25-439a-a98b-d16a5338cdbf)
  
  *Example: Clustering of rotated vs. original samples for digit 0 over epochs.*

- **Digit 1**
  
  ![digit1](https://github.com/user-attachments/assets/0155e494-9870-42da-8978-f9c825928956)
  
  *Example: Clustering of rotated vs. original samples for digit 1 over epochs.*

- **Digit 2**
  
  ![digit2](https://github.com/user-attachments/assets/a2cf07d2-8fbb-4bb8-8cf8-ca79d151e60e)
  
  *Example: Clustering of rotated vs. original samples for digit 2 over epochs.*

- **Digit 3**
  
  ![digit3](https://github.com/user-attachments/assets/eb72bd63-8f30-4e7d-af20-1a90347111c3)
  
  *Example: Clustering of rotated vs. original samples for digit 3 over epochs.*

- **Digit 4**
  
  ![digit4](https://github.com/user-attachments/assets/97cfaf9b-8556-43f0-a877-b16bcb4a1639)
  
  *Example: Clustering of rotated vs. original samples for digit 4 over epochs.*

- **Digit 5**
  
  ![digit5](https://github.com/user-attachments/assets/c79827aa-d7b8-4d5e-a917-239a30d886c6)
  
  *Example: Clustering of rotated vs. original samples for digit 5 over epochs.*

- **Digit 6**
  
  ![digit6](https://github.com/user-attachments/assets/f0a977c7-0e33-45a1-a392-07b825db2f23)
  
  *Example: Clustering of rotated vs. original samples for digit 6 over epochs.*

- **Digit 7**
  
  ![digit7](https://github.com/user-attachments/assets/098e9685-b75a-4150-82a3-e5b43570a978)
  
  *Example: Clustering of rotated vs. original samples for digit 7 over epochs.*

- **Digit 8**
  
  ![digit8](https://github.com/user-attachments/assets/e2d30f4a-99a8-4f0e-95d0-de99577bf5f0)
  
  *Example: Clustering of rotated vs. original samples for digit 8 over epochs.*

- **Digit 9**
  
  ![digit9](https://github.com/user-attachments/assets/d0eba4cd-0cef-419c-a300-45e9c99a847d)
  
  *Example: Clustering of rotated vs. original samples for digit 9 over epochs.*

---

## References

- **Source Code**: [Colab Notebook](https://colab.research.google.com/drive/1HScciFVXf6Vmg4XuZU3zFqbMCyNeGTs8)
- **Related Works**:
  - Pocevičiūtė et al. (2022). Generalisation effects of predictive uncertainty estimation.
  - Wei et al. (2023). Convex Bounds on the Softmax Function.
