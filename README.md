# Explainable AI: Confidence and Uncertainty

Welcome to the repository for the project **Explainable AI: Confidence and Uncertainty Based on Minimax and Relative Uniformity of Softmax Output**. This repository includes the code, resources, and documentation for analyzing the confidence and uncertainty of machine learning models using Softmax probabilities.

## Table of Contents

- [Overview](#overview)
- [Formula](#formula)
- [Visualizations](#visualizations)
- [References](#references)

---

## Formula

**Inputs:**
- $\mathbf{p} \in \mathbb{R}^{n \times k}$: The predicted probabilities, where $n$ is the number of samples and $k$ is the number of classes. Each element $p_{i,j}$ represents the predicted probability of the $i$-th sample belonging to the $j$-th class.

**Definitions:**
- $N = k$ (number of classes)
- $a_i = \max(\mathbf{p}_i)$ (maximum predicted probability for the $i$-th sample)
- $b_i = \min(\mathbf{p}_i)$ (minimum predicted probability for the $i$-th sample)
- $\text{inf} = -\log_2(\epsilon)$ (where $\epsilon$ is the smallest representable number for the data type)

**Computation of Confidence:**

$$
c_i = \frac{\log_2(N-1) - \log_2(N) - \log_2(1 - a_i)}{\text{inf}}
$$

**Computation of Uncertainty:**

$$
u_i = \frac{2^{-\log_2(N) - \log_2(b_i) - c_i} - 1}{2^{-\log_2(N) - \log_2(b_i) - c_i} + 1}
$$

**Outputs:**
- $c \in \mathbb{R}^{n}$ (confidence for each sample)
- $u \in \mathbb{R}^{n}$ (uncertainty for each sample)


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
  
  ![label_0](https://github.com/user-attachments/assets/05707f46-317c-42e1-af9d-fe4421b193d4)
  
  *Example: Clustering of rotated vs. original samples for digit 0 over epochs.*

- **Digit 1**
  
  ![label_1](https://github.com/user-attachments/assets/94837afa-56f5-4ad7-853d-97ad3828d696)

  *Example: Clustering of rotated vs. original samples for digit 1 over epochs.*

- **Digit 2**
  
  ![label_2](https://github.com/user-attachments/assets/65d440ac-0eb6-4760-aab8-f6decaedbfcc)

  *Example: Clustering of rotated vs. original samples for digit 2 over epochs.*

- **Digit 3**
  
  ![label_3](https://github.com/user-attachments/assets/16a48798-daa1-4938-9c44-57e24a44823c)

  *Example: Clustering of rotated vs. original samples for digit 3 over epochs.*

- **Digit 4**
  
  ![label_4](https://github.com/user-attachments/assets/933a6501-6436-45a0-acd1-ad31db8c7e31)
  
  *Example: Clustering of rotated vs. original samples for digit 4 over epochs.*

- **Digit 5**
  
  ![label_5](https://github.com/user-attachments/assets/7cb60a3f-985c-413d-b9b8-eae6a95972de)
  
  *Example: Clustering of rotated vs. original samples for digit 5 over epochs.*

- **Digit 6**
  
  ![label_6](https://github.com/user-attachments/assets/b75e3b28-10e6-4d5b-8d2e-21acf8adc953)
  
  *Example: Clustering of rotated vs. original samples for digit 6 over epochs.*

- **Digit 7**
  
  ![label_7](https://github.com/user-attachments/assets/ec2bed27-9015-41bf-aede-5f6fcfd495f4)
  
  *Example: Clustering of rotated vs. original samples for digit 7 over epochs.*

- **Digit 8**
  
  ![label_8](https://github.com/user-attachments/assets/2fcd7d09-3d9b-48b9-b10d-e03a30c28b2a)
  
  *Example: Clustering of rotated vs. original samples for digit 8 over epochs.*

- **Digit 9**
  
  ![label_9](https://github.com/user-attachments/assets/1265128f-2ef9-4ed2-a75b-b03d3ed2d719)
  
  *Example: Clustering of rotated vs. original samples for digit 9 over epochs.*

---

## References

- **Related Works**:
  - Pocevičiūtė et al. (2022). Generalisation effects of predictive uncertainty estimation.
  - Wei et al. (2023). Convex Bounds on the Softmax Function.
