# Explainable AI: Confidence and Uncertainty

Welcome to the repository for the project **Explainable AI: Confidence and Uncertainty Based on Minimax and Relative Uniformity of Softmax Output**. This repository includes the code, resources, and documentation for analyzing the confidence and uncertainty of machine learning models using Softmax probabilities.

## Table of Contents

- [Overview](#overview)
- [Formula](#formula)
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

## Formula

### Inputs:
- Let $\mathbf{p} \in \mathbb{R}^{n_{\text{samples}} \times n_{\text{classes}}}$ be the input matrix of predicted probabilities, where:
  - $n_{\text{samples}}$ is the number of samples.
  - $n_{\text{classes}}$ is the number of classes (i.e., the number of possible predictions per sample).
  - Each element $p_{i,j} \in [0, 1]$ represents the predicted probability for sample $i$ and class $j$, and for each $i$, we have $\sum_{j=1}^{n_{\text{classes}}} p_{i,j} = 1$.

---

### Definitions:
- Let $N = n_{\text{classes}}$ denote the number of classes.
- $\text{inf} = -\log_2(\epsilon)$ (where $\epsilon$ is the smallest representable number for the input's data type)
- For each sample $i$, define:
  - The maximum predicted probability $a_i = \max_j (p_{i,j})$.
  - The minimum predicted probability $b_i = \min_j (p_{i,j})$.

---

### Confidence:
For each sample $i$, the confidence is defined as:

$$
c_i = \log_2(N - 1) - \log_2(N) - \log_2(1 - a_i)
$$

where $a_i = \max_j (p_{i,j})$ is the maximum predicted probability for sample $i$.

**Specifically:**
The confidence for each sample is normalized as:

$$
c_i' = \frac{c_i}{\inf}
$$

where $\inf$ is a large constant, representing the inverse of the smallest representable number of the data type, ensuring that the confidence is mapped to the range $[0, 1]$.

---

### Uncertainty:
The uncertainty for each sample $i$ is defined as:

$$
u_i = -\log_2(N) - \log_2(b_i) - c_i
$$

where $b_i = \min_j (p_{i,j})$ is the minimum predicted probability for sample $i$, and $c_i$ is the confidence for sample $i$.

**Specifically:**
The uncertainty for each sample is normalized as:

$$
u_i' = \frac{u_i}{\inf}
$$

where $\inf$ is the same large constant used for normalizing the confidence.

Then, the uncertainty is transformed as:

$$
u_i'' = \frac{2^{u_i'} - 1}{2^{u_i'} + 1}
$$

This maps the uncertainty values to the range $[0, 1]$.

---

### Outputs:
- The output consists of two arrays:
  1. **Confidence array $\mathbf{c}$**: A normalized confidence value for each sample, where $c_i' \in [0, 1]$.
  2. **Uncertainty array $\mathbf{u}$**: A normalized uncertainty value for each sample, where $u_i'' \in [0, 1]$.
  
The final output is a matrix $\mathbf{output} \in \mathbb{R}^{n_{\text{samples}} \times 2}$, where each row contains the normalized confidence and uncertainty values for each sample.

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
