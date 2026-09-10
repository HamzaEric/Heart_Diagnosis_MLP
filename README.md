# Multilayer Perceptron (MLP) Theory & Binary Classification

An educational repository dedicated to demystifying the mathematical foundations of **Feedforward Neural Networks (FNNs)**. This project breaks down the mechanics of **Multilayer Perceptrons (MLPs)** for binary classification, bridging the gap between raw calculus and PyTorch implementations.

---

## Core Mathematical Concepts

### 1. The Forward Pass (Linear Algebra & Activations)

Understanding how data flows through a neural network using matrix multiplication and nonlinear transformations.

#### Linear Transformations

Each layer applies a linear transformation:

$$
z = xW + b
$$

where:

* $x$ = input
* $W$ = weight matrix
* $b$ = bias vector
* $z$ = pre-activation output

#### Shape Intuition

Tracking matrix dimensions helps explain how information moves through the network, from the input features to the hidden layers and ultimately to the output.

For example:

$$
(1, 2)(2, 3) + (1, 3) = (1, 3)
$$

This represents:

```text
2 input features → 3 hidden neurons
```

The final output layer then reduces the hidden representation to a single scalar:

$$
(1, 3)(3, 1) + (1, 1) = (1, 1)
$$

#### Activation Functions

The hidden layers use the **ReLU** activation function:

$$
\operatorname{ReLU}(z) = \max(0,z)
$$

ReLU introduces nonlinearity, allowing the network to learn nonlinear decision boundaries.

For binary classification, the final logit can be transformed into a probability using the **Sigmoid** function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The resulting value lies between $0$ and $1$ and can be interpreted as the predicted probability of the positive class.

---

### 2. Backpropagation (Calculus & the Chain Rule)

Backpropagation explains how the neural network learns from its prediction errors by propagating gradients backward through the computational graph.

The process involves calculating partial derivatives and applying the **chain rule**.

#### Loss Gradient

First, calculate how the loss changes with respect to the model's prediction:

$$
\frac{\partial L}{\partial \hat{y}}
$$

#### Chain Rule

Gradients are then propagated backward through each operation:

$$
\frac{\partial L}{\partial W_1}
=
\frac{\partial L}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial h}
\frac{\partial h}{\partial z_1}
\frac{\partial z_1}{\partial W_1}
$$

This isolates the contribution of a particular weight to the final loss.

#### Activation Derivatives

For ReLU:

$$
\operatorname{ReLU}'(z)
=
\begin{cases}
1 & z > 0 \\
0 & z < 0
\end{cases}
$$

Therefore, neurons with negative pre-activation values block the gradient, while active neurons allow the gradient to pass through.

---

### 3. Binary Classification Dynamics

#### Logits vs. Probabilities

The final linear layer produces a **logit**, which is an unrestricted real-valued number.

The sigmoid function converts the logit into a probability:

$$
p = \sigma(z)
$$

where:

$$
0 \leq p \leq 1
$$

For example:

```text
Logit → Sigmoid → Probability

  2.5  →   σ(2.5)   →   0.924
 -1.2  →   σ(-1.2)  →   0.231
```

#### Binary Cross Entropy

Binary Cross Entropy (BCE) measures how well the predicted probability matches the true binary label.

$$L = -\left[y\log(p)+(1-y)\log(1-p)\right]$$

where:

* $y$ = true label
* $p$ = predicted probability

Incorrect and overly confident predictions receive a larger penalty.

---

## Repository Structure

The repository is organized as follows:

```text
├── DataFiles/
│   └── heart.csv
│       # Clinical tabular dataset with 13 input features
│
├── Models/
│   ├── custom_model_weights.pth
│   │   # Serialized PyTorch model parameters
│   │
│   └── feature_scaler.pkl
│       # Fitted StandardScaler used to standardize input features
│
└── Notebooks/
    ├── Backpropagation.ipynb
    │   # Calculus, chain rule, and partial derivatives
    │
    ├── Binary_Classification.ipynb
    │   # Activation functions, logits, probabilities, and decision boundaries
    │
    ├── Feed_Forward_Neural_Network.ipynb
    │   # Matrix multiplication, weights, biases, and shape intuition
    │
    └── Pytorch_MLP.ipynb
        # Translating the mathematical model into a PyTorch architecture
        # with regularization
```

---

