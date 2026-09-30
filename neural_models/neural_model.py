"""
Neural Models Lab
Learning, Depth, Activations, and Output Layers

Single-file implementation of the executable parts of the laboratory:
1. Binary XOR with a 2-2-1 network
2. Backpropagation / gradient inspection
3. Zero-initialisation symmetry experiment
4. Hidden-activation experiment: sigmoid, tanh, ReLU
5. Three-class extension with 3 logits + CrossEntropyLoss
6. Optional linear-only XOR baseline

Requirements:
    Python 3
    NumPy
    PyTorch

Run:
    python neural_model.py
"""

import numpy as np
import torch
import torch.nn as nn


# Configuration

SEED = 42
LEARNING_RATE = 0.1
STEPS = 5000
EARLY_GRADIENT_STEP = 10

torch.set_printoptions(precision=6, sci_mode=False, linewidth=140)


# Reproducibility

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


# Dataset

X_XOR = torch.tensor(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    dtype=torch.float32,
)

Y_XOR = torch.tensor(
    [
        [0.0],
        [1.0],
        [1.0],
        [0.0],
    ],
    dtype=torch.float32,
)

# Three-class version:
# 0 = both sensors inactive
# 1 = sensors disagree
# 2 = both sensors active
Y_3CLASS = torch.tensor([0, 1, 1, 2], dtype=torch.long)


# Models

class XORBinaryNet(nn.Module):
    """
    2 inputs -> 2 hidden units -> 1 output logit.

    BCEWithLogitsLoss is used instead of explicitly applying sigmoid
    during training. Sigmoid is applied only when reporting probabilities.
    """

    def __init__(self, activation="sigmoid"):
        super().__init__()

        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)

        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(
                "activation must be 'sigmoid', 'tanh', or 'relu'"
            )

        self.activation_name = activation

    def forward(self, x):
        h = self.hidden(x)
        h = self.activation(h)
        logits = self.output(h)
        return logits


class LinearXORNet(nn.Module):
    """
    Single affine transformation followed by one output logit.

    This is included as the linear baseline. It cannot represent XOR
    exactly because the decision boundary is a single hyperplane.
    """

    def __init__(self):
        super().__init__()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        return self.output(x)


class ThreeClassNet(nn.Module):
    """
    2 inputs -> 2 hidden units -> 3 output logits.
    """

    def __init__(self, activation="sigmoid"):
        super().__init__()

        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 3)

        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(
                "activation must be 'sigmoid', 'tanh', or 'relu'"
            )

        self.activation_name = activation

    def forward(self, x):
        h = self.activation(self.hidden(x))
        return self.output(h)


# Utility functions

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def binary_results(model):
    """
    Return probabilities and thresholded binary predictions.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_XOR)
        probabilities = torch.sigmoid(logits).squeeze(1)
        predictions = (probabilities >= 0.5).long()

    return probabilities, predictions


def count_correct_binary(predictions):
    targets = Y_XOR.squeeze(1).long()
    return int((predictions == targets).sum().item())


def gradient_norm(model):
    """
    Euclidean norm of the first-layer gradient.
    """
    grad = model.hidden.weight.grad
    if grad is None:
        return float("nan")
    return grad.norm().item()


def print_binary_results(model, initial_loss, final_loss):
    probabilities, predictions = binary_results(model)

    print(f"Initial loss: {initial_loss:.8f}")
    print(f"Final loss:   {final_loss:.8f}")

    print("\nInput      Target   Probability   Prediction")
    print("-" * 48)

    for i in range(len(X_XOR)):
        x = X_XOR[i].tolist()
        target = int(Y_XOR[i].item())
        probability = probabilities[i].item()
        prediction = int(predictions[i].item())

        print(
            f"{x!s:<10} {target:<8} "
            f"{probability:<13.6f} {prediction}"
        )

    correct = count_correct_binary(predictions)
    print(f"\nCorrect: {correct}/4")


# Task 1: Linear baseline

def run_linear_baseline():
    print_header("TASK 1 / LINEAR BASELINE")

    set_seed(SEED)

    model = LinearXORNet()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    initial_loss = loss_fn(model(X_XOR), Y_XOR).item()

    for _ in range(STEPS):
        optimizer.zero_grad()

        logits = model(X_XOR)
        loss = loss_fn(logits, Y_XOR)

        loss.backward()
        optimizer.step()

    final_loss = loss_fn(model(X_XOR), Y_XOR).item()

    print(
        "A single affine layer followed by sigmoid is a linear classifier. "
        "It cannot represent XOR exactly."
    )

    print_binary_results(model, initial_loss, final_loss)



# Task 3 + 4: Binary XOR training
def train_binary(
    activation="sigmoid",
    seed=SEED,
    learning_rate=LEARNING_RATE,
    steps=STEPS,
    zero_initialize=False,
    record_early_gradient=False,
):
    """
    Train the 2-2-1 binary XOR model.

    Returns:
        model
        initial_loss
        final_loss
        early_gradient_norm
        history
    """

    set_seed(seed)

    model = XORBinaryNet(activation)

    if zero_initialize:
        with torch.no_grad():
            model.hidden.weight.zero_()
            model.hidden.bias.zero_()
            model.output.weight.zero_()
            model.output.bias.zero_()

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    with torch.no_grad():
        initial_loss = loss_fn(model(X_XOR), Y_XOR).item()

    early_gradient = None
    history = []

    for step in range(1, steps + 1):
        optimizer.zero_grad()

        logits = model(X_XOR)
        loss = loss_fn(logits, Y_XOR)

        loss.backward()

        if record_early_gradient and step == EARLY_GRADIENT_STEP:
            early_gradient = gradient_norm(model)

        optimizer.step()

        if step == 1 or step % 500 == 0 or step == steps:
            history.append((step, loss.item()))

    with torch.no_grad():
        final_loss = loss_fn(model(X_XOR), Y_XOR).item()

    return model, initial_loss, final_loss, early_gradient, history


def run_binary_xor():
    print_header("TASK 3 / BASIC XOR LEARNING")

    # Seed is fixed for reproducibility. The architecture, data, and loss
    # remain exactly those specified by the laboratory.
    model, initial_loss, final_loss, early_gradient, history = train_binary(
        activation="sigmoid",
        seed=SEED,
        record_early_gradient=True,
    )

    print("Architecture: 2 inputs -> 2 sigmoid hidden units -> 1 output logit")
    print("Loss: BCEWithLogitsLoss")
    print("Optimiser: SGD")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Training steps: {STEPS}")

    print_binary_results(model, initial_loss, final_loss)

    print("\nFirst-layer weight gradient after backward():")
    print(model.hidden.weight.grad)


    print("\nLoss checkpoints:")
    for step, loss in history:
        print(f"  step {step:4d}: {loss:.8f}")


# Task 4B: Explicit gradient check
def run_gradient_check():
    print_header("TASK 4B / BACKPROPAGATION GRADIENT CHECK")

    set_seed(SEED)

    model = XORBinaryNet("sigmoid")
    loss_fn = nn.BCEWithLogitsLoss()

    logits = model(X_XOR)
    loss = loss_fn(logits, Y_XOR)

    loss.backward()

    print(f"Loss before parameter update: {loss.item():.8f}")
    print("\nFirst-layer weights:")
    print(model.hidden.weight)

    print("\nFirst-layer gradient dL/dW^(1):")
    print(model.hidden.weight.grad)

    print(
        "\nGradient norm:",
        f"{model.hidden.weight.grad.norm().item():.8f}"
    )

    print(
        "\nBecause BCEWithLogitsLoss uses a mean over the four examples, "
        "the reported gradient is the mean contribution from the four "
        "training examples after applying the chain rule through the network."
    )


# Task 4C: Symmetry experiment
def run_symmetry_experiment():
    print_header("TASK 4C / ZERO-INITIALISATION SYMMETRY")

    set_seed(SEED)

    model = XORBinaryNet("sigmoid")

    # Set all parameters to zero as required by the lab's symmetry experiment.
    with torch.no_grad():
        model.hidden.weight.zero_()
        model.hidden.bias.zero_()
        model.output.weight.zero_()
        model.output.bias.zero_()

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print("Initial hidden-layer weight matrix:")
    print(model.hidden.weight)

    print("\nHidden-layer rows over training:")
    print("-" * 70)

    for step in range(1, 11):
        optimizer.zero_grad()

        logits = model(X_XOR)
        loss = loss_fn(logits, Y_XOR)

        loss.backward()

        optimizer.step()

        row0 = model.hidden.weight[0].detach().clone()
        row1 = model.hidden.weight[1].detach().clone()

        identical = torch.allclose(row0, row1, atol=1e-12)

        print(
            f"step {step:2d}: "
            f"row 1 = {row0.numpy()}, "
            f"row 2 = {row1.numpy()}, "
            f"identical = {identical}"
        )

    print(
        "\nInterpretation: the two hidden units start with identical parameters. "
        "They therefore compute the same hidden activation and receive the same "
        "gradient. Gradient descent preserves the symmetry instead of causing "
        "the units to learn distinct features."
    )


# Task 4D: Activation experiment
def run_activation_experiment():
    print_header("TASK 4D / ACTIVATION EXPERIMENT")

    results = []

    for activation in ["sigmoid", "tanh", "relu"]:
        model, initial_loss, final_loss, early_gradient, _ = train_binary(
            activation=activation,
            seed=SEED,
            record_early_gradient=True,
        )

        probabilities, predictions = binary_results(model)
        correct = count_correct_binary(predictions)

        results.append(
            {
                "activation": activation,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "correct": correct,
                "early_gradient_norm": early_gradient,
                "probabilities": probabilities,
            }
        )

    print(
        f"{'Hidden activation':<20}"
        f"{'Final loss':<15}"
        f"{'4/4 correct?':<15}"
        f"{'Early ||grad||':<18}"
    )
    print("-" * 68)

    for result in results:
        correct_text = (
            "Yes" if result["correct"] == 4 else "No"
        )

        print(
            f"{result['activation']:<20}"
            f"{result['final_loss']:<15.8f}"
            f"{correct_text:<15}"
            f"{result['early_gradient_norm']:<18.8f}"
        )

    print("\nFinal probabilities for each activation:")

    for result in results:
        print(f"\n{result['activation']}:")
        for x, probability in zip(X_XOR, result["probabilities"]):
            print(
                f"  {x.tolist()} -> {probability.item():.6f}"
            )



# Task 5: Three-class extension
def train_three_class(
    activation="sigmoid",
    seed=SEED,
    learning_rate=LEARNING_RATE,
    steps=STEPS,
):
    set_seed(seed)

    model = ThreeClassNet(activation)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    with torch.no_grad():
        initial_loss = loss_fn(model(X_XOR), Y_3CLASS).item()

    for _ in range(steps):
        optimizer.zero_grad()

        logits = model(X_XOR)
        loss = loss_fn(logits, Y_3CLASS)

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = loss_fn(model(X_XOR), Y_3CLASS).item()

    return model, initial_loss, final_loss


def run_three_class_experiment():
    print_header("TASK 5 / THREE-CLASS EXTENSION")

    model, initial_loss, final_loss = train_three_class(
        activation="sigmoid",
        seed=SEED,
    )

    model.eval()

    with torch.no_grad():
        logits = model(X_XOR)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    print("Architecture: 2 inputs -> 2 hidden units -> 3 output logits")
    print("Loss: CrossEntropyLoss")

    print(f"\nInitial loss: {initial_loss:.8f}")
    print(f"Final loss:   {final_loss:.8f}")

    print("\nFinal output-layer weight shape:")
    print(model.output.weight.shape)

    print(
        "\nInput      Target   P(class 0)   P(class 1)   P(class 2)   Prediction"
    )
    print("-" * 80)

    for i in range(len(X_XOR)):
        x = X_XOR[i].tolist()
        target = Y_3CLASS[i].item()
        probs = probabilities[i]
        prediction = predictions[i].item()

        print(
            f"{str(x):<10} "
            f"{target:<8} "
            f"{probs[0].item():<13.6f}"
            f"{probs[1].item():<13.6f}"
            f"{probs[2].item():<13.6f}"
            f"{prediction}"
        )

    correct = int((predictions == Y_3CLASS).sum().item())
    print(f"\nCorrect: {correct}/4")

    # Explicit probability-sum verification for one example.
    example_index = 1
    probability_vector = probabilities[example_index]
    probability_sum = probability_vector.sum().item()

    print("\nSoftmax probability vector for example:")
    print(f"Input: {X_XOR[example_index].tolist()}")
    print(probability_vector)
    print(f"Sum: {probability_sum:.12f}")

    # Optional numerical-stability experiment from the lab.
    with torch.no_grad():
        shifted_logits = logits[example_index] + 100.0
        shifted_probabilities = torch.softmax(shifted_logits, dim=0)

    print("\nOptional +100 logit-shift experiment:")
    print("Original probabilities:")
    print(probabilities[example_index])

    print("Probabilities after adding 100 to every logit:")
    print(shifted_probabilities)

    print(
        "\nMaximum absolute difference:",
        torch.max(
            torch.abs(probabilities[example_index] - shifted_probabilities)
        ).item(),
    )


# Task 5: p - y gradient demonstration
def run_softmax_gradient_demo():
    print_header("TASK 5 / p - y LOGIT-GRADIENT DEMONSTRATION")

    set_seed(SEED)

    model = ThreeClassNet("sigmoid")
    loss_fn = nn.CrossEntropyLoss()

    logits = model(X_XOR)
    loss = loss_fn(logits, Y_3CLASS)

    loss.backward()

    with torch.no_grad():
        probabilities = torch.softmax(logits, dim=1)

        # One-hot representation of the labels.
        one_hot = torch.zeros_like(probabilities)
        one_hot.scatter_(1, Y_3CLASS.unsqueeze(1), 1.0)

        # For a mean CrossEntropyLoss over N examples:
        # dL/dz = (p - y) / N
        theoretical_gradient = (probabilities - one_hot) / len(X_XOR)

    print("For CrossEntropyLoss with mean reduction:")
    print("dL/dz = (p - y) / N")
    print("\nAutograd gradient of output bias:")
    print(model.output.bias.grad)

    print("\nSum of example-wise (p-y)/N contributions:")
    print(theoretical_gradient.sum(dim=0))


def main():
    print_header("NEURAL MODELS LAB - COMPLETE PYTHON IMPLEMENTATION")

    print(
        "Dataset:\n"
        "  (0,0) -> 0\n"
        "  (0,1) -> 1\n"
        "  (1,0) -> 1\n"
        "  (1,1) -> 0\n"
    )


    run_linear_baseline()
    run_binary_xor()
    run_gradient_check()
    run_symmetry_experiment()
    run_activation_experiment()
    run_three_class_experiment()
    run_softmax_gradient_demo()




if __name__ == "__main__":
    main()

