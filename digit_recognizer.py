

import numpy as np


DIGIT_BITMAPS = {
    0: ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    1: ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    2: ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    3: ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    4: ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    5: ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    6: ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    7: ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    8: ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    9: ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}


def bitmap_to_vector(rows):
    
    return np.array([int(ch) for row in rows for ch in row], dtype=float)


def add_noise(vector, flip_count, rng):
    
    noisy = vector.copy()
    flip_idx = rng.choice(len(vector), size=flip_count, replace=False)
    noisy[flip_idx] = 1 - noisy[flip_idx]
    return noisy


def make_dataset(samples_per_digit, max_flips, seed):
   
    rng = np.random.default_rng(seed)
    X, y = [], []
    for digit, rows in DIGIT_BITMAPS.items():
        clean = bitmap_to_vector(rows)
        for _ in range(samples_per_digit):
            flips = rng.integers(0, max_flips + 1)
            X.append(add_noise(clean, flips, rng) if flips else clean.copy())
            y.append(digit)
    X = np.array(X)
    y_onehot = np.zeros((len(y), 10))
    y_onehot[np.arange(len(y)), y] = 1
    return X, y_onehot, np.array(y)


def print_bitmap(vector, width=5):
   
    pixels = vector.reshape(-1, width)
    for row in pixels:
        print("".join("#" if p > 0.5 else "." for p in row))


class DigitRecognizer:
   

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5, seed=0):
        rng = np.random.default_rng(seed)
        self.lr = learning_rate
        self.W1 = rng.normal(0, 0.5, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(0, 0.5, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def softmax(z):
        
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y_onehot):
        m = X.shape[0]

       
        dz2 = self.a2 - y_onehot
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.a1 * (1 - self.a1)  
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y_onehot, epochs, print_every=200):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y_onehot)
            if epoch % print_every == 0:
                loss = -np.mean(np.sum(y_onehot * np.log(self.a2 + 1e-9), axis=1))
                acc = self.accuracy(X, y_onehot)
                print(f"Epoch {epoch:>5} | loss: {loss:.4f} | train accuracy: {acc:.2%}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y_onehot):
        preds = self.predict(X)
        labels = np.argmax(y_onehot, axis=1)
        return np.mean(preds == labels)


if __name__ == "__main__":
   
    X_train, y_train_onehot, _ = make_dataset(samples_per_digit=60, max_flips=4, seed=1)

    
    X_test, y_test_onehot, y_test_labels = make_dataset(samples_per_digit=20, max_flips=4, seed=99)

    net = DigitRecognizer(input_size=35, hidden_size=20, output_size=10,
                           learning_rate=0.5, seed=7)

    print("Training the digit recognizer...\n")
    net.train(X_train, y_train_onehot, epochs=2000, print_every=200)

    test_acc = net.accuracy(X_test, y_test_onehot)
    print(f"\nTest accuracy on unseen noisy digits: {test_acc:.2%}")

    print("\nA few sample predictions on noisy, unseen digits:\n")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=4, replace=False)
    predictions = net.predict(X_test[sample_idx])
    for idx, pred in zip(sample_idx, predictions):
        print(f"Actual: {y_test_labels[idx]}   Predicted: {pred}")
        print_bitmap(X_test[idx])
        print()
