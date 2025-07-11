import argparse
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from PIL import Image


def train_model(epochs: int = 10):
    """Train a simple digit recognition model and display progress."""
    digits = load_digits()
    X = digits.images.reshape(len(digits.images), -1)
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = SGDClassifier(
        loss="log_loss",
        max_iter=1,
        learning_rate="constant",
        eta0=0.01,
        random_state=42,
        warm_start=True,
    )

    for epoch in range(epochs):
        clf.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"Epoch {epoch+1}/{epochs} - Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

    print("Final evaluation on test set:")
    print(classification_report(y_test, clf.predict(X_test)))

    joblib.dump({"scaler": scaler, "model": clf}, "digit_model.joblib")
    print("Model saved to digit_model.joblib")
    return scaler, clf


def predict_image(image_path: str, scaler, model):
    """Classify a single image using the trained model."""
    img = Image.open(image_path).convert("L")
    img = img.resize((8, 8), Image.LANCZOS)
    img_array = np.array(img, dtype=float)
    img_array = img_array / 255.0 * 16.0
    img_array = scaler.transform(img_array.reshape(1, -1))
    pred = model.predict(img_array)[0]
    print(f"Predicted digit: {pred}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a digit recognition model and optionally predict an image"
    )
    parser.add_argument(
        "--image",
        help="Path to an image to classify after training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    args = parser.parse_args()

    scaler, model = train_model(epochs=args.epochs)

    if args.image:
        predict_image(args.image, scaler, model)


if __name__ == "__main__":
    main()
