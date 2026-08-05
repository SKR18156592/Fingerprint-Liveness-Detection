import pandas as pd
import matplotlib.pyplot as plt


# Load training history
history = pd.read_csv(
    "outputs/training_log.csv"
)


# ============================
# Accuracy Plot
# ============================

plt.figure(figsize=(8,5))

plt.plot(
    history["epoch"],
    history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history["epoch"],
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.grid(True)

plt.savefig(
    "outputs/accuracy_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# ============================
# Loss Plot
# ============================

plt.figure(figsize=(8,5))

plt.plot(
    history["epoch"],
    history["loss"],
    label="Train Loss"
)

plt.plot(
    history["epoch"],
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.grid(True)

plt.savefig(
    "outputs/loss_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# ============================
# Precision Recall Plot
# ============================

plt.figure(figsize=(8,5))

plt.plot(
    history["epoch"],
    history["precision"],
    label="Precision"
)

plt.plot(
    history["epoch"],
    history["recall"],
    label="Recall"
)

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Precision and Recall")
plt.legend()

plt.grid(True)

plt.savefig(
    "outputs/precision_recall_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()