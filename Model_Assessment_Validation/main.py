import matplotlib.pyplot as plt
import numpy as np

from generateData import generateData
from rightCV import rightCV
from wrongCV import wrongCV


def main():
    # Generate data once
    X, y = generateData()
    wrong_correlations = wrongCV(X, y)
    right_correlations = rightCV(X, y)

    # Plot histograms
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.hist(wrong_correlations, bins=20, range=(-1, 1), color="red", alpha=0.7)
    ax1.set_title("Wrong way")
    ax1.set_xlabel("Correlations of Selected Predictors with Outcome")
    ax1.set_ylabel("Frequency")
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(0, 30)

    ax2.hist(right_correlations, bins=20, range=(-1, 1), color="green", alpha=0.7)
    ax2.set_title("Right way")
    ax2.set_xlabel("Correlations of Selected Predictors with Outcome")
    ax2.set_ylabel("Frequency")
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(0, 30)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
