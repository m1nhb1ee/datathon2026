import pandas as pd
import matplotlib.pyplot as plt


def load_data(path):
    """Load data from local path or URL"""
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def plot_comparison(file1, file2, column):
    # Load data
    df1 = load_data(file1)
    df2 = load_data(file2)

    if df1 is None or df2 is None:
        print("Failed to load data")
        return

    # Check column
    if column not in df1.columns or column not in df2.columns:
        print(f"Column '{column}' not found in both files")
        return

    x = range(len(df1))

    # --- Line plot ---
    plt.figure()
    plt.plot(x, df1[column], label='File 1')
    plt.plot(x, df2[column], label='File 2')
    plt.title("Line Comparison")
    plt.legend()
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.show()

    # --- Histogram ---
    plt.figure()
    plt.hist(df1[column], bins=50, alpha=0.5, label='File 1')
    plt.hist(df2[column], bins=50, alpha=0.5, label='File 2')
    plt.title("Distribution Comparison")
    plt.legend()
    plt.show()

    # --- Scatter plot ---
    plt.figure()
    plt.scatter(df1[column], df2[column])
    plt.title("Correlation between 2 files")
    plt.xlabel("File 1")
    plt.ylabel("File 2")
    plt.show()

    # --- Basic stats ---
    print("=== Stats File 1 ===")
    print(df1[column].describe())

    print("\n=== Stats File 2 ===")
    print(df2[column].describe())


if __name__ == "__main__":
    file1 = "C:\\Project\\Personal Project\\Datathon 2026\\submission_test_tuned.csv"
    file2 = "C:\\Project\\Personal Project\\Datathon 2026\\submission.csv"
    column = "Revenue"  # đổi lại theo file của bạn

    plot_comparison(file1, file2, column)