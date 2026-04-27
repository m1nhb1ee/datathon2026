"""
Time-based train/test split for downstream time-series modeling.

This script uses the analytical daily sales table because it contains the
modeling target columns (`Revenue`, `COGS`) at the daily prediction granularity.
It does not compute rolling, aggregate, or lag features, so no future data can
leak into the train split.
"""

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "dataset"

SOURCE_FILE = RAW_DATA_DIR / "sales.csv"
TIME_COLUMN = "Date"

TRAIN_END = pd.Timestamp("2017-12-31")
VALID_START = pd.Timestamp("2018-01-01")
VALID_END = pd.Timestamp("2019-12-31")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2022-12-31")


def inspect_time_columns(data_dir: Path) -> None:
    """Print date-like columns in every CSV so the time-column choice is explicit."""
    print("Inspecting date-like columns in raw tables:")
    for path in sorted(data_dir.glob("*.csv")):
        sample = pd.read_csv(path, nrows=5)
        date_like_columns = [
            col
            for col in sample.columns
            if "date" in col.lower() or col.lower() in {"date", "timestamp", "datetime"}
        ]
        if not date_like_columns:
            print(f"- {path.name}: no date-like column")
            continue

        print(f"- {path.name}: {date_like_columns}")

    print(
        "\nSelected time column: sales.csv::Date "
        "(daily analytical target table for Revenue/COGS forecasting)\n"
    )


def load_sales(path: Path) -> pd.DataFrame:
    """Load the target table and enforce datetime ordering."""
    df = pd.read_csv(path)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="raise")
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    return df


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by strict calendar boundaries without shuffling."""
    train_mask = df[TIME_COLUMN].le(TRAIN_END)
    valid_mask = df[TIME_COLUMN].between(VALID_START, VALID_END, inclusive="both")
    test_mask = df[TIME_COLUMN].between(TEST_START, TEST_END, inclusive="both")

    train = df.loc[train_mask].copy()
    valid = df.loc[valid_mask].copy()
    test = df.loc[test_mask].copy()

    return train, valid, test


def validate_split(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print and assert split boundaries and date overlap constraints."""
    train_min = train[TIME_COLUMN].min()
    train_max = train[TIME_COLUMN].max()
    valid_min = valid[TIME_COLUMN].min()
    valid_max = valid[TIME_COLUMN].max()
    test_min = test[TIME_COLUMN].min()
    test_max = test[TIME_COLUMN].max()

    print("Split validation:")
    print(f"- Train rows: {len(train):,} ({train_min.date()} -> {train_max.date()})")
    print(f"- Valid rows: {len(valid):,} ({valid_min.date()} -> {valid_max.date()})")
    print(f"- Test rows : {len(test):,} ({test_min.date()} -> {test_max.date()})")

    train_dates = set(train[TIME_COLUMN])
    valid_dates = set(valid[TIME_COLUMN])
    test_dates = set(test[TIME_COLUMN])

    assert not train.empty, "Train split is empty."
    assert not valid.empty, "Validation split is empty."
    assert not test.empty, "Test split is empty."
    assert train_max <= TRAIN_END, "Train split contains records after 2017-12-31."
    assert valid_min >= VALID_START, "Validation split contains records before 2018-01-01."
    assert valid_max <= VALID_END, "Validation split contains records after 2019-12-31."
    assert test_min >= TEST_START, "Test split contains records before 2020-01-01."
    assert test_max <= TEST_END, "Test split contains records after 2022-12-31."
    assert train_dates.isdisjoint(valid_dates), "Train and validation splits overlap."
    assert train_dates.isdisjoint(test_dates), "Train and test splits overlap."
    assert valid_dates.isdisjoint(test_dates), "Validation and test splits overlap."

    print("- Assertions passed: boundaries are valid and dates do not overlap.\n")


def save_outputs(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> None:
    """Persist split outputs in efficient Parquet format."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    valid.to_parquet(OUTPUT_DIR / "valid.parquet", index=False)
    test.to_parquet(OUTPUT_DIR / "test.parquet", index=False)

    print(f"Saved: {OUTPUT_DIR / 'train.parquet'}")
    print(f"Saved: {OUTPUT_DIR / 'valid.parquet'}")
    print(f"Saved: {OUTPUT_DIR / 'test.parquet'}")


def main() -> None:
    inspect_time_columns(RAW_DATA_DIR)
    sales = load_sales(SOURCE_FILE)
    train, valid, test = split_by_time(sales)
    validate_split(train, valid, test)
    save_outputs(train, valid, test)


if __name__ == "__main__":
    main()
