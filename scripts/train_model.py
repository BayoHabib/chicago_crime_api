"""Train crime prediction model with MLflow tracking.

Uses temporal train/test split (time series best practice) and
compares against baseline models.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style for plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_eda(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, plt.Figure]:
    """Generate time-series focused EDA plots for crime forecasting.

    Returns dict of figure name -> matplotlib figure for MLflow logging.
    """
    figures = {}

    # 1. TIME SERIES DECOMPOSITION - Overall trend and seasonality
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Aggregate to city-wide weekly totals
    weekly_total = (
        df.groupby("week_of_year")["crime_count"].agg(["sum", "mean", "std"]).reset_index()
    )

    # Total crimes over time
    axes[0].plot(
        weekly_total["week_of_year"],
        weekly_total["sum"],
        "b-",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    axes[0].fill_between(weekly_total["week_of_year"], weekly_total["sum"], alpha=0.3)
    axes[0].set_ylabel("Total Weekly Crimes")
    axes[0].set_title("City-Wide Crime Time Series")
    axes[0].grid(True, alpha=0.3)

    # Mean per cell with confidence band
    axes[1].plot(weekly_total["week_of_year"], weekly_total["mean"], "g-", linewidth=2)
    axes[1].fill_between(
        weekly_total["week_of_year"],
        weekly_total["mean"] - weekly_total["std"],
        weekly_total["mean"] + weekly_total["std"],
        alpha=0.3,
        color="green",
        label="±1 std",
    )
    axes[1].set_ylabel("Mean Crimes per Cell")
    axes[1].set_title("Average Crime per Grid Cell (with variance)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Number of active cells (cells with >0 crimes)
    active_cells = df[df["crime_count"] > 0].groupby("week_of_year").size()
    axes[2].bar(active_cells.index, active_cells.values, alpha=0.7, color="orange")
    axes[2].set_xlabel("Week of Year")
    axes[2].set_ylabel("Active Grid Cells")
    axes[2].set_title("Number of Cells with Crime Activity")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    figures["eda_time_series_decomposition"] = fig

    # 2. AUTOCORRELATION ANALYSIS - Critical for time series models
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Sample a few grid cells for ACF visualization
    top_cells = df.groupby("grid_id")["crime_count"].sum().nlargest(5).index.tolist()

    # Plot ACF for top cells
    for i, cell_id in enumerate(top_cells[:4]):
        ax = axes[i // 2, i % 2]
        cell_data = df[df["grid_id"] == cell_id].sort_values("week_of_year")["crime_count"].values

        # Manual ACF calculation
        n = len(cell_data)
        if n > 10:
            max_lag = min(15, n - 1)
            acf_values = []
            for lag in range(max_lag + 1):
                if lag == 0:
                    acf_values.append(1.0)
                else:
                    corr = np.corrcoef(cell_data[:-lag], cell_data[lag:])[0, 1]
                    acf_values.append(corr if not np.isnan(corr) else 0)

            ax.bar(range(len(acf_values)), acf_values, alpha=0.7)
            ax.axhline(y=0, color="black", linestyle="-")
            ax.axhline(y=1.96 / np.sqrt(n), color="red", linestyle="--", alpha=0.5)
            ax.axhline(y=-1.96 / np.sqrt(n), color="red", linestyle="--", alpha=0.5)
            ax.set_xlabel("Lag (weeks)")
            ax.set_ylabel("Autocorrelation")
            ax.set_title(f"ACF - Grid Cell {cell_id} (high crime)")
            ax.set_ylim(-0.5, 1.1)

    plt.tight_layout()
    figures["eda_autocorrelation"] = fig

    # 3. STATIONARITY CHECK - Distribution stability over time
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Split data into quarters
    weeks = sorted(df["week_of_year"].unique())
    n_weeks = len(weeks)
    q1_weeks = weeks[: n_weeks // 4]
    q2_weeks = weeks[n_weeks // 4 : n_weeks // 2]
    q3_weeks = weeks[n_weeks // 2 : 3 * n_weeks // 4]
    q4_weeks = weeks[3 * n_weeks // 4 :]

    quarters = [
        ("Q1 (Early)", df[df["week_of_year"].isin(q1_weeks)]["crime_count"]),
        ("Q2", df[df["week_of_year"].isin(q2_weeks)]["crime_count"]),
        ("Q3", df[df["week_of_year"].isin(q3_weeks)]["crime_count"]),
        ("Q4 (Recent)", df[df["week_of_year"].isin(q4_weeks)]["crime_count"]),
    ]

    # Distribution comparison
    ax = axes[0, 0]
    for name, data in quarters:
        ax.hist(data, bins=30, alpha=0.5, label=f"{name} (μ={data.mean():.2f})", density=True)
    ax.set_xlabel("Crime Count")
    ax.set_ylabel("Density")
    ax.set_title("Distribution Stability Across Time")
    ax.legend()

    # Box plot by quarter
    ax = axes[0, 1]
    box_data = [q[1].values for q in quarters]
    bp = ax.boxplot(box_data, labels=[q[0] for q in quarters], patch_artist=True)
    for patch, color in zip(bp["boxes"], plt.cm.viridis(np.linspace(0.2, 0.8, 4)), strict=False):
        patch.set_facecolor(color)
    ax.set_ylabel("Crime Count")
    ax.set_title("Crime Distribution by Time Period")

    # Rolling mean stability
    ax = axes[1, 0]
    if "crime_count_rolling_mean_4" in df.columns:
        rm_by_week = df.groupby("week_of_year")["crime_count_rolling_mean_4"].mean()
        ax.plot(rm_by_week.index, rm_by_week.values, "b-", linewidth=2)
        ax.axhline(
            rm_by_week.mean(),
            color="red",
            linestyle="--",
            label=f"Overall mean: {rm_by_week.mean():.2f}",
        )
        ax.set_xlabel("Week of Year")
        ax.set_ylabel("Avg Rolling Mean (4-week)")
        ax.set_title("Rolling Mean Stability Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Variance over time
    ax = axes[1, 1]
    var_by_week = df.groupby("week_of_year")["crime_count"].var()
    ax.plot(var_by_week.index, var_by_week.values, "purple", linewidth=2, marker="o", markersize=4)
    ax.axhline(
        var_by_week.mean(),
        color="red",
        linestyle="--",
        label=f"Mean variance: {var_by_week.mean():.2f}",
    )
    ax.set_xlabel("Week of Year")
    ax.set_ylabel("Variance")
    ax.set_title("Crime Variance Over Time (Heteroscedasticity Check)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    figures["eda_stationarity"] = fig

    # 4. LAG FEATURE DIAGNOSTICS - How well do lags predict?
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    lag_features = [
        "crime_count_lag1",
        "crime_count_lag2",
        "crime_count_lag3",
        "crime_count_lag4",
        "crime_count_rolling_mean_4",
        "crime_trend",
    ]

    for idx, lag_col in enumerate(lag_features):
        if lag_col in df.columns:
            ax = axes[idx // 3, idx % 3]

            # Hexbin for dense data
            hb = ax.hexbin(df[lag_col], df["crime_count"], gridsize=30, cmap="YlOrRd", mincnt=1)
            plt.colorbar(hb, ax=ax, label="Count")

            # Add regression line
            z = np.polyfit(df[lag_col].dropna(), df.loc[df[lag_col].notna(), "crime_count"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df[lag_col].min(), df[lag_col].max(), 100)
            ax.plot(x_line, p(x_line), "b--", linewidth=2, label=f"y={z[0]:.2f}x+{z[1]:.2f}")

            # Correlation
            corr = df[[lag_col, "crime_count"]].corr().iloc[0, 1]
            ax.set_xlabel(lag_col.replace("_", " ").title())
            ax.set_ylabel("Crime Count")
            ax.set_title(f"r = {corr:.3f}")
            ax.legend(loc="upper left")

    plt.suptitle("Lag Feature Predictive Power", fontsize=14, fontweight="bold")
    plt.tight_layout()
    figures["eda_lag_diagnostics"] = fig

    # 5. FORECAST HORIZON ANALYSIS - How far ahead can we predict?
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Correlation decay with lag
    ax = axes[0]
    max_lag = 12
    lag_corrs = []
    for lag in range(1, max_lag + 1):
        lag_col = f"crime_count_lag{lag}" if f"crime_count_lag{lag}" in df.columns else None
        if lag_col:
            corr = df[[lag_col, "crime_count"]].corr().iloc[0, 1]
        else:
            # Calculate manually
            shifted = df.groupby("grid_id")["crime_count"].shift(lag)
            corr = df["crime_count"].corr(shifted)
        lag_corrs.append(corr if not np.isnan(corr) else 0)

    ax.bar(range(1, max_lag + 1), lag_corrs, alpha=0.7, color="steelblue")
    ax.axhline(y=0.5, color="red", linestyle="--", label="r=0.5 threshold")
    ax.set_xlabel("Forecast Horizon (weeks)")
    ax.set_ylabel("Correlation with Target")
    ax.set_title("Predictability Decay by Forecast Horizon")
    ax.set_xticks(range(1, max_lag + 1))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE by crime level (error analysis preview)
    ax = axes[1]
    if "crime_count_rolling_mean_4" in df.columns:
        df_temp = df.copy()
        df_temp["baseline_pred"] = df_temp["crime_count_rolling_mean_4"]
        df_temp["error"] = np.abs(df_temp["crime_count"] - df_temp["baseline_pred"])
        df_temp["crime_bin"] = pd.cut(
            df_temp["crime_count"],
            bins=[0, 1, 3, 5, 10, 50],
            labels=["0-1", "1-3", "3-5", "5-10", "10+"],
        )

        error_by_bin = df_temp.groupby("crime_bin", observed=True)["error"].agg(
            ["mean", "std", "count"]
        )

        bars = ax.bar(
            range(len(error_by_bin)),
            error_by_bin["mean"],
            yerr=error_by_bin["std"],
            capsize=5,
            alpha=0.7,
            color="coral",
        )
        ax.set_xticks(range(len(error_by_bin)))
        ax.set_xticklabels(error_by_bin.index)
        ax.set_xlabel("Actual Crime Count Range")
        ax.set_ylabel("Mean Absolute Error (Baseline)")
        ax.set_title("Baseline Model Error by Crime Volume")

        for _, (bar, count) in enumerate(zip(bars, error_by_bin["count"], strict=False)):
            ax.annotate(
                f"n={int(count)}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    figures["eda_forecast_horizon"] = fig

    return figures


def plot_model_evaluation(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    feature_names: list[str],
    feature_importances: np.ndarray | None,
    test_df: pd.DataFrame,
) -> dict[str, plt.Figure]:
    """Generate model evaluation plots.

    Returns dict of figure name -> matplotlib figure for MLflow logging.
    """
    figures = {}

    # 1. Feature Importance (if available)
    if feature_importances is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": feature_importances}
        ).sort_values("importance", ascending=True)

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))
        ax.barh(importance_df["feature"], importance_df["importance"], color=colors)
        ax.set_xlabel("Feature Importance")
        ax.set_title("Random Forest Feature Importance")
        plt.tight_layout()
        figures["model_feature_importance"] = fig

    # 2. Actual vs Predicted (Train and Test)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Train
    axes[0].scatter(y_train, y_train_pred, alpha=0.1, s=5)
    max_val = max(y_train.max(), y_train_pred.max())
    axes[0].plot([0, max_val], [0, max_val], "r--", label="Perfect prediction")
    axes[0].set_xlabel("Actual Crime Count")
    axes[0].set_ylabel("Predicted Crime Count")
    axes[0].set_title(f"Train: Actual vs Predicted (R²={r2_score(y_train, y_train_pred):.3f})")
    axes[0].legend()

    # Test
    axes[1].scatter(y_test, y_test_pred, alpha=0.3, s=10)
    max_val = max(y_test.max(), y_test_pred.max())
    axes[1].plot([0, max_val], [0, max_val], "r--", label="Perfect prediction")
    axes[1].set_xlabel("Actual Crime Count")
    axes[1].set_ylabel("Predicted Crime Count")
    axes[1].set_title(f"Test: Actual vs Predicted (R²={r2_score(y_test, y_test_pred):.3f})")
    axes[1].legend()

    plt.tight_layout()
    figures["model_actual_vs_predicted"] = fig

    # 3. Residual Analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    residuals = y_test - y_test_pred

    # Residual distribution
    axes[0].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Residual (Actual - Predicted)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Residual Distribution (Mean: {residuals.mean():.3f})")

    # Residuals vs Predicted
    axes[1].scatter(y_test_pred, residuals, alpha=0.3, s=10)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted Crime Count")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs Predicted Values")

    # Q-Q plot approximation
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = np.linspace(0.01, 0.99, len(sorted_residuals))
    theoretical_values = np.quantile(
        np.random.normal(0, residuals.std(), 10000), theoretical_quantiles
    )
    axes[2].scatter(theoretical_values[: len(sorted_residuals)], sorted_residuals, alpha=0.3, s=5)
    lims = [
        min(axes[2].get_xlim()[0], axes[2].get_ylim()[0]),
        max(axes[2].get_xlim()[1], axes[2].get_ylim()[1]),
    ]
    axes[2].plot(lims, lims, "r--")
    axes[2].set_xlabel("Theoretical Quantiles")
    axes[2].set_ylabel("Sample Quantiles")
    axes[2].set_title("Q-Q Plot (Normality Check)")

    plt.tight_layout()
    figures["model_residual_analysis"] = fig

    # 4. Prediction Error Over Time
    fig, ax = plt.subplots(figsize=(12, 6))

    error_by_week = test_df.copy()
    error_by_week["prediction"] = y_test_pred
    error_by_week["abs_error"] = np.abs(y_test - y_test_pred)

    weekly_error = (
        error_by_week.groupby("week_of_year")
        .agg({"abs_error": "mean", "crime_count": "mean", "prediction": "mean"})
        .reset_index()
    )

    ax.plot(
        weekly_error["week_of_year"],
        weekly_error["crime_count"],
        "b-",
        label="Actual (mean)",
        linewidth=2,
    )
    ax.plot(
        weekly_error["week_of_year"],
        weekly_error["prediction"],
        "g--",
        label="Predicted (mean)",
        linewidth=2,
    )
    ax.fill_between(
        weekly_error["week_of_year"],
        weekly_error["prediction"] - weekly_error["abs_error"],
        weekly_error["prediction"] + weekly_error["abs_error"],
        alpha=0.3,
        color="green",
        label="±MAE",
    )
    ax.set_xlabel("Week of Year")
    ax.set_ylabel("Crime Count (mean per grid cell)")
    ax.set_title("Model Predictions Over Time (Test Set)")
    ax.legend()

    plt.tight_layout()
    figures["model_temporal_performance"] = fig

    # 5. Error by Crime Volume (binned)
    fig, ax = plt.subplots(figsize=(10, 6))

    error_df = pd.DataFrame(
        {
            "actual": y_test,
            "predicted": y_test_pred,
            "abs_error": np.abs(y_test - y_test_pred),
            "pct_error": np.where(y_test > 0, np.abs(y_test - y_test_pred) / y_test * 100, 0),
        }
    )
    error_df["crime_bin"] = pd.cut(
        error_df["actual"], bins=[0, 1, 3, 5, 10, 100], labels=["0-1", "1-3", "3-5", "5-10", "10+"]
    )

    bin_stats = (
        error_df.groupby("crime_bin", observed=True)
        .agg({"abs_error": ["mean", "std"], "actual": "count"})
        .reset_index()
    )
    bin_stats.columns = ["crime_bin", "mae", "mae_std", "count"]

    bars = ax.bar(
        range(len(bin_stats)), bin_stats["mae"], yerr=bin_stats["mae_std"], capsize=5, alpha=0.7
    )
    ax.set_xticks(range(len(bin_stats)))
    ax.set_xticklabels(bin_stats["crime_bin"])
    ax.set_xlabel("Actual Crime Count Range")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Prediction Error by Crime Volume")

    # Add count annotations
    for _, (bar, count) in enumerate(zip(bars, bin_stats["count"], strict=False)):
        ax.annotate(
            f"n={count}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    figures["model_error_by_volume"] = fig

    return figures


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error."""
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def baseline_last_week(
    df: pd.DataFrame, target_col: str = "crime_count"
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline: predict using last week's value (lag1)."""
    return df[target_col].values, df["crime_count_lag1"].values


def baseline_rolling_mean(
    df: pd.DataFrame, target_col: str = "crime_count"
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline: predict using 4-week rolling mean."""
    return df[target_col].values, df["crime_count_rolling_mean_4"].values


def temporal_train_test_split(
    df: pd.DataFrame,
    test_weeks: int = 8,
    val_weeks: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data temporally: train -> validation -> test.

    For time series, we MUST split by time to avoid data leakage.
    """
    # Sort by time
    df = df.sort_values(["year", "week_of_year"]).reset_index(drop=True)

    # Get unique time periods
    time_periods = (
        df[["year", "week_of_year"]].drop_duplicates().sort_values(["year", "week_of_year"])
    )
    n_periods = len(time_periods)

    # Calculate split points
    test_start = n_periods - test_weeks
    val_start = test_start - val_weeks

    # Get the year/week values for splits
    val_cutoff = time_periods.iloc[val_start] if val_start > 0 else time_periods.iloc[0]
    test_cutoff = time_periods.iloc[test_start]

    # Split data
    train_mask = (df["year"] < val_cutoff["year"]) | (
        (df["year"] == val_cutoff["year"]) & (df["week_of_year"] < val_cutoff["week_of_year"])
    )
    test_mask = (df["year"] > test_cutoff["year"]) | (
        (df["year"] == test_cutoff["year"]) & (df["week_of_year"] >= test_cutoff["week_of_year"])
    )
    val_mask = ~train_mask & ~test_mask

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    return train_df, val_df, test_df


def train_model(
    input_path: Path,
    model_output_path: Path,
    model_type: str = "random_forest",
    test_weeks: int = 8,
) -> None:
    """Train a crime prediction model and log to MLflow.

    Uses temporal train/test split and compares against baselines.

    Args:
        input_path: Path to processed features (Parquet)
        model_output_path: Path to save the trained model
        model_type: Type of model to train ('random_forest' or 'poisson')
        test_weeks: Number of weeks to hold out for testing
    """
    print(f"Loading processed data from {input_path}...")

    # Load data
    if input_path.suffix == ".parquet":
        df = pl.read_parquet(input_path).to_pandas()
    else:
        df = pl.read_csv(input_path).to_pandas()

    print(f"Loaded {len(df)} records")
    print(f"Unique grid cells: {df['grid_id'].nunique()}")
    print(
        f"Date range: Year {df['year'].min()}-{df['year'].max()}, "
        f"Weeks {df['week_of_year'].min()}-{df['week_of_year'].max()}"
    )

    # Define features and target
    feature_columns = [
        # Lag features (autoregressive - most important)
        "crime_count_lag1",
        "crime_count_lag2",
        "crime_count_lag3",
        "crime_count_lag4",
        # Rolling statistics
        "crime_count_rolling_mean_4",
        "crime_count_rolling_std_4",
        "crime_count_rolling_mean_8",
        # Trend
        "crime_trend",
        # Seasonality (Fourier encoding - better than raw week)
        "week_sin",
        "week_cos",
        "week_sin2",
        "week_cos2",
        # Time context
        "month",
        "is_weekend_ratio",
    ]
    target_column = "crime_count"

    # Handle grid_id to lat/lon bins if needed
    if "grid_id" in df.columns and "lat_bin" not in df.columns:
        grid_size = 50
        df["lat_bin"] = df["grid_id"] // grid_size
        df["lon_bin"] = df["grid_id"] % grid_size

    # Temporal split (NO random split for time series!)
    print("\n--- Temporal Train/Val/Test Split ---")
    train_df, val_df, test_df = temporal_train_test_split(df, test_weeks=test_weeks, val_weeks=4)

    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    if len(train_df) < 10:
        print("WARNING: Very small training set. Results may be unreliable.")

    # Prepare features
    features_train = train_df[feature_columns]
    target_train = train_df[target_column]
    features_test = test_df[feature_columns]
    target_test = test_df[target_column]

    # Calculate baseline metrics on test set
    print("\n--- Baseline Models ---")
    baselines: dict[str, dict[str, float]] = {}

    # Baseline 1: Last week's value
    y_true, y_pred_baseline1 = baseline_last_week(test_df)
    baselines["baseline_last_week"] = {
        "mae": mean_absolute_error(y_true, y_pred_baseline1),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred_baseline1)),
        "r2": r2_score(y_true, y_pred_baseline1),
        "mape": compute_mape(y_true, y_pred_baseline1),
    }
    print(
        f"Baseline (last week): MAE={baselines['baseline_last_week']['mae']:.2f}, "
        f"R²={baselines['baseline_last_week']['r2']:.4f}"
    )

    # Baseline 2: Rolling mean
    y_true, y_pred_baseline2 = baseline_rolling_mean(test_df)
    baselines["baseline_rolling_mean"] = {
        "mae": mean_absolute_error(y_true, y_pred_baseline2),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred_baseline2)),
        "r2": r2_score(y_true, y_pred_baseline2),
        "mape": compute_mape(y_true, y_pred_baseline2),
    }
    print(
        f"Baseline (4-week avg): MAE={baselines['baseline_rolling_mean']['mae']:.2f}, "
        f"R²={baselines['baseline_rolling_mean']['r2']:.4f}"
    )

    # Set up MLflow
    mlflow.set_experiment("chicago-crime-prediction")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_weeks", test_weeks)
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("n_training_samples", len(train_df))
        mlflow.log_param("n_test_samples", len(test_df))
        mlflow.log_param("n_grid_cells", df["grid_id"].nunique())
        mlflow.log_param("split_type", "temporal")

        # === EDA PLOTS (Before Training) ===
        print("\n--- Generating EDA Plots ---")
        eda_figures = plot_eda(df, feature_columns)
        for fig_name, fig in eda_figures.items():
            mlflow.log_figure(fig, f"plots/{fig_name}.png")
            plt.close(fig)
        print(f"  Logged {len(eda_figures)} EDA figures to MLflow")

        # Create and train model
        print(f"\n--- Training {model_type} Model ---")
        model: Any
        if model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 15)
            mlflow.log_param("min_samples_split", 5)
        elif model_type == "poisson":
            model = PoissonRegressor(alpha=1.0, max_iter=1000)
            mlflow.log_param("alpha", 1.0)
            mlflow.log_param("max_iter", 1000)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(features_train, target_train)

        # Make predictions
        pred_train = model.predict(features_train)
        pred_test = model.predict(features_test)

        # Calculate metrics
        metrics: dict[str, float] = {
            "train_mae": mean_absolute_error(target_train, pred_train),
            "train_rmse": np.sqrt(mean_squared_error(target_train, pred_train)),
            "train_r2": r2_score(target_train, pred_train),
            "test_mae": mean_absolute_error(target_test, pred_test),
            "test_rmse": np.sqrt(mean_squared_error(target_test, pred_test)),
            "test_r2": r2_score(target_test, pred_test),
            "test_mape": compute_mape(target_test.values, pred_test),
        }

        # Compare to baselines
        best_baseline_mae = min(
            baselines["baseline_last_week"]["mae"],
            baselines["baseline_rolling_mean"]["mae"],
        )
        metrics["improvement_vs_baseline"] = (
            (best_baseline_mae - metrics["test_mae"]) / best_baseline_mae * 100
        )

        # Log metrics
        print("\n--- Model Performance ---")
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"  {metric_name}: {metric_value:.4f}")

        # Log baseline metrics
        for baseline_name, baseline_metrics in baselines.items():
            for metric_name, metric_value in baseline_metrics.items():
                mlflow.log_metric(f"{baseline_name}_{metric_name}", metric_value)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        # Save model locally
        output_file = Path(model_output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_file)
        print(f"\nModel saved to {output_file}")

        # Save model metadata
        run = mlflow.active_run()
        run_id = run.info.run_id if run else "unknown"
        metadata = {
            "model_type": model_type,
            "version": run_id[:8],
            "features": feature_columns,
            "metrics": metrics,
            "baselines": baselines,
            "training_samples": len(train_df),
            "test_samples": len(test_df),
            "n_grid_cells": int(df["grid_id"].nunique()),
            "split_type": "temporal",
        }

        metadata_file = output_file.parent / "model_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_file}")

        # Log feature importance for Random Forest
        feature_importances = None
        if model_type == "random_forest":
            feature_importances = model.feature_importances_
            importance = dict(zip(feature_columns, feature_importances, strict=True))
            print("\nFeature Importance:")
            for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
                print(f"  {feat}: {imp:.4f}")
            mlflow.log_dict(importance, "feature_importance.json")

        # === MODEL EVALUATION PLOTS (After Training) ===
        print("\n--- Generating Model Evaluation Plots ---")
        eval_figures = plot_model_evaluation(
            y_train=target_train.values,
            y_train_pred=pred_train,
            y_test=target_test.values,
            y_test_pred=pred_test,
            feature_names=feature_columns,
            feature_importances=feature_importances,
            test_df=test_df,
        )
        for fig_name, fig in eval_figures.items():
            mlflow.log_figure(fig, f"plots/{fig_name}.png")
            plt.close(fig)
        print(f"  Logged {len(eval_figures)} evaluation figures to MLflow")

        # Summary
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Model: {model_type}")
        print(f"Test MAE: {metrics['test_mae']:.2f} crimes/week")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        print(f"Best Baseline MAE: {best_baseline_mae:.2f}")
        if metrics["improvement_vs_baseline"] > 0:
            print(f"✓ Model beats baseline by {metrics['improvement_vs_baseline']:.1f}%")
        else:
            print(f"✗ Model underperforms baseline by {-metrics['improvement_vs_baseline']:.1f}%")
        print(f"MLflow run ID: {run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crime prediction model")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/features.parquet"),
        help="Input path for processed features",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/crime_model.joblib"),
        help="Output path for trained model",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "poisson"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--test-weeks",
        type=int,
        default=8,
        help="Number of weeks to hold out for testing",
    )

    args = parser.parse_args()
    train_model(args.input, args.output, args.model_type, args.test_weeks)
