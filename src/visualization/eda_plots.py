
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_engine_life_distribution(
    df: pd.DataFrame,
) -> None:
    life = (
        df.groupby("engine_id")["cycle"]
        .max()
    )

    plt.figure(figsize=(10, 5))

    sns.histplot(
        life,
        bins=20,
        kde=True,
    )

    plt.title("Engine Life Distribution")
    plt.xlabel("Operational cycles until failure")
    plt.ylabel("Number of engines")

    plt.tight_layout()
    plt.show()


def plot_sensor_distribution(
    df: pd.DataFrame,
) -> None:
    sensor_columns = [
        c for c in df.columns
        if c.startswith("sensor_")
    ]

    melted = df[sensor_columns].melt(
        var_name="sensor",
        value_name="measurement",
    )

    plt.figure(figsize=(16, 7))

    sns.boxplot(
        data=melted,
        x="sensor",
        y="measurement",
    )

    plt.xticks(rotation=90)
    plt.title("Sensor Measurement Distributions")
    plt.tight_layout()
    plt.show()


def plot_sensor_over_cycles(
    df: pd.DataFrame,
    engine_id: int,
    sensors: list[str] | None = None,
) -> None:
    engine_data = df[
        df["engine_id"] == engine_id
    ].sort_values("cycle")

    if engine_data.empty:
        raise ValueError(
            f"Engine {engine_id} does not exist."
        )

    if sensors is None:
        sensors = [
            "sensor_2",
            "sensor_3",
            "sensor_4",
        ]

    fig, axes = plt.subplots(
        len(sensors),
        1,
        figsize=(12, 3 * len(sensors)),
        sharex=True,
    )

    if len(sensors) == 1:
        axes = [axes]

    for ax, sensor in zip(axes, sensors):
        ax.plot(
            engine_data["cycle"],
            engine_data[sensor],
        )

        ax.set_ylabel(sensor)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Operational cycle")

    fig.suptitle(
        f"Sensor Evolution — Engine {engine_id}"
    )

    plt.tight_layout()
    plt.show()


def plot_sensor_correlation(
    df: pd.DataFrame,
) -> None:
    sensor_columns = [
        c for c in df.columns
        if c.startswith("sensor_")
    ]

    corr = df[sensor_columns].corr()

    plt.figure(figsize=(14, 11))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=False,
    )

    plt.title("Sensor Correlation Matrix")
    plt.tight_layout()
    plt.show()


def plot_sensor_mean_by_cycle(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
) -> None:
    if sensors is None:
        sensors = [
            "sensor_2",
            "sensor_3",
            "sensor_4",
            "sensor_7",
        ]

    mean_df = (
        df.groupby("cycle")[sensors]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    for sensor in sensors:
        plt.plot(
            mean_df["cycle"],
            mean_df[sensor],
            label=sensor,
        )

    plt.xlabel("Operational cycle")
    plt.ylabel("Mean sensor measurement")
    plt.title("Mean Sensor Trends Across Engines")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
