"""Analyze student event feedback and generate charts plus an insights report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REQUIRED_COLUMNS = {"event_name", "overall_rating", "feedback"}
POSITIVE_WORDS = {"excellent", "great", "good", "helpful", "engaging", "clear", "useful", "amazing", "well"}
NEGATIVE_WORDS = {"poor", "bad", "boring", "late", "confusing", "unhelpful", "disorganized", "slow"}


def classify_sentiment(text: object) -> str:
    """Return a simple, transparent sentiment label for one feedback response."""
    words = set(str(text).lower().replace(".", " ").replace(",", " ").split())
    score = len(words & POSITIVE_WORDS) - len(words & NEGATIVE_WORDS)
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


def load_feedback(path: str | Path) -> pd.DataFrame:
    """Load a CSV, standardize column names, and remove invalid responses."""
    feedback = pd.read_csv(path)
    feedback.columns = feedback.columns.str.strip().str.lower().str.replace(" ", "_")
    missing = REQUIRED_COLUMNS - set(feedback.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    feedback = feedback.dropna(subset=["event_name", "overall_rating", "feedback"]).copy()
    feedback["overall_rating"] = pd.to_numeric(feedback["overall_rating"], errors="coerce")
    feedback = feedback.loc[feedback["overall_rating"].between(1, 5)].copy()
    feedback["sentiment"] = feedback["feedback"].map(classify_sentiment)
    return feedback.reset_index(drop=True)


def build_summary(feedback: pd.DataFrame) -> pd.DataFrame:
    """Calculate response count and average satisfaction for each event."""
    return (
        feedback.groupby("event_name", as_index=False)
        .agg(responses=("overall_rating", "size"), average_rating=("overall_rating", "mean"))
        .sort_values("average_rating", ascending=False)
    )


def write_outputs(feedback: pd.DataFrame, output_dir: str | Path) -> pd.DataFrame:
    """Save cleaned data, two charts, and a Markdown insights report."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = build_summary(feedback)
    feedback.to_csv(output / "cleaned_feedback.csv", index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=summary, x="average_rating", y="event_name", color="#4C78A8")
    plt.xlim(0, 5)
    plt.xlabel("Average rating (out of 5)")
    plt.ylabel("Event")
    plt.title("Student Satisfaction by Event")
    plt.tight_layout()
    plt.savefig(output / "event_satisfaction.png", dpi=160)
    plt.close()

    sentiment_order = ["positive", "neutral", "negative"]
    sentiment_counts = feedback["sentiment"].value_counts().reindex(sentiment_order, fill_value=0)
    plt.figure(figsize=(7, 4))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, hue=sentiment_counts.index, legend=False)
    plt.xlabel("Sentiment")
    plt.ylabel("Responses")
    plt.title("Feedback Sentiment Distribution")
    plt.tight_layout()
    plt.savefig(output / "sentiment_distribution.png", dpi=160)
    plt.close()

    best, lowest = summary.iloc[0], summary.iloc[-1]
    rating_columns = [column for column in feedback.columns if column.endswith("_rating")]
    category_means = feedback[rating_columns].mean(numeric_only=True).sort_values()
    recommendations = [
        f"Maintain the practices used in {best.event_name}, the highest-rated event ({best.average_rating:.2f}/5).",
        f"Prioritize improvements for {lowest.event_name}, the lowest-rated event ({lowest.average_rating:.2f}/5).",
    ]
    if not category_means.empty:
        recommendations.append(
            f"Focus on {category_means.index[0].replace('_', ' ')}, the lowest-rated survey category ({category_means.iloc[0]:.2f}/5)."
        )

    report = [
        "# Student Feedback Insights", "",
        f"- Responses analyzed: {len(feedback)}",
        f"- Overall average rating: {feedback.overall_rating.mean():.2f}/5",
        f"- Positive feedback: {(feedback.sentiment == 'positive').mean():.0%}", "",
        "## Recommendations",
        *[f"- {item}" for item in recommendations], "",
        "## Event Summary", "", summary.round(2).to_csv(index=False).rstrip(),
    ]
    (output / "insights.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze college event feedback.")
    parser.add_argument("--input", required=True, help="Path to the survey CSV file.")
    parser.add_argument("--output", default="outputs", help="Directory for generated files.")
    args = parser.parse_args()
    feedback = load_feedback(args.input)
    summary = write_outputs(feedback, args.output)
    print(f"Analyzed {len(feedback)} responses across {len(summary)} events.")
    print(f"Results saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
