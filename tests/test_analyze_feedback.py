import pandas as pd
import pytest

from analyze_feedback import classify_sentiment, load_feedback


def test_classify_sentiment():
    assert classify_sentiment("Excellent and helpful session") == "positive"
    assert classify_sentiment("Boring and confusing session") == "negative"
    assert classify_sentiment("The event was on Friday") == "neutral"


def test_load_feedback_rejects_missing_columns(tmp_path):
    csv_file = tmp_path / "feedback.csv"
    pd.DataFrame({"event_name": ["Workshop"]}).to_csv(csv_file, index=False)
    with pytest.raises(ValueError, match="overall_rating"):
        load_feedback(csv_file)
