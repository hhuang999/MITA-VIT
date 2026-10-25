import pandas as pd
import pytest

from mita_probe.evaluate import compare_runs


def test_compare_runs_aligns_images_not_row_order():
    left = pd.DataFrame({"index": [1, 2, 3], "agree": [1, 1, 0]})
    right = pd.DataFrame({"index": [3, 2, 1], "agree": [0, 0, 1]})
    result = compare_runs(left, right)
    assert result["difference"] == pytest.approx(1 / 3)
    assert result["ci_low"] == 0 and result["ci_high"] == 1


@pytest.mark.parametrize("problem", ["duplicate", "missing", "original"])
def test_compare_runs_rejects_incomparable_results(problem):
    left = pd.DataFrame({"index": [1, 2], "agree": [1, 0], "original": [5, 6]})
    right = left.copy()
    if problem == "duplicate":
        right.loc[1, "index"] = 1
    elif problem == "missing":
        right = right.iloc[:1]
    else:
        right.loc[0, "original"] = 9
    with pytest.raises(ValueError):
        compare_runs(left, right)
