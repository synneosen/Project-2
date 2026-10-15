import json
from pathlib import Path
from time import perf_counter

from sklearn.linear_model import LogisticRegression

from data import load_mnist
from metrics import accuracy


def main():
    (x_train, y_train), (x_val, y_val), _ = load_mnist(seed=2026)

    model = LogisticRegression(solver="saga", max_iter=1000, random_state=2026)
    start = perf_counter()
    model.fit(x_train, y_train)
    time_s = perf_counter() - start

    y_pred = model.predict(x_val)
    result = {
        "model": "sklearn LogisticRegression (solver=saga)",
        "n_train": int(len(y_train)),
        "val_accuracy": float(accuracy(y_val, y_pred)),
        "n_iter": int(model.n_iter_[0]),
        "max_iter": int(model.max_iter),
        "time_s": float(time_s),
        "seed": 2026,
    }

    results_path = Path(__file__).resolve().parent.parent / "results" / "logreg_baseline.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()