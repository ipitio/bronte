import logging
import os
import subprocess
import sys

reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
installed = [r.decode().split("==")[0] for r in reqs.split()]
failure = []
with open("requirements.txt", "r") as dependencies:
    for pkg in dependencies.readlines():  # to ignore errors
        # if pkg is a comment or empty line
        if pkg.startswith("#") or len(pkg.strip()) == 0:
            continue
        pkg = pkg.strip()
        args = f"pip install {pkg}"
        if pkg not in installed and subprocess.call(args.split()) != 0:
            failure.append(pkg)

if len(failure) > 0 and failure[0]:
    print("Failed to install the following packages: " + str(failure))
    print("Try installing them manually.")
    raise SystemExit(1)

import joblib
import numpy as np
import pandas as pd
from dask.distributed import Client
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, pool
from .src.main import Bronte

os.chdir(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__)

engine = create_engine(
    "sqlite:///data/data.db",
    connect_args={"check_same_thread": False},
    poolclass=pool.SingletonThreadPool,
    echo=True,
)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data/data.db"
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"connect_args": {"check_same_thread": False}}
db = SQLAlchemy(app)
prefix = "bronte_"
idx = 0


@app.route("/clear", methods=["GET"])
def clear():
    cursor = db.cursor()
    for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';"):
        cursor.execute(f"DROP TABLE {row[0]}")
    return "Database cleared"


@app.route("/load", methods=["POST"])
def load():
    data = request.get_json()["data"]
    df = pd.read_json(data)
    global idx
    df.to_sql(f"{prefix}{idx}", db, if_exists="replace", index=False)
    idx += 1
    return jsonify({"message": f"Data loaded into table {prefix}{idx}"}), 200


@app.route("/fit", methods=["POST"])
def fit():
    models = request.get_json()

    printedDashLink = False

    for task in models["tasks"]:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", db)

        for arch in models["archs"]:
            arch["client"] = Client(
                n_workers=task["n_workers"], silence_logs=logging.CRITICAL
            )
            if arch["client"] is not None and not task["colab"] and not printedDashLink:
                print(f'Dask dashboard is available at {arch["client"].dashboard_link}')
                printedDashLink = True

            trainer = Bronte(task | arch)
            with joblib.parallel_backend("dask"):
                for i, table in enumerate(tables["name"].tolist()):
                    data = pd.DataFrame()
                    if prefix in table:
                        data = pd.read_sql(f"SELECT * FROM {table}", db)
                        if task["verbose"]:
                            print("\nSplitting data and transforming features...")

                        df = data.copy()
                        if 0 < task["sample_size"] < 1:
                            df = df.sample(frac=task["sample_size"], random_state=42)
                        if 1 <= task["sample_size"] < len(df):
                            rg = np.random.default_rng(42)
                            n = rg.choice(len(df) - 1 - int(task["sample_size"]))
                            df = df.iloc[n : n + task["sample_size"], :].copy()

                        for col in df.columns:
                            if df[col].dtype == "object":
                                try:
                                    df[col] = pd.to_numeric(df[col])
                                except ValueError:
                                    pass

                        data = pd.get_dummies(df, dtype=np.int8, sparse=True)

                        if isinstance(task["targets"], str):
                            task["targets"] = [task["targets"]]
                        X = data.drop(columns=task["targets"])
                        y = data[task["targets"]]
                        if trainer.fit(X, y) != 0 or i >= task["samples"] - 1:
                            break

            if arch["client"] is not None:
                arch["client"].close()
                arch["client"] = None

    return jsonify({"message": "Models trained successfully"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    X = request.get_json()  # Get new data from the request

    models = []
    for root, dirs, files in os.walk("models"):
        if "checkpoints" in dirs:
            dirs.remove("checkpoints")
        for file in files:
            if file.endswith(".pt"):
                trainer = Bronte(path=os.path.join(root, file))
                models.append(trainer.model)

    print("Loaded models:")
    for model in models:
        print(f"\t{model.typeof}: {model.name} ({model.fits} fit(s))")

    predictions = [model.predict(X) for model in models]

    return jsonify(predictions), 200


if __name__ == "__main__":
    app.run(debug=True)
