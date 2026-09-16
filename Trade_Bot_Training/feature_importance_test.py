# filename: Trade_Bot_Traning/feature_importance_test.py

import numpy as np
import pandas as pd

from config_model import DATA_PATH, SYMBOLS
from feature_engine import FeatureTransformer
from training_utils import compute_weights, create_model


def train_feature_importance():
    df = pd.read_csv(DATA_PATH)
    features = FeatureTransformer().get_feature_list()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=features + ["target_class", "target_best_r", "symbol"], inplace=True)

    for symbol in SYMBOLS:
        print("\n" + "═" * 80)
        print(f"📈 FEATURE IMPORTANCE: {symbol}")
        data = df[df["symbol"] == symbol].copy()
        y = data["target_class"].to_numpy(dtype=np.int32)
        model = create_model()
        model.fit(
            data[features],
            y,
            sample_weight=compute_weights(y, data["target_best_r"].to_numpy()),
            verbose=False,
        )
        importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        for i, (feat, val) in enumerate(importance.head(20).items(), 1):
            print(f"{i:02d}. {feat:<30} {val:.6f}")


if __name__ == "__main__":
    train_feature_importance()
