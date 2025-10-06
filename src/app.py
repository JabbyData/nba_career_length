"""
Module implementing API predicting if player worth investing
"""
# Dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # adding working directory to python path

class PlayerStats(BaseModel):
    gp: int
    min: float
    pts: float
    fga: float
    fg_percent: float
    three_pa: float
    three_p_percent: float
    fta: float
    ft_percent: float
    oreb: float
    reb: float
    ast: float
    stl: float
    blk: float
    tov: float

app = FastAPI()

path_run_configs = os.path.join("src","run_configs.json")

@app.post("/predict/")
async def predict(player_stats: PlayerStats):

    # Load stats
    stats_dict = player_stats.model_dump()
    features = np.array([
        stats_dict['gp'],
        stats_dict['min'],
        stats_dict['pts'],
        stats_dict['fga'],
        stats_dict['fg_percent'],
        stats_dict['three_pa'],
        stats_dict['three_p_percent'],
        stats_dict['fta'],
        stats_dict['ft_percent'],
        stats_dict['oreb'],
        stats_dict['reb'],
        stats_dict['ast'],
        stats_dict['stl'],
        stats_dict['blk'],
        stats_dict['tov']
    ]).reshape(1, -1)

    # Load model
    with open(path_run_configs) as f:
        run_configs = json.load(f)["test"]
        weights_path = os.path.join(run_configs["weights_folder_path"],run_configs["model_name"]+".joblib")
        pipeline = joblib.load(weights_path)

    prediction = pipeline.predict(features)
    prediction_proba = pipeline.predict_proba(features)

    prediction_map = {
        0: "Career < 5Yrs",
        1: "Career >= 5Yrs"
    }
    
    return {
        "prediction": prediction_map[int(prediction[0])],
        "prediction_probability": {
            "Career < 5Yrs": float(prediction_proba[0][0]),
            "Career >= 5Yrs": float(prediction_proba[0][1])
        }
    }

