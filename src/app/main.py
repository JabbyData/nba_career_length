"""
Module implementing API
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import json

class PlayerStats(BaseModel):
    gp: int
    min: float
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
    print(prediction)
    # return {"prediction": prediction}

