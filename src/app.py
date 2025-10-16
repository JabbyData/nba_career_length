"""
Module implementing API predicting if player worth investing
"""
# Dependencies
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # adding working directory to python path

class PlayerStats(BaseModel):
    gp: int = Field(..., ge=0, description="Games played", example=70)
    min: float = Field(..., ge=0, description="Minutes played per game", example=32.5)
    pts: float = Field(..., ge=0, description="Points per game", example=20.1)
    fga: float = Field(..., ge=0, description="Field goals attempted per game", example=15.3)
    fg_percent: float = Field(..., ge=0,le=100, description="Field goal percentage (0-100)", example=48.5)
    three_pa: float = Field(..., ge=0, description="Three-pointers attempted per game", example=6.2)
    three_p_percent: float = Field(..., ge=0, le=100, description="Three-point percentage (0-100)", example=37.2)
    fta: float = Field(..., ge=0, description="Free throws attempted per game", example=5.1)
    ft_percent: float = Field(..., ge=0, le=100, description="Free throw percentage (0-100)", example=85.0)
    oreb: float = Field(..., ge=0, description="Offensive rebounds per game", example=1.8)
    reb: float = Field(..., ge=0, description="Total rebounds per game", example=7.5)
    ast: float = Field(..., ge=0, description="Assists per game", example=5.3)
    stl: float = Field(..., ge=0, description="Steals per game", example=1.2)
    blk: float = Field(..., ge=0, description="Blocks per game", example=0.5)
    tov: float = Field(..., ge=0, description="Turnovers per game", example=2.1)

app = FastAPI()

path_run_configs = os.path.join("src","run_configs.json")

@app.post(
        "/predict/",
        summary="Predict NBA Player Career Length",
        response_description="Prediction of whether the player's career will last at least 5 years, with probabilities.",
        responses={
            status.HTTP_200_OK: {
                "description": "Successful prediction",
                "content": {
                    "application/json": {
                        "example": {
                            "prediction": "Career >= 5Yrs",
                            "prediction_probability": {
                                "Career < 5Yrs": 0.25,
                                "Career >= 5Yrs": 0.75
                            }
                        }
                    }
                }
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error (e.g., negative value, invalid percentage, or offensive rebounds > total rebounds)"
            }
        }
)
async def predict(player_stats: PlayerStats):
    """
    Predicts the career longevity of a basketball player based on their statistics.
    Args:
        player_stats (PlayerStats): An object containing the player's season statistics.
    Returns:
        dict: A dictionary with the predicted career category ("Career < 5Yrs" or "Career >= 5Yrs") and the associated probabilities for each category.
    Raises:
        AssertionError: If any input value is negative, if percentage values exceed 100, or if offensive rebounds exceed total rebounds.
    Notes:
        - The function loads a pre-trained model pipeline to make predictions.
        - Input statistics are validated for coherency before prediction.
    """

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

    if stats_dict['oreb'] > stats_dict['reb']:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"More offensive rebounds ({stats_dict['oreb']}) than total rebounds ({stats_dict['reb']})"
        )

    # Load model
    try:
        with open(path_run_configs) as f:
            run_configs = json.load(f)["test"]
            weights_path = os.path.join(run_configs["weights_folder_path"], run_configs["model_name"] + ".joblib")
            pipeline = joblib.load(weights_path)
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model configuration or weights file not found."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {str(e)}"
        )

    # Prediction
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

