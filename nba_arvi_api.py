from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import tempfile
from nba_arvi_win_model import run_arvi_model

app = FastAPI()

@app.post("/api/run-arvi-model")
async def run_arvi_model_api(
    player_stats: UploadFile = File(...),
    team_wins: UploadFile = File(...)
):
    try:
        # Save uploaded files to temporary paths
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_player_file:
            player_stats_data = await player_stats.read()
            temp_player_file.write(player_stats_data)
            player_stats_path = temp_player_file.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_team_file:
            team_wins_data = await team_wins.read()
            temp_team_file.write(team_wins_data)
            team_wins_path = temp_team_file.name

        # Run model
        run_arvi_model(player_stats_path, team_wins_path)

        return JSONResponse(content={"status": "success", "message": "ARVI model run completed."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})