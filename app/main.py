import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
from pymoo.optimize import minimize
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from helper.random import Random
from helper.reporter import Reporter
from pymoo_operators.menu_planning_problem import MenuPlanningProblem
from pymoo_runner import get_algorithm

app = FastAPI(debug=True)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
script_dir = os.path.dirname(__file__)
st_abs_file_path = os.path.join(script_dir, "static/")
templates_abs_file_path = os.path.join(script_dir, "templates/")

app.mount("/static", StaticFiles(directory=st_abs_file_path), name="static")

templates = Jinja2Templates(directory=templates_abs_file_path)


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/generate/")
async def create_upload_file(file: UploadFile):
    try:
        df = pd.read_csv(file.file)
        problem = MenuPlanningProblem(None, df=df)
        algorithm = get_algorithm(problem.conf.ALGORITHM, problem.conf.NUMBER_OF_POPULATION)
        reporter = Reporter(problem.conf)
        rand = Random()
        res = minimize(problem,
                       algorithm,
                       ('n_evals', problem.conf.MAXIMUM_EVALUATION),
                       seed=rand.random.get_state()[1][0],
                       save_history=True)
        ind_fitnesses = [x[0].total_fitness for x in res.X]
        best_sol = res.X[np.argmin(ind_fitnesses)]
        best_sol = best_sol[0]
        json_response = reporter.generate_solution_json(best_sol)
        return JSONResponse(content=json_response, status_code=200)
    except Exception as e:
        return {"error": e}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", log_level="info")
