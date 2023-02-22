import os
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pyunpack import Archive
import pandas as pd
from .MetaDbBuilder import Searcher

zip_embedded_df_fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), "embedded_dataframes", "embedded_dataframes.7z")
embedded_df_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "embedded_dataframes")
Archive(zip_embedded_df_fp).extractall(directory=embedded_df_dir)



md_df_fp = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "metadata_dataframes", "2023-02-18-GMD-metadata.pkl")

if os.path.exists(md_df_fp):
    df = pd.read_pickle(md_df_fp)
else:
    raise FileNotFoundError(f'{md_df_fp} not found. Use `DataSourceBuilder` to build and pickle the dataframe first.')

searcher = Searcher(df = df, model = 'multi-qa-MiniLM-L6-cos-v1')

app = FastAPI(
    title="NDSH MetaSearch API",
    description="currently only support CEDA metadata",
    version="0.0.1",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "ChrisLam",
        "email": "chris.lam@manchester.ac.uk",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    }
)

class Query(BaseModel):
    query: str = Field(default="find me datasets showing precipitation in the uk for the last 20 years", example="find me datasets showing precipitation in the uk for the last 20 years", description="semtantic search query")
    query_col: str = Field(default="abstract", example="abstract", description="column to search on, if the column had not been embedded, it will be embeded on the fly.")
    topk: int = Field(default=5, example = 5, description="number of results to return")
    show_columns: Optional[list[str]] = Field(default=None, example=["identifier", "title", "abstract", "abstract_emb", "scores"], description="optional, list of columns to show in the results, available column: {df.columns}, embedding column pattern: {col}_emb")


class SearchResult(BaseModel):
    result: list[dict] = Field(exmaple = [{"identifier": "123", "title": "test", "abstract": "test", "abstract_emb": [0.1, 0.2, 0.3], "scores": 0.9}, {"identifier": "123", "title": "test", "abstract": "test", "abstract_emb": [0.1, 0.2, 0.3], "scores": 0.9}, {"identifier": "123", "title": "test", "abstract": "test", "abstract_emb": [0.1, 0.2, 0.3], "scores": 0.9}], description="list of results")

@app.get("/")
def home():
    return {
        "greeting": "Welcome to the CEDA Metadata Search API, please visit /docs for experiment",
        "current model": searcher.model_name,
        "health_check": "OK", 
        }


@app.post("/search", response_model=SearchResult)
def search(payload: Query):
    print(payload)
    results: pd.DataFrame = searcher.search(
        query = payload.query,
        col = payload.query_col,
        k = payload.topk)
    
    if payload.show_columns:
        results = results[payload.show_columns]
    
    return {'result': results.to_dict(orient="records")}
    