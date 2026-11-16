"""Score raw listing CSVs without any price-derived input columns."""
import argparse
from pathlib import Path
import joblib
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    # Only open locally trained/trusted joblib files; deserialization executes Python.
    model=joblib.load(a.model)
    d=pd.read_csv(a.input)
    d["predicted_price_usd"]=model.predict(d)
    d["predicted_price_usd_m2"]=d.predicted_price_usd/model.observed_area(d)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    d.to_csv(a.output,index=False)
    print(a.output)

if __name__=="__main__":main()
