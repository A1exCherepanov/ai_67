"""Recompute exported scores and validate experiment provenance without retraining."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from core import metrics,load_data,time_split

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"

def main():
    s=json.loads((OUT/"selection_frozen.json").read_text())
    assert s["source_sha256"]==hashlib.sha256((ROOT/"data/off_plan.csv").read_bytes()).hexdigest()
    raw=load_data(ROOT/"data/off_plan.csv");_,va,te=time_split(raw)
    search=pd.read_csv(OUT/"validation_search.csv")
    for f,target in s["target_per_family"].items():
        z=search[search.family==f]
        assert z.sort_values("MAE_USD").iloc[0].target==target
    pp=pd.read_csv(OUT/"predictions.csv");mm=pd.read_csv(OUT/"metrics.csv")
    checks=0
    for (f,target,seed,split),d in pp.groupby(["family","target","seed","split"]):
        truth=(va if split=="val" else te).set_index("id").loc[d.id]
        np.testing.assert_array_equal(d.actual_usd.to_numpy(),truth.price_usd.to_numpy())
        got=metrics(d.actual_usd,d.predicted_usd,d.area_m2)
        expected=mm[(mm.family==f)&(mm.target==target)&(mm.seed==seed)&(mm.split==split)].iloc[0]
        for key,value in got.items():np.testing.assert_allclose(value,expected[key],rtol=1e-9,atol=1e-7)
        checks+=1
    for name in ["split_assignments.csv","group_split_assignments.csv"]:
        d=pd.read_csv(OUT/name)
        assert d.id.nunique()==len(raw)
        assert d.groupby("id").split.nunique().max()==1
        if name.startswith("group"):
            assert d.groupby("project_id").split.nunique().max()==1
    x=pd.read_csv(OUT/"robustness_inputs_runs.csv")
    assert len(x)==3*3*10*6
    assert x.groupby(["kind","share","repeat"]).family.nunique().eq(6).all()
    y=pd.read_csv(OUT/"robustness_labels_runs.csv")
    assert len(y)==5*4*3
    assert y.groupby(["family","share"]).size().eq(3).all()
    print(f"PASS source hash; frozen validation selection; {checks} metric groups; disjoint splits; 540 input-noise runs; 60 label-noise records")

if __name__=="__main__":main()
