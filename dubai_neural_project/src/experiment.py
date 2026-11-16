"""CLI for the full study. Stages intentionally separate selection and test evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from core import (FAMILIES, TARGETS, SEEDS, LEVELS, PriceModel, candidates, corrupt,
                  features, load_data, metrics, time_split)

ROOT = Path(__file__).resolve().parents[1]


def dump_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def audit(raw, out):
    tr, va, te = time_split(raw)
    a = {
        "rows": len(raw), "columns": len(raw.columns), "unique_projects": raw.project_id.nunique(),
        "duplicate_ids": int(raw.id.duplicated().sum()), "missing_cells": int(raw.isna().sum().sum()),
        "min_date": str(raw.date_listed.min().date()), "max_date": str(raw.date_listed.max().date()),
        "listed_before_launch_year": int((raw.launch_year > raw.date_listed.dt.year).sum()),
        "handover_before_launch_year": int((raw.handover_year < raw.launch_year).sum()),
        "area_conversion_max_difference_m2": float(np.abs(raw.area_m2 - .09290304 * raw.area_sqft).max()),
        "price_min": float(raw.price_usd.min()), "price_median": float(raw.price_usd.median()),
        "price_max": float(raw.price_usd.max()), "splits": {},
        "train_test_project_overlap": len(set(tr.project_id) & set(te.project_id)),
    }
    for name, d in [("train",tr),("val",va),("test",te)]:
        a["splits"][name] = {"n": len(d), "projects": d.project_id.nunique(),
            "start": str(d.date_listed.min().date()), "end": str(d.date_listed.max().date()),
            "median_price": float(d.price_usd.median())}
    dump_json(out / "data_audit.json", a)
    assign = raw[["id", "project_id", "date_listed"]].copy()
    assign["split"] = np.select([raw.date_listed < "2024-01-01",raw.date_listed < "2025-01-01"],
                                  ["train","val"], default="test")
    assign.to_csv(out / "split_assignments.csv", index=False)


def tune(raw, out):
    train, val, _ = time_split(raw)
    rows, selected = [], {}
    for family in FAMILIES:
        for target in TARGETS:
            best = np.inf
            for i, params in enumerate(candidates(family)):
                log(f"TUNE {family} {target} candidate {i+1}")
                m = PriceModel(family, target, params, 42).fit(train, val)
                score = metrics(val.price_usd, m.predict(val), val.area_m2)
                rows.append(dict(family=family,target=target,candidate=i,params=json.dumps(params),
                                 best_iteration=m.best_iteration,seconds=m.fit_seconds,**score))
                log(f"VAL MAE {score['MAE_USD']:,.0f}; {m.best_iteration=}; time={m.fit_seconds:.1f}s")
                if score["MAE_USD"] < best:
                    best = score["MAE_USD"]
                    selected[f"{family}_{target}"] = {"params":params,"val_MAE_USD":best,
                                                       "best_iteration":m.best_iteration}
                    joblib.dump(m, out / "models" / f"{family}_{target}_42.joblib", compress=3)
                pd.DataFrame(rows).to_csv(out / "validation_search.csv", index=False)
    family_targets = {f: min(TARGETS, key=lambda t:selected[f"{f}_{t}"]["val_MAE_USD"]) for f in FAMILIES}
    best_neural = min(["MLP", "TabResNet"], key=lambda f:selected[f"{f}_{family_targets[f]}"]["val_MAE_USD"])
    best_overall = min(FAMILIES, key=lambda f:selected[f"{f}_{family_targets[f]}"]["val_MAE_USD"])
    # This manifest is written BEFORE any test metric is computed by this program.
    manifest = {"selection_rule":"lowest 2024 validation MAE in USD; seed 42; no test scores",
        "timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "source_sha256":hashlib.sha256((ROOT/"data/off_plan.csv").read_bytes()).hexdigest(),
        "seed":42,"replicate_seeds":SEEDS,"input_noise_repeats":10,"label_noise_repeats":3,
        "noise_levels":LEVELS,"params":selected,"target_per_family":family_targets,
        "best_neural":best_neural,"best_overall":best_overall,
        "augmentation_policy":{"area_share":.1,"missing_share":.05,"copies":1},
        "neural_max_epochs":220,"neural_patience":25,"batch_size":256}
    dump_json(out / "selection_frozen.json", manifest)
    log(f"SELECTION FROZEN: {family_targets}; best neural={best_neural}; overall={best_overall}")


def get_model(out, family, target, seed=42):
    return joblib.load(out / "models" / f"{family}_{target}_{seed}.joblib")


def evaluate(raw, out, selection):
    train, val, test = time_split(raw)
    rows, predictions, histories = [], [], []
    for family in FAMILIES:
        for target in TARGETS:
            for seed in (SEEDS if family in ["MLP","TabResNet"] else [42]):
                log(f"EVAL {family} {target} seed={seed}")
                if seed == 42:
                    m = get_model(out,family,target,seed)
                else:
                    p = selection["params"][f"{family}_{target}"]["params"]
                    m = PriceModel(family,target,p,seed).fit(train,val)
                    joblib.dump(m,out/"models"/f"{family}_{target}_{seed}.joblib",compress=3)
                for split,d in [("train",train),("val",val),("test",test)]:
                    pred = m.predict(d)
                    rows.append(dict(family=family,target=target,seed=seed,split=split,
                         best_iteration=m.best_iteration,fit_seconds=m.fit_seconds,
                         input_dim=m.input_dim,parameters=getattr(m,"parameter_count",None),
                         **metrics(d.price_usd,pred,d.area_m2)))
                    if split in ["val","test"]:
                        predictions.append(pd.DataFrame({"id":d.id.to_numpy(),"family":family,
                            "target":target,"seed":seed,"split":split,"actual_usd":d.price_usd.to_numpy(),
                            "predicted_usd":pred,"area_m2":d.area_m2.to_numpy(),
                            "project_id":d.project_id.to_numpy(),"date_listed":d.date_listed.to_numpy()}))
                histories.extend([dict(family=family,target=target,seed=seed,**h) for h in m.history])
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/"predictions.csv",index=False)
    pd.DataFrame(histories).to_csv(out/"learning_curves.csv",index=False)


def robustness(raw,out,selection):
    train,val,test = time_split(raw)
    pool = {"view":set(train.view)}
    rows,labels = [],[]
    clean_models={f:get_model(out,f,selection["target_per_family"][f]) for f in FAMILIES}
    clean_maes={f:metrics(test.price_usd,m.predict(test))["MAE_USD"] for f,m in clean_models.items()}
    for kind in ["area","view","missing"]:
        for share in LEVELS:
            for repeat in range(10):
                # A single corrupted table is reused across ALL models at this repeat.
                xn = corrupt(test,kind,share,8100+repeat,train_pools=pool)
                for family in FAMILIES:
                    target = selection["target_per_family"][family]
                    m = clean_models[family]
                    base = clean_maes[family]
                    mm = metrics(test.price_usd,m.predict(xn))
                    rows.append(dict(kind=kind,share=share,repeat=repeat,family=family,target=target,
                            clean_MAE_USD=base,relative_growth_pct=100*(mm["MAE_USD"]/base-1),**mm))
            log(f"INPUT NOISE {kind} {share} complete")
    pd.DataFrame(rows).to_csv(out/"robustness_inputs_runs.csv",index=False)
    pd.DataFrame(rows).groupby(["kind","share","family","target"])[
        ["MAE_USD","RMSE_USD","MAPE_pct","R2","relative_growth_pct"]].agg(["mean","std"]).to_csv(out/"robustness_inputs_summary.csv")
    for family in FAMILIES[1:]:
        target=selection["target_per_family"][family]
        params=selection["params"][f"{family}_{target}"]["params"]
        for repeat,seed in enumerate(SEEDS):
            if family in ["MLP","TabResNet"]:
                base=get_model(out,family,target,seed)
            else:
                base=(get_model(out,family,target) if seed==42 else PriceModel(family,target,params,seed).fit(train,val))
            clean=metrics(test.price_usd,base.predict(test))["MAE_USD"]
            for share in [0.]+LEVELS:
                log(f"LABEL NOISE {family} {share} repeat {repeat}")
                if share==0:
                    model=base
                else:
                    noisy=corrupt(train,"price",share,9100+repeat)
                    model=PriceModel(family,target,params,seed).fit(noisy,val)
                mm=metrics(test.price_usd,model.predict(test))
                labels.append(dict(family=family,target=target,share=share,repeat=repeat,seed=seed,
                    clean_MAE_USD=clean,relative_growth_pct=100*(mm["MAE_USD"]/clean-1),
                    best_iteration=model.best_iteration,**mm))
                pd.DataFrame(labels).to_csv(out/"robustness_labels_runs.csv",index=False)
    pd.DataFrame(labels).groupby(["share","family","target"])[["MAE_USD","MAPE_pct","relative_growth_pct"]].agg(["mean","std"]).to_csv(out/"robustness_labels_summary.csv")
    # Fixed additional experiment. Hyperparameters and corruption policy are not tuned on test.
    family=selection["best_neural"];target=selection["target_per_family"][family]
    params=selection["params"][f"{family}_{target}"]["params"]
    noisy=corrupt(train,"area",.1,12000)
    noisy=corrupt(noisy,"missing",.05,12001)
    augtrain=pd.concat([train,noisy],ignore_index=True)
    aug=PriceModel(family,target,params,42).fit(augtrain,val)
    joblib.dump(aug,out/"models"/"augmented_neural.joblib",compress=3)
    rr=[]
    base=get_model(out,family,target)
    for kind in ["clean","area","view","missing"]:
        for share in ([0.] if kind=="clean" else LEVELS):
            for repeat in range(10 if kind!="clean" else 1):
                xn=test if kind=="clean" else corrupt(test,kind,share,8100+repeat,pool)
                for name,m in [("standard",base),("augmented",aug)]:
                    rr.append(dict(training=name,family=family,target=target,kind=kind,share=share,
                                   repeat=repeat,**metrics(test.price_usd,m.predict(xn))))
    pd.DataFrame(rr).to_csv(out/"augmentation.csv",index=False)


def diagnostics(raw,out,selection):
    train,val,test=time_split(raw)
    neural=selection["best_neural"]
    families=list(dict.fromkeys(["Ridge","LightGBM",neural]))
    models={f:get_model(out,f,selection["target_per_family"][f]) for f in families}
    groups={
        "area":["area_m2"],
        "geography":["community","zone","lat","lon","metro_station","metro_line",
                     "metro_distance_min","to_burj_khalifa_km"],
        "property":["bedrooms","property_category","property_type"],
        "project":["developer","developer_tier","launch_year","handover_year","total_project_units","payment_plan"],
        "view":["view"],
        "calendar_and_rate":["date_listed","mortgage_rate_at_listing"],
    }
    # Importance on validation only; permute correlated raw columns jointly.
    rows=[]
    for f,m in models.items():
        base=metrics(val.price_usd,m.predict(val))["MAE_USD"]
        for label,cols in groups.items():
            for repeat in range(10):
                rng=np.random.default_rng(11000+repeat)
                perm=rng.permutation(len(val));x=val.copy()
                x[cols]=val[cols].iloc[perm].to_numpy()
                change=metrics(val.price_usd,m.predict(x))["MAE_USD"]-base
                rows.append(dict(family=f,group=label,repeat=repeat,delta_MAE_USD=change))
    pd.DataFrame(rows).to_csv(out/"permutation_groups_val.csv",index=False)
    # Cohort diagnostics: no refits or architecture selection using these errors.
    cohorts=[];worst=[]
    for f,m in models.items():
        p=m.predict(test);z=test.copy();z["prediction_usd"]=p;z["error_usd"]=p-z.price_usd
        z["absolute_error_usd"]=np.abs(z.error_usd);z["APE_pct"]=100*z.absolute_error_usd/z.price_usd
        z["quarter"]=z.date_listed.dt.to_period("Q").astype(str)
        thresholds=train.price_usd.quantile([.5,.9]).to_numpy()
        z["price_band"]=pd.cut(z.price_usd,[0,*thresholds,np.inf],labels=["below_train_p50","train_p50_p90","above_train_p90"])
        z["chronology_valid"]=((z.launch_year<=z.date_listed.dt.year)&(z.handover_year>=z.launch_year)).map({True:"consistent",False:"inconsistent"})
        for group in ["quarter","property_category","price_band","chronology_valid"]:
            for value,d in z.groupby(group,observed=True):
                cohorts.append(dict(family=f,group=group,value=str(value),**metrics(d.price_usd,d.prediction_usd,d.area_m2)))
        top=z.nlargest(20,"absolute_error_usd").copy();top["family"]=f;worst.append(top)
    pd.DataFrame(cohorts).to_csv(out/"cohort_metrics.csv",index=False)
    pd.concat(worst).to_csv(out/"worst_errors.csv",index=False)
    # Project-cluster bootstrap paired delta: neural minus LightGBM.
    proj=test.project_id.to_numpy();unique=np.unique(proj)
    abs_errors={f:np.abs(m.predict(test)-test.price_usd.to_numpy()) for f,m in models.items()}
    sums={f:np.array([e[proj==g].sum() for g in unique]) for f,e in abs_errors.items()}
    counts=np.array([(proj==g).sum() for g in unique])
    rng=np.random.default_rng(13000);idx=rng.integers(0,len(unique),(2000,len(unique)))
    boot={f:s[idx].sum(axis=1)/counts[idx].sum(axis=1) for f,s in sums.items()}
    intervals=[]
    for f,b in boot.items():
        lo,hi=np.quantile(b,[.025,.975]);intervals.append(dict(metric="MAE_USD",family=f,
            point=float(abs_errors[f].mean()),low=float(lo),high=float(hi),cluster="project_id",clusters=len(unique),repeats=2000))
    delta=boot[neural]-boot["LightGBM"];lo,hi=np.quantile(delta,[.025,.975])
    intervals.append(dict(metric="paired_MAE_difference_USD",family=neural+" minus LightGBM",
        point=float(abs_errors[neural].mean()-abs_errors["LightGBM"].mean()),low=float(lo),high=float(hi),
        cluster="project_id",clusters=len(unique),repeats=2000))
    pd.DataFrame(intervals).to_csv(out/"bootstrap_intervals.csv",index=False)
    # Out-of-project diagnostic across all dates (NOT a temporal forecast).
    gss=GroupShuffleSplit(n_splits=1,train_size=.7,random_state=42)
    ti,tmp=next(gss.split(raw,groups=raw.project_id));other=raw.iloc[tmp]
    vi,ei=next(GroupShuffleSplit(n_splits=1,train_size=.5,random_state=43).split(other,groups=other.project_id))
    gt,gv,ge=raw.iloc[ti],other.iloc[vi],other.iloc[ei]
    assign=pd.concat([d[["id","project_id"]].assign(split=s) for s,d in [("train",gt),("val",gv),("test",ge)]])
    assign.to_csv(out/"group_split_assignments.csv",index=False)
    rows=[]
    for f in families:
        target=selection["target_per_family"][f];params=selection["params"][f"{f}_{target}"]["params"]
        log(f"PROJECT HOLDOUT {f}")
        m=PriceModel(f,target,params,42).fit(gt,gv)
        rows.append(dict(family=f,target=target,n_train=len(gt),n_val=len(gv),n_test=len(ge),
            projects_train=gt.project_id.nunique(),projects_val=gv.project_id.nunique(),
            projects_test=ge.project_id.nunique(),**metrics(ge.price_usd,m.predict(ge),ge.area_m2)))
    pd.DataFrame(rows).to_csv(out/"group_holdout_metrics.csv",index=False)
    # Frozen-feature scenarios use validation examples and observed support checks.
    base=val.sample(n=min(400,len(val)),random_state=42).copy();rows=[]
    for f,m in models.items():
        original=m.predict(base)
        for feature,grid in [("area_m2",[.8,.9,1.,1.1,1.2]),
                            ("mortgage_rate_at_listing",[1.9,3.,4.5,5.9,6.9])]:
            for value in grid:
                x=base.copy()
                x[feature]=base[feature]*value if feature=="area_m2" else value
                pred=m.predict(x)
                support=train[feature]
                rows.append(dict(family=f,feature=feature,value=value,
                    median_prediction_usd=float(np.median(pred)),
                    median_paired_change_pct=float(np.median(100*(pred/original-1))),
                    outside_train_range_pct=float(100*np.mean((x[feature]<support.min())|(x[feature]>support.max())))))
    pd.DataFrame(rows).to_csv(out/"scenarios.csv",index=False)
    # In-distribution ablation of contextual variables on the identical temporal split.
    rows=[]
    for f in ["LightGBM",neural]:
        target=selection["target_per_family"][f];params=selection["params"][f"{f}_{target}"]["params"]
        at,av,ae=train.copy(),val.copy(),test.copy()
        # Constants remove rate and calendar information while keeping the preprocessing schema.
        for d in [at,av,ae]:
            d["date_listed"]=pd.Timestamp("2021-01-01")
            d["mortgage_rate_at_listing"]=float(train.mortgage_rate_at_listing.median())
        log(f"CONTEXT ABLATION {f}")
        m=PriceModel(f,target,params,42).fit(at,av)
        rows.append(dict(family=f,target=target,**metrics(ae.price_usd,m.predict(ae),ae.area_m2)))
    pd.DataFrame(rows).to_csv(out/"context_ablation.csv",index=False)


def sensitivity(raw,out,selection):
    """Exploratory sensitivity to date-quality rules; frozen model configurations."""
    good=(raw.launch_year<=raw.date_listed.dt.year)&(raw.handover_year>=raw.launch_year)
    clean=raw.loc[good].copy();tr,va,te=time_split(clean)
    rows=[]
    for f in ["Ridge","LightGBM",selection["best_neural"]]:
        target=selection["target_per_family"][f];params=selection["params"][f"{f}_{target}"]["params"]
        log(f"STRICT CHRONOLOGY {f}")
        model=PriceModel(f,target,params,42).fit(tr,va)
        rows.append(dict(family=f,target=target,removed=int((~good).sum()),n_train=len(tr),n_val=len(va),
                         n_test=len(te),train_projects=tr.project_id.nunique(),
                         test_unseen_projects=len(set(te.project_id)-set(tr.project_id)),
                         **metrics(te.price_usd,model.predict(te),te.area_m2)))
    pd.DataFrame(rows).to_csv(out/"strict_chronology_metrics.csv",index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["all","tune","evaluate","robustness","diagnostics","sensitivity"],default="all")
    p.add_argument("--out",type=Path,default=ROOT/"outputs")
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);(out/"models").mkdir(exist_ok=True)
    raw=load_data(ROOT/"data/off_plan.csv")
    if args.stage in ["all","tune"]:
        audit(raw,out);tune(raw,out)
    s=json.loads((out/"selection_frozen.json").read_text())
    if args.stage in ["all","evaluate"]:evaluate(raw,out,s)
    if args.stage in ["all","robustness"]:robustness(raw,out,s)
    if args.stage in ["all","diagnostics"]:diagnostics(raw,out,s)
    if args.stage in ["all","sensitivity"]:sensitivity(raw,out,s)
    import sklearn,torch,lightgbm,scipy
    dump_json(out/"environment.json",dict(python=sys.version,platform=platform.platform(),
        numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,torch=torch.__version__,
        lightgbm=lightgbm.__version__,scipy=scipy.__version__,torch_threads=torch.get_num_threads()))
    log("COMPLETE "+args.stage)


if __name__=="__main__":main()
