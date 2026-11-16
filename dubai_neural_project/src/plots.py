"""Publication figures directly from exported experiment tables; no hand-entered metrics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import load_data

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"
FIG=OUT/"figures"
NAMES={"Median":"Медиана", "Ridge":"Ridge", "RandomForest":"Random Forest", "LightGBM":"LightGBM", "MLP":"MLP", "TabResNet":"TabResNet"}
COLORS={"Median":"#999999","Ridge":"#888055","RandomForest":"#638d74","LightGBM":"#e18c35","MLP":"#3077a4","TabResNet":"#9a4e79"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
    "axes.spines.right":False,"axes.grid":True,"grid.alpha":.2,"figure.dpi":150,"savefig.dpi":220})

def save(fig,name):
    fig.tight_layout(pad=1.5)
    fig.savefig(FIG/name,bbox_inches="tight",facecolor="white")
    plt.close(fig)

def main():
    FIG.mkdir(parents=True,exist_ok=True)
    sel=json.loads((OUT/"selection_frozen.json").read_text());neural=sel["best_neural"]
    raw=load_data(ROOT/"data/off_plan.csv")
    fig,ax=plt.subplots(1,2,figsize=(10,3.2))
    ax[0].hist(np.log10(raw.price_usd),bins=40,color="#3077a4",edgecolor="white")
    ax[0].set(xlabel="log₁₀ цены в USD",ylabel="Число объектов",title="Распределение полной цены")
    d=raw.groupby(raw.date_listed.dt.to_period("M")).price_usd.median()
    ax[1].plot(d.index.to_timestamp(),d.to_numpy()/1e6,color="#3077a4",lw=1.6)
    for date in ["2024-01-01","2025-01-01"]:ax[1].axvline(pd.Timestamp(date),color="#777",ls="--",lw=.9)
    ax[1].set(xlabel="Месяц размещения",ylabel="Медиана цены, млн USD",title="Состав объявлений по времени")
    ax[1].tick_params(axis="x",rotation=25)
    save(fig,"01_data.png")

    m=pd.read_csv(OUT/"metrics.csv");t=m[(m.split=="test")&(m.seed==42)]
    families=list(NAMES)
    fig,ax=plt.subplots(1,2,figsize=(10,3.4))
    for i,(target,title) in enumerate([("total","Обучение на полной цене"),("unit","Обучение на цене за м²")]):
        z=t[t.target==target].set_index("family").loc[families]
        ax[i].barh(range(len(z)),z.MAE_USD/1000,color=[COLORS[f] for f in families])
        ax[i].set(yticks=range(len(z)),yticklabels=[NAMES[f] for f in families],xlabel="MAE полной цены, тыс. USD",title=title)
        ax[i].invert_yaxis()
    save(fig,"02_comparison.png")

    lc=pd.read_csv(OUT/"learning_curves.csv")
    fig,ax=plt.subplots(1,2,figsize=(10,3.3))
    for f in ["MLP","TabResNet"]:
        tar=sel["target_per_family"][f];z=lc[(lc.family==f)&(lc.target==tar)&(lc.seed==42)]
        ax[0].plot(z.epoch,z.train_standardized_MSE,label=f,color=COLORS[f])
        ax[1].plot(z.epoch,z.val_MAE_USD/1000,label=f,color=COLORS[f])
        best=m[(m.family==f)&(m.target==tar)&(m.seed==42)&(m.split=="val")].iloc[0]
        ax[1].scatter(best.best_iteration,best.MAE_USD/1000,color=COLORS[f],s=25,zorder=4)
    ax[0].set(xlabel="Эпоха",ylabel="MSE стандартизованной log цели",yscale="log",title="Ошибка на обучении с dropout")
    ax[1].set(xlabel="Эпоха",ylabel="MAE, тыс. USD",title="Валидация и выбранная эпоха")
    for a in ax:a.legend(frameon=False)
    save(fig,"03_learning.png")

    pp=pd.read_csv(OUT/"predictions.csv")
    fig,ax=plt.subplots(1,2,figsize=(9.4,4))
    for a,f in zip(ax,[neural,"LightGBM"]):
        tar=sel["target_per_family"][f];z=pp[(pp.family==f)&(pp.target==tar)&(pp.seed==42)&(pp.split=="test")]
        a.scatter(z.actual_usd/1e6,z.predicted_usd/1e6,s=6,alpha=.4,color=COLORS[f],rasterized=True)
        lim=max(z.actual_usd.max(),z.predicted_usd.max())/1e6
        a.plot([.02,lim],[.02,lim],color="#444",ls="--",lw=1)
        a.set(xlabel="Заданная цена, млн USD",ylabel="Прогноз, млн USD",title=f+" на отложенном тесте")
    save(fig,"04_predictions.png")

    rr=pd.read_csv(OUT/"robustness_inputs_runs.csv")
    fig,ax=plt.subplots(1,3,figsize=(11,3.5))
    for a,(kind,title) in zip(ax,[("area","Ошибка площади ±30%"),("view","Замена вида из окна"),("missing","Пропуски площади и метро")]):
        for f in ["RandomForest","LightGBM","MLP","TabResNet"]:
            z=rr[(rr.kind==kind)&(rr.family==f)];v=z.groupby("share").MAE_USD.agg(["mean","std"])
            base=z.clean_MAE_USD.iloc[0]
            x=np.r_[0,v.index.to_numpy()*100];y=np.r_[base,v["mean"]]/1000;sd=np.r_[0,v["std"]]/1000
            a.plot(x,y,"o-",markersize=3,label=f,color=COLORS[f]);a.fill_between(x,y-sd,y+sd,color=COLORS[f],alpha=.1)
        a.set(title=title,xlabel="Искажённые строки, %",ylabel="MAE, тыс. USD",xticks=[0,5,10,20])
    ax[1].legend(frameon=False,fontsize=8)
    save(fig,"05_input_noise.png")

    rr=pd.read_csv(OUT/"robustness_labels_runs.csv")
    fig,ax=plt.subplots(1,2,figsize=(10,3.3))
    for f in ["RandomForest","LightGBM","MLP","TabResNet"]:
        z=rr[rr.family==f].groupby("share").agg(mae=("MAE_USD","mean"),sd=("MAE_USD","std"),growth=("relative_growth_pct","mean"))
        x=z.index.to_numpy()*100
        ax[0].errorbar(x,z.mae/1000,yerr=z.sd/1000,marker="o",ms=3,capsize=2,label=f,color=COLORS[f])
        ax[1].plot(x,z.growth,"o-",ms=3,label=f,color=COLORS[f])
    ax[0].set(xlabel="Повреждённые цены train, %",ylabel="MAE, тыс. USD",title="Среднее и SD трёх повторов")
    ax[1].set(xlabel="Повреждённые цены train, %",ylabel="Изменение MAE, %",title="Парное сравнение с чистым обучением")
    ax[0].legend(frameon=False,fontsize=8)
    save(fig,"06_label_noise.png")

    imp=pd.read_csv(OUT/"permutation_groups_val.csv")
    labels={"area":"Площадь","geography":"География и доступность","property":"Тип и спальни","project":"Параметры проекта","view":"Вид из окна","calendar_and_rate":"Календарь и ставка"}
    fig,ax=plt.subplots(1,2,figsize=(10,3.7))
    for a,f in zip(ax,[neural,"LightGBM"]):
        z=imp[imp.family==f].groupby("group").delta_MAE_USD.agg(["mean","std"]).sort_values("mean")
        a.barh(range(len(z)),z["mean"]/1000,xerr=z["std"]/1000,color=COLORS[f],capsize=2)
        a.set(yticks=range(len(z)),yticklabels=[labels[k] for k in z.index],xlabel="Прирост MAE на validation, тыс. USD",title=f)
    save(fig,"07_importance.png")

    scen=pd.read_csv(OUT/"scenarios.csv")
    fig,ax=plt.subplots(1,2,figsize=(10,3.3))
    for a,feature in zip(ax,["area_m2","mortgage_rate_at_listing"]):
        for f in [neural,"LightGBM","Ridge"]:
            z=scen[(scen.family==f)&(scen.feature==feature)]
            a.plot(z.value,z.median_paired_change_pct,"o-",label=f,color=COLORS[f])
        a.set(ylabel="Изменение прогноза, %")
    ax[0].set(xlabel="Множитель площади",title="Локальный сценарий площади")
    ax[1].set(xlabel="Ипотечная ставка в наборе, %",title="Диагностика реакции на ставку")
    ax[0].legend(frameon=False,fontsize=8)
    save(fig,"08_scenarios.png")

    aug=pd.read_csv(OUT/"augmentation.csv")
    fig,ax=plt.subplots(1,2,figsize=(10,3.3))
    for a,kind,title in zip(ax,["area","missing"],["Ошибка площади","Пропуски"]):
        for name,label,color in [("standard","Исходное обучение",COLORS[neural]),("augmented","С аугментацией","#328175")]:
            z=aug[(aug.training==name)&(aug.kind==kind)].groupby("share").MAE_USD.mean()
            base=aug[(aug.training==name)&(aug.kind=="clean")].MAE_USD.iloc[0]
            a.plot(np.r_[0,z.index.to_numpy()*100],np.r_[base,z.to_numpy()]/1000,"o-",label=label,color=color)
        a.set(title=title,xlabel="Искажённые строки, %",ylabel="MAE, тыс. USD")
    ax[0].legend(frameon=False,fontsize=8)
    save(fig,"09_augmentation.png")
    print("Figures:",len(list(FIG.glob('*.png'))))

if __name__=="__main__":main()
