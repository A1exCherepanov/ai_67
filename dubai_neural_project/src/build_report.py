"""Russian academic report. Every numerical result comes from exported CSV/JSON."""
from pathlib import Path
import json
import shutil
import pandas as pd
import numpy as np
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"
REPORT=ROOT/"report"
FIG=OUT/"figures"
NAMES={"Median":"Медианный ориентир","Ridge":"Ridge","RandomForest":"Random Forest", "LightGBM":"LightGBM","MLP":"MLP","TabResNet":"TabResNet"}
TARGET_LABEL={"total":"P","unit":"P/A"}
NUM_WORD={2:"два",3:"три"}

def fmt(x,d=0):
    return f"{x:,.{d}f}".replace(","," ").replace(".",",")

def fe(x,d=0):
    return f"{x:,.{d}f}"

def main():
    REPORT.mkdir(exist_ok=True)
    sel=json.loads((OUT/"selection_frozen.json").read_text())
    audit=json.loads((OUT/"data_audit.json").read_text())
    met=pd.read_csv(OUT/"metrics.csv")
    inp=pd.read_csv(OUT/"robustness_inputs_runs.csv");lab=pd.read_csv(OUT/"robustness_labels_runs.csv")
    aug=pd.read_csv(OUT/"augmentation.csv");boot=pd.read_csv(OUT/"bootstrap_intervals.csv")
    groups=pd.read_csv(OUT/"group_holdout_metrics.csv");coh=pd.read_csv(OUT/"cohort_metrics.csv")
    strict=pd.read_csv(OUT/"strict_chronology_metrics.csv");abl=pd.read_csv(OUT/"context_ablation.csv")
    scenarios=pd.read_csv(OUT/"scenarios.csv");worst=pd.read_csv(OUT/"worst_errors.csv")
    perm=pd.read_csv(OUT/"permutation_groups_val.csv")
    neural=sel["best_neural"]
    n_train,n_val,n_test=[audit["splits"][s]["n"] for s in ("train","val","test")]
    def result(f,target=None,split="test",seed=42):
        if target is None:target=sel["target_per_family"][f]
        return met[(met.family==f)&(met.target==target)&(met.split==split)&(met.seed==seed)].iloc[0]
    def conf(f):
        t=sel["target_per_family"][f];return sel["params"][f"{f}_{t}"]["params"]
    boost=result("LightGBM");nn=result(neural);mlp=result("MLP");ridge=result("Ridge")
    input_dim=int(boost.input_dim)
    importance=perm.groupby(["family","group"]).delta_MAE_USD.mean()

    doc=Document();sec=doc.sections[0]
    sec.page_width=Cm(21);sec.page_height=Cm(29.7)
    sec.top_margin=Cm(1.9);sec.bottom_margin=Cm(1.8);sec.left_margin=Cm(2.5);sec.right_margin=Cm(1.7)
    sec.header_distance=Cm(.7);sec.footer_distance=Cm(.7)
    for name in ["Normal","Title","Subtitle","Heading 1","Heading 2","Caption"]:
        s=doc.styles[name];s.font.name="Times New Roman";s.font.color.rgb=RGBColor(0,0,0)
        s.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"),"Times New Roman")
    normal=doc.styles["Normal"];normal.font.size=Pt(12)
    normal.paragraph_format.line_spacing=1.15;normal.paragraph_format.space_after=Pt(6)
    normal.paragraph_format.first_line_indent=Cm(.7)
    for h,size in [("Title",17),("Heading 1",14),("Heading 2",12)]:
        s=doc.styles[h];s.font.size=Pt(size);s.font.bold=True
        s.paragraph_format.first_line_indent=Cm(0)
        s.paragraph_format.space_before=Pt(10 if h!="Title" else 6);s.paragraph_format.space_after=Pt(6 if h=="Heading 2" else 9)
    doc.styles["Title"].paragraph_format.space_after=Pt(12)
    doc.styles["Caption"].font.size=Pt(10)
    foot=sec.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    fld=OxmlElement("w:fldSimple");fld.set(qn("w:instr"),"PAGE");foot._p.append(fld)
    foot.paragraph_format.first_line_indent=Cm(0)
    doc.core_properties.title="Сравнение нейросетевых моделей оценки недвижимости и их устойчивости к шуму"
    doc.core_properties.subject="Учебный проект по нейронным сетям в магистратуре"
    doc.core_properties.author=""
    doc.core_properties.keywords="нейронные сети, TabResNet, MLP, устойчивость, недвижимость, Ясницкий"
    md=[];counter={"table":0,"figure":0}

    def para(text,small=False,indent=True):
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
        if not indent:p.paragraph_format.first_line_indent=Cm(0)
        if p._p.getprevious() is not None and p._p.getprevious().tag==qn("w:tbl"):
            p.paragraph_format.space_before=Pt(7)
        r=p.add_run(text)
        if small:r.font.size=Pt(10.5)
        md.append(text+"\n");return p
    def head(text,level=1):
        doc.add_heading(text,level=level);md.append("#"*(level+1)+" "+text+"\n")
    def formula(text):
        p=doc.add_paragraph(text);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.space_after=Pt(6)
        md.append("`"+text+"`\n")
    def table(caption,headers,rows,widths=None,size=10):
        counter["table"]+=1;caption=f"Таблица {counter['table']} — {caption}"
        p=doc.add_paragraph(caption,style="Caption");p.paragraph_format.keep_with_next=True
        p.paragraph_format.first_line_indent=Cm(0)
        t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
        if widths is None:widths=[16.8/len(headers)]*len(headers)
        for c,w in zip(t.columns,widths):c.width=Cm(w)
        for c,h in zip(t.rows[0].cells,headers):c.text=str(h)
        header=OxmlElement("w:tblHeader");t.rows[0]._tr.get_or_add_trPr().append(header)
        for row in rows:
            cells=t.add_row().cells
            for cell,value in zip(cells,row):cell.text=str(value)
        borders=OxmlElement("w:tblBorders")
        for edge in ["top","left","bottom","right","insideH","insideV"]:
            el=OxmlElement("w:"+edge);el.set(qn("w:val"),"single");el.set(qn("w:sz"),"4");el.set(qn("w:color"),"D9D9D9");borders.append(el)
        t._tbl.tblPr.append(borders)
        for ri,row in enumerate(t.rows):
            trpr=row._tr.get_or_add_trPr();nr=OxmlElement("w:cantSplit");trpr.append(nr)
            for ci,cell in enumerate(row.cells):
                cell.width=Cm(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tcpr=cell._tc.get_or_add_tcPr();marg=OxmlElement("w:tcMar")
                for side,v in [("top",75),("bottom",75),("left",90),("right",90)]:
                    v_el=OxmlElement("w:"+side);v_el.set(qn("w:w"),str(v));v_el.set(qn("w:type"),"dxa");marg.append(v_el)
                tcpr.append(marg)
                sh=OxmlElement("w:shd");sh.set(qn("w:fill"),"E7EDF3" if ri==0 else ("F7F8FA" if ri%2==0 else "FFFFFF"));tcpr.append(sh)
                for pp in cell.paragraphs:
                    pp.paragraph_format.keep_with_next=ri<len(t.rows)-1
                    pp.paragraph_format.first_line_indent=Cm(0);pp.paragraph_format.space_after=Pt(0)
                    pp.paragraph_format.space_before=Pt(0);pp.paragraph_format.line_spacing=1.0
                    pp.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER
                    for rr in pp.runs:rr.font.size=Pt(size);rr.bold=ri==0
        md.append(caption+"\n\n| "+" | ".join(headers)+" |\n|"+"|".join(["---"]*len(headers))+"|\n"+"\n".join("| "+" | ".join(map(str,row))+" |" for row in rows)+"\n")
        return t
    def figure(name,caption,width=16.3):
        counter["figure"]+=1;caption=f"Рисунок {counter['figure']} — {caption}"
        p=doc.add_paragraph();p.paragraph_format.first_line_indent=Cm(0);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next=True;p.paragraph_format.space_after=Pt(3)
        p.add_run().add_picture(str(FIG/name),width=Cm(width))
        cp=doc.add_paragraph(caption,style="Caption");cp.alignment=WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.first_line_indent=Cm(0)
        md.append(f"![{caption}](../outputs/figures/{name})\n")

    # ---------------------------------------------------------------- title, annotation
    p=doc.add_paragraph("УДК 004.032.26");p.paragraph_format.first_line_indent=Cm(0);md.append("УДК 004.032.26\n")
    title=doc.add_paragraph("Сравнение нейросетевых моделей оценки недвижимости и их устойчивости к шуму",style="Title")
    title.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for r in title.runs:
        r.font.name="Times New Roman";r._element.get_or_add_rPr().rFonts.set(qn("w:cs"),"Times New Roman")
    p=doc.add_paragraph("Учебный проект по нейронным сетям, магистратура\nНоябрь 2026 г.",style="Subtitle")
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Cm(0);p.runs[0].font.size=Pt(11)
    md.append("# "+title.text+"\n")
    head("Аннотация")
    para(f"Рассматривается задача оценки стоимости строящейся недвижимости по табличным данным {fmt(audit['rows'])} объявлений Дубая. Две нейросети — многослойный персептрон (MLP) и сеть с остаточными связями (TabResNet) — сравниваются с линейной регрессией Ridge, случайным лесом, градиентным бустингом LightGBM и медианным ориентиром. Для каждой модели сопоставлены две цели обучения: полная цена и цена за квадратный метр. Качество оценено на временной выборке (2025 — апрель 2026 года) по MAE, RMSE, MAPE и R², устойчивость — при ошибках площади и категориального признака, пропусках и ошибках обучающих цен. По критерию validation выбрана LightGBM на цене за м²: MAE {fmt(boost.MAE_USD)} USD, MAPE {fmt(boost.MAPE_pct,2)}%, R² {fmt(boost.R2,4)}. Лучшая нейросеть {neural} получила MAE {fmt(nn.MAE_USD)} USD и R² {fmt(nn.R2,4)}. Методика исследования опирается на работы Л. Н. и В. Л. Ясницких [1–3]. Набор предположительно синтетический, поэтому выводы относятся к сравнению моделей, а не к рынку Дубая.")
    para("Ключевые слова: нейронные сети, MLP, остаточные связи, TabResNet, оценка недвижимости, цена за квадратный метр, временная валидация, устойчивость к шуму.",small=True)

    # ---------------------------------------------------------------- introduction
    head("Введение")
    para("Массовая оценка недвижимости — типичная задача регрессии на табличных данных: по характеристикам объекта и времени размещения нужно оценить его цену. Л. Н. и В. Л. Ясницкие предложили для неё комплексные нейросетевые модели, которые учитывают одновременно строительно-эксплуатационные параметры квартиры и показатели экономической среды [1, с. 72–74]. Прогностические свойства такой модели проверяются на примерах, не использованных при обучении, после чего обученную сеть исследуют вычислительными экспериментами: оценивают значимость входов и изменяют один входной параметр при фиксации остальных — методом «замораживания» [1, с. 75–79; 3, с. 5]. Во введении монографии [2, с. 6–7] отмечено, что модели без учёта внешней экономической среды теряют актуальность, а регрессионные модели закладывают допущение о характере зависимости и могут работать некорректно при переломе рыночной тенденции.")
    para("В настоящей работе этот подход применён к современному табличному набору и дополнен сравнением архитектур. Для табличных данных нейросети не всегда превосходят ансамбли деревьев; в работе [4] ResNet-подобная сеть названа сильным базовым вариантом, поэтому наряду с MLP исследована компактная сеть с остаточными связями TabResNet. Градиентный бустинг, случайный лес и линейная регрессия служат контролем: они показывают, даёт ли усложнение сети выигрыш на этих данных.")
    para("Цель работы — определить, как архитектура модели, постановка цели обучения и ошибки в данных влияют на качество оценки стоимости. Задачи: 1) проверить данные и подготовить признаки без утечки цены; 2) сравнить шесть подходов при двух целях — полной цене P и цене за квадратный метр P/A; 3) оценить устойчивость моделей к ошибкам площади, категориального признака, пропускам и ошибкам обучающих цен; 4) исследовать обученные модели методами Ясницких; 5) проверить перенос на поздний период и на новые проекты. Увеличение числа нейронов само по себе не считается улучшением: в работах [1, с. 73] и [3, с. 2] оптимальной оказывается небольшая сеть (три и четыре скрытых нейрона), а структуру выбирают испытанием вариантов. Поэтому модели сравниваются по ошибкам на отложенных данных и по разбросу повторных запусков.")

    # ---------------------------------------------------------------- 1 data
    head("1 Данные и подготовка признаков")
    head("1.1 Исходные данные",2)
    para(f"Исходный архив содержит пять таблиц: off_plan.csv ({fmt(audit['rows'])} объявлений о строящихся объектах), secondary_sales.csv (50 000), rentals.csv (25 000), area_prices_monthly.csv (6 384) и metro_stations.csv (55). Эксперимент выполнен на off_plan.csv; продажи вторичного жилья и аренда относятся к другим процессам ценообразования и с ним не объединяются. Структура файлов соответствует набору Dubai Real Estate: Sales & Rentals (2020–2026) на Kaggle [5]. Основная таблица содержит {audit['columns']} столбцов и {audit['unique_projects']} проекта; период объявлений — {audit['min_date']}–{audit['max_date']}. Медиана цены составляет {fmt(audit['price_median'])} USD, минимум — {fmt(audit['price_min'])} USD, максимум — {fmt(audit['price_max'])} USD; пропусков и повторов идентификатора нет. Оценивается заданная в наборе цена объявления в долларах США; она не тождественна цене зарегистрированной сделки.")
    figure("01_data.png","Распределение цен и месячные медианы исходного набора")
    para("Цены распределены в широком диапазоне, поэтому гистограмма на рис. 1 построена в логарифмической шкале. Месячная медиана зависит и от состава объектов, поэтому её нельзя читать как индекс рынка. Модели обучаются на логарифме цены, но все ошибки далее считаются в исходных USD.")
    head("1.2 Аудит данных",2)
    para(f"Проверка согласованности выявила {fmt(audit['listed_before_launch_year'])} записей ({fmt(100*audit['listed_before_launch_year']/audit['rows'],2)}%), у которых год объявления раньше года запуска проекта, и {audit['handover_before_launch_year']} записей, у которых передача раньше запуска. Для реального рынка такие нарушения неправдоподобны и указывают на предположительно синтетическое происхождение набора. Поэтому результаты характеризуют сравнение моделей на учебном наборе, а не рынок Дубая. Основной эксперимент сохраняет все {fmt(audit['rows'])} строк, чтобы не менять состав выборки после просмотра ошибок; влияние нарушений оценено отдельно (раздел 6).")
    para("Из входов исключены price_per_sqft_usd и price_per_m2_usd: они вычислены через искомую цену и дали бы утечку ответа. Идентификаторы объявления и проекта, а также название проекта тоже не используются. Из двух дублирующих столбцов площади оставлен area_m2.")
    head("1.3 Признаки и предобработка",2)
    table("Группы входных признаков",["Группа","Содержание"],[
        ["Размер и тип","Логарифм площади, спальни, категория и тип объекта"],
        ["Расположение","Широта, долгота, community, zone, расстояние до Burj Khalifa"],
        ["Доступность","Станция и линия метро, log(1 + время до метро)"],
        ["Проект","Застройщик, его категория, годы запуска и передачи, log(1 + число объектов), схема оплаты"],
        ["Характеристика объекта","Вид из окна"],
        ["Контекст","Число месяцев от января 2021 года, sin и cos месяца, ипотечная ставка"],],[4.,12.8])
    para(f"Признаки строятся детерминированно, без обращения к цене. Используются 13 числовых и 10 категориальных признаков; после one-hot кодирования вход имеет размерность {input_dim} и одинаков для всех моделей. Пропуски заполняются медианой, числовые признаки стандартизируются (StandardScaler); словарь категорий, медианы и масштабы оцениваются только на обучающей части, неизвестные категории кодируются нулями. Календарная точка отсчёта (январь 2021 года) задана константой. Таблицы цен по районам не присоединяются, так как момент доступности их агрегатов неизвестен; ипотечная ставка берётся из самого объявления.")

    # ---------------------------------------------------------------- 2 method
    head("2 Методика эксперимента")
    head("2.1 Разделение данных и выбор настроек",2)
    table("Временное разделение данных",["Часть","Период","Объектов","Назначение"],[
        ["Train","2021–2023",fmt(n_train),"Обучение и предобработка"],
        ["Validation","2024",fmt(n_val),"Настройки и ранняя остановка"],
        ["Test",f"2025–{audit['splits']['test']['end'][8:].lstrip('0')} апреля 2026",fmt(n_test),"Итоговая оценка"],],[2.5,5.2,2.4,6.7])
    para(f"Данные делятся по времени (табл. 2): так проверяется перенос на более поздний период, подобно проверке прогностических свойств на примерах, не участвовавших в обучении [1, с. 74]. Для каждой обучаемой модели и каждой цели на validation перебираются три заранее заданные конфигурации; всего проверено 32 сочетания алгоритма, цели и настроек (медианный ориентир настроек не имеет). Побеждает конфигурация с наименьшей MAE полной цены на 2024 год при seed 42; выбор записан в selection_frozen.json до расчёта тестовых метрик. Тестовый период известен команде по предыдущей версии проекта, поэтому он не является новой внешней выборкой, но его ответы не использовались ни для подбора настроек, ни для остановки, ни для выбора уровней шума. Одна validation-выборка служит и для остановки, и для выбора конфигурации: это ограничение небольшого поиска, а не вложенная кросс-валидация. Объём обучающей части ({fmt(n_train)}) значительно превышает 7·N_x + 15 = {fmt(7*input_dim+15)} при N_x = {input_dim} входов — критерий представительности из [1, с. 75].")
    head("2.2 Цели обучения и метрики",2)
    para("Для положительных цены P и площади A обучаются две цели: z = ln(P) и z = ln(P/A). Полная цена восстанавливается как exp(z_pred) и A_obs · exp(z_pred) соответственно, где A_obs — наблюдаемая (возможно, искажённая) площадь. Обе постановки оцениваются на одних и тех же объектах по полной цене; цена за м² — альтернативная цель на тех же строках, а не второй набор данных.")
    formula("MAE = mean(|P − P_pred|)")
    formula("RMSE = √mean((P − P_pred)²)")
    formula("MAPE = 100 · mean(|P − P_pred| / P)")
    formula("R² = 1 − sum((P − P_pred)²) / sum((P − mean(P))²)")
    para("MAE и RMSE выражены в USD, MAPE — в процентах; дополнительно считается доля прогнозов с ошибкой не более 20%. R² не является долей точных прогнозов. Определения соответствуют стандартным регрессионным метрикам [6]. В [1, с. 73–74] ошибка сети измеряется среднеквадратичной погрешностью, нормированной на размах цен; её проценты нельзя напрямую сравнивать с MAPE. Обратное экспоненцирование не даёт несмещённой оценки среднего цены; поправка на смещение не вводилась.")
    head("2.3 Модели",2)
    rows=[]
    for f in NAMES:
        t=sel["target_per_family"][f];c=conf(f);r=result(f)
        if f=="Median":desc="Медиана удельной цены train × наблюдаемая площадь"
        elif f=="Ridge":desc=f"L2-регуляризация, α = {fmt(c['alpha'])}"
        elif f=="RandomForest":desc=f"220 деревьев, min_samples_leaf = {c['min_samples_leaf']}, max_features = {fmt(c['max_features'],1)}"
        elif f=="LightGBM":desc=f"{c['num_leaves']} листьев, min_child_samples = {c['min_child_samples']}, шаг 0,03, {int(r.best_iteration)} итераций"
        elif f=="MLP":desc=f"{input_dim} → "+" → ".join(map(str,c['hidden']))+f" → 1, ReLU, dropout {fmt(c['dropout'],2)}"
        else:desc=f"{input_dim} → {c['width']}, {NUM_WORD.get(c['blocks'],c['blocks'])} остаточных блока, выход 1, dropout {fmt(c['dropout'],2)}"
        rows.append([NAMES[f],TARGET_LABEL[t],desc])
    table("Модели и конфигурации, выбранные по validation",["Модель","Цель","Конфигурация"],rows,[3.2,1.4,12.2])
    para("Медианный ориентир показывает качество без обучения зависимости от характеристик объекта. Ridge — линейная регрессия с L2-регуляризацией; случайный лес и LightGBM описывают нелинейные взаимодействия. Всем моделям подаются одинаковые one-hot входы, поэтому нативные категориальные возможности LightGBM не используются.")
    para("MLP — последовательность полносвязных слоёв с ReLU и dropout и линейным выходом; в отличие от сигмоидного персептрона из [1, с. 73] она глубже и обучается с dropout. TabResNet проецирует вход в пространство размерности 128 и применяет два остаточных блока; каждый блок содержит LayerNorm, Linear 128→256, ReLU, dropout, Linear 256→128 и dropout. После блоков следуют LayerNorm, ReLU и Linear 128→1.")
    formula("h_next = h + F(h)")
    para(f"Остаточная связь позволяет блоку обучать поправку к текущему представлению и сохраняет прямой путь сигнала; LayerNorm нормализует скрытые признаки внутри одного объекта. Реализация собственная: она использует LayerNorm вместо BatchNorm из [4] и не заявляется как точное воспроизведение архитектуры и результатов авторов. Выбранные MLP и TabResNet содержат {fmt(mlp.parameters)} и {fmt(nn.parameters)} обучаемых параметров; сеть, в четыре раза более крупная, должна подтвердить преимущество на отложенных данных.")
    head("2.4 Обучение нейросетей",2)
    para("Сети обучаются на MSE стандартизованного логарифма цели оптимизатором AdamW [8]: learning rate 0,001, weight decay 0,0001, batch size 256, ограничение нормы градиента 5, не более 220 эпох. После каждой эпохи вычисляется MAE на validation в исходных USD; сохраняются веса лучшей эпохи, обучение останавливается после 25 эпох без улучшения. LightGBM использует ту же метрику и ожидание в 100 итераций [7].")

    # ---------------------------------------------------------------- 3 results
    head("3 Результаты сравнения моделей")
    head("3.1 Качество на временном тесте",2)
    rows=[]
    for f in NAMES:
        for target in ["total","unit"]:
            r=result(f,target)
            rows.append([NAMES[f],TARGET_LABEL[target],fmt(r.MAE_USD/1000,1),fmt(r.RMSE_USD/1000,1),fmt(r.MAPE_pct,2),fmt(r.R2,4)])
    table(f"Полная цена на {fmt(n_test)} тестовых объектах при seed 42",["Модель","Цель","MAE тыс USD","RMSE тыс USD","MAPE %","R²"],rows,[3.2,1.2,3.1,3.1,2.7,2.5],9.5)
    def gain(f):
        a,b=result(f,"total").MAE_USD,result(f,"unit").MAE_USD
        return 100*(a-b)/a
    val_ridge=sel["params"]["Ridge_total"]["val_MAE_USD"]
    nn_gain=100*(mlp.MAE_USD-nn.MAE_USD)/mlp.MAE_USD
    para(f"В табл. 4 и на рис. 2 показаны ошибки полной цены при seed 42. По validation выбрана LightGBM с целью P/A: MAE {fmt(boost.MAE_USD)} USD, MAPE {fmt(boost.MAPE_pct,2)}%, R² {fmt(boost.R2,4)}. Цена за м² как цель снижает MAE полной цены у LightGBM на {fmt(gain('LightGBM'),1)}% и у случайного леса на {fmt(gain('RandomForest'),1)}%, улучшает результат MLP ({fmt(result('MLP','total').MAE_USD/1000,1)} → {fmt(mlp.MAE_USD/1000,1)} тыс. USD); для TabResNet по validation выбрана полная цена. Линейная Ridge на validation ошибается на {fmt(val_ridge/1000,1)} тыс. USD, а на тесте — на {fmt(ridge.MAE_USD/1000,1)} тыс. USD, то есть в {fmt(ridge.MAE_USD/val_ridge,1)} раза сильнее: зависимость, найденная на 2021–2023 годах, не переносится на 2025–2026. Выбранная TabResNet на {fmt(nn_gain,1)}% точнее выбранной MLP ({fmt(nn.MAE_USD/1000,1)} против {fmt(mlp.MAE_USD/1000,1)} тыс. USD), но её ошибка в {fmt(nn.MAE_USD/boost.MAE_USD,1)} раза больше, чем у LightGBM. Для ориентира: в [1, с. 74] на тесте получен R² = 0,87; у наших сетей он равен {fmt(mlp.R2,2)} и {fmt(nn.R2,2)}, но наборы данных несопоставимы.")
    figure("02_comparison.png","Ошибка полной стоимости при двух целях обучения")
    head("3.2 Обучение и разброс инициализаций",2)
    best_mlp,best_nn=int(mlp.best_iteration),int(nn.best_iteration)
    para(f"При seed 42 лучшие эпохи MLP и TabResNet равны {best_mlp} и {best_nn} (рис. 3). После них MAE на validation растёт, хотя ошибка на train продолжает снижаться: это ограничение обобщения, а не повод увеличивать число эпох. Из-за dropout кривая train не совпадает с метрикой сети в режиме оценки.")
    figure("03_learning.png","Ход обучения выбранных нейросетей и контроль на 2024 год")
    rows=[]
    for f in ["MLP","TabResNet"]:
        for target in ["total","unit"]:
            d=met[(met.family==f)&(met.target==target)&(met.split=="test")]
            rows.append([f,TARGET_LABEL[target],fmt(d.MAE_USD.mean()/1000,1)+" ± "+fmt(d.MAE_USD.std()/1000,1),fmt(d.R2.mean(),3)+" ± "+fmt(d.R2.std(),3)])
    table("Тест по трём инициализациям (seed 42, 43 и 44)",["Сеть","Цель","MAE тыс USD среднее ± SD","R² среднее ± SD"],rows,[3.1,1.5,7.1,5.1])
    tr_p=met[(met.family=="TabResNet")&(met.target=="total")&(met.split=="test")].MAE_USD
    para(f"Настройки повторных запусков фиксированы, лучший seed по тесту не выбирался. В основной таблице показан заранее заданный seed 42, а табл. 5 характеризует чувствительность к инициализации: например, MAE TabResNet на P равна {fmt(tr_p.mean()/1000,1)} ± {fmt(tr_p.std()/1000,1)} тыс. USD. Разброс сопоставим с различиями между сетями, поэтому рост сложности не гарантирует улучшения в каждом запуске. SD по трём запускам не является доверительным интервалом.")
    head("3.3 Структура ошибок",2)
    para(f"У LightGBM RMSE ({fmt(boost.RMSE_USD)} USD) в {fmt(boost.RMSE_USD/boost.MAE_USD,1)} раза выше MAE ({fmt(boost.MAE_USD)} USD): отдельные дорогие объекты дают крупные денежные отклонения. При этом {fmt(boost.within20_pct,2)}% прогнозов LightGBM имеют относительную ошибку не более 20%, а у TabResNet такая доля равна {fmt(nn.within20_pct,2)}% (рис. 4). Высокий R² нейросети поэтому нельзя отождествлять с высокой точностью оценки каждого объекта.")
    figure("04_predictions.png","Индивидуальные прогнозы выбранной нейросети и бустинга")
    br=boot[boot.metric=="MAE_USD"]
    rows=[[r.family,fmt(r.point/1000,1),fmt(r.low/1000,1)+"–"+fmt(r.high/1000,1)] for r in br.itertuples()]
    table("Кластерный bootstrap MAE при фиксированных моделях",["Модель","MAE тыс USD","95% интервал тыс USD"],rows,[4.,5.2,7.6])
    delta=boot[boot.metric=="paired_MAE_difference_USD"].iloc[0]
    para(f"Выполнено 2 000 повторных выборок по project_id: проект выбирается целиком со всеми его тестовыми объявлениями, что учитывает общность объектов одного проекта. Парная разность MAE TabResNet минус LightGBM равна {fmt(delta.point/1000,1)} тыс. USD, 95% интервал — {fmt(delta.low/1000,1)}–{fmt(delta.high/1000,1)} тыс. USD: преимущество бустинга устойчиво к составу проектов. Интервал условен на обученных моделях; он не учитывает изменение рынка, подбор настроек и синтетическую природу данных и не является прогнозным интервалом для отдельного объекта.")

    # ---------------------------------------------------------------- 4 robustness
    head("4 Устойчивость к ошибкам данных")
    head("4.1 Протокол",2)
    table("Воздействия на данные",["Воздействие","Что изменяется","Что остаётся эталоном"],[
        ["Ошибка площади test","У p строк площадь умножается на U[0,7; 1,3]","Исходная полная цена P"],
        ["Ошибка категории test","У p строк view заменяется другой категорией из train","Исходная полная цена P"],
        ["Пропуски test","По p строк отдельно для площади и времени до метро получают NaN","Исходная полная цена P"],
        ["Ошибка цены train","У p обучающих строк P умножается на U[0,5; 1,5]","Чистые validation и test"],],[3.4,7.4,6.])
    k=[round(s*n_test) for s in (.05,.1,.2)];kt=[round(s*n_train) for s in (.05,.1,.2)]
    para(f"Устойчивость проверяется четырьмя независимыми воздействиями (табл. 7). Уровень p = 5, 10 и 20% задан заранее и означает долю искажаемых строк, а не стандартное отклонение шума. Шум здесь — контролируемое воздействие для сравнения моделей: уровни не подбирались, чтобы понизить или улучшить метрики. Маски вложены (строки уровня 5% входят в 10% и 20%) и одинаковы для всех моделей; истинные цены теста не меняются. На тесте искажается {k[0]}, {k[1]} и {k[2]} строк в каждом поле, в обучающей части — {fmt(kt[0])}, {fmt(kt[1])} и {fmt(kt[2])} цен.")
    para("Ошибки входов повторяются 10 раз при фиксированной модели seed 42; разброс отражает маску повреждений. Модели используют цель и настройки, выбранные на чистой validation, не переобучаются, пропуски заполняются статистиками train. В опыте с пропусками два столбца получают независимые маски, поэтому доля строк хотя бы с одним пропуском может превышать p. Для моделей P/A полная цена восстанавливается по наблюдаемой — ошибочной или пропущенной (тогда подставляется медиана train) — площади: ошибка влияет и на признаки, и на множитель. При ошибках обучающих цен модель переобучается с теми же настройками, чистой validation и тем же правилом остановки; для каждого уровня выполнены три парных повтора (seed 42, 43, 44 с разными масками), результат сравнивается с чистым обучением при том же seed.")
    para("Устойчивость оценивается по абсолютной MAE и её относительному изменению 100·(MAE_noise/MAE_clean − 1): небольшой относительный рост у изначально неточной модели не означает хорошей оценки стоимости.")
    head("4.2 Ошибки входных характеристик",2)
    rows=[]
    for f in NAMES:
        r=result(f);vals=[]
        for kind in ["area","view","missing"]:
            z=inp[(inp.family==f)&(inp.kind==kind)&(inp.share==.2)]
            vals.append(fmt(z.MAE_USD.mean()/1000,1))
        rows.append([NAMES[f],fmt(r.MAE_USD/1000,1),*vals])
    figure("05_input_noise.png","Реакция моделей на три вида повреждения входов")
    table("MAE тыс USD при 20% повреждении каждого указанного поля",["Модель","Без шума","Площадь","Вид","Пропуски"],rows,[4.,3.2,3.2,3.2,3.2])
    z_area=inp[(inp.family=="LightGBM")&(inp.kind=="area")&(inp.share==.2)]
    z_miss=inp[(inp.family=="LightGBM")&(inp.kind=="missing")&(inp.share==.2)]
    z_nn_miss=inp[(inp.family==neural)&(inp.kind=="missing")&(inp.share==.2)]
    para(f"Для LightGBM (табл. 8, рис. 5) ошибка площади у 20% объектов повышает среднюю MAE до {fmt(z_area.MAE_USD.mean())} USD (на {fmt(z_area.relative_growth_pct.mean(),1)}%), а пропуски в площади и времени до метро — до {fmt(z_miss.MAE_USD.mean())} USD (на {fmt(z_miss.relative_growth_pct.mean(),1)}%): пропущенная площадь заменяется типичным значением train и особенно неточна для крупных дорогих объектов. Относительный рост у нейросетей меньше (у {neural} при пропусках — {fmt(z_nn_miss.relative_growth_pct.mean(),1)}%), но по абсолютной ошибке даже при 20% пропусков LightGBM ({fmt(z_miss.MAE_USD.mean()/1000,1)} тыс. USD) точнее {neural} без шума ({fmt(nn.MAE_USD/1000,1)} тыс. USD).")
    para("Замена вида из окна — узкий категориальный эксперимент: ошибки застройщика, района или типа объекта не моделировались, так как их массовая случайная замена создаёт противоречивые описания. Ошибка площади является ошибкой измерения, а не реальным изменением объекта, поэтому цена в этом опыте не меняется. Отсутствие монотонного роста у отдельной модели допустимо: искажение может случайно компенсировать смещение прогноза. Среднее ± SD по повторам для всех уровней сохранено в outputs/robustness_inputs_summary.csv.")
    head("4.3 Ошибки обучающих цен",2)
    rows=[];L={}
    for f in list(NAMES)[1:]:
        clean=lab[(lab.family==f)&(lab.share==0)];z=lab[(lab.family==f)&(lab.share==.2)]
        L[f]=(clean.MAE_USD.mean(),z.MAE_USD.mean(),z.MAE_USD.std(),z.relative_growth_pct.mean())
        rows.append([NAMES[f],fmt(L[f][0]/1000,1),fmt(L[f][1]/1000,1)+" ± "+fmt(L[f][2]/1000,1),fmt(L[f][3],1)])
    figure("06_label_noise.png","Переобучение при повреждении целевой переменной train")
    table("Результат трёх парных повторов при 20% ошибочных цен",["Модель","Чистое обучение MAE тыс USD","Шум MAE тыс USD среднее ± SD","Изменение MAE %"],rows,[3.7,4.1,5.6,3.4])
    para(f"При 20% ошибочных обучающих цен (табл. 9, рис. 6) средняя MAE LightGBM растёт на {fmt(L['LightGBM'][3],1)}% ({fmt(L['LightGBM'][0]/1000,1)} → {fmt(L['LightGBM'][1]/1000,1)} тыс. USD, SD {fmt(L['LightGBM'][2]/1000,1)} тыс.) и случайного леса — на {fmt(L['RandomForest'][3],1)}%, тогда как Ridge и MLP почти не меняются (+{fmt(L['Ridge'][3],1)}% и +{fmt(L['MLP'][3],1)}%). У {neural} средняя MAE снижается на {fmt(-L[neural][3],1)}% ({fmt(L[neural][0]/1000,1)} → {fmt(L[neural][1]/1000,1)} тыс. USD), но SD ({fmt(L[neural][2]/1000,1)} тыс.) превышает разницу средних. Этот результат не означает, что портить цены полезно: возможны регуляризация, смена эпохи остановки и случайная компенсация смещения на позднем периоде. Три повтора дают лишь ограниченную оценку разброса, и уровень шума по такому результату не подбирался. Validation остаётся чистой, а правило остановки сохраняется, поэтому эффект ошибок меток не смешивается с изменением процедуры обучения. Воздействие — мультипликативная ошибка цены на части строк, а не гауссовский шум.")
    head("4.4 Аугментация обучающих входов",2)
    para(f"Проверен один заранее заданный способ повысить устойчивость выбранной нейросети. К train добавлена копия строк, у 10% которых площадь искажена в пределах ±30%, а у 5% площадь и время до метро пропущены; исходные строки и цены сохранены ({fmt(2*n_train)} обучающих примеров), validation чистая, архитектура и правило остановки прежние.")
    rows=[];A={}
    for kind,share,label in [("clean",0.,"Чистый test"),("area",.2,"20% ошибок площади"),("missing",.2,"20% пропусков полей")]:
        vals=[]
        for training in ["standard","augmented"]:
            z=aug[(aug.kind==kind)&(aug.share==share)&(aug.training==training)]
            vals.append(float(z.MAE_USD.mean()))
        A[kind]=100*(vals[1]/vals[0]-1)
        rows.append([label,fmt(vals[0]/1000,1),fmt(vals[1]/1000,1),fmt(A[kind],1)])
    figure("09_augmentation.png","Обычное обучение и обучение с аугментацией входов")
    table("MAE выбранной нейросети тыс USD",["Условие","Исходное обучение","Аугментация","Изменение %"],rows,[5.4,4.,4.,3.4])
    para(f"На чистом тесте (табл. 10, рис. 7) MAE выросла на {fmt(A['clean'],1)}%, при 20% ошибок площади — на {fmt(A['area'],1)}%, а при 20% пропусков снизилась на {fmt(-A['missing'],1)}%: выигрыш в устойчивости достигнут ценой исходной точности, и оценивать эти значения нужно совместно. Опыт выполнен для одной сети и одного seed, поэтому не устанавливает универсального эффекта; это дополнительная интервенция, а не новый выбор победителя. Поскольку TabResNet обучается на полной цене, искажение площади не затрагивает цель обучения.")

    # ---------------------------------------------------------------- 5 interpretation
    head("5 Исследование обученных моделей")
    head("5.1 Значимость групп признаков",2)
    def imp(f,g):return importance[(f,g)]
    para(f"В [1, с. 75] значимость входа оценивается по росту ошибки тестирования при его исключении и сопоставляется с корреляциями Пирсона [1, с. 75–77]. Здесь использован близкий метод без переобучения — групповая перестановка признаков на validation: связанные поля (например, community, zone, координаты и метро) переставляются между объектами совместно, для каждой группы выполнено 10 перестановок, измеряется прирост MAE в USD (рис. 8). Наибольший прирост у LightGBM даёт площадь (+{fmt(imp('LightGBM','area')/1e6,2)} млн USD; для моделей P/A частично потому, что площадь участвует и в восстановлении цены), затем расположение (+{fmt(imp('LightGBM','geography')/1e6,2)} млн), тип и число спален (+{fmt(imp('LightGBM','property')/1e6,2)} млн), вид из окна (+{fmt(imp('LightGBM','view')/1e6,2)} млн) и параметры проекта (+{fmt(imp('LightGBM','project')/1e6,2)} млн); порядок у {neural} тот же. Календарь и ставка при перестановке внутри 2024 года ошибку не меняют ({fmt(imp('LightGBM','calendar_and_rate'))} USD у LightGBM): внутри одного года они почти не различаются, а их роль проявляется при переносе на другой период (п. 5.2). Значимость — чувствительность конкретной модели, а не причинный вклад фактора; перестановка внутри группы сохраняет связи между её полями, но может нарушать связи между группами.")
    figure("07_importance.png","Групповая перестановочная значимость на validation")
    head("5.2 Роль экономического контекста",2)
    rows=[];ab={}
    for f in ["LightGBM",neural]:
        base=result(f).MAE_USD;r=abl[abl.family==f].iloc[0];ab[f]=100*(r.MAE_USD/base-1)
        rows.append([f,fmt(base/1000,1),fmt(r.MAE_USD/1000,1),fmt(ab[f],1)])
    table("Абляция календаря и ипотечной ставки",["Модель","Исходная MAE тыс USD","Без контекста тыс USD","Изменение %"],rows,[3.6,4.7,5.,3.5])
    para(f"Чтобы проверить роль экономической среды, о которой говорится в [2, с. 6–7], месяц, календарный тренд и ипотечная ставка заменены константами train во всех частях, после чего модель переобучена с зафиксированными настройками. У LightGBM (табл. 11) исключение контекста увеличивает MAE на {fmt(ab['LightGBM'],1)}%: контекст нужен для переноса на 2025–2026 годы. У {neural}, напротив, MAE снижается на {fmt(-ab[neural],1)}%: сеть некорректно экстраполирует календарную зависимость и без неё переносится лучше. Наличие полезной информации не гарантирует, что архитектура использует её правильно. Результат не даёт оснований считать календарь и ставку причинными драйверами цены.")
    head("5.3 Сценарные вычислительные эксперименты",2)
    def rate(f,v):
        z=scenarios[(scenarios.family==f)&(scenarios.feature=="mortgage_rate_at_listing")&np.isclose(scenarios.value,v)]
        return z.median_paired_change_pct.iloc[0]
    area_out=scenarios[(scenarios.feature=="area_m2")&(scenarios.value.round(1).isin([.8,1.2]))].outside_train_range_pct
    para("Метод «замораживания» [1, с. 77–79; 3, с. 5] применён к 400 случайным объектам validation (seed 42): один вход изменяется, остальные фиксируются; для каждого объекта вычислено парное процентное изменение прогноза, затем найдена его медиана (рис. 9).")
    figure("08_scenarios.png","Локальная реакция прогнозов на площадь и ипотечную ставку")
    rows=[]
    for f in [neural,"LightGBM"]:
        z=scenarios[(scenarios.family==f)&(scenarios.feature=="area_m2")]
        for value in [.8,1.2]:
            r=z[np.isclose(z.value,value)].iloc[0]
            rows.append([f,fmt(value,1),fmt(r.median_paired_change_pct,2),fmt(r.outside_train_range_pct,2)])
    table("Парное изменение прогноза при изменении площади",["Модель","Множитель площади","Изменение цены %","За диапазоном train %"],rows,[3.4,4.5,4.4,4.5])
    para(f"У моделей удельной цены (табл. 12) заметная часть реакции на площадь заложена в формуле P_pred = A_obs · exp(z_pred), поэтому почти пропорциональный рост полной цены не доказывает, что модель открыла экономический закон. Доля объектов, у которых изменённая площадь выходит за диапазон train, составляет {fmt(area_out.min(),1)}–{fmt(area_out.max(),1)}%. Сценарий ставки (1,9–6,9%, диапазон train) служит только диагностикой: при ставке 1,9% медианный прогноз LightGBM ниже на {fmt(-rate('LightGBM',1.9),1)}%, а прогноз {neural} выше на {fmt(rate(neural,1.9),1)}%; знаки реакции различаются, а бустинг даёт ступенчатую зависимость. Ставка тесно связана с датой, поэтому её изменение при замороженном календаре создаёт сочетания, которых не было в обучении. Аналогичный эксперимент с объёмом кредитования в [1, с. 78–79] трактуется как прогноз влияния кредитной политики на рынок; для нашего учебного набора такая интерпретация не обоснована.")

    # ---------------------------------------------------------------- 6 transfer & limits
    head("6 Перенос на поздние даты и новые проекты, ограничения")
    head("6.1 Ошибка во времени",2)
    rows=[]
    quarters=coh[(coh.family=="LightGBM")&(coh.group=="quarter")]
    for r in quarters.itertuples():
        nr=coh[(coh.family==neural)&(coh.group=="quarter")&(coh.value==r.value)].iloc[0]
        rows.append([r.value+('*' if r.value=='2026Q2' else ''),r.n,fmt(r.MAE_USD/1000,1),fmt(nr.MAE_USD/1000,1)])
    table("Ошибка во времени на основном test",["Период","N","LightGBM MAE тыс USD","TabResNet MAE тыс USD"],rows,[3.,2.,5.9,5.9])
    nq=coh[(coh.family==neural)&(coh.group=="quarter")].MAE_USD
    lq=quarters.MAE_USD
    para(f"* Во втором квартале 2026 года представлен только апрель. Ошибка {neural} растёт с удалением от периода обучения (табл. 13; с {fmt(nq.iloc[0]/1000,1)} до {fmt(nq.iloc[-1]/1000,1)} тыс. USD), тогда как MAE LightGBM остаётся в пределах {fmt(lq.min()/1000,1)}–{fmt(lq.max()/1000,1)} тыс. USD. Это согласуется с проблемой экстраполяции календарной зависимости нейросетью (п. 5.2), но само по себе не доказывает единственную причину.",small=True)
    head("6.2 Новые проекты и качество дат",2)
    rows=[[r.family,r.n_train,r.n_val,r.n_test,fmt(r.MAE_USD/1000,1),fmt(r.R2,4)] for r in groups.itertuples()]
    table("Проверка с непересекающимися project_id",["Модель","Train","Val","Test","MAE тыс USD","R²"],rows,[3.5,2.,2.,2.,4.1,3.2])
    g=groups.iloc[0];gm=groups.set_index("family")
    para(f"Во втором эксперименте (табл. 14) {g.projects_train} проектов отданы в обучение, {g.projects_val} — в validation и {g.projects_test} — в test; даты внутри частей перемешаны. Это перенос на новые проекты при представленном во всех частях временном диапазоне. Параметры моделей и цель перенесены из основного опыта, повторного поиска нет. Здесь {neural} точнее LightGBM ({fmt(gm.loc[neural,'MAE_USD']/1000,1)} против {fmt(gm.loc['LightGBM','MAE_USD']/1000,1)} тыс. USD): порядок моделей зависит от вида обобщения, так как перенос на поздние даты и на новые проекты — разные задачи. Результат получен на одном групповом разбиении и одном seed и не доказывает общего превосходства нейросетей.")
    ss=strict.iloc[0]
    para(f"Дополнительная проверка качества дат удаляет из всех частей {fmt(ss.removed)} строк с нарушением запуска или передачи. Остаются {fmt(ss.n_train)} train, {fmt(ss.n_val)} validation и {fmt(ss.n_test)} test; в test представлены {int(ss.test_unseen_projects)} проектов, отсутствующих в очищенном train. MAE составляет: "+"; ".join(f"{r.family} — {fmt(r.MAE_USD/1000,1)} тыс. USD" for r in strict.itertuples())+". Изменились объём и покрытие обучения, поэтому эти ошибки нельзя объяснять только эффектом очистки; основной результат ими не заменяется.")
    head("6.3 Крупные ошибки",2)
    rows=[]
    for f in ["LightGBM",neural]:
        for r in worst[worst.family==f].head(3).itertuples():
            rows.append([f,r.id,fmt(r.price_usd/1e6,2),fmt(r.prediction_usd/1e6,2),fmt(r.APE_pct,1)])
    table("Три наибольшие денежные ошибки каждой модели",["Модель","ID","Цена млн USD","Прогноз млн USD","Ошибка %"],rows,[3.4,2.6,3.7,3.9,2.8])
    high=coh[(coh.family=="LightGBM")&(coh.group=="price_band")&(coh.value=="above_train_p90")].iloc[0]
    low=coh[(coh.family=="LightGBM")&(coh.group=="price_band")&(coh.value=="below_train_p50")].iloc[0]
    para(f"Наибольшие денежные ошибки (табл. 15) приходятся на самые дорогие объекты; объекты из test после анализа не удалялись. У LightGBM MAE в сегменте выше 90-го процентиля цены train равна {fmt(high.MAE_USD)} USD, а ниже медианы train — {fmt(low.MAE_USD)} USD при близкой MAPE ({fmt(high.MAPE_pct,2)}% и {fmt(low.MAPE_pct,2)}%): денежная ошибка растёт с ценой, а относительная точность остаётся прежней. Границы сегментов заданы по train и не подстраивались под тестовые цены.")
    head("6.4 Ограничения",2)
    para("Главное ограничение — предположительно синтетическое происхождение набора и нарушения дат: высокая точность может отражать восстановление правил генератора, а не закономерности реального рынка, а отсутствие пропусков не подтверждает качество реальных данных. Набор не содержит проверенного ряда сделок и истории публикации каждого признака. Кроме того, поиск настроек ограничен тремя конфигурациями, использован один основной временной разрез, три инициализации сетей, три повтора искажения цен и одна конфигурация аугментации, проверены лишь некоторые виды ошибок входов, а все временные части содержат одни и те же проекты. Для оценки прикладной пригодности нужен внешний набор с подтверждённым происхождением, ценами реальных сделок и несколькими последовательными временными разрезами. По имеющимся данным корректно сравнивать алгоритмы и их устойчивость, но нельзя использовать модель как инструмент инвестиционной или оценочной практики.")

    # ---------------------------------------------------------------- conclusion
    head("Заключение")
    ml_missing=z_miss.relative_growth_pct.mean()
    para(f"Выполнен полный цикл моделирования: аудит данных, подготовка признаков, разделение во времени, подбор настроек по validation, итоговая оценка, исследование моделей и проверка устойчивости. Среди шести подходов по критерию validation выбрана LightGBM на цене за квадратный метр: на тесте MAE равна {fmt(boost.MAE_USD)} USD, MAPE — {fmt(boost.MAPE_pct,2)}%, R² — {fmt(boost.R2,4)}. Переход к цене за м² снизил ошибку полной цены у LightGBM, случайного леса и MLP.")
    para(f"Остаточная сеть {neural} точнее MLP (MAE {fmt(nn.MAE_USD)} против {fmt(mlp.MAE_USD)} USD при seed 42), но уступает бустингу: её ошибка растёт к концу тестового периода, а исключение календаря её улучшает. На новых проектах порядок моделей меняется. Заметный разброс по инициализациям не позволяет считать усложнение архитектуры улучшением само по себе, что согласуется с подходом Ясницких: структура сети определяется проверкой на отложенных данных [1, с. 73; 3, с. 2].")
    para(f"Наиболее вредны для моделей пропуски и ошибки площади (у LightGBM при 20% пропусков MAE выше на {fmt(ml_missing)}%); 20% ошибок в обучающих ценах повышают MAE бустинга на {fmt(L['LightGBM'][3],1)}%. Аугментация входов повысила устойчивость сети к пропускам ценой точности на чистых данных. Методы Ясницких — отдельная проверка, оценка значимости входов и «замораживание» — реализованы как вычислительные эксперименты; наблюдаемые реакции не объявляются причинными законами, что особенно важно для предположительно синтетических данных.")
    para("Код полного цикла, обученные модели, индивидуальные прогнозы и таблицы воспроизводятся командой python src/experiment.py --stage all (графики — python src/plots.py), версии библиотек сохранены в outputs/environment.json, исходный seed — 42. Проверки (python -m unittest discover -s tests и python src/verify_results.py) подтверждают разделение выборок, исключение ценовых производных, отсутствие обучения предобработки на будущем и корректность искажения площади. Все таблицы отчёта формируются из результатов выполнения кода.")

    # ---------------------------------------------------------------- bibliography
    head("Библиографический список")
    refs=[
        "1. Ясницкий Л. Н., Ясницкий В. Л. Разработка и применение комплексных нейросетевых моделей массовой оценки и прогнозирования стоимости жилых объектов на примере рынков недвижимости Екатеринбурга и Перми // Имущественные отношения в Российской Федерации. 2017. № 3 (186). С. 68–84. URL: https://publications.hse.ru/articles/212942576.",
        "2. Ясницкий В. Л., Алексеев А. О., Ясницкий Л. Н. Массовая оценка и сценарное прогнозирование рыночной стоимости городской недвижимости на основе технологии нейросетевого моделирования: монография. М.: РУСАЙНС, 2019. ISBN 978-5-4365-4474-8. (Использован фрагмент: введение, с. 6–7.)",
        "3. Ясницкий Л. Н., Внукова О. В., Черепанов Ф. М. Прогноз результатов Олимпиады-2014 в мужском одиночном фигурном катании методами искусственного интеллекта // Современные проблемы науки и образования. 2014. № 1. URL: http://www.science-education.ru/115-11339.",
        "4. Gorishniy Y., Rubachev I., Khrulkov V., Babenko A. Revisiting Deep Learning Models for Tabular Data // Advances in Neural Information Processing Systems. 2021. Vol. 34. arXiv:2106.11959. URL: https://arxiv.org/abs/2106.11959.",
        "5. sergionefedov. Dubai Real Estate: Sales & Rentals (2020–2026) [Электронный ресурс] // Kaggle. URL: https://www.kaggle.com/datasets/sergionefedov/dubai-real-estate-sales-and-rentals-20202026. Локальная версия данных — CSV из предоставленного архива; условия генерации набора не подтверждены.",
        "6. Scikit-learn. Metrics and scoring: quantifying the quality of predictions. Regression metrics [Электронный ресурс]. URL: https://scikit-learn.org/stable/modules/model_evaluation.html (scikit-learn 1.8.0).",
        "7. LightGBM. early_stopping [Электронный ресурс]. URL: https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html (LightGBM 4.7.0).",
        "8. PyTorch. AdamW [Электронный ресурс]. URL: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html (PyTorch 2.6.0).",
    ]
    for r in refs:
        p=para(r,small=True,indent=False);p.alignment=WD_ALIGN_PARAGRAPH.LEFT
    para("Дата обращения к веб-ресурсам: ноябрь 2026 года.",small=True,indent=False)

    # ---------------------------------------------------------------- english annotation
    p=doc.add_paragraph("NEURAL NETWORK MODELS FOR REAL ESTATE VALUATION AND THEIR ROBUSTNESS TO NOISE")
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.space_before=Pt(14)
    p.runs[0].bold=True;md.append("**NEURAL NETWORK MODELS FOR REAL ESTATE VALUATION AND THEIR ROBUSTNESS TO NOISE**\n")
    para(f"The paper compares neural and classical regression models for estimating the price of off-plan properties on a tabular dataset of {fe(audit['rows'])} Dubai listings. A multilayer perceptron (MLP) and a residual network (TabResNet) are compared with Ridge regression, random forest, LightGBM and a median baseline; each model is trained on two targets, the full price and the price per square metre. The models are evaluated on a temporal test set (2025 – April 2026) by MAE, RMSE, MAPE and R², and their robustness is studied under area errors, categorical errors, missing values and corrupted training prices. LightGBM trained on the price per square metre was selected on validation (test MAE {fe(boost.MAE_USD)} USD, MAPE {fe(boost.MAPE_pct,2)}%, R² {fe(boost.R2,4)}); the best neural network, {neural}, reached MAE {fe(nn.MAE_USD)} USD and R² {fe(nn.R2,4)}. The methodology follows the work of L. N. and V. L. Yasnitsky on complex neural network models for mass real estate valuation. The dataset is presumably synthetic, so the conclusions concern the comparison of models rather than the real Dubai market.",small=True)
    para("Keywords: neural networks, MLP, residual network, TabResNet, real estate valuation, price per square metre, temporal validation, robustness to noise.",small=True)

    # Explicitly remove template borders and theme fonts; assignments alone do not override them in LO.
    for style in doc.styles:
        if style.type==1:
            style.font.color.rgb=RGBColor(0,0,0)
        for element in style.element.iter():
            if element.tag==qn("w:pBdr"):
                element.getparent().remove(element)
            elif element.tag==qn("w:rFonts"):
                for attr in list(element.attrib):
                    if "theme" in attr.lower():del element.attrib[attr]
    for p in doc.paragraphs:
        for border in list(p._p.iter(qn("w:pBdr"))):border.getparent().remove(border)
    doc.save(REPORT/"project_report.docx")
    (REPORT/"project_report.md").write_text("\n".join(md),encoding="utf-8")
    shutil.copy2(REPORT/"project_report.docx",ROOT.parent/"project_report.docx")
    print(REPORT/"project_report.docx")

if __name__=="__main__":main()
