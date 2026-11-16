"""Create and execute the review notebook against existing experiment artifacts."""
from pathlib import Path
import sys
import os
import base64
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT=Path(__file__).resolve().parents[1]

def main():
    cells=[]
    def md(t):cells.append(nbf.v4.new_markdown_cell(t))
    def code(t):cells.append(nbf.v4.new_code_cell(t))
    md("""# Нейросетевые модели стоимости недвижимости и устойчивость к шуму

Завершённый учебный эксперимент на `off_plan.csv`: MLP и TabResNet в сравнении с Ridge,
Random Forest, LightGBM и медианным ориентиром. Две цели обучения: `ln(P)` и `ln(P/A)`.

Основной протокол: train 2021–2023, validation 2024, test 2025–апрель 2026.
Все результаты ниже получены реальным запуском; ноутбук читает сохранённые CSV и модели.
Повторный полный расчёт включается явно в последней ячейке и может занять несколько минут.

Набор рассматривается как учебный с предполагаемым синтетическим происхождением.
Выводы не переносятся на действительный рынок недвижимости без внешней проверки.
""")
    code("""from pathlib import Path
import json, sys, subprocess
import pandas as pd
from IPython.display import display, Image

ROOT = Path.cwd()
if not (ROOT / 'src/core.py').exists():
    if (ROOT / 'dubai_neural_project/src/core.py').exists():
        ROOT = ROOT / 'dubai_neural_project'
    else:
        raise FileNotFoundError('Распакуйте полный ZIP и откройте ноутбук внутри dubai_neural_project')
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'outputs'
selection = json.loads((OUT / 'selection_frozen.json').read_text())
print('Общий победитель по validation:', selection['best_overall'])
print('Выбранная нейросеть по validation:', selection['best_neural'])
display(pd.Series(selection['target_per_family'], name='Цель').to_frame())
""")
    md("## Аудит и воспроизводимость\nИсходный файл проверяется по SHA-256; статистики предобработки рассчитываются только на train. Ценовые производные и идентификаторы не являются входами.")
    code("""audit = json.loads((OUT / 'data_audit.json').read_text())
display(pd.DataFrame(audit['splits']).T)
print('Объявлений раньше года запуска:', audit['listed_before_launch_year'])
print('Передача раньше запуска:', audit['handover_before_launch_year'])
display(pd.Series(json.loads((OUT / 'environment.json').read_text()), name='Версия').to_frame())
display(Image(filename=str(OUT / 'figures/01_data.png')))
""")
    md("## Итоговый временной тест\nОсновной seed 42 зафиксирован заранее. Цель внутри каждого семейства выбрана по MAE полной цены на validation. Следующая таблица не выбирает модель по тесту.")
    code("""metrics = pd.read_csv(OUT / 'metrics.csv')
test = metrics[(metrics['split']=='test') & (metrics['seed']==42)].copy()
chosen = test[test.apply(lambda r: r['target']==selection['target_per_family'][r['family']], axis=1)]
display(chosen[['family','target','n','MAE_USD','RMSE_USD','MAPE_pct','R2','within20_pct']].round(3))
display(test[['family','target','MAE_USD','RMSE_USD','MAPE_pct','R2']].round(3))
display(Image(filename=str(OUT / 'figures/02_comparison.png')))
display(Image(filename=str(OUT / 'figures/04_predictions.png')))
""")
    md("## Повторные инициализации нейросетей\nСреднее и стандартное отклонение по seeds 42, 43 и 44. Выбор лучшего seed по тесту не выполняется.")
    code("""neural = metrics[(metrics['split']=='test') & metrics.family.isin(['MLP','TabResNet'])]
display(neural.groupby(['family','target'])[['MAE_USD','MAPE_pct','R2']].agg(['mean','std']).round(3))
display(Image(filename=str(OUT / 'figures/03_learning.png')))
""")
    md("## Ошибки входов\nПо отдельности: площадь ±30% у p строк, замена view у p строк, пропуски в площади и времени до метро по p строк каждого поля. 10 одинаковых для всех моделей масок на уровень. Истинные цены сохраняются. Для P/A восстановление полной цены использует наблюдаемую, в том числе повреждённую, площадь.")
    code("""inputs = pd.read_csv(OUT / 'robustness_inputs_runs.csv')
display(inputs[inputs.share==0.2].groupby(['family','kind'])[['MAE_USD','relative_growth_pct']].agg(['mean','std']).round(2))
display(Image(filename=str(OUT / 'figures/05_input_noise.png')))
""")
    md("## Ошибки обучающих цен\nУ p строк train цена умножается на U[0.5, 1.5]. Чистая validation и правило ранней остановки сохраняются. Три парных повтора; baseline для каждого seed считается отдельно.")
    code("""labels = pd.read_csv(OUT / 'robustness_labels_runs.csv')
display(labels.groupby(['family','share'])[['MAE_USD','relative_growth_pct']].agg(['mean','std']).round(2))
display(Image(filename=str(OUT / 'figures/06_label_noise.png')))
display(Image(filename=str(OUT / 'figures/09_augmentation.png')))
""")
    md("## Неопределённость и альтернативные проверки\nBootstrap выбирает целые тестовые проекты и сравнивает фиксированные модели. Групповой holdout проверяет новые проекты и смешивает даты. Строгая хронология меняет объём и покрытие train; результаты не заменяют основной benchmark.")
    code("""display(pd.read_csv(OUT / 'bootstrap_intervals.csv').round(2))
display(pd.read_csv(OUT / 'group_holdout_metrics.csv').round(3))
display(pd.read_csv(OUT / 'strict_chronology_metrics.csv').round(3))
display(pd.read_csv(OUT / 'context_ablation.csv').round(3))
""")
    md("## Интерпретация и вычислительные сценарии\nГрупповая permutation importance рассчитывается на validation. Метод изменения одного входа связан с подходом Ясницких, но реакция функции модели не является причинным эффектом.")
    code("""display(Image(filename=str(OUT / 'figures/07_importance.png')))
display(Image(filename=str(OUT / 'figures/08_scenarios.png')))
display(pd.read_csv(OUT / 'scenarios.csv').round(3))
""")
    md("## Применение сохранённой модели\nТри демонстрационные строки взяты из исходного набора и не являются новым тестом. Поля истинной цены и ценовых производных убраны из входа.")
    code("""import joblib
family = selection['best_overall']
target = selection['target_per_family'][family]
model = joblib.load(OUT / 'models' / f'{family}_{target}_42.joblib')
demo = pd.read_csv(ROOT / 'data/off_plan.csv').head(3).drop(columns=['price_usd','price_per_sqft_usd','price_per_m2_usd'])
forecast = demo[['id','date_listed','community','area_m2']].copy()
forecast['predicted_price_usd'] = model.predict(demo)
display(forecast.round(2))
""")
    md("## Проверки сохранённых результатов\nПроверяются временные границы, отсутствие утечки цены в признаки, обучение предобработки только на train, правила повреждения и пересчёт метрик из индивидуальных прогнозов.")
    code("""for command in [
    [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
    [sys.executable, 'src/verify_results.py'],
]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    print(completed.stdout)
    print(completed.stderr)
""")
    md("## Полный повтор эксперимента\nСохранённые результаты доступны сразу. Чтобы заново обучить все модели и повторы, измените `RUN_FULL_EXPERIMENT` на `True`. Требуются зависимости из `requirements.txt`. В Colab сначала распакуйте весь архив в рабочий каталог; одной загрузки ipynb недостаточно.")
    code("""RUN_FULL_EXPERIMENT = False
if RUN_FULL_EXPERIMENT:
    subprocess.run([sys.executable, 'src/experiment.py', '--stage', 'all'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, 'src/plots.py'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, 'src/verify_results.py'], cwd=ROOT, check=True)
else:
    print('Используются сохранённые результаты. Полный пересчёт выключен.')
""")
    md("## Выводы\nНа временном тесте LightGBM с удельной целью превосходит проверенные нейросети. Остаточная сеть при заранее заданном seed 42 лучше выбранной MLP, но чувствительна к позднему периоду; результаты по трём seed показывают разброс. На новых проектах порядок моделей меняется. Ошибки площади и особенно её пропуски ухудшают полную денежную оценку, поэтому устойчивость рассматривается совместно с исходной точностью. Подробная аргументация и ссылки на Ясницкого находятся в `report/project_report.docx` и `report/decision_log.md`.")
    nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
    path=ROOT/"project.ipynb"
    nbf.write(nb,path)
    # Execute actual cells in-process; no TCP/IPC listener is needed to review saved results.
    shell=InteractiveShell.instance()
    os.chdir(ROOT)
    count=0
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        count+=1
        with capture_output() as captured:
            executed=shell.run_cell(cell.source,store_history=True)
        cell.execution_count=count
        cell.outputs=[]
        for stream,value in [('stdout',captured.stdout),('stderr',captured.stderr)]:
            if value:cell.outputs.append(nbf.v4.new_output('stream',name=stream,text=value))
        for rich in captured.outputs:
            data={key:(base64.b64encode(value).decode() if isinstance(value,bytes) else value)
                  for key,value in rich.data.items()}
            cell.outputs.append(nbf.v4.new_output('display_data',data=data,metadata=rich.metadata))
        error=executed.error_before_exec or executed.error_in_exec
        if error is not None:raise error
    nb.metadata['execution']={'engine':'IPython in-process','all_cells_succeeded':True}
    nbf.validate(nb)
    nbf.write(nb,path)
    print(path)
    print('Executed code cells:',sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells))

if __name__=="__main__":main()
