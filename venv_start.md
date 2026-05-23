Для переходу в виртульное окружение проекта из вашего рабочего каталога:

powershellcd D:\_project\Search-doc
.\venv-embed-test\Scripts\Activate.ps1

После активации в начале строки появится префикс (venv-embed-test) - это индикатор, что окружение активно:
(venv-embed-test) PS D:\_project\Search-doc>

Дальше можно сразу запускать:
powershellpython pre_mvp_eval.py --eval eval\eval_queries_v2.yaml

Если PowerShell вдруг ругнётся на политику выполнения скриптов (после перезагрузки иногда сбрасывается) - однократно:

powershellSet-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

И повторите активацию.
