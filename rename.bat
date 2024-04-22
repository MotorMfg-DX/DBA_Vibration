@echo off
setlocal enabledelayedexpansion

echo このバッチファイルは、指定したワークNoをファイル名に追加してリネームします。
echo.

set /p input_folder="フォルダのパスを入力してください: "
cd "%input_folder%"

set /p input_workNo="ワークNoを入力してください: "
echo.

set /a count=0

echo ファイルのリネームを開始します...

for %%F in (*.tdms) do (
    set /a count+=1
    set filename=%%~nF
    set extension=%%~xF
    ren %%F No!input_workNo!_ST1_R!count!_!filename!!extension!
)

echo ファイルのリネームが完了しました。
pause