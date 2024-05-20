@echo off
setlocal enabledelayedexpansion

echo このバッチファイルは、指定したワークNoをファイル名に追加してリネームします。
echo.

set /p input_folder="フォルダのパスを入力してください: "
cd "%input_folder%"

set /p input_workNo="ワークNoを入力してください: "
echo.

rem set /a count=0
set /a count=510

echo ファイルのリネームを開始します...

for %%F in (*.tdms) do (
rem set /a count+=1
    set /a count+=100
    set filename=%%~nF
    set extension=%%~xF
rem    ren %%F No!input_workNo!_ST1_R!count!_!filename!!extension!
    ren %%F No!input_workNo!_rpm!count!_!filename!!extension!
    echo !filename!
)

echo ファイルのリネームが完了しました。
pause