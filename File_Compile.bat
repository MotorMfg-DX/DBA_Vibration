@echo off

REM 現在のディレクトリとそのすべてのサブディレクトリに存在するすべてのファイルを、現在のディレクトリに移動させる

for /r %%i in (*) do move "%%i" "%cd%"