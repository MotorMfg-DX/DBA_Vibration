@echo off
setlocal enabledelayedexpansion

echo ���̃o�b�`�t�@�C���́A�w�肵�����[�NNo���t�@�C�����ɒǉ����ă��l�[�����܂��B
echo.

set /p input_folder="�t�H���_�̃p�X����͂��Ă�������: "
cd "%input_folder%"

set /p input_workNo="���[�NNo����͂��Ă�������: "
echo.

rem set /a count=0
set /a count=510

echo �t�@�C���̃��l�[�����J�n���܂�...

for %%F in (*.tdms) do (
rem set /a count+=1
    set /a count+=100
    set filename=%%~nF
    set extension=%%~xF
rem    ren %%F No!input_workNo!_ST1_R!count!_!filename!!extension!
    ren %%F No!input_workNo!_rpm!count!_!filename!!extension!
    echo !filename!
)

echo �t�@�C���̃��l�[�����������܂����B
pause