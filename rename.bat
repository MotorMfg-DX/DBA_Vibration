@echo off
setlocal enabledelayedexpansion

echo ���̃o�b�`�t�@�C���́A�w�肵�����[�NNo���t�@�C�����ɒǉ����ă��l�[�����܂��B
echo.

set /p input_folder="�t�H���_�̃p�X����͂��Ă�������: "
cd "%input_folder%"

set /p input_workNo="���[�NNo����͂��Ă�������: "
echo.

set /a count=0

echo �t�@�C���̃��l�[�����J�n���܂�...

for %%F in (*.tdms) do (
    set /a count+=1
    set filename=%%~nF
    set extension=%%~xF
    ren %%F No!input_workNo!_ST1_R!count!_!filename!!extension!
)

echo �t�@�C���̃��l�[�����������܂����B
pause