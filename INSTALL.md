# llmmd lite Rework: инструкция по установке

llmmd lite для работы требует установленного Docker Desktop и LM Studio.

## Docker Desktop - установка

Установить Docker Desktop можно с официального сайта: **https://www.docker.com/products/docker-desktop/** (дата обращения 23.07.2026)
Для использования программы регистрация и авторизация не требуются.

## Установка WSL

Помимо самого Docker, для операционной системы Windows 10 (Windows 11) потребуется установка WSL 2, она производится через терминал. 

1. Откройте командную строку Windows;
2. Введите команду: `wsl --install`

Примечание: установка может занять 10 минут. При этом в командой строке может быть указана установка ububntu - это не ошибка;
3. После установки перезагрузите систему.

## Настройка WSL

Для работы llmmd-lite необходим одновременный запуск Docker, LM Studio и самого пайплайна (llmmd-lite). Настоятельно рекомендуется ограничить вычсилительные мощности компьютера, доступные Docker. Ограничения завистя от характеристик компьютера. Далее будет рассмотрена настройка ограничений для ноутбука со следующими характеристиками: процессор Ryzen 7 5700U (8 ядер), ОЗУ 16 Гб DDR4, накопитель SSD на 512 Гб (форм-фактор M.2), интегрированный видеоадаптер AMD Radeon 2 Гб, операционная система Windows 10. 

1. Закрыть Docker Desktop: нажать правой кнопкой мыши по иконке с китом в системном трее (возле часов).
2. Отркыть блокнот, создать файл со следующим содержимым:
```
[wsl2]
memory=6GB
processors=4
swap=2GB
localhostForwarding=true
```
3. Сохранить файл под именем `.wslconfig` в вашей пользовательской папке: `C:\Users\ваше_имя_пользователя\`

(!) Убедитесь, что файл называется именно `.wslconfig`, а не `.wslconfig.txt` или иначе.
4. Откройте PowerShell или командную строку и выполните команду `wsl --shotdown`
5. Запустите Docker Desktop обычным способом.

Примечание: шаги 4 и 5 нужны для вступления изменений в силу.

## Установка LM Studio

Для запуска локальной большой языковой модели (local LLM) потребуется установка LM Studio от Bionic. Ссылка на официальный сайт: **https://lmstudio.ai/** (дата обращения 23.07.2026)
Для использования LM Studio обязательная регистрация и авторизация не требуется.

## Скачивание локальной LLM

Для работы пайплайна llmmd-lite Rework потребуется скачать большую языковую модель (LLM) в LM Studio. Выбор конкретной модели, опять же, зависит от характеристик компьютера. При наличии современной дискретной видеокарты можно использовать модели на 7-8 миллиардов парамеров. При отсуствии видеокарты выбор модели ограничен. В качестве примера, под ранее описанный ноутбук можно установить модель на 1 миллиард параметров, например **google/gemma-3-1b**. При желании можно подобрать и другую LLM под характеристики конкретного компьютера.

Скачивание локальной языковой модели достаточно просто:
1. Открыть LM Studio;
2. В окне программы зайти в "settings" - в левом нижем углу;
3. Пролистать список слева вниз, под заголовком "Local Models" зайти в "Explore";
4. Набрать название выбранной модели в поиске, или выбрать модель из предложенных;
5. Зайти на страницу выбранной модели, скачать её (синяя кнопка "Download");
6. Дождаться окончания установки;
7. В списке слева поднаяться вверх, под заголовком "Settings" зайти в "General";
8. В окне "Genaral" в строке "Root Model" выбрать скачанную модель.

## Настройка LM Studio

Для избежания приостановок работы системы llmmd-lite Rework рекомендуется установить настройки LM Studio согласно перечню:

0. Раздел настроек (список слева) / Пункт настроек / Заголовок (Область окна справа) / Строка настройки - состояние;
1. Settings / General / Chat / Exploration agents - включено;
2. Devices / LM Link / - / Enable LM Link - включено;
3. Devices / LM Link / This device / Share local models - включено;
4. Local Models / Local Model API / Behavior / Just-in-time model loading - включено;
5. Local Models / Local Model API / Behavior / CORS - включено;

## Установка GIT (для установки llmmd-lite)

Один из вариантов скачивания и установки системы llmmd-lite Rework - использование программы Git. Далее будут рассмотрены шаги для установки и настройки Git.

1. Зайти на официальный сайт Git: **https://git-scm.com/** (дата обращения 24.07.2026)
2. Скачать установочный файл;
3. Запустить установочный файл;
4. В окне "Select Components" из списка выбрать следующее: Window Explorer Integration, Open Git Bash here, Open Git GUI here, Git LFS (Large File Support), Associate .git* configuration files with the default text editor, Associate .sh files to be run with Bash, Check daily for Git for Windows updates, Add a Git Bash Profile to Windows Terminal, Scalar (Git add-on to manage large-scale repositories);
5. В окне "Start Menu Folder" можно ничего не менять;
6. В окне "Choosing the default editor used by Git" выбрать "Use Visual Code as Git's default editor";

Примечание: Visual Studio Code - среда разработки, устанавливается отдельно.
7. В окне "Adjusting the name of the initial branch in new repositories" выбрать "Override the default branch name for new repositories" и оставить значение "main";
8. В окне "Adjusting your PATH enviroment" выбрать "Git from the command line and also from 3rd-party software";
9. В окне "Choosing the SSH executable" выбрать "Use bundled OpenSSH";
10. В окне "Choosing HTTPS transport backend" выбрать "Use the OpenSSL library";
11. В окне "Configuring the line ending conversions" выбрать "Checkout Windows-style, commit Unix-style line endings";
12. В окне "Configuring the terminal emulator to use with Git Bash" выбрать "Use MinTTY (the default terminal of MSYS2)";
13. В окне "Choose the default behaviour of 'git pull'" выбрать "Merge";
14. В окне "Choose a credential helper" выбрать "Git Credential Manager";
15. В окне "Configuring extra options" выбрать только "Enable file system caching".

После установки можно провести проверку: открыть терминал и набрать команду `git --version`, в ответ должна прийти информация об установленной версии, например `git version 2.55.0.windows.1`.

## Заголовок.