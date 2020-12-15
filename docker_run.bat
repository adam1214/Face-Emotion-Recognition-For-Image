@echo OFF
set FERI_ROOT="%cd%"
set PROJECT_NAME="fer_image"

IF [%1] == [] (
	echo NO arguments passed
	docker run --rm ^
	-v %FERI_ROOT%/feri_input:/media/%PROJECT_NAME%/feri_input ^
	-v %FERI_ROOT%/feri_output:/media/%PROJECT_NAME%/feri_output ^
	-v %FERI_ROOT%/feri_finished:/media/%PROJECT_NAME%/feri_finished ^
	-v %FERI_ROOT%/feri_model:/media/%PROJECT_NAME%/feri_model ^
	-v %FERI_ROOT%/feri_result:/media/%PROJECT_NAME%/feri_result ^
	biicgitlab.ee.nthu.edu.tw:5050/prod/engineer/fer_image:latest ^
	/bin/bash -c "python3 main.py feri_input/ feri_output/ feri_result/ feri_finished/ feri_model/ 720p"
	) ELSE	(
	echo Arguments passed
	docker run --rm  ^
	-v %FERI_ROOT%/feri_input:/media/%PROJECT_NAME%/feri_input ^
	-v %FERI_ROOT%/feri_output:/media/%PROJECT_NAME%/feri_output ^
	-v %FERI_ROOT%/feri_finished:/media/%PROJECT_NAME%/feri_finished ^
	-v %FERI_ROOT%/feri_model:/media/%PROJECT_NAME%/feri_model ^
	-v %FERI_ROOT%/feri_result:/media/%PROJECT_NAME%/feri_result ^
	biicgitlab.ee.nthu.edu.tw:5050/prod/engineer/fer_image:latest ^
	/bin/bash -c "python3 main.py feri_input/%1 feri_output/%1 feri_result/%1 feri_finished/%1 feri_model/ 720p"
	)
pause

