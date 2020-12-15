# FER Image
> **Engineer Team**  
> last update: 2020.09.28    
> 
> @ author: Chun-Yu Chen  
> @ email: adam@gapp.nthu.edu.tw  
> @ mattermost: chen-chun-yu  


## Setting
* Image    
	1. install docker in your device
	2. use your account login to gitlab `docker login biicgitlab.ee.nthu.edu.tw:5050`
	3. download the image `docker pull biicgitlab.ee.nthu.edu.tw:5050/prod/engineer/fer_image`

* Code Structure
	1. `git clone https://biicgitlab.ee.nthu.edu.tw/prod/engineer/fer_image.git`

## Run 

### Mode1 > Assign Folder 

1. Go to the path where you clone the repository, run `cd  <your_path>`

2. Put your files in `feri_input/<your_folder>/` 
  
3. Execute **docker_run.sh** or **docker_run.bat** depends on your OS:     
     
   Only do files in `feri_input/<your_folder>`    
   * if your OS is UBUNTU, `bash docker_run.sh  <your_folder>`   
   * if your OS is WINDOWS, `docker_run.bat  <your_folder>`

4. Check result in the `feri_output` and `feri_result` folder 
5. The original files and folder will be moved to 
	
	`feri_finished/<your_folder>/*.jpg`  

### Mode2 > Do All files

1. Go to the path where you clone the repository, run `cd  <your_path>`

2. Put all the files in `feri_input/` 

3. Execute **docker_run.sh** or **docker_run.bat** depends on your OS:     
     
   do all files in `feri_input/` without input folder name
	* if your OS is UBUNTU, run `bash docker_run.sh`
	* if your OS is WINDOWS, run `docker_run.bat` 

4. Check result in the `feri_output` and `feri_result` folder 
5. The original files and folder will be moved to `feri_finished/*`

## Input Data Requirement
1. Only process `.jpg` and `.png` files which are 720p and above.

    (`.png` file we will convert to `.jpg` file and then processing)

## Output Format
2. There are `.csv` files in the `feri_result` folder. And each colume in the `.csv` file is:
    (If the resolution of the input `.jpg` file isn't 720p and above, `.csv` file will only be printed error msg.)
    * file name
    * face_x
    * face_y
    * face_w
    * face_h
    * emotion
    * angry_prob
    * disgust_prob
    * fear_prob
    * happy_prob
    * sad_prob
    * surprise_prob
    * neutral_pr
3. There are `.jpg` files in the `feri_output` folder. They show the recognized face frame and the current emotion of the face.
4. After the `.jpg` file in the `feri_input` folder is processed, it will be moved to `feri_finished` folder.
