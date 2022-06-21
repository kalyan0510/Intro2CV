# Intro2CV - Indtroduction to Computer Vision - Udacity (Georgia Tech - CS 6476)

Introduction to CV is a course taught by professor Dr. Aaron Bobick, offered free online by Udacity  
(https://www.udacity.com/course/introduction-to-computer-vision--ud810).

#### This repository contains:

1. Octave code solutions to 'in the class' quizzes
2. Solution to all [7 problem sets](https://docs.google.com/spreadsheets/d/1ecUGIyhYOfQPi3HPXb-7NndrLgpX_zgkwsqzfqHPaus/pubhtml) +
   with all the extra credit problems solved
3. Problem set solution reports added as README under each problem set directory  
4. Experimentation methods that helps understand the effect of parameters on certain algorithms (included in the ps_x.py files + observations added to ps_x/observations/ dir)
5. A [generic helper class](https://github.com/kalyan0510/Intro2CV/blob/main/ps_hepers/helpers.py) that contains tools to operate on images

| Links      |
| :---        |
| [Problem Set 0: Images as Functions](https://github.com/kalyan0510/Intro2CV/tree/main/ps_0)     |
| [Problem Set 1: Edges and Lines](https://github.com/kalyan0510/Intro2CV/tree/main/ps_1)     |
| [Problem Set 2: Window-based Stereo Matching](https://github.com/kalyan0510/Intro2CV/tree/main/ps_2)     |
| [Problem Set 3: Geometry](https://github.com/kalyan0510/Intro2CV/tree/main/ps_3)     |
| [Problem Set 4: Harris, SIFT, RANSAC](https://github.com/kalyan0510/Intro2CV/tree/main/ps_4)     |
| [Problem Set 5: Optic Flow](https://github.com/kalyan0510/Intro2CV/tree/main/ps_5)     |
| [Problem Set 6: Particle Tracking](https://github.com/kalyan0510/Intro2CV/tree/main/ps_6)     |
| [Problem Set 7: Motion History Images](https://github.com/kalyan0510/Intro2CV/tree/main/ps_7)     |

### How to run

1. Clone the repository
   ```
   git clone --recursive https://github.com/kalyan0510/Intro2CV.git
   cd Intro2CV/
   ```
2. Install the dependent libraries from requirements.txt ```prefer Python 3.9.5```
   ```
   pip3 install -r requirements.txt
   ```
3. To run solutions of a problem set X, run the python file ./ps_X/ps_X.py (make sure the sub problem is not commented
   out)
   ```
   python3 ps_X/ps_X.py
   ```
