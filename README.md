# UCL-ELEC0134_assignment

UCL ELEC0134 course final assignment

  

This readme file shows the structure of the codes and how to use it.

  

### Python libraries used

- tensorflow
- scikit-learn
- numpy
- matplotlib
- cmake
- dlib
- opencv-python
- xgboost
  

Dlib is quite complicated to install. This website is quite helpful:[How to Install dlib Library for python in windows 10 - GeeksforGeeks](https://www.geeksforgeeks.org/how-to-install-dlib-library-for-python-in-windows-10/)


 





### Program structure

-- AMLS_22-23_SN22212102


&emsp; -- main.py


&emsp; -- A1


&emsp;&emsp; -- CNN.py


&emsp;&emsp; -- landmark_XGBoost.py


&emsp;&emsp; -- landmark_MLP.py

  
&emsp; -- A2


&emsp;&emsp; -- landmark_MLP.py

  

&emsp; -- B1

  

&emsp;&emsp; -- EfficientNet.py

&emsp;&emsp; -- VGG.py


&emsp; -- B2

  

&emsp;&emsp; -- EfficientNet.py

&emsp;&emsp; -- EfficientNet_Cropping.py

  



  

### Program run instruction

The main function is composed of several sub function and you can run each function separately by commenting out the functions you don't want to run.

&emsp; -- solve_A1_landmark_XGBoost()

&emsp; -- solve_A1_landmark_MLP()

&emsp; -- solve_A1_CNN()

&emsp; -- solve_A2_MLP()

&emsp; -- solve_B1_VGG()

&emsp; -- solve_B1_EfficientNet_V2()

&emsp; -- solve_B2_EfficientNet()

&emsp; -- solve_B2_EfficientNet_Cropping()

Without commenting anything, all tasks will be run one by one when you run "main.py". Please remember to change the working dirctory to the loacation of the folder "AMLS_22-23_SN22081179".