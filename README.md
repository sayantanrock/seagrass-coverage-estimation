## Overview
In this project we examined the coverage estimation of seagrass from underwater videos recorded by scuba divers in Danish coastal waters. There were total 14 video transects from strategically different part of a fjord ( Roskilde, Denmark). A feature extractor,  line segment detector to detect number of egdes from video frames is used to extract two features: **number of lines detected** and **length of lines detected** and are reported in the folder *Raw extracted data* along with gps coordinates and they form the meta-data used for this analysis. Two algorithms are proposed for coverage estimations and a case study to validate the model on the dataset from : [HS Fulda](https://www.hs-fulda.de/fileadmin/user_upload/FB_ET/Projekte_Forschung/Enview_Jaeger/EnView_News_2018-04/Conference_Kobe_2018_Seagrass.pdf) .
This project has been done at the Department of Applied Mathematics and Computer Sciences, Technical University of Denmark.

### Involved People
* Sayantan Sengupta (Technical University of Denmark)
* Anders Stockmarr (Technical University of Denmark)
* Danish Hydrauics Institute (Data provider)

### Paper

This work is an extension of our previous work on Seagrass detection from underwater videos: [Sengupta et al 2020](https://www.sciencedirect.com/science/article/pii/S1574954120300339?via%3Dihub) , where we use similar features from the same video transects to estimate *presence/absence* status of seagrass.

## Software 
#### Dependencies
* Python 2.7 ( Anaconda installation)
* pylsd 0.0.2 ([pip install pylsd](https://pypi.org/project/pylsd/))
* Opencv 3.4.0.14 ([pip install opencv-python](https://pypi.org/project/opencv-python/3.4.0.14/))
* Scipy
* Matplotlib
* Numpy
* Sklearn

[![Watch the video]([https://i.sstatic.net/Vp2cE.png](https://img.youtube.com/vi/NZkDht-DgbA?si=KC79qoxkzSE8Ddr7/hqdefault.jpg))](https://www.youtube.com/watch?v=NZkDht-DgbA&list=PLgOY2SnZ2Tu7iDmDGWtV7YrQaWbuJlmbk)

https://img.youtube.com/vi/NZkDht-DgbA?si=KC79qoxkzSE8Ddr7/hqdefault.jpg
