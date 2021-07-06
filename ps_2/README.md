# Problem Set 2: Window-based Stereo Matching
[link to problems](https://docs.google.com/document/d/1WcljLaRxL-Pj3VWYz7JtYysYoZtRZoLIrTG2x48uVWE/pub?embedded=true)

### 1. Disparity on Synthetic textured image
a) Disparity with respect to left & Disparity with respect to right 

><img src="output/ps2-1-a-1.png" height="250"> <img src="output/ps2-1-a-2.png" height="250"> 

### 2. Disparity on Real images
a)  
><img src="output/ps2-2-a-1.png" height="250"> <img src="output/ps2-2-a-2.png" height="250"> 

Ground Truth
><img src="input/pair1-D_L.png" height="250"> <img src="input/pair1-D_R.png" height="250"> 

b) "Description of the differences between your results and ground truth"
  
    The calculated disparity cannot be as good as ground truth as finding correspondences for triangulation is very hard

### 3. Disparity on Noisy Images
a) Gaussian Noise over one image

><img src="output/ps2-3-a-1.png" height="250"> <img src="output/ps2-3-a-2.png" height="250"> 

b) Contrast Difference

><img src="input/ps2-3-b-1.png" height="250"> <img src="output/ps2-3-b-2.png" height="250"> 

    The used similarity method SSD is sensitive to noise (gaussian or contrasting change) and hence can cause error in 
    calculated disparities

### 4. Normalized Correlation for finding correspondences

a) Disparity with respect to left & Disparity with respect to right  
><img src="output/ps2-4-a-1.png" height="250"> <img src="output/ps2-4-a-2.png" height="250">  

b) 
N-Corr disparity finding on image pair with Gaussing Noise on one image
Left Disparity Map,   Right Disparity Map
><img src="output/ps2-4-b-1.png" height="250"> <img src="output/ps2-4-b-2.png" height="250">  

N-Corr disparity finding on image pair with Gaussing Noise on one image
><img src="output/ps2-4-b-3.png" height="250"> <img src="output/ps2-4-b-4.png" height="250">  

#### 5. Disparity on second pair of images
a)  
><img src="output/ps2-5-a-1.png" height="250"> <img src="output/ps2-5-a-2.png" height="250">  

Ground Truth
><img src="input/pair2-D_L.png" height="250"> <img src="input/pair2-D_R.png" height="250"> 

    Images in pair 2 does not have good texture and thus has less correspondences available to find the disparity.