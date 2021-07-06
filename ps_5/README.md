# Problem Set 5: Optic Flow
[link to problems](https://docs.google.com/document/d/1Bi2_CThMfoLEf4TCMhyFR7cls6t3LAiSz7YYYV89eoE/pub?embedded=true)

### 1. Lucas Kanade Optic Flow
a) Flow estimation on Synthetic image pairs  
><img src="output/ps5-1-a-1.png" height="250">  <img src="output/ps5-1-a-2.png" height="250">  

b) LK on Larger Shifts
><img src="output/ps5-1-b-1.png" height="250">  <img src="output/ps5-1-b-2.png" height="250">  <img src="output/ps5-1-b-3.png" height="250">  

"Does it still work? Does it fall apart on any of the pairs?"
    
    It does fall apart on all of heavily shifted images.
    Motion flow can only be estimated when shift in pixels are small. Since, all the lk flow detector does is compare 
    gradients ix and iy to time gradient, when time gradient fails (i.e, when delta is too high), it falls apart.   
    
### 2. Gaussian and Laplacian Pyramids
a) Gaussian Pyramid
><img src="output/ps5-2-a-1.png" height="250">  <img src="output/ps5-2-a-2.png" height="250">  
><img src="output/ps5-2-a-3.png" height="250">  <img src="output/ps5-2-a-4.png" height="250">  

b) Laplacian Pyramid
><img src="output/ps5-2-b-1.png" height="250">  <img src="output/ps5-2-b-2.png" height="250">  
><img src="output/ps5-2-b-3.png" height="250">  <img src="output/ps5-2-b-4.png" height="250">  

### 3.  Warping by flow
a) 
DataSeq1 warping & diff  
><img src="output/ps5-3-a-1.png" height="250">  <img src="output/ps5-3-a-2.png" height="250">   
DataSeq2 warping & diff  
><img src="output/ps5-3-a-3.png" height="250">  <img src="output/ps5-3-a-4.png" height="250">   

### 4. Hierarchical LK optic flow
a) Over synthetic images
the displacement images between the warped I2 and the original I1 
><img src="output/ps5-4-a-1.png" height="250"> 

the difference images between the warped I2 and the original I1
><img src="output/ps5-4-a-2.png" height="250"> 

b) Over DataSeq1
the displacement images between the warped I2 and the original I1 
><img src="output/ps5-4-b-1.png" height="250"> 

the difference images between the warped I2 and the original I1
><img src="output/ps5-4-b-2.png" height="250"> 

c) Over DataSeq2
the displacement images between the warped I2 and the original I1 
><img src="output/ps5-4-c-1.png" height="250"> 

the difference images between the warped I2 and the original I1
><img src="output/ps5-4-c-2.png" height="250"> 

### 4. Extra Credit for Juggle
a)  
the displacement images between the warped I2 and the original I1  
><img src="output/ps5-5-a-1.png" height="250"> 

the difference images between the warped I2 and the original I1  
><img src="output/ps5-5-a-2.png" height="250"> 

        With some changes I was able to derive motion flow for this sequence. It was hard because the shift in pixels 
        is too high for lk detection (~35 pixels).
        The detected optical flow was not too bad, as it shows the approximate magnitude and direction of shift. 
        
        Notice that there was some flow detected over the static pixels around the ball. This is because, the actual 
        flow was only detected in highest pyramid level and since the image is scaled multiple fold, the pixels around a
        moving object are also affected, not to mention the noise added by missing/new pixels at boundary of moving 
        object.     


