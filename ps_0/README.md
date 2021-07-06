# Problem Set 0: Images as Functions
[link to problems](https://docs.google.com/document/d/1PO9SuHMYhx6nDbB38ByB1QANasP1UaEiXaeGeHmp3II/pub?embedded=true)



### 1. Input images
a0) Wide image  a1) Tall image  
>
><img src="ps_0/output/ps0-1-a-1.png" height="250">
><img src="ps_0/output/ps0-1-a-2.png" height="250">


### 2. Color  planes
a) Red and blue channels swapped  b) Monochrome from green channel  c) Monochrome from red channel   
><img src="ps_0/output/ps0-2-a-1.png" height="250">
><img src="ps_0/output/ps0-2-b-1.png" height="250">
><img src="ps_0/output/ps0-2-c-1.png" height="250">  
d) Which looks more like what you’d expect a monochrome image to look like?
 
    In case of Lena's image green channel provides a better gray scale image than the red channel. This can because of 
    low saturation of red in the image than that of green.
    But what might not be a coincidence is a standard way of converting RGB images to gray scale gives more weight to
    the green channel.
    Y = 0.299 R + 0.587 G + 0.114 B (formula used by open cv's cv.COLOR_RGB2GRAY conversion)
    This is probably because of the way human's vision system is designed to perceive colors. 
    Ref: https://respuestas.me/q/deteccion-de-la-vision-humana-del-punto-de-luz-debil-parpadeante-o-en-movim-60506457491
    http://cadik.posvete.cz/color_to_gray_evaluation/ 
    
    
    Would you expect a computer vision algorithm to work on one better than the other?
    Of course, yes. A segmentation algorithm will produce different results over different channels of color. For 
    example, an apple (red) and a lemon(yellow = green + red) might just look alike in red channel but are very 
    different in green channel.   

### 3. Replacement of pixels
a) Inserting 100x100 center square crop of image 1 into image 2  
><img src="ps_0/output/ps0-3-a-1.png" height="250">  

### 4. Arithmetic and Geometric operations
a)  Min, Max, Mean & Std

    max of pixels 248
    min of pixels 3
    average of pixels 99.05121612548828
    std of pixels 52.87751732904626

b)  "Subtract the mean from all pixels, then divide by standard deviation, then multiply by 10 (if your image is 0 to 255) or by 0.05 (if your image ranges from 0.0 to 1.0). Now add the mean back in"   
><img src="ps_0/output/ps0-4-b-1.png" height="250">  
  
    This step just decreased the contrast as 10.0 < np.std(im)(=52.8)  
c)  "Shift img1_green to the left by 2 pixels"  d)  "Subtract the shifted version of img1_green from the original"  
><img src="ps_0/output/ps0-4-c-1.png" height="250">
><img src="ps_0/output/ps0-4-d-1.png" height="250">  
What do negative pixel values mean anyways?  

    Negative pixel values occur when I(x,y)<I(x+shift, y). So, a negative value indicates a edge with increasing 
    intensity towards x direction.  

### 5. Noise Addition
a)  Noise in green channel (sigma = 15.0) b) Noise in blue channel  
><img src="ps_0/output/ps0-5-a-1.png" height="250">
><img src="ps_0/output/ps0-5-b-1.png" height="250">  
c) Which looks better?  

    The image with noise in the blue channel looks better than the one with green noise.
   Why?

    This definitely has something to human color perception. The number of cones sensitive to color blue is a lot 
    lesser than that of color green. 
    The cones distribution in human eye is (Red - 64%, Green - 32%, Blue - 2%)
    So, humans are less sensitive to noise in blue channel.
  
