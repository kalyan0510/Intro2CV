# Problem Set 4: Harris, SIFT, RANSAC
[link to problems](https://docs.google.com/document/d/1DlyziyQB163r1Lx3F4-Tanm8Oq4O9-W3X5Hpdw4QGUE/pub?embedded=true)

### 1. Harris Corners
a) Input, Gradient x and Gradient y

><img src="output/ps4-1-a-1.png" height="250">  
><img src="output/ps4-1-a-2.png" height="250">  


b) Harris Response
><img src="output/ps4-1-b-1.png" height="250">  <img src="output/ps4-1-b-2.png" height="250">  
><img src="output/ps4-1-b-3.png" height="250">  <img src="output/ps4-1-b-4.png" height="250">  

b) Corners from Harris Response
><img src="output/ps4-1-c-1.png" height="250">  <img src="output/ps4-1-c-2.png" height="250">  
><img src="output/ps4-1-c-3.png" height="250">  <img src="output/ps4-1-c-4.png" height="250">  

"Describe the behavior of your corner detector including anything surprising, such as points not found in both images of a pair."
 
    Corner detection is sensitive to noise. There were corners detected at places with sharp intensity changes. So, 
    often when there is sharp noise, a corner is detected and the same corner is not detected in the pair image. 
    
    Also, few corners that can be clearly perceived by humans are not detected by the harris algo. For example, the 
    rectangular corners of the lawn in front of the campus building in simA.jpg is not detected, but it looks well 
    like an interest point. 
    The reason could be because, the lawn and the ground has similar pixel intensities. But the reason why humans see it
    might be because we are aware of the entire lawn as a separate entity from ground and so, are able to see cornerness
    at those pixels. Also, such pixels could cause only smaller harris responses and are prone to thresholding done
    before the non maximal suppression step.
    
### 2. SIFT Features

a) Interest Points Visualized
><img src="output/ps4-2-a-1.png" height="250">  
><img src="output/ps4-2-a-2.png" height="250">   

b) Putative Pairs
Darker lines signify higher distance between point descriptors  
transA–transB Pair
><img src="output/ps4-2-b-1.png" height="250">  

simA–simB Pair
><img src="output/ps4-2-b-2.png" height="250">  


### 3. RANSAC
a) Consensus Set between translated image pair  
><img src="output/ps4-3-a-1.png" height="250">  

b) Consensus Set between image pair with similarity transformation  
><img src="output/ps4-3-b-1.png" height="250">  

c) Consensus Set between image pair with similarity transformation, but assuming that the trasformation is affine  
><img src="output/ps4-3-c-1.png" height="250">  

d) Stitching similarity images with warping
><img src="output/ps4-3-d-1.png" height="250">  
><img src="output/ps4-3-d-2.png" height="250">  

e) Stitching affine images with warping
><img src="output/ps4-3-e-1.png" height="250">  
><img src="output/ps4-3-e-2.png" height="250">  

"Comment as to whether using the similarity transform or the affine one gave better results, and why or why not."
  
    Using the similarity transform gave better alignment. This could be because, in calculating similarity transform 
    matrix we only need to solve for 4 unknowns where as in affine we solve for 6. Solution for the two extra unknowns
    can come along with some error. Probably that can have an effect. 
    But in the above case, there are too many factors that effect the efficiency of alignment. Random sampling can 
    produce different results each time and the way of consensus voting hugely affects ransac.  
