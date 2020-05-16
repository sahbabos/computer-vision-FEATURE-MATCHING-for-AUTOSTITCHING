FEATURE MATCHING for AUTOSTITCHING:
===================================

#### I took some some pictures. For the first one I used a chess board as arefrnce because it's planar. For the second part I took three stationary pictures with different angles

![](image/chess7.jpg) ![](image/chess6.jpg) ![](image/sign.jfif)
![](image/ind1.jpg) ![](image/ind2.jpg) ![](image/ind3.jpg)

Recover Homographies:
---------------------

#### First nedd to recover the homography transformation matrix. This way we are able to find how to move the pixles. By using p' = Hp and 4 control points in each image, we can solve for the matrix H.

#### We can simply want to buid the matrix below. Then we can solve it using least squared:

![](image/homog.png)

Warp the Images (Rectifying)
----------------------------

#### I applied to the homographic transformations on first three pictures, but since I didnt have second reference points I used the for corners or just estimated where they should form a rectangle

![](image/sahba3.jpg) ![](image/sahba5.jpg) ![](image/sahba6.jpg)
![](image/sahba2.jpg)

Blend the images into a mosaic
------------------------------

#### Blending was trickier than i thought. Although the result is not too bad but the edge where the images were connected together is visible. I think using a mask would produce better results.

![](image/left_A.jpg) ![](image/middle_A.jpg) ![](image/right_A.jpg)
![](image/result.jpg)

What I learned?
---------------

#### After doing the project things that looked IMPOSIBLE are easy to do. I think there are alot of application that can be implemented using Homography. Taking linear algebra classes paid off :).

Detecting corner features in an image:
--------------------------------------

#### For this part we use the provided Harris code to get the points and their H value if we plot them we get:

![](image/ind_left.png) ![](image/ind_middle.png)
![](image/ind_right.png)

#### now that we have our points we need to use ANMS and find best points we can use to make out patch, these points:

![](image/indoor_left_ANMS.png) ![](image/indoor_middle_ANMS.png)
![](image/indoor_right_ANMS.png)

#### OK! so we have out points we need to find some features and for each point so we can find a good match in the other image. the featuers look like these.

![](image/ind_batch0.jpg) ![](image/ind_batch1.jpg)
![](image/ind_batch2.jpg) ![](image/ind_batch6.jpg)
![](image/ind_batch8.jpg) ![](image/ind_batch8.jpg)

#### we got the pathes and now we should go over each point withs its patch and compare agains every patch in the other image. afther that we should have some matches as these:

![](image/1.png) ![](image/2.png) ![](image/4.png)

#### Now after we find the all matching points we need to pick the best ones that are close enough to compute the homography and warp the images. we are so close:

![](image/left_final_ransac.png) ![](image/middle_final_ransac.png)
![](image/middle_final_2_ransac.png) ![](image/right_final_ransac.png)

#### We are done now only thing left to do is use the final points and make a homography and warp all the images and baaaam:

![](image/auto_left.jpg) ![](image/auto_middle.jpg)
![](image/auto_right.jpg)

#### the result:

![](image/auto_finish.jpg)
