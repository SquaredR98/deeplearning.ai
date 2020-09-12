# **Foundations of CNN**

## **Computer vision**

- Computer vision is one of the applications that are rapidly active thanks to deep learning.
- Some of the applications of computer vision that are using deep learning includes:
  - **Self driving cars**.
  - **Face recognition**.
- Deep learning is also enabling new types of art to be created.
- Rapid changes to computer vision are making new applications that weren't possible a few years ago.
- Computer vision deep leaning techniques are always evolving making a new architectures which can help us in other areas other than computer vision.
  - For example, Andrew Ng took some ideas of computer vision and applied it in speech recognition.
- Examples of a computer vision problems includes:
  - Image classification.
  - Object detection.
    - Detect object and localize them.
  - Neural style transfer
    - Changes the style of an image using another image.
- One of the challenges of computer vision problem that images can be so large and we want a fast and accurate algorithm to work with that.
  - For example, a **1000 x 1000** image will represent 3 million feature/input to the full connected neural network. If the following hidden layer contains 1000 units, then we will want to learn weights of the shape **[ 1000, 3 million ]** which is 3 billion parameter only in the first layer and that's so computationally expensive!
- One of the solutions is to build this using **convolution layers** instead of the **fully connected layers**.


## **Edge detection example**

- The convolution operation is one of the fundamentals blocks of a CNN. One of the examples about convolution is the image edge detection operation.

  ![](images/conv.gif)

  Source: [Intuitively Understanding Convolutions for Deep Learning](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)

- Early layers of CNN might detect edges then the middle layers will detect parts of objects and the later layers will put the these parts together to produce an output.

- In an image we can detect vertical edges, horizontal edges, or full edge detector.

- Vertical edge detection:

  - An example of convolution operation to detect vertical edges:
    ![](images/edgedetection.png)
  - In the last example a `6x6` matrix convolved with `3x3` filter/kernel gives us a `4x4` matrix.
  - If you make the convolution operation in TensorFlow you will find the function **`tf.nn.conv2d`**. In keras you will find **`Conv2d`** function.
  - The vertical edge detection filter will find a `3x3` place in an image where there are a bright region followed by a dark region.
  - If we applied this filter to a white region followed by a dark region, it should find the edges in between the two colours as a positive value. But if we applied the same filter to a dark region followed by a white region it will give us negative values. To solve this we can use the abs function to make it positive.
  
- For horizontal edge detection filter/kernel would be like this

  ```
   1	 1	 1
   0	 0	 0
   -1	-1	-1
  ```
  
- There are a lot of ways we can put number inside the horizontal or vertical edge detections. For example here are the vertical **Sobel** filter (The idea is taking care of the middle row):

  ```
   1	0	-1
   2	0	-2
   1	0	-1
  ```

- Also something called **Scharr** filter (The idea is taking great care of the middle row):

  ```
    3	0	 -3
   10	0	-10
    3	0	 -3
  ```

- What we learned in the deep learning is that we don't need to hand craft these numbers, we can treat them as weights and then learn them. It can learn horizontal, vertical, angled, or any edge type automatically rather than getting them by hand.

## Padding

- In order to to use deep neural networks we really need to use **paddings**.

- In the last section we saw that a 6 x 6 matrix convolved with 3 x 3 filter/kernel gives us a 4 x 4 matrix.

- To give it a general rule, if a matrix <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\textbf&space;n&space;\times&space;\textbf&space;n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\textbf&space;n&space;\times&space;\textbf&space;n" title="\textbf n \times \textbf n" /></a> is convolved with <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\textbf&space;f&space;\times&space;\textbf&space;f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\textbf&space;f&space;\times&space;\textbf&space;f" title="\textbf f \times \textbf f" /></a> filter/kernel give us <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;(\textbf&space;n&space;-&space;\textbf&space;f&space;&plus;&space;\textbf&space;1)&space;\times&space;(\textbf&space;n&space;-&space;\textbf&space;f&space;&plus;&space;\textbf&space;1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;(\textbf&space;n&space;-&space;\textbf&space;f&space;&plus;&space;\textbf&space;1)&space;\times&space;(\textbf&space;n&space;-&space;\textbf&space;f&space;&plus;&space;\textbf&space;1)" title="(\textbf n - \textbf f + \textbf 1) \times (\textbf n - \textbf f + \textbf 1)" /></a> matrix. 

- The convolution operation shrinks the matrix if <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\textbf&space;f&space;>&space;\textbf&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\textbf&space;f&space;>&space;\textbf&space;1" title="\textbf f > \textbf 1" /></a>.

- We want to apply convolution operation multiple times, but if the image shrinks we will lose a lot of data on this process. Also the edges pixels are used less than other pixels in an image.

- So the problems with convolutions are:

  - Shrinks output.
  - throwing away a lot of information that are in the edges.

- To solve these problems we can pad the input image before convolution by adding some rows and columns to it. We will call the padding amount <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\textbf&space;p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\textbf&space;p" title="\textbf p" /></a> the number of row/columns that we will insert in top, bottom, left and right of the image.

- In almost all the cases the padding values are zeros.

- The general rule now,  if a matrix <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\textbf&space;n&space;\times&space;\textbf&space;n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\textbf&space;n&space;\times&space;\textbf&space;n" title="\textbf n \times \textbf n" /></a> is convolved with <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\textbf&space;f&space;\times&space;\textbf&space;f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\textbf&space;f&space;\times&space;\textbf&space;f" title="\textbf f \times \textbf f" /></a> filter/kernel and padding <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\textbf&space;p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\textbf&space;p" title="\textbf p" /></a> give us <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;(\textbf&space;n&space;&plus;&space;\textbf&space;2&space;\textbf&space;p&space;-\textbf&space;f&space;&plus;&space;\textbf&space;1)&space;\times&space;(\textbf&space;n&space;&plus;&space;\textbf&space;2&space;\textbf&space;p&space;-\textbf&space;f&space;&plus;&space;\textbf&space;1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;(\textbf&space;n&space;&plus;&space;\textbf&space;2&space;\textbf&space;p&space;-\textbf&space;f&space;&plus;&space;\textbf&space;1)&space;\times&space;(\textbf&space;n&space;&plus;&space;\textbf&space;2&space;\textbf&space;p&space;-\textbf&space;f&space;&plus;&space;\textbf&space;1)" title="(\textbf n + \textbf 2 \textbf p -\textbf f + \textbf 1) \times (\textbf n + \textbf 2 \textbf p -\textbf f + \textbf 1)" /></a> matrix. 

- If n = 6, f = 3, and p = 1 Then the output image will have 

  n + 2p - f + 1 = 6 + 2 - 3 + 1 = 6

  We maintain the size of the image.

- **Same convolutions** is a convolution with a pad so that output size is the same as the input size. Its given by the equation:

  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\textbf&space;p&space;=&space;(\textbf&space;f-&space;\textbf&space;1)&space;/&space;\textbf&space;2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\textbf&space;p&space;=&space;(\textbf&space;f-&space;\textbf&space;1)&space;/&space;\textbf&space;2" title="\textbf p = (\textbf f- \textbf 1) / \textbf 2" /></a>

- In computer vision **f** is usually odd. Some of the reasons is that it will have a centre value.

## Strided convolution

- Strided convolution is another piece that are used in CNNs.

- We will call stride **`S`**.

- When we are making the convolution operation we used **`S`** to tell us the number of pixels we will jump when we are convolving filter/kernel. The last examples we described S was 1.

- Now the general rule is:

  - if a matrix <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\textbf&space;n&space;\times&space;\textbf&space;n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\textbf&space;n&space;\times&space;\textbf&space;n" title="\textbf n \times \textbf n" /></a> is convolved with <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\textbf&space;f&space;\times&space;\textbf&space;f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\textbf&space;f&space;\times&space;\textbf&space;f" title="\textbf f \times \textbf f" /></a> filter/kernel and padding <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\textbf&space;p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\textbf&space;p" title="\textbf p" /></a> and stride <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\LARGE&space;\textbf&space;s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\LARGE&space;\textbf&space;s" title="\LARGE \textbf s" /></a> it give us 

    <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\frac{(\textbf&space;n&space;&plus;&space;\textbf&space;2&space;\textbf&space;p&space;-&space;\textbf&space;f)}{\textbf&space;s}&space;&plus;&space;\textbf&space;1,\frac{(\textbf&space;n&space;&plus;&space;\textbf&space;2&space;\textbf&space;p&space;-&space;\textbf&space;f)}{\textbf&space;s}&space;&plus;&space;\textbf&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\frac{(\textbf&space;n&space;&plus;&space;\textbf&space;2&space;\textbf&space;p&space;-&space;\textbf&space;f)}{\textbf&space;s}&space;&plus;&space;\textbf&space;1,\frac{(\textbf&space;n&space;&plus;&space;\textbf&space;2&space;\textbf&space;p&space;-&space;\textbf&space;f)}{\textbf&space;s}&space;&plus;&space;\textbf&space;1" title="\frac{(\textbf n + \textbf 2 \textbf p - \textbf f)}{\textbf s} + \textbf 1,\frac{(\textbf n + \textbf 2 \textbf p - \textbf f)}{\textbf s} + \textbf 1" /></a>

    matrix. 

- In case <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{150}&space;\frac{(\textbf&space;n&space;&plus;&space;\textbf&space;2&space;\textbf&space;p&space;-&space;\textbf&space;f)}{\textbf&space;s}&space;&plus;&space;\textbf&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{150}&space;\frac{(\textbf&space;n&space;&plus;&space;\textbf&space;2&space;\textbf&space;p&space;-&space;\textbf&space;f)}{\textbf&space;s}&space;&plus;&space;\textbf&space;1" title="\frac{(\textbf n + \textbf 2 \textbf p - \textbf f)}{\textbf s} + \textbf 1" /></a> is fraction we can take **floor** of this value.

- In math textbooks the conv operation is filpping the filter before using it. What we were doing is called cross-correlation operation but the state of art of deep learning is using this as conv operation.

- Same convolutions is a convolution with a padding so that output size is the same as the input size. Its given by the equation:

  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\textbf&space;p&space;=&space;(\textbf&space;n&space;\times&space;\textbf&space;s&space;-&space;\textbf&space;n&space;&plus;&space;\textbf&space;f&space;-&space;\textbf&space;s)&space;/&space;2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\textbf&space;p&space;=&space;(\textbf&space;n&space;\times&space;\textbf&space;s&space;-&space;\textbf&space;n&space;&plus;&space;\textbf&space;f&space;-&space;\textbf&space;s)&space;/&space;2" title="\large \textbf p = (\textbf n \times \textbf s - \textbf n + \textbf f - \textbf s) / 2" /></a>

  when

  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\textbf&space;s&space;=&space;\textbf&space;1&space;\implies&space;\textbf&space;p&space;=&space;(\textbf&space;f&space;-&space;\textbf&space;1)&space;/&space;\textbf&space;2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\textbf&space;s&space;=&space;\textbf&space;1&space;\implies&space;\textbf&space;p&space;=&space;(\textbf&space;f&space;-&space;\textbf&space;1)&space;/&space;\textbf&space;2" title="\large \textbf s = \textbf 1 \implies \textbf p = (\textbf f - \textbf 1) / \textbf 2" /></a>

## Convolutions over volumes

- We see how convolution works with 2D images, now lets see if we want to convolve 3D images (RGB image)
- We will convolve an image of height, width, # of channels with a filter of a height, width, same # of channels. Hint that the image number channels and the filter number of channels are the same.
- We can call this as stacked filters for each channel!
- Example:
  - Input image: 6 x 6 x 3
  - Filter: 3 x 3 x 3
  - Result image: 4 x 4 x 1
  - In the last result <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\textbf&space;p&space;=&space;\textbf&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\textbf&space;p&space;=&space;\textbf&space;0" title="\large \textbf p = \textbf 0" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{120}&space;\large&space;\textbf&space;s&space;=&space;\textbf1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{120}&space;\large&space;\textbf&space;s&space;=&space;\textbf1" title="\large \textbf s = \textbf1" /></a>
- Hint the output here is only 2D.
- We can use multiple filters to detect multiple features or edges. Example.
  - Input image: 6 x 6 x 3
  - 10 Filters: 3 x 3 x 3
  - Result image: 4 x 4 x 10
  - In the last result <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\textbf&space;p&space;=&space;\textbf&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\textbf&space;p&space;=&space;\textbf&space;0" title="\large \textbf p = \textbf 0" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{120}&space;\large&space;\textbf&space;s&space;=&space;\textbf1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{120}&space;\large&space;\textbf&space;s&space;=&space;\textbf1" title="\large \textbf s = \textbf1" /></a>