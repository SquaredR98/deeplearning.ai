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

- To give it a general rule, if a matrix <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;n\times&space;n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;n&space;\times&space;n" title="n \times  n" /></a> is convolved with <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;f&space;\times&space;f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;f&space;\times&space;f" title="f \times f" /></a> filter/kernel give us <a href="https://www.codecogs.com/eqnedit.php?latex=(n-f&plus;1)&space;\times&space;(n-f&plus;1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(n-f&plus;1)&space;\times&space;(n-f&plus;1)" title="(n-f+1) \times (n-f+1)" /></a>matrix. 

- The convolution operation shrinks the matrix if <a href="https://www.codecogs.com/eqnedit.php?latex=f&space;>&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f&space;>&space;1" title="f > 1" /></a>.

- We want to apply convolution operation multiple times, but if the image shrinks we will lose a lot of data on this process. Also the edges pixels are used less than other pixels in an image.

- So the problems with convolutions are:

  - Shrinks output.
  - throwing away a lot of information that are in the edges.

- To solve these problems we can pad the input image before convolution by adding some rows and columns to it. We will call the padding amount <a href="https://www.codecogs.com/eqnedit.php?latex=p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p" title="p" /></a>the number of row/columns that we will insert in top, bottom, left and right of the image.

- In almost all the cases the padding values are zeros.

- The general rule now,  if a matrix <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;n\times&space;n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;n&space;\times&space;n" title="n \times  n" /></a>is convolved with  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;f&space;\times&space;f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;f&space;\times&space;f" title="f \times f" /></a>filter/kernel and padding <a href="https://www.codecogs.com/eqnedit.php?latex=p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p" title="p" /></a> give us <a href="https://www.codecogs.com/eqnedit.php?latex=(n&space;&plus;&space;2p&space;-&space;f&space;&plus;&space;1)\times(n&space;&plus;&space;2p&space;-&space;f&space;&plus;&space;1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(n&space;&plus;&space;2p&space;-&space;f&space;&plus;&space;1)\times(n&space;&plus;&space;2p&space;-&space;f&space;&plus;&space;1)" title="(n + 2p - f + 1)\times(n + 2p - f + 1)" /></a> matrix. 

- If n = 6, f = 3, and p = 1 Then the output image will have 

  <a href="https://www.codecogs.com/eqnedit.php?latex=(n&space;&plus;&space;2p&space;-&space;f&space;&plus;&space;1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(n&space;&plus;&space;2p&space;-&space;f&space;&plus;&space;1)" title="(n + 2p - f + 1)" /></a> = 6 + 2 - 3 + 1 = 6

  We maintain the size of the image.

- **Same convolutions** is a convolution with a pad so that output size is the same as the input size. Its given by the equation:

  <a href="https://www.codecogs.com/eqnedit.php?latex=p&space;=&space;\frac{(f&space;-&space;1)}{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p&space;=&space;\frac{(f&space;-&space;1)}{2}" title="p = \frac{(f - 1)}{2}" /></a>

- In computer vision <a href="https://www.codecogs.com/eqnedit.php?latex=f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f" title="f" /></a> is usually odd. Some of the reasons is that it will have a centre value.

## Strided convolution

- Strided convolution is another piece that are used in CNNs.

- We will call stride <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;s" title="\large s" /></a>.

- When we are making the convolution operation we used <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;s" title="\large s" /></a> to tell us the number of pixels we will jump when we are convolving filter/kernel. The last examples we described <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;s" title="\large s" /></a> was 1.

- Now the general rule is:

  - if a matrix <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;n\times&space;n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;n&space;\times&space;n" title="n \times  n" /></a> is convolved with <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;f&space;\times&space;f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;f&space;\times&space;f" title="f \times f" /></a> filter/kernel and padding <a href="https://www.codecogs.com/eqnedit.php?latex=p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p" title="p" /></a> and stride <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;s" title="\large s" /></a> it give us 

    <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\left(\frac{n&plus;2p-f}{s}&plus;1&space;\right&space;)\times&space;\left(\frac{n&plus;2p-f}{s}&plus;1&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\left(\frac{n&plus;2p-f}{s}&plus;1&space;\right&space;)\times&space;\left(\frac{n&plus;2p-f}{s}&plus;1&space;\right&space;)" title="\large \left(\frac{n+2p-f}{s}+1 \right )\times \left(\frac{n+2p-f}{s}+1 \right )" /></a>

    matrix. 

- In case <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\left(\frac{n&plus;2p-f}{s}&plus;1&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\left(\frac{n&plus;2p-f}{s}&plus;1&space;\right&space;)" title="\large \left(\frac{n+2p-f}{s}+1 \right )" /></a> is fraction we can take **floor** of this value.

- In math textbooks the conv operation is filpping the filter before using it. What we were doing is called cross-correlation operation but the state of art of deep learning is using this as conv operation.

- Same convolutions is a convolution with a padding so that output size is the same as the input size. Its given by the equation:

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;p&space;=&space;\frac{(n&space;\times&space;s&space;-&space;n&space;&plus;&space;f&space;-&space;s)}{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;p&space;=&space;\frac{(n&space;\times&space;s&space;-&space;n&space;&plus;&space;f&space;-&space;s)}{2}" title="\large p = \frac{(n \times s - n + f - s)}{2}" /></a>

  when

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;s&space;=&space;1&space;\implies&space;p&space;=&space;\frac{(f&space;-&space;1)}{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;s&space;=&space;1&space;\implies&space;p&space;=&space;\frac{(f&space;-&space;1)}{2}" title="\large s = 1 \implies p = \frac{(f - 1)}{2}" /></a>

## Convolutions over volumes

- We see how convolution works with 2D images, now lets see if we want to convolve 3D images (RGB image)
- We will convolve an image of height, width, # of channels with a filter of a height, width, same # of channels. Hint that the image number channels and the filter number of channels are the same.
- We can call this as stacked filters for each channel!
- Example:
  - Input image: 6 x 6 x 3
  - Filter: 3 x 3 x 3
  - Result image: 4 x 4 x 1
  - In the last result <a href="https://www.codecogs.com/eqnedit.php?latex=p&space;=&space;0,&space;s&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p&space;=&space;0,&space;s&space;=&space;1" title="p = 0, s = 1" /></a>
- Hint the output here is only 2D.
- We can use multiple filters to detect multiple features or edges. Example.
  - Input image: 6 x 6 x 3
  - 10 Filters: 3 x 3 x 3
  - Result image: 4 x 4 x 10
  - In the last result <a href="https://www.codecogs.com/eqnedit.php?latex=p&space;=&space;0,&space;s&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p&space;=&space;0,&space;s&space;=&space;1" title="p = 0, s = 1" /></a>

### One Layer of a Convolutional Network

- First we convolve some filters to a given input and then add a bias to each filter output and then get RELU of the result. Example:

  - Input image: 6 x 6 x 3        <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;a^{[0]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;a^{[0]}" title="\large a^{[0]}" /></a>
  - 10 Filters: 3 x 3 x 3              <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;W^{[1]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;W^{[1]}" title="\large W^{[1]}" /></a>
  - Result image: 4 x 4 x 10      <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;W^{[1]}a^{[0]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;W^{[1]}a^{[0]}" title="\large W^{[1]}a^{[0]}" /></a>
  - Add <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\textbf&space;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\textbf&space;b" title="\textbf b" /></a> (bias) with 10 x 1  will get us : 4 x 4 x 10 image      <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;W^{[1]}a^{[0]}&space;&plus;&space;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;W^{[1]}a^{[0]}&space;&plus;&space;b" title="\large W^{[1]}a^{[0]} + b" /></a>
  - Apply RELU will get us: 4 x 4 x 10 image                <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;A1=relu(W^{[1]}a^{[0]}&space;&plus;&space;b)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;A1=relu(W^{[1]}a^{[0]}&space;&plus;&space;b)" title="\large A1=relu(W^{[1]}a^{[0]} + b)" /></a>
  - In the last result <a href="https://www.codecogs.com/eqnedit.php?latex=p&space;=&space;0,&space;s&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p&space;=&space;0,&space;s&space;=&space;1" title="p = 0, s = 1" /></a>
  - Hint number of parameters here are: **(3 x 3 x 3 x 10) + 10 = 280**

- The last example forms a layer in the CNN.

- Hint: no matter the size of the input, the number of the parameters is same if filter size is same. That makes it less prone to over-fitting.

- Here are some notations we will use. If layer l is a conv layer:

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\textbf&space;f^{\hspace{.1cm}&space;\textbf&space;[\hspace{.05cm}&space;l&space;\hspace{.05cm}&space;\textbf&space;]}&space;=&space;filter&space;\hspace{.1cm}&space;size" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;f^{\hspace{.1cm}&space;\textbf&space;[\hspace{.05cm}&space;l&space;\hspace{.05cm}&space;\textbf&space;]}&space;=&space;filter&space;\hspace{.1cm}&space;size" title="\large f^{\hspace{.1cm} \textbf [\hspace{.05cm} l \hspace{.05cm} \textbf ]} = filter \hspace{.1cm} size" /></a> 						  	

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;p^{\hspace{.1cm}&space;\textbf&space;[\hspace{.05cm}&space;l&space;\hspace{.05cm}&space;\textbf&space;]}&space;=&space;padding&space;\hspace{.2cm}&space;(default&space;\hspace{.2cm}&space;0)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;p^{\hspace{.1cm}&space;\textbf&space;[\hspace{.05cm}&space;l&space;\hspace{.05cm}&space;\textbf&space;]}&space;=&space;padding&space;\hspace{.2cm}&space;(default&space;\hspace{.2cm}&space;0)" title="\large p^{\hspace{.1cm} \textbf [\hspace{.05cm} l \hspace{.05cm} \textbf ]} = padding \hspace{.2cm} (default \hspace{.2cm} 0)" /></a>      

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;s^{\hspace{.1cm}&space;\textbf&space;[\hspace{.05cm}&space;l&space;\hspace{.05cm}&space;\textbf&space;]}&space;=&space;stride" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;s^{\hspace{.1cm}&space;\textbf&space;[\hspace{.05cm}&space;l&space;\hspace{.05cm}&space;\textbf&space;]}&space;=&space;stride" title="\large \textbf s^{\hspace{.1cm} \textbf [\hspace{.05cm} l \hspace{.05cm} \textbf ]} = stride" /></a> 									  

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;n_c^{\textbf&space;[\hspace{0.05cm}l\hspace{0.05cm}\textbf&space;]}&space;=&space;number&space;\hspace{0.2cm}&space;of&space;\hspace{0.2cm}&space;filters" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;n_c^{\textbf&space;[\hspace{0.05cm}l\hspace{0.05cm}\textbf&space;]}&space;=&space;number&space;\hspace{0.2cm}&space;of&space;\hspace{0.2cm}&space;filters" title="\large n_c^{\textbf [\hspace{0.05cm}l\hspace{0.05cm}\textbf ]} = number \hspace{0.2cm} of \hspace{0.2cm} filters" /></a> 		Each filter is: <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\large&space;f^{[l]}&space;\times&space;f^{[l]}&space;\times&space;n_c^{[l-1]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\large&space;f^{[l]}&space;\times&space;f^{[l]}&space;\times&space;n_c^{[l-1]}" title="\large \large f^{[l]} \times f^{[l]} \times n_c^{[l-1]}" /></a> 

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;Input&space;=&space;n_{H}^{[l-1]}\times&space;n_{W}^{[l-1]}&space;\times&space;n_{c}^{[l-1]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;Input&space;=&space;n_{H}^{[l-1]}\times&space;n_{W}^{[l-1]}&space;\times&space;n_{c}^{[l-1]}" title="\large Input = n_{H}^{[l-1]}\times n_{W}^{[l-1]} \times n_{c}^{[l-1]}" /></a>
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;Output&space;=&space;n_{H}^{[l]}\times&space;n_{W}^{[l]}&space;\times&space;n_{c}^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;Output&space;=&space;n_{H}^{[l]}\times&space;n_{W}^{[l]}&space;\times&space;n_{c}^{[l]}" title="\large Output = n_{H}^{[l]}\times n_{W}^{[l]} \times n_{c}^{[l]}" /></a>
  
  where <a href="https://www.codecogs.com/eqnedit.php?latex=n_{H/W}^{[l]}&space;=&space;\left&space;\lfloor&space;\left(&space;\frac{n_{H/W}^{[l-1]}&space;&plus;&space;2p^{[l]}&space;-&space;f^{[l]}}{s^{[l]}}&space;&plus;&space;1&space;\right)\right&space;\rfloor" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_{H/W}^{[l]}&space;=&space;\left&space;\lfloor&space;\left(&space;\frac{n_{H/W}^{[l-1]}&space;&plus;&space;2p^{[l]}&space;-&space;f^{[l]}}{s^{[l]}}&space;&plus;&space;1&space;\right)\right&space;\rfloor" title="n_{H/W}^{[l]} = \left \lfloor \left( \frac{n_{H/W}^{[l-1]} + 2p^{[l]} - f^{[l]}}{s^{[l]}} + 1 \right)\right \rfloor" /></a>
  
  Activations:
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;a^{[l]}&space;=&space;n_H^{[l]}&space;\times&space;n_W^{[l]}&space;\times&space;n_c^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;a^{[l]}&space;=&space;n_H^{[l]}&space;\times&space;n_W^{[l]}&space;\times&space;n_c^{[l]}" title="\large a^{[l]} = n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}" /></a>
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;A^{[l]}&space;=&space;m&space;\times&space;n_H^{[l]}&space;\times&space;n_W^{[l]}&space;\times&space;n_c^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;A^{[l]}&space;=&space;m&space;\times&space;n_H^{[l]}&space;\times&space;n_W^{[l]}&space;\times&space;n_c^{[l]}" title="\large A^{[l]} = m \times n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}" /></a>
  
  Weights: <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;f^{[l]}&space;\times&space;f^{[l]}&space;\times&space;n_c^{[l-1]}&space;\times&space;n_c^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;f^{[l]}&space;\times&space;f^{[l]}&space;\times&space;n_c^{[l-1]}&space;\times&space;n_c^{[l]}" title="\large f^{[l]} \times f^{[l]} \times n_c^{[l-1]} \times n_c^{[l]}" /></a>
  
  bias: <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;(1,&space;1,&space;1,&space;n_c^{[l]})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;(1,&space;1,&space;1,&space;n_c^{[l]})" title="\large (1, 1, 1, n_c^{[l]})" /></a> 

### A simple convolution network example

- Lets build a big example.
  - Input Image are:   <a href="https://www.codecogs.com/eqnedit.php?latex=a^{[0]}&space;=&space;39\times&space;39\times&space;3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{[0]}&space;=&space;39\times&space;39\times&space;3" title="a^{[0]} = 39\times 39\times 3" /></a>
    - <a href="https://www.codecogs.com/eqnedit.php?latex=n^{[0]}&space;=&space;39" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^{[0]}&space;=&space;39" title="n^{[0]} = 39" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=n_c^{[0]}&space;=&space;3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_c^{[0]}&space;=&space;3" title="n_c^{[0]} = 3" /></a>
  - First layer (Conv layer):
    - <a href="https://www.codecogs.com/eqnedit.php?latex=f^{[1]}=3,s^{[1]}=1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f^{[1]}=3,s^{[1]}=1" title="f^{[1]}=3,s^{[1]}=1" /></a>, and <a href="https://www.codecogs.com/eqnedit.php?latex=p^{[1]}=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p^{[1]}=0" title="p^{[1]}=0" /></a>
    - number of filters = 10
    - Then output are <a href="https://www.codecogs.com/eqnedit.php?latex=a^{[1]}=37&space;\times&space;37&space;\times&space;10" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{[1]}=37&space;\times&space;37&space;\times&space;10" title="a^{[1]}=37 \times 37 \times 10" /></a>
      - <a href="https://www.codecogs.com/eqnedit.php?latex=n^{[1]}=37" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^{[1]}=37" title="n^{[1]}=37" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=n_c^{[1]}=10" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_c^{[1]}=10" title="n_c^{[1]}=10" /></a>
  - Second layer (Conv layer):
    - <a href="https://www.codecogs.com/eqnedit.php?latex=f^{[2]}=5,s^{[2]}=2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f^{[2]}=5,s^{[2]}=2" title="f^{[2]}=5,s^{[2]}=2" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=p^{[2]}=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p^{[2]}=0" title="p^{[2]}=0" /></a> 
    - number of filters = 20
    - The output are <a href="https://www.codecogs.com/eqnedit.php?latex=a^{[2]}=17\times&space;17&space;\times&space;20" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{[2]}=17\times&space;17&space;\times&space;20" title="a^{[2]}=17\times 17 \times 20" /></a> 
      - <a href="https://www.codecogs.com/eqnedit.php?latex=n^{[2]}=17,&space;n_c^{[2]}=20" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^{[2]}=17,&space;n_c^{[2]}=20" title="n^{[2]}=17, n_c^{[2]}=20" /></a> 
    - Hint shrinking goes much faster because the stride is 2
  - Third layer (Conv layer):
    - <a href="https://www.codecogs.com/eqnedit.php?latex=f^{[3]}=5,&space;s^{[3]}=2,&space;p^{[3]}=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f^{[3]}=5,&space;s^{[3]}=2,&space;p^{[3]}=0" title="f^{[3]}=5, s^{[3]}=2, p^{[3]}=0" /></a>
    - number of filters = 40
    - The output are <a href="https://www.codecogs.com/eqnedit.php?latex=a^{[3]}=&space;7&space;\times&space;7&space;\times&space;40" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{[3]}=&space;7&space;\times&space;7&space;\times&space;40" title="a^{[3]}= 7 \times 7 \times 40" /></a>
      - <a href="https://www.codecogs.com/eqnedit.php?latex=n^{[3]}=&space;7,&space;n_c^{[3]}=&space;40" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^{[3]}=&space;7,&space;n_c^{[3]}=&space;40" title="n^{[3]}= 7, n_c^{[3]}= 40" /></a>
  - Forth layer (Fully connected Softmax)
    - <a href="https://www.codecogs.com/eqnedit.php?latex=a^{[3]}&space;=&space;7&space;\times&space;7&space;\times&space;40&space;=&space;1960" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{[3]}&space;=&space;7&space;\times&space;7&space;\times&space;40&space;=&space;1960" title="a^{[3]} = 7 \times 7 \times 40 = 1960" /></a>  as a vector..
- In the last example you seen that the image are getting smaller after each layer and thats the trend now.
- Types of layer in a convolutional network:
  - Convolution :- **`Conv`**
  - Pooling :- **`Pool`**
  - Fully connected :- **`FC`**

### Pooling layers

- Other than the conv layers, CNNs often uses pooling layers to reduce the size of the inputs, speed up computation, and to make some of the features it detects more robust.

- Max pooling example:

  ![](images/pooling.png)

  - This example has <a href="https://www.codecogs.com/eqnedit.php?latex=f=2,&space;s=2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f=2,&space;s=2" title="f=2, s=2" /></a>, and <a href="https://www.codecogs.com/eqnedit.php?latex=p&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p&space;=&space;0" title="p = 0" /></a> hyperparameters

- The max pooling is saying, if the feature is detected anywhere in this filter then keep a high number. But the main reason why people are using pooling because its works well in practice and reduce computations.

- Max pooling has no parameters to learn.

- Example of Max pooling on 3D input:

  - Input: 4 x 4 x 10
  - Max pooling size = 2  and stride = 2
  - Output: 2 x 2 x 10

- Average pooling is taking the averages of the values instead of taking the max values.

- Max pooling is used more often than average pooling in practice.

- If stride of pooling equals the size, it will then apply the effect of shrinking.

- Hyperparameters summary

  - <a href="https://www.codecogs.com/eqnedit.php?latex=f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f" title="f" /></a> : filter size.
  - <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a>  : stride.
  - Padding are rarely uses here.
  - Max or average pooling.

### Convolutional neural network example

- Now we will deal with a full CNN example. This example is something like the ***LeNet-5*** that was invented by Yann Lecun.

  - Input Image are:   <a href="https://www.codecogs.com/eqnedit.php?latex=a^{[0]}=32\times&space;32\times&space;3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{[0]}=32\times&space;32\times&space;3" title="a^{[0]}=32\times 32\times 3" /></a>
    - <a href="https://www.codecogs.com/eqnedit.php?latex=n^{[0]}=32" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^{[0]}=32" title="n^{[0]}=32" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=n_c^{[0]}=3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_c^{[0]}=3" title="n_c^{[0]}=3" /></a>
  - First layer (Conv layer):       **`#Conv1`**
    - <a href="https://www.codecogs.com/eqnedit.php?latex=f^{[1]}=5,&space;s^{[1]}=1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f^{[1]}=5,&space;s^{[1]}=1" title="f^{[1]}=5, s^{[1]}=1" /></a>, and <a href="https://www.codecogs.com/eqnedit.php?latex=p^{[1]}=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p^{[1]}=0" title="p^{[1]}=0" /></a>
    - number of filters = 6
    - Then output are <a href="https://www.codecogs.com/eqnedit.php?latex=a^{[1]}=28\times&space;28\times&space;6" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{[1]}=28\times&space;28\times&space;6" title="a^{[1]}=28\times 28\times 6" /></a>
      - <a href="https://www.codecogs.com/eqnedit.php?latex=n^{[1]}=28" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^{[1]}=28" title="n^{[1]}=28" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=n_c^{[1]}=6" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_c^{[1]}=6" title="n_c^{[1]}=6" /></a>
    - Then apply (Max pooling):         **`#Pool1`**
      - <a href="https://www.codecogs.com/eqnedit.php?latex=f_p^{[1]}=2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_p^{[1]}=2" title="f_p^{[1]}=2" /></a>, and <a href="https://www.codecogs.com/eqnedit.php?latex=s_p^{[1]}=2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_p^{[1]}=2" title="s_p^{[1]}=2" /></a>
      - The output are <a href="https://www.codecogs.com/eqnedit.php?latex=a^{[1]}=14&space;\times&space;14&space;\times&space;6" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{[1]}=14&space;\times&space;14&space;\times&space;6" title="a^{[1]}=14 \times 14 \times 6" /></a>
  - Second layer (Conv layer):   **`#Conv2`**
    - <a href="https://www.codecogs.com/eqnedit.php?latex=f^{[2]}&space;=&space;5,&space;s^{[2]}&space;=&space;1,&space;p^{[2]}&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f^{[2]}&space;=&space;5,&space;s^{[2]}&space;=&space;1,&space;p^{[2]}&space;=&space;0" title="f^{[2]} = 5, s^{[2]} = 1, p^{[2]} = 0" /></a>
    - number of filters = 16
    - The output are <a href="https://www.codecogs.com/eqnedit.php?latex=a^{[2]}&space;=&space;10\times&space;10\times&space;16" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{[2]}&space;=&space;10\times&space;10\times&space;16" title="a^{[2]} = 10\times 10\times 16" /></a>
      - <a href="https://www.codecogs.com/eqnedit.php?latex=n^{[2]}&space;=&space;10,&space;n_c^{[2]}&space;=&space;16" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n^{[2]}&space;=&space;10,&space;n_c^{[2]}&space;=&space;16" title="n^{[2]} = 10, n_c^{[2]} = 16" /></a>
    - Then apply (Max pooling):**`#Pool2`**
      - <a href="https://www.codecogs.com/eqnedit.php?latex=f^{[2]}_p&space;=&space;2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f^{[2]}_p&space;=&space;2" title="f^{[2]}_p = 2" /></a>, and <a href="https://www.codecogs.com/eqnedit.php?latex=s^{[2]}_p&space;=&space;2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s^{[2]}_p&space;=&space;2" title="s^{[2]}_p = 2" /></a>
      - The output are <a href="https://www.codecogs.com/eqnedit.php?latex=a^{[2]}=5\times&space;5\times&space;16" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{[2]}=5\times&space;5\times&space;16" title="a^{[2]}=5\times 5\times 16" /></a>
  - Third layer (Fully connected)   **`#FC3`**
    - Number of neurons are 120
    - The output <a href="https://www.codecogs.com/eqnedit.php?latex=a^{[3]}=120\times&space;3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{[3]}=120\times&space;3" title="a^{[3]}=120\times 3" /></a> . 400 came from `5x5x16`
  - Forth layer (Fully connected)  **`#FC4`**
    - Number of neurons are 84
    - The output <a href="https://www.codecogs.com/eqnedit.php?latex=a^{[4]}=84\times&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^{[4]}=84\times&space;1" title="a^{[4]}=84\times 1" /></a>.
  - Fifth layer (Softmax)
    - Number of neurons is 10 if we need to identify for example the 10 digits.

- Hint a Conv1 and Pool1 is treated as one layer.

- Some statistics about the last example:

  |                                                              | Activation Shape | Activation Size | #parameters |
  | :----------------------------------------------------------: | :--------------: | :-------------: | :---------: |
  |                            Input:                            |   (32, 32, 3)    |      3072       |      0      |
  | CONV1(<a href="https://www.codecogs.com/eqnedit.php?latex=f=5,&space;s=1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f=5,&space;s=1" title="f=5, s=1" /></a>) |   (28, 28, 8)    |      6272       |     208     |
  |                            POOL1                             |   (14, 14, 8)    |      1568       |      0      |
  | CONV2(<a href="https://www.codecogs.com/eqnedit.php?latex=f=5,&space;s=1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f=5,&space;s=1" title="f=5, s=1" /></a>) |   (10, 10, 16)   |      1600       |     416     |
  |                            POOL2                             |    (5, 5, 16)    |       400       |      0      |
  |                             FC3                              |     (120, 1)     |       120       |   48,001    |
  |                             FC4                              |     (84, 1)      |       84        |   10,081    |
  |                           Softmax                            |      (10,1)      |       10        |     841     |

- Hyperparameters are a lot. For choosing the value of each you should follow the guideline that we will discuss later or check the literature and takes some ideas and numbers from it.

- Usually the input size decreases over layers while the number of filters increases.

- A CNN usually consists of one or more convolution (Not just one as the shown examples) followed by a pooling.

- Fully connected layers has the most parameters in the network.

- To consider using these blocks together you should look at other working examples firsts to get some intuitions.

### Why convolutions?

- Two main advantages of Convs are:

  - Parameter sharing.
    - A feature detector (such as a vertical edge detector) that's useful in one part of the image is probably useful in another part of the image.
  - sparsity of connections.
    - In each layer, each output value depends only on a small number of inputs which makes it translation invariance.

- Putting it all together:

  ![](images/alltogether.png)