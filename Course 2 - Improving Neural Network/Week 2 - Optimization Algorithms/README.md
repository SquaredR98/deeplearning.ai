## Optimization algorithms

### Mini-batch gradient descent

- Training NN with a large data is slow. So to find an optimization algorithm that runs faster is a good idea.

- Suppose we have <a href="https://www.codecogs.com/eqnedit.php?latex=m&space;=&space;50" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m&space;=&space;50" title="m = 50" /></a> million. To train this data it will take a huge processing time for one step.

  - because 50 million won't fit in the memory at once we need other processing to make such a thing.

- It turns out you can make a faster algorithm to make gradient descent process some of your items even before you finish the 50 million items.

- Suppose we have split m to **mini batches** of size 1000.

  - <a href="https://www.codecogs.com/eqnedit.php?latex=X^{\{1\}}&space;=&space;0&space;...&space;1000" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X^{\{1\}}&space;=&space;0&space;...&space;1000" title="X^{\{1\}} = 0 ... 1000" /></a>
  - <a href="https://www.codecogs.com/eqnedit.php?latex=X^{\{2\}}&space;=&space;0&space;...&space;1000" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X^{\{2\}}&space;=&space;0&space;...&space;1000" title="X^{\{2\}} = 0 ... 1000" /></a>
  - ...
  - <a href="https://www.codecogs.com/eqnedit.php?latex=X^{\{bs\}}&space;=&space;..." target="_blank"><img src="https://latex.codecogs.com/gif.latex?X^{\{bs\}}&space;=&space;..." title="X^{\{bs\}} = ..." /></a>

- We similarly split <a href="https://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X" title="X" /></a> & <a href="https://www.codecogs.com/eqnedit.php?latex=Y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Y" title="Y" /></a>.

- So the definition of mini batches <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\implies&space;t:&space;X^{\{t\}},&space;Y^{\{t\}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\implies&space;t:&space;X^{\{t\}},&space;Y^{\{t\}}" title="\implies t: X^{\{t\}}, Y^{\{t\}}" /></a>

- In **Batch gradient descent** we run the gradient descent on the whole dataset.

- While in **Mini-Batch gradient descent** we run the gradient descent on the mini datasets.

- Mini-Batch algorithm pseudo code:

  for <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;t&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;t&space;=&space;1" title="\large t = 1" /></a>: No_of_batches { 							# this is called an epoch

  ​	Forward prop on   <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;X^{\{t\}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;X^{\{t\}}" title="\large X^{\{t\}}" /></a>                      	

  ​			<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;Z^{[1]}=W^{[1]}X^{\{t\}}&plus;b^{[1]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;Z^{[1]}=W^{[1]}X^{\{t\}}&plus;b^{[1]}" title="\large Z^{[1]}=W^{[1]}X^{\{t\}}+b^{[1]}" /></a>
  ​			<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;A^{[1]}=g^{[1]}(Z^{[1]})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;A^{[1]}=g^{[1]}(Z^{[1]})" title="\large A^{[1]}=g^{[1]}(Z^{[1]})" /></a>

  ​			... ... ...

  ​			<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;A^{[l]}=g^{[l]}(Z^{[l]})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;A^{[l]}=g^{[l]}(Z^{[l]})" title="\large A^{[l]}=g^{[l]}(Z^{[l]})" /></a>

  ​			Compute Cost <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\large&space;J^{\{t\}}=\frac{1}{1000}\sum_{i=1}^{l}\mathcal{L}(\hat{y}^{(i)},y^{(i)})&plus;\frac{\lambda}{2(1000)}||w^{[l]}||_{F}^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\large&space;J^{\{t\}}=\frac{1}{1000}\sum_{i=1}^{l}\mathcal{L}(\hat{y}^{(i)},y^{(i)})&plus;\frac{\lambda}{2(1000)}||w^{[l]}||_{F}^{2}" title="\large \large J^{\{t\}}=\frac{1}{1000}\sum_{i=1}^{l}\mathcal{L}(\hat{y}^{(i)},y^{(i)})+\frac{\lambda}{2(1000)}||w^{[l]}||_{F}^{2}" /></a>

  ​			Backprop to compute gradients wrt <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\large&space;J^{\{t\}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\large&space;J^{\{t\}}" title="\large \large J^{\{t\}}" /></a> (using <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\left&space;(X^{\{t\}},&space;Y^{\{t\}}&space;\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\left&space;(X^{\{t\}},&space;Y^{\{t\}}&space;\right)" title="\large \left (X^{\{t\}}, Y^{\{t\}} \right)" /></a>)

  ​			<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\large&space;W^{[l]}&space;:=&space;W^{[l]}&space;-&space;\alpha&space;dW^{[l]},b^{[l]}&space;:=&space;b^{[l]}&space;-&space;\alpha&space;db^{[l]}," target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\large&space;W^{[l]}&space;:=&space;W^{[l]}&space;-&space;\alpha&space;dW^{[l]},b^{[l]}&space;:=&space;b^{[l]}&space;-&space;\alpha&space;db^{[l]}," title="\large \large W^{[l]} := W^{[l]} - \alpha dW^{[l]},b^{[l]} := b^{[l]} - \alpha db^{[l]}," /></a>

  }

- The code inside an epoch should be vectorized.

- Mini-batch gradient descent works much faster in the large datasets.

### Understanding mini-batch gradient descent

- In mini-batch algorithm, the cost won't go down with each step as it does in batch algorithm. It could contain some ups and downs but generally it has to go down (unlike the batch gradient descent where cost function decreases on each iteration).
  ![](images/batch_vs_mini_batch_cost.png)
- Mini-batch size:
  - <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;mini-batch&space;\hspace{0.2cm}&space;size&space;=&space;m&space;\implies" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;mini-batch&space;\hspace{0.2cm}&space;size&space;=&space;m&space;\implies" title="\large mini-batch \hspace{0.2cm} size = m \implies" /></a> Batch gradient descent
  - <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;mini-batch&space;\hspace{0.2cm}&space;size&space;=&space;1&space;\implies" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;mini-batch&space;\hspace{0.2cm}&space;size&space;=&space;1&space;\implies" title="\large mini-batch \hspace{0.2cm} size = 1 \implies" /></a>     Stochastic gradient descent (SGD)
  -  ![](images/CodeCogsEqn.gif) Mini-batch gradient descent
- Batch gradient descent:
  - too long per iteration (epoch)
- Stochastic gradient descent:
  - too noisy regarding cost minimization (can be reduced by using smaller learning rate)
  - won't ever converge (reach the minimum cost)
  - lose speedup from vectorization
- Mini-batch gradient descent:
  1. faster learning:
     - you have the vectorization advantage
     - make progress without waiting to process the entire training set
  2. doesn't always exactly converge (oscillate in a very small region, but we can reduce learning rate)
- Guidelines for choosing mini-batch size:
  1. If small training set (< 2000 examples) - use batch gradient descent.
  2. It has to be a power of 2 (because of the way computer memory is layed out and accessed, sometimes code runs faster if mini-batch size is a power of 2):
     64, 128, 256, 512, 1024, ...
  3. Make sure that mini-batch fits in CPU/GPU memory.
- Mini-batch size is a **hyperparameter**.

### Exponentially weighted averages

- There are optimization algorithms that are better than **gradient descent**, but you should first learn about Exponentially weighted averages.

- If we have data like the temperature of day through the year it could be like this:

  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;t(1)&space;=&space;40" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;t(1)&space;=&space;40" title="\large t(1) = 40" /></a>

  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;t(2)&space;=&space;49" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;t(2)&space;=&space;49" title="\large t(1) = 49" /></a>

  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;t(3)&space;=&space;45" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;t(3)&space;=&space;45" title="\large t(1) = 45" /></a>

  ... ... ... ... ...

  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;t(180)&space;=&space;60" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;t(180)&space;=&space;60" title="\large t(180) = 60" /></a>

  ... ... ... ... ...

- This data is small in winter and big in summer. If we plot this data we will find it some noisy.

- Now lets compute the Exponentially weighted averages:

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;v_0&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;v_0&space;=&space;0" title="\large v_0 = 0" /></a>

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;v_1&space;=&space;0.9&space;\hspace{0.1cm}&space;v_0&space;&plus;&space;0.1&space;\hspace{0.1cm}&space;t(1)&space;=&space;4" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;v_1&space;=&space;0.9&space;\hspace{0.1cm}&space;v_0&space;&plus;&space;0.1&space;\hspace{0.1cm}&space;t(1)&space;=&space;4" title="\large v_1 = 0.9 \hspace{0.1cm} v_0 + 0.1 \hspace{0.1cm} t(1) = 4" /></a>								# 0.9 and 0.1 are hyperparameters

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;v_2&space;=&space;0.9&space;\hspace{0.1cm}&space;v_1&space;&plus;&space;0.1&space;\hspace{0.1cm}&space;t(2)&space;=&space;8.5" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;v_2&space;=&space;0.9&space;\hspace{0.1cm}&space;v_1&space;&plus;&space;0.1&space;\hspace{0.1cm}&space;t(2)&space;=&space;8.5" title="\large v_2 = 0.9 \hspace{0.1cm} v_1 + 0.1 \hspace{0.1cm} t(2) = 8.5" /></a>

  ... ... ... ... ... ... ... ... ... ... ... ...

- General equation

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;v_{(t)}&space;=&space;\beta&space;\hspace{0.1cm}&space;v_{(t-1)}&space;&plus;&space;(1-\beta)&space;\hspace{0.1cm}&space;\theta_{(t)}&space;=&space;8.5" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;v_{(t)}&space;=&space;\beta&space;\hspace{0.1cm}&space;v_{(t-1)}&space;&plus;&space;(1-\beta)&space;\hspace{0.1cm}&space;\theta_{(t)}&space;=&space;8.5" title="\large v_{(t)} = \beta \hspace{0.1cm} v_{(t-1)} + (1-\beta) \hspace{0.1cm} \theta_{(t)} = 8.5" /></a>

- If we plot this it will represent averages over <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\approx&space;\left(&space;\frac{1}{1-\beta}\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\approx&space;\left(&space;\frac{1}{1-\beta}\right&space;)" title="\large \approx \left( \frac{1}{1-\beta}\right )" /></a> entries:

  - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\beta&space;=" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\beta&space;=" title="\large \beta =" /></a> 0.9 will average last 10 entries
  - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\beta&space;=" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\beta&space;=" title="\large \beta =" /></a> 0.98 will average last 50 entries
  - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\beta&space;=" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\beta&space;=" title="\large \beta =" /></a> 0.5 will average last 2 entries

- Best beta average for our case is between 0.9 and 0.98

- **Intuition**: The reason why exponentially weighted averages are useful for further optimizing gradient descent algorithm is that it can give different weights to recent data points (<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\theta" title="\large \theta" /></a>) based on value of <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\beta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\beta" title="\large \beta" /></a>. If <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\beta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\beta" title="\large \beta" /></a> is high (around 0.9), it smoothens out the averages of skewed data points (oscillations w.r.t. Gradient descent terminology). So this reduces oscillations in gradient descent and hence makes faster and smoother path towards minima.

### Understanding exponentially weighted averages

- We can implement this algorithm with more accurate results using a moving window. But the code is more efficient and faster using the exponentially weighted averages algorithm.

- Algorithm is very simple:

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;v&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;v&space;=&space;0" title="\large v = 0" /></a> 

  Repeat {

  ​			Get next <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\theta_{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\theta_{t}" title="\large \theta_{t}" /></a> 

  ​			<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;v_{\theta}:=&space;\beta&space;v_{\theta}&space;&plus;&space;(1-\beta)\theta_{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;v_{\theta}:=&space;\beta&space;v_{\theta}&space;&plus;&space;(1-\beta)\theta_{t}" title="\large v_{\theta}:= \beta v_{\theta} + (1-\beta)\theta_{t}" /></a>

  }

### Bias correction in exponentially weighted averages

- The bias correction helps make the exponentially weighted averages more accurate.

- Because <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;v_{0}=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;v_{0}=&space;0" title="\large v_{0}= 0" /></a>, the bias of the weighted averages is shifted and the accuracy suffers at the start.

- To solve the bias issue we have to use this equation:

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;v_{t}=&space;\frac{\beta&space;\times&space;v_{(t-1)}&space;&plus;&space;(1-\beta)\times&space;\theta_{t}}{1&space;-&space;\beta^{t}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;v_{t}=&space;\frac{\beta&space;\times&space;v_{(t-1)}&space;&plus;&space;(1-\beta)\times&space;\theta_{t}}{1&space;-&space;\beta^{t}}" title="\large v_{t}= \frac{\beta \times v_{(t-1)} + (1-\beta)\times \theta_{t}}{1 - \beta^{t}}" /></a>

- As <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;t" title="\large t" /></a> becomes larger the  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;(1&space;-&space;\beta^{t})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;(1&space;-&space;\beta^{t})" title="\large (1 - \beta^{t})" /></a> becomes close to 1.

### Gradient descent with momentum

- The momentum algorithm almost always works faster than standard gradient descent.

- The simple idea is to calculate the exponentially weighted averages for your gradients and then update your weights with the new values.

- Pseudo code:

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;v_{dW}&space;=&space;0,&space;v_{db}&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;v_{dW}&space;=&space;0,&space;v_{db}&space;=&space;0" title="\large v_{dW} = 0, v_{db} = 0" /></a>

  On iteration <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;t" title="\large t" /></a>:

  ​				Compute <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;dW,db" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;dW,db" title="dW,db" /></a> on the current mini-batch

  ​				<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;v_{dW}&space;=&space;\beta&space;v_{dW}&plus;(1-\beta)dW" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;v_{dW}&space;=&space;\beta&space;v_{dW}&plus;(1-\beta)dW" title="\large v_{dW} = \beta v_{dW}+(1-\beta)dW" /></a> 

  ​				<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;v_{db}&space;=&space;\beta&space;v_{db}&plus;(1-\beta)db" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;v_{db}&space;=&space;\beta&space;v_{db}&plus;(1-\beta)db" title="\large v_{db} = \beta v_{db}+(1-\beta)db" /></a> 

  ​				<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;W&space;=&space;W-\alpha&space;v_{dW},&space;\hspace{0.3cm}&space;b&space;=&space;b-\alpha&space;v_{db}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;W&space;=&space;W-\alpha&space;v_{dW},&space;\hspace{0.3cm}&space;b&space;=&space;b-\alpha&space;v_{db}" title="\large W = W-\alpha v_{dW}, \hspace{0.3cm} b = b-\alpha v_{db}" /></a>	

- Momentum helps the cost function to go to the minimum point in a more fast and consistent way.

- <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\beta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\beta" title="\large \beta" /></a> is another **hyperparameter**. <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\beta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\beta" title="\large \beta" /></a> = 0.9 is very common and works very well in most cases.

- In practice people don't bother implementing **bias correction**.

### RMSprop

- Stands for **Root mean square prop**.

- This algorithm speeds up the gradient descent.

- Pseudo code:

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;S_{dW}&space;=&space;0,&space;S_{db}=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;S_{dW}&space;=&space;0,&space;S_{db}=0" title="\large S_{dW} = 0, S_{db}=0" /></a> 

  on iteration <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;t" title="\large t" /></a>:

  ​			*can be mini-batch or batch gradient descent*

  ​			Compute <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;dW,db" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;dW,db" title="dW,db" /></a> on the current mini-batch

  ​			<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;S_{dW}=\beta&space;(S_{dW})&plus;(1-\beta)dW^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;S_{dW}=\beta&space;(S_{dW})&plus;(1-\beta)dW^2" title="\large S_{dW}=\beta (S_{dW})+(1-\beta)dW^2" /></a>			(Element Wise Squaring)

  ​			<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;S_{db}=\beta&space;(S_{db})&plus;(1-\beta)db^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;S_{db}=\beta&space;(S_{db})&plus;(1-\beta)db^2" title="\large S_{db}=\beta (S_{db})+(1-\beta)db^2" /></a>					(Element Wise Squaring)

  ​			<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;W&space;=&space;W&space;-&space;\alpha&space;\left(&space;\frac{dW}{\sqrt{S_{dW}}}&space;\right&space;),b&space;=&space;b&space;-&space;\alpha&space;\left(&space;\frac{db}{\sqrt{S_{db}}}&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;W&space;=&space;W&space;-&space;\alpha&space;\left(&space;\frac{dW}{\sqrt{S_{dW}}}&space;\right&space;),b&space;=&space;b&space;-&space;\alpha&space;\left(&space;\frac{db}{\sqrt{S_{db}}}&space;\right&space;)" title="\large W = W - \alpha \left( \frac{dW}{\sqrt{S_{dW}}} \right ),b = b - \alpha \left( \frac{db}{\sqrt{S_{db}}} \right )" /></a>

- RMSprop will make the cost function move slower on the vertical direction and faster on the horizontal direction in the following example:
  ![](images/RMSprop.png)

- Ensure that <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;S_{dW}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;S_{dW}" title="S_{dW}" /></a> is not zero by adding a small value <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\epsilon" title="\epsilon" /></a> (e.g. <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\epsilon&space;=&space;10^{-8}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\epsilon&space;=&space;10^{-8}" title="\epsilon = 10^{-8}" /></a>) to it:   
  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;W&space;=&space;W&space;-&space;\alpha&space;\left(&space;\frac{dW}{\sqrt{S_{dW}&space;&plus;&space;\epsilon}}&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;W&space;=&space;W&space;-&space;\alpha&space;\left(&space;\frac{dW}{\sqrt{S_{dW}&space;&plus;&space;\epsilon}}&space;\right&space;)" title="\large W = W - \alpha \left( \frac{dW}{\sqrt{S_{dW} + \epsilon}} \right )" /></a>

- With RMSprop you can increase your learning rate.

- Developed by Geoffrey Hinton and firstly introduced on [Coursera.org](https://www.coursera.org/) course.

### Adam optimization algorithm

- Stands for **Adaptive Moment Estimation**.

- Adam optimization and RMSprop are among the optimization algorithms that worked very well with a lot of NN architectures.

- Adam optimization simply puts RMSprop and momentum together!

- Pseudo code:

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;v_{dW}&space;=&space;0,&space;v_{db}&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;v_{dW}&space;=&space;0,&space;v_{db}&space;=&space;0" title="\large v_{dW} = 0, v_{db} = 0" /></a>

  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;S_{dW}&space;=&space;0,&space;S_{db}=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;S_{dW}&space;=&space;0,&space;S_{db}=0" title="\large S_{dW} = 0, S_{db}=0" /></a> 

  on iteration <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;t" title="\large t" /></a>:

  ​				*can be mini-batch or batch gradient descent*

  ​				Compute <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;dW,db" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;dW,db" title="dW,db" /></a> on the current mini-batch

  ​				

  ​				<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;v_{dW}&space;=&space;\beta_{1}&space;(v_{dW})&plus;(1-\beta_{}1)dW" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;v_{dW}&space;=&space;\beta_{1}&space;(v_{dW})&plus;(1-\beta_{1})dW" title="\large v_{dW} = \beta v_{dW}+(1-\beta)dW" /></a> 					*Momentum*

  ​				<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;v_{db}&space;=&space;\beta_1&space;(v_{db})&plus;(1-\beta_1)db" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;v_{db}&space;=&space;\beta_1&space;(v_{db})&plus;(1-\beta_1)db" title="\large v_{db} = \beta v_{db}+(1-\beta)db" /></a> 							*Momentum*

  

  ​				<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;S_{dW}=\beta_2&space;(S_{dW})&plus;(1-\beta_2)dW^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;S_{dW}=\beta_2&space;(S_{dW})&plus;(1-\beta_2)dW^2" title="\large S_{dW}=\beta (S_{dW})+(1-\beta)dW^2" /></a>					*RMS Prop*

  ​				<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;S_{db}=\beta_2&space;(S_{db})&plus;(1-\beta_2)db^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;S_{db}=\beta_2&space;(S_{db})&plus;(1-\beta_2)db^2" title="\large S_{db}=\beta (S_{db})+(1-\beta)db^2" /></a>							*RMS Prop*

  

  ​				<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;v_{dW}&space;=&space;\frac{v_{dW}}{1-\beta_1^{t}}&space;\hspace{2cm}&space;v_{db}&space;=&space;\frac{v_{db}}{1-\beta_1^{t}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;v_{dW}&space;=&space;\frac{v_{dW}}{1-\beta_1^{t}}&space;\hspace{2cm}&space;v_{db}&space;=&space;\frac{v_{db}}{1-\beta_1^{t}}" title="\large v_{dW} = \frac{v_{dW}}{1-\beta_1^{t}} \hspace{2cm} v_{db} = \frac{v_{db}}{1-\beta_1^{t}}" /></a>		*Fixing Bias*

  

  ​				<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;S_{dW}&space;=&space;\frac{S_{dW}}{1-\beta_2^{t}}&space;\hspace{2cm}&space;S_{db}&space;=&space;\frac{S_{db}}{1-\beta_2^{t}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;S_{dW}&space;=&space;\frac{S_{dW}}{1-\beta_2^{t}}&space;\hspace{2cm}&space;S_{db}&space;=&space;\frac{S_{db}}{1-\beta_2^{t}}" title="\large S_{dW} = \frac{S_{dW}}{1-\beta_2^{t}} \hspace{2cm} S_{db} = \frac{S_{db}}{1-\beta_2^{t}}" /></a>	  *Fixing Bias*

  

  ​				<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;W&space;=&space;W&space;-&space;\alpha&space;\left(&space;\frac{v_{dW}}{\sqrt{S_{dW}&space;&plus;&space;\epsilon}}&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;W&space;=&space;W&space;-&space;\alpha&space;\left(&space;\frac{v_{dW}}{\sqrt{S_{dW}&space;&plus;&space;\epsilon}}&space;\right&space;)" title="\large W = W - \alpha \left( \frac{v_{dW}}{\sqrt{S_{dW} + \epsilon}} \right )" /></a> 

  ​				<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;b&space;=&space;b&space;-&space;\alpha&space;\left(&space;\frac{v_{db}}{\sqrt{S_{db}&space;&plus;&space;\epsilon}}&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;b&space;=&space;b&space;-&space;\alpha&space;\left(&space;\frac{v_{db}}{\sqrt{S_{db}&space;&plus;&space;\epsilon}}&space;\right&space;)" title="\large b = b - \alpha \left( \frac{v_{db}}{\sqrt{S_{db} + \epsilon}} \right )" /></a>

- Hyperparameters for Adam:

  - Learning rate: needed to be tuned.
  - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\beta_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\beta_1" title="\large \beta_1" /></a>: parameter of the momentum - 0.9 is recommended by default.
  - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\beta_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\beta_2" title="\large \beta_2" /></a>: parameter of the RMSprop - 0.999 is recommended by default.
  - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\epsilon&space;:&space;10^{-8}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\epsilon&space;:&space;10^{-8}" title="\large \epsilon : 10^{-8}" /></a> is recommended by default.

### Learning rate decay

- Slowly reduce learning rate.

- As mentioned before mini-batch gradient descent won't reach the optimum point (converge). But by making the learning rate decay with iterations it will be much closer to it because the steps (and possible oscillations) near the optimum are smaller.

- One technique equations is
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\alpha&space;=&space;\left&space;(&space;\frac{1}{1&space;&plus;&space;(decay\_rate&space;\times&space;no&space;{\hspace{0.2cm}}&space;of&space;{\hspace{.2cm}}&space;epochs)}&space;\right&space;)\alpha_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\alpha&space;=&space;\left&space;(&space;\frac{1}{1&space;&plus;&space;(decay\_rate&space;\times&space;no&space;{\hspace{0.2cm}}&space;of&space;{\hspace{.2cm}}&space;epochs)}&space;\right&space;)\alpha_0" title="\large \alpha = \left ( \frac{1}{1 + (decay\_rate \times no {\hspace{0.2cm}} of {\hspace{.2cm}} epochs)} \right )\alpha_0" /></a>
  
  - **no of epoch** is over all data (not a single mini-batch).
  
- Other learning rate decay methods (continuous):
  - <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\alpha&space;=&space;(0.95^{(epoch\_num)})\alpha_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\alpha&space;=&space;(0.95^{(epoch\_num)})\alpha_0" title="\large \alpha = (0.95^{(epoch\_num)})\alpha_0" /></a>
  - <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\alpha&space;=&space;\frac{k}{\sqrt{no\_of\_epochs}}\alpha_0&space;\hspace{1cm}&space;or&space;\hspace{1cm}&space;\frac{k}{\sqrt{t}}\alpha_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\alpha&space;=&space;\frac{k}{\sqrt{no\_of\_epochs}}\alpha_0&space;\hspace{1cm}&space;or&space;\hspace{1cm}&space;\frac{k}{\sqrt{t}}\alpha_0" title="\large \alpha = \frac{k}{\sqrt{no\_of\_epochs}}\alpha_0 \hspace{1cm} or \hspace{1cm} \frac{k}{\sqrt{t}}\alpha_0" /></a>
  
- Some people perform learning rate decay discretely - repeatedly decrease after some number of epochs.

- Some people are making changes to the learning rate manually.

- **decay_rate** is another **hyperparameter**.

- For Andrew Ng, learning rate decay has less priority.

### The problem of local optima

- The normal local optima is not likely to appear in a deep neural network because data is usually high dimensional. For point to be a local optima it has to be a local optima for each of the dimensions which is highly unlikely.
- It's unlikely to get stuck in a bad local optima in high dimensions, it is much more likely to get to the saddle point rather to the local optima, which is not a problem.
- Plateaus can make learning slow:
  - Plateau is a region where the derivative is close to zero for a long time.
  - This is where algorithms like momentum, RMSprop or Adam can help.