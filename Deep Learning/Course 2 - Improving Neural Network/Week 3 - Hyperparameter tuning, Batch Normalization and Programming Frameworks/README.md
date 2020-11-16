## Hyperparameter tuning, Batch Normalization and Programming Frameworks

### Tuning process

- We need to tune our hyperparameters to get the best out of them.
- Hyperparameters importance are (as for Andrew Ng):
  1. Learning rate.
  2. Momentum beta.
  3. Mini-batch size.
  4. No. of hidden units.
  5. No. of layers.
  6. Learning rate decay.
  7. Regularization lambda.
  8. Activation functions.
  9. Adam <a href="https://www.codecogs.com/eqnedit.php?latex=\beta_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta_1" title="\beta_1" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\beta_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta_2" title="\beta_2" /></a> & <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /></a>.
- Its hard to decide which hyperparameter is the most important in a problem. It depends a lot on the problem.
- One of the ways to tune is to sample a grid with **N** hyperparameter settings and then try all settings combinations in the problem.
- Try random values: don't use a grid.
- You can use **Coarse to fine sampling scheme**:
  - When you find some hyperparameters values that give you a better performance - zoom into a smaller region around these values and sample more densely within this space.
- These methods can be automated.

### Using an appropriate scale to pick hyperparameters

- Let's say you have a specific range for a hyperparameter from **a** to **b**. It's better to search for the right ones using the logarithmic scale rather then in linear scale:

  - Calculate: `a_log = log(a)  # e.g. a = 0.0001 then a_log = -4`

  - Calculate: `b_log = log(b)  # e.g. b = 1  then b_log = 0`

  - Then:

    ```
    r = (a_log - b_log) * np.random.rand() + b_log
    # In the example the range would be from [-4, 0] because rand range [0,1)
    result = 10^r
    ```

    It uniformly samples values in log scale from [a,b].

- If we want to use the last method on exploring on the "momentum beta":

  - Beta best range is from 0.9 to 0.999.

  - You should search for 1 - <a href="https://www.codecogs.com/eqnedit.php?latex=\beta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta" title="\beta" /></a> in range 0.001 to 0.1 (1 - 0.9 and 1 - 0.999)  and then use  a = 0.00 and b = 0.1. Then:

    ```
    a_log = -3
    b_log = -1
    r = (a_log - b_log) * np.random.rand() + b_log
    beta = 1 - 10^r   # because 1 - beta = 10^r
    ```

### Hyperparameters tuning in practice: Pandas vs. Caviar 

- Intuitions about hyperparameter settings from one application area may or may not transfer to a different one.
- If you don't have much computational resources you can use the "babysitting model":
  - Day 0 you might initialize your parameter as random and then start training.
  - Then you watch your learning curve gradually decrease over the day.
  - And each day you nudge your parameters a little during training.
  - Called panda approach.
- If you have enough computational resources, you can run some models in parallel and at the end of the day(s) you check the results.
  - Called Caviar approach.

### Normalizing activations in a network

- In the rise of deep learning, one of the most important ideas has been an algorithm called **batch normalization**, created by two researchers, Sergey Ioffe and Christian Szegedy.

- Batch Normalization speeds up learning.

- Before we normalized input by subtracting the mean and dividing by variance. This helped a lot for the shape of the cost function and for reaching the minimum point faster.

- The question is: *for any hidden layer can we normalize <a href="https://www.codecogs.com/eqnedit.php?latex=A^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A^{[l]}" title="A^{[l]}" /></a> to train <a href="https://www.codecogs.com/eqnedit.php?latex=W^{[l&plus;1]},b^{[l&plus;1]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W^{[l&plus;1]},b^{[l&plus;1]}" title="W^{[l+1]},b^{[l+1]}" /></a> faster?* This is what batch normalization is about.

- There are some debates in the deep learning literature about whether you should normalize values before the activation function <a href="https://www.codecogs.com/eqnedit.php?latex=Z^{[1]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z^{[1]}" title="Z^{[1]}" /></a> or after applying the activation function <a href="https://www.codecogs.com/eqnedit.php?latex=A^{[1]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A^{[1]}" title="A^{[1]}" /></a>. In practice, normalizing <a href="https://www.codecogs.com/eqnedit.php?latex=Z^{[1]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z^{[1]}" title="Z^{[1]}" /></a> is done much more often and that is what Andrew Ng presents.

- Algorithm:

  - Given <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Z^{[l]}&space;=&space;[z^{[1]},...,z^{[m]}]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Z^{[l]}&space;=&space;[z^{[1]},...,z^{[m]}]" title="Z^{[l]} = [z^{[1]},...,z^{[m]}]" /></a>, *i* = *1* to *m* (for each input)
  - Compute <a href="https://www.codecogs.com/eqnedit.php?latex=\mu&space;=&space;\frac{1}{m}&space;\sum&space;z^{[\textbf&space;i]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu&space;=&space;\frac{1}{m}&space;\sum&space;z^{[\textbf&space;i]}" title="\mu = \frac{1}{m} \sum z^{[\textbf i]}" /></a>
  - Compute <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma&space;=&space;\frac{1}{m}&space;\sum&space;(z^{[\textbf&space;i]}-\mu)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma&space;=&space;\frac{1}{m}&space;\sum&space;(z^{[\textbf&space;i]}-\mu)^2" title="\sigma = \frac{1}{m} \sum (z^{[\textbf i]}-\mu)^2" /></a>
  - Then <a href="https://www.codecogs.com/eqnedit.php?latex=Z_{norm}^{[i]}&space;=&space;\frac{(z^{[i]}-\mu)}{\sqrt{\sigma&space;&plus;&space;\epsilon}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z_{norm}^{[i]}&space;=&space;\frac{(z^{[i]}-\mu)}{\sqrt{\sigma&space;&plus;&space;\epsilon}}" title="Z_{norm}^{[i]} = \frac{(z^{[i]}-\mu)}{\sqrt{\sigma + \epsilon}}" /></a> (add <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /></a> for numerical stability if variance = 0)
    - Forcing the inputs to a distribution with zero mean and variance of 1.
  - Then <a href="https://www.codecogs.com/eqnedit.php?latex=\widetilde{z}^{[i]}&space;=&space;\gamma&space;&plus;&space;z_{norm}^{[i]}&plus;\beta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widetilde{z}^{[i]}&space;=&space;\gamma&space;&plus;&space;z_{norm}^{[i]}&plus;\beta" title="\widetilde{z}^{[i]} = \gamma + z_{norm}^{[i]}+\beta" /></a>
    - To make inputs belong to other distribution (with other mean and variance).
    - gamma and beta are learnable parameters of the model.
    - Making the NN learn the distribution of the outputs.
    - _Note:_ if <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma&space;=&space;\sqrt{\sigma&space;&plus;&space;\epsilon}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma&space;=&space;\sqrt{\sigma&space;&plus;&space;\epsilon}" title="\gamma = \sqrt{\sigma + \epsilon}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\beta&space;=&space;\mu" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta&space;=&space;\mu" title="\beta = \mu" /></a> then <a href="https://www.codecogs.com/eqnedit.php?latex=\widetilde{z}^{[i]}&space;=&space;z^{[i]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widetilde{z}^{[i]}&space;=&space;z^{[i]}" title="\widetilde{z}^{[i]} = z^{[i]}" /></a>

  ### Fitting Batch Normalization into a neural network

  - Using batch norm in 3 hidden layers NN:
    <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\Large&space;X\xrightarrow[]{W^{[1]},b^{[1]}}&space;z^{[1]}\xrightarrow[batch\hspace{0.2cm}norm]{\beta^{[1]},\gamma^{[1]}}\widetilde{z}^{[1]}\xrightarrow[]{}a^{[1]}=g^{[1]}(\widetilde{z}^{[1]})\xrightarrow[]{W^{[2]},b^{[2]}}z^{[2]}\xrightarrow[batch\hspace{0.2cm}norm]{\beta^{[2]},\gamma^{[2]}}\widetilde{z}^{[2]}\rightarrow&space;a^{[2]}\rightarrow&space;..." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\Large&space;X\xrightarrow[]{W^{[1]},b^{[1]}}&space;z^{[1]}\xrightarrow[batch\hspace{0.2cm}norm]{\beta^{[1]},\gamma^{[1]}}\widetilde{z}^{[1]}\xrightarrow[]{}a^{[1]}=g^{[1]}(\widetilde{z}^{[1]})\xrightarrow[]{W^{[2]},b^{[2]}}z^{[2]}\xrightarrow[batch\hspace{0.2cm}norm]{\beta^{[2]},\gamma^{[2]}}\widetilde{z}^{[2]}\rightarrow&space;a^{[2]}\rightarrow&space;..." title="\Large X\xrightarrow[]{W^{[1]},b^{[1]}} z^{[1]}\xrightarrow[batch\hspace{0.2cm}norm]{\beta^{[1]},\gamma^{[1]}}\widetilde{z}^{[1]}\xrightarrow[]{}a^{[1]}=g^{[1]}(\widetilde{z}^{[1]})\xrightarrow[]{W^{[2]},b^{[2]}}z^{[2]}\xrightarrow[batch\hspace{0.2cm}norm]{\beta^{[2]},\gamma^{[2]}}\widetilde{z}^{[2]}\rightarrow a^{[2]}\rightarrow ..." /></a>

  - Our NN parameters will be:

    - <a href="https://www.codecogs.com/eqnedit.php?latex=W^{[1]},&space;b^{[1]},&space;...,&space;W^{[L]},&space;b^{[L]},&space;\beta^{[1]},&space;\gamma^{[1]},&space;...,&space;\beta^{[L]},&space;\gamma^{[L]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W^{[1]},&space;b^{[1]},&space;...,&space;W^{[L]},&space;b^{[L]},&space;\beta^{[1]},&space;\gamma^{[1]},&space;...,&space;\beta^{[L]},&space;\gamma^{[L]}" title="W^{[1]}, b^{[1]}, ..., W^{[L]}, b^{[L]}, \beta^{[1]}, \gamma^{[1]}, ..., \beta^{[L]}, \gamma^{[L]}" /></a>
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\beta^{[1]},&space;\gamma^{[1]},&space;...,&space;\beta^{[L]},&space;\gamma^{[L]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta^{[1]},&space;\gamma^{[1]},&space;...,&space;\beta^{[L]},&space;\gamma^{[L]}" title="\beta^{[1]}, \gamma^{[1]}, ..., \beta^{[L]}, \gamma^{[L]}" /></a> are updated using any optimization algorithms (like GD, RMSprop, Adam)

  - If you are using a deep learning framework, you won't have to implement batch norm yourself:

    - Ex. in Tensorflow you can add this line: `tf.nn.batch-normalization()`

  - Batch normalization is usually applied with mini-batches.

  - If we are using batch normalization parameters <a href="https://www.codecogs.com/eqnedit.php?latex=b^{[1]},...,b^{[L]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b^{[1]},...,b^{[L]}" title="b^{[1]},...,b^{[L]}" /></a> doesn't count because they will be eliminated after mean subtraction step, so:

    <a href="https://www.codecogs.com/eqnedit.php?latex=Z^{[l]}&space;=&space;W^{[l]}A^{[l-1]}&space;&plus;&space;b^{[l]}&space;\implies&space;Z^{[l]}&space;=&space;W^{[l]}A^{[l-1]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z^{[l]}&space;=&space;W^{[l]}A^{[l-1]}&space;&plus;&space;b^{[l]}&space;\implies&space;Z^{[l]}&space;=&space;W^{[l]}A^{[l-1]}" title="Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]} \implies Z^{[l]} = W^{[l]}A^{[l-1]}" /></a>

    <a href="https://www.codecogs.com/eqnedit.php?latex=Z_{norm}^{[l]}&space;=&space;..." target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z_{norm}^{[l]}&space;=&space;..." title="Z_{norm}^{[l]} = ..." /></a>

    <a href="https://www.codecogs.com/eqnedit.php?latex=\widetilde{Z}^{[l]}&space;=&space;\gamma^{[l]}&space;*&space;Z_{norm}^{[l]}&space;&plus;&space;\beta^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widetilde{Z}^{[l]}&space;=&space;\gamma^{[l]}&space;*&space;Z_{norm}^{[l]}&space;&plus;&space;\beta^{[l]}" title="\widetilde{Z}^{[l]} = \gamma^{[l]} * Z_{norm}^{[l]} + \beta^{[l]}" /></a>

    - Taking the mean of a constant <a href="https://www.codecogs.com/eqnedit.php?latex=b^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b^{[l]}" title="b^{[l]}" /></a> will eliminate the <a href="https://www.codecogs.com/eqnedit.php?latex=b^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b^{[l]}" title="b^{[l]}" /></a>

  - So if you are using batch normalization, you can remove <a href="https://www.codecogs.com/eqnedit.php?latex=b^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b^{[l]}" title="b^{[l]}" /></a> or make it always zero.

  - So the parameters will be <a href="https://www.codecogs.com/eqnedit.php?latex=W^{[l]},&space;\beta^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W^{[l]},&space;\beta^{[l]}" title="W^{[l]}, \beta^{[l]}" /></a>, and<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha^{[l]}" title="\alpha^{[l]}" /></a>.

  - Shapes:

    - `Z[l]       - (n[l], m)`
    - `beta[l]    - (n[l], m)`
    - `gamma[l]   - (n[l], m)`