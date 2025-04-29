# machine-learning-exercise-8-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning Exercise 8 Solved](https://www.ankitcodinghub.com/product/machine-learning-labs-solved-5/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;110201&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning  Exercise 8 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
&nbsp;

(Neural Networks &amp; PyTorch Introduction)

Goals. The goals of this exercise are to:

• Introduce you to the PyTorch deep learning framework.

• Explore the representational capacity of neural networks by approximating 2d functions.

• Train a fully connected neural network to classify digits using PyTorch.

• Explore the effects of weight initialization on activation and gradient magnitudes.

• Mathematically analyze basic components of neural network training.

Problem 1 (PyTorch Introduction and Neural Networks):

The accompaning Jupyter Notebook contains a brief introduction to PyTorch along with three neural network exercises. We recommend running the notebook on Google Colab which provides you with a free GPU and does not require installing any packages.

1. Open the colab link for the lab 8:

https://colab.research.google.com/github/epfml/ML_course/blob/master/labs/ex08/template/ex08.ipynb 2. To save your progress, click on “File &gt; Save a Copy in Drive” to get your own copy of the notebook.

3. Click ‘connect’ on top right to make the notebook executable (or ‘open in playground’).

4. Work your way through the introduction and exercises.

Alternatively you can download the notebook from GitHub and install PyTorch locally, see the instructions on pytorch.org.

Additional Tutorials: If you plan on using PyTorch in your own projects, we recommend additionally going through the official tutorials after the exercise session:

• Deep Learning with PyTorch: a 60-minute Blitz

• Learning PyTorch with Examples

Problem 2 (Variance Preserving Weight Initialization for ReLUs):

When training neural networks it is desirable to keep the variance of activations roughly constant across layers.

Let’s assume we have y = x⊤w where x = ReLU(z), z ∈ Rd and w ∈ Rd. Further assume that all elements and are independent with wi ∼ N(0,σ) and zi ∼ N(0,1).

Derive Var[y] as a function of d and σ i.e. how should we set σ to have Var[y] = 1.

Hint: Remember the law of total expecation i.e. EA[A] = EB[EA[A|B]]

Problem 3 (Softmax Cross Entropy):

In the notebook exercises we performed multiclass classification using softmax-cross-entropy as our loss. The softmax of a vector x = [x1,…,xd]⊤ is a vector z = [z1,…,zd]⊤ with:

(1)

The label y is an integer denoting the target class. To turn y into a probability distribution for use with crossentropy, we use one-hot encoding:

⊤ k onehot(y) = y = [y1,…,yd] where(2)

The cross-entropy is given by:

(3)

We ask you to do the following:

1. Equation 1 potentially computes exp of large positive numbers which is numerically unstable. Modify Eq. 1 to avoid positive numbers in exp. Hint: Use maxj(xj).

3. What values of xi minimize the softmax-cross-entropy loss? To avoid complications, practitioners sometimes use a trick called label smoothing where y is replaced by yˆ for some small value e.g. ϵ = 0.1.

Problem 4 (Computation and Memory Requirements of Neural Networks):

Let’s consider a fully connected neural network with L layers in total, all of width K. The input is a mini-batch of size n × K. For this exercise we will only consider the matrix multiplications, ignoring biases and activation functions.

• How many multiplications are needed in a forward pass (inference)?

• How many multiplications are needed in a forward + backward pass (training)?

• How much memory is needed for an inference forward pass? The memory needed is the sum of the memory needed for activations and weights. Assume activations are deleted as soon as they are no longer required.

• How much memory is needed for a training forward + backward pass? Note that during training we additionally use the forward activations for the computation of the derivatives.

2
