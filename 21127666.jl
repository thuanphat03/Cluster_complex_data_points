### A Pluto.jl notebook ###
# v0.19.29

#> [frontmatter]
#> title = "Lab 05: Support-Vector Networks"
#> date = "2023-12-02"
#> tags = ["Machine Learning", "Statistical Learning Theory", "Classification", "Intro2ML ", "Lab5 "]
#> description = "Implement primal/ kernel SVM"
#> license = "Copyright © Dept. of CS, VNUHCM-University of Science, 2023. This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License"

using Markdown
using InteractiveUtils

# ╔═╡ 287aba39-66d9-4ff6-9605-1bca094a1ce5
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI # visualization purpose
	TableOfContents(title="📚 Table of Contents", indent=true, depth=3, aside=true)
end
  ╠═╡ =#

# ╔═╡ 08c5b8cf-be45-4d11-bd24-204b364a8278
using Plots, Distributions, LinearAlgebra, Random

# ╔═╡ e48c1538-f9b2-4b7d-b855-1e3cab86567b
# START YOUR CODE
using LIBSVM

# ╔═╡ f1b2170a-9595-4a91-9035-16d7b7f9aeb2
using Printf

# ╔═╡ dc09fd89-b904-45f6-a5b4-66c148922f8d
using ScikitLearn: fit_transform!

# ╔═╡ 86f24f32-d8ee-49f2-b71a-bd1c5cd7a28f
# edit the code below to set your name and student identity number (i.e. the number without @student.hcmus.edu.vn)

student = (name = "Trần Thuận Phát", id = "21127666")

# you might need to wait until all other cells in this notebook have completed running.
# scroll around the page to see what's up

# ╔═╡ eeae9e2c-aaf8-11ed-32f5-934bc393b3e6
md"""
Submission by: **_$(student.name)_** ($(student.id)@student.hcmus.edu.vn)
"""

# ╔═╡ cab7c88c-fe3f-40c4-beea-eea924d67975
md"""
# **Homework 5**: Support Vector Networks
`CSC14005`, Introduction to Machine Learning

This notebook was built for FIT@HCMUS student to learn about Support Vector Machines/or Support Vector Networks in the course CSC14005 - Introduction to Machine Learning. 

## Instructions for homework and submission

It's important to keep in mind that the teaching assistants will use a grading support application, so you must strictly adhere to the guidelines outlined in the instructions. If you are unsure, please ask the teaching assistants or the lab instructors as soon as you can. **Do not follow your personal preferences at stochastically**

### Instructions for doing homework

- You will work directly on this notebook; the word **TODO** indicates the parts you need to do.
- You can discuss the ideas as well as refer to the documents, but *the code and work must be yours*.

### Instructions for submission

- Before submitting, save this file as `<ID>.jl`. For example, if your ID is 123456, then your file will be `123456.jl`. Submit that file on Moodle.
	
!!! danger
	**Note that you will get 0 point for the wrong submit**.

### Content of the assignment

- Recall: Perceptron & Geometriy Margin
- Linear support vector machine (Hard-margin, soft-margin)
- Popular non-linear kernels
- Computing SVM: Primal, Dual
- Multi-class SVM

### Others

Other advice for you includes:
- Starting early and not waiting until the last minute
- Proceed with caution and gentleness. 

"Living 'Slow' just means doing everything at the right speed – quickly, slowly, or at whatever pace delivers the best results." Carl Honoré.

- Avoid sources of interference, such as social networks, games, etc.

"""

# ╔═╡ 99329d11-e709-48f0-96b5-32ae0cac1f50
Random.seed!(0)

# ╔═╡ bbccfa2d-f5b6-49c7-b11e-53e419808c1b
html"""
<p align="center">
  <img src="https://lnhutnam.github.io/resources/images/yinyang.png" />
</p>
"""

# ╔═╡ 4321f49b-1057-46bc-8d67-be3122be7a68
md"""
## Problem statement

Let $\mathcal{D} = \{(x_i, y_i) | x_i \in \mathbb{R}^{d}, y_i \in \{-1, 1\}\}_{i=1}^{n}$ be a dataset which is a set of pairs where $x_i \in \mathbb{R}^d$ is *data point* in some $d$-dimension vector space, and $y_i \in \{-1, 1\}$ is a *label* of the corespondent $x_i$ data point classifying it to one of the two classes.

The model is trained on $\mathcal{D}$ after which it is present with $x_{i+1}$, and is asked to predict the label of this previously unseen data point.

The prediction function is donated by $f(x) : \mathbb{R}^d \rightarrow \{-1, 1\}$
"""

# ╔═╡ 4fdaeeda-beee-41e1-a5f0-3209151a880d
md"""
## Recall: Perceptron & Geometry Margin (Maximum 2.5 points)

In fact, it is always possible to come up with such a "perfect" binary function if training samples are distinct. However, it is unclear whether such rules are applicable to data that does not exist in the training set. We don't need "learn-by-heart" learners; we need "intelligent" learners. More especially, such trivial rules do not suffice because our task is not to correctly classify the training set. Our task is to find a rule that works well for all new samples we would encounter in the access control setting; the training set is merely a helpful source of information to find such a function. We would like to find a classifier that "generalizes" well.

The key to finding a generalized classifier is to constrain the set of possible binary functions we can entertain. In other words, we would like to find a class of classifier functions such that if a function in this class works well on the training set, it is also likely to work well on the unseen images. This problem is considered a key problem named "model selection" in machine learning.
"""

# ╔═╡ ec906d94-8aed-4df1-932c-fa2263e6325d
md"""
### Linear classifiers through origin

For simplicity, we will just fix the function class for now. We will only consider a type of *linear classifiers*. For more formally, we consider the function of the form:

$f(\mathbf{x}, \theta) = \text{sign}(\theta_1\mathbf{x}_1 + \theta_2\mathbf{x}_2 + \dots + \theta_d\mathbf{x}_d) = \text{sign}(\theta^\top\mathbf{x})$
where $\theta = [\theta_1, \theta_2, \dots, \theta_d]^\top$ is a column vector of real valued parameters.

Different settings of the parameters give different functions in this class, i.e., functions whose value or output in $\{-1, 1\}$ could be different for some input $\mathbf{x}$.
"""

# ╔═╡ 42882ca3-8df6-4fb8-8884-29337373bac5
md"""
### Perceptron Learning Algorithms

After chosen a class of functions, we still have to find a specific function in this class that works well on the training set. This task often refers to estimation problem in machine learning. We would like to find $\theta$ that minimize the *training error*, i.e we would like to find a linear classifier that make fewest mistake in the training set.

$\mathcal{L}(\theta) = \frac{1}{n}\sum_{t=1}^n\left(1-\delta(y_t, f(\mathbf{x}; \theta))\right) = \frac{1}{n}\sum_{t=1}^n\text{Loss}(y_t, f(\mathbf{x}; \theta))$
where $\delta(y, y') = 1$ if $y=y'$ and $0$ if otherwise.

Perceptron update rule: Let $k$ donates the number of parameter updates we have performed and $\theta^{(k)}$ is the parameter vector after $k$ updates. Initially $k=0$, and $\theta^{(k)} = 0$. We the loop through all the training instances $(\mathbf{x}_t, y)t)$, and updates the parameters only in response to mistakes,

$$\begin{cases}
\theta^{(k+1)} \leftarrow \theta^{(k)} + y_t\mathbf{x}_t \text{ if } y_t(\theta^{(k+1)})^\top\mathbf{x}_t < 0 \\
\text{The parameters unchanged}\end{cases}$$

![Geometry intuition of Perceptron](https://lnhutnam.github.io/assets/images_posts/pla/linear_classfier.png)
"""

# ╔═╡ e78094ff-6565-4e9d-812e-3a36f78731ed
begin
	n = 1000 # sample size
	d = 2; # dimensionality of data
	μ = 5 # mean
	Σ = 8 # variance
end

# ╔═╡ d0ecb21c-6189-4b10-9162-1b94424f49ce
points1ₜᵣₐᵢₙ = rand(MvNormal([Σ, μ], 5 .* [2 (μ - d)/Σ; (μ - d)/Σ d]), n ÷ 2)

# ╔═╡ 921e8d15-e751-4976-bb80-2cc09e6c950e
points2ₜᵣₐᵢₙ = rand(MvNormal([-μ+d, Σ+d], 5 .* [3 (μ - d)/μ; (μ - d)/μ d]), n ÷ 2)

# ╔═╡ 4048a66b-a89f-4e37-a89f-6fe57519d5d7
points1ₜₑₛₜ = rand(MvNormal([Σ, μ], 5 .* [2 (μ - d)/Σ; (μ - d)/Σ d]), n ÷ 2)

# ╔═╡ 17663f65-1aa1-44c4-8eae-f4bc6e24fe98
points2ₜₑₛₜ = rand(MvNormal([-μ+d, Σ+d], 5 .* [3 (μ - d)/μ; (μ - d)/μ d]), n ÷ 2)

# ╔═╡ 16390a59-9ef0-4b05-8412-7eef4dfb13ee
md"""
!!! todo
 Your task here is implement the PLA (1 point). You can modify your own code in the area bounded by START YOUR CODE and END YOUR CODE.
"""

# ╔═╡ 43dee3c9-88f7-4c79-b4a3-6ab2cc3bba2e
"""
	Perceptron learning algorithm (PLA) implement function.

### Fields
- pos_data::Matrix{Float64}: Input features for postive class (+1)
- neg_data::Matrix{Float64}: Input features for negative class (-1)
- n_epochs::Int64=10000: Maximum training epochs. Default is 10000
- η::Float64=0.03: Learning rate. Default is 0.03
"""
function pla(pos_data::Matrix{Float64}, neg_data::Matrix{Float64}, 
	n_epochs::Int64=10000, η::Float64=0.03)
	# START YOUR CODE
	# Init θ 
	pos_data = hcat(pos_data', ones(Base.size(pos_data, 2)))
	pos_data = hcat(pos_data, ones(Base.size(pos_data, 1)))
	neg_data = hcat(neg_data', ones(Base.size(neg_data, 2)))
	neg_data = hcat(neg_data, -1 .* ones(Base.size(neg_data, 1)))
	D = vcat(pos_data, neg_data)
	θ = D[rand(1:Base.size(D, 1)), 1:end-1]

	# Repeat until satisfied
	for i in 1:n_epochs
		D = D[shuffle(1:end), :]
		for row in 1:Base.size(D, 1)
			x = D[row, 1:end-1]
			y = D[row, end]
			if sign(θ' * x) .!= y
				θ = θ + η * y * x
			end
		end
		# Check converences
		if all(sign.(D[:, 1:end-1] * θ) .== D[:, end])
			break
		end
	end
	
	# END YOUR CODE
	return θ
end

# ╔═╡ 2d7dde2b-59fc-47c0-a2d0-79dcd48d8041
θₘₗ = pla(points1ₜᵣₐᵢₙ, points2ₜᵣₐᵢₙ)

# ╔═╡ 8e06040d-512e-4ff6-a035-f121e9d73eb4
"""
	Decision boundary visualization function for PLA

### Fields
- θ: PLA paramters
- pos_data::Matrix{Float64}: Input features for postive class (+1)
- neg_data::Matrix{Float64}: Input features for negative class (-1)
"""
function draw_pla(θ, pos_data::Matrix{Float64}, neg_data::Matrix{Float64})
	plt = scatter(pos_data[1, :], pos_data[2, :], label="y = 1")
  	scatter!(plt, neg_data[1, :], neg_data[2, :], label="y = -1")

	b = θ[3]
	θₘₗ = θ[1:2]

	decision(x) = θₘₗ' * x + b
	
	D = ([
	  tuple.(eachcol(pos_data), 1)
	  tuple.(eachcol(neg_data), -1)
	])

	xₘᵢₙ = minimum(map((p) -> p[1][1], D))
  	yₘᵢₙ = minimum(map((p) -> p[1][2], D))
  	xₘₐₓ = maximum(map((p) -> p[1][1], D))
 	yₘₐₓ = maximum(map((p) -> p[1][2], D))
	
	contour!(plt, xₘᵢₙ:0.1:xₘₐₓ, yₘᵢₙ:0.1:yₘₐₓ,
			(x, y) -> decision([x, y]),
			levels=[0], linestyles=:solid, label="Decision boundary", colorbar_entry=false, color=:green)
end

# ╔═╡ f40fbc75-2879-4bf8-a2ba-7b9356149dcd
# Uncomment this line below when you finish your implementation
draw_pla(θₘₗ, points1ₜᵣₐᵢₙ, points2ₜᵣₐᵢₙ)

# ╔═╡ da9c313e-4623-4370-9d5f-0560d62deb51
"""
	Calculating values for True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)

### Fields
- y_test: Actual labels
- y_pred: Predicted labels
"""
function tpfptnfn_cal(y_test, y_pred, positive_class=1)
	true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

	# Calculate true positives, false positives, false negatives, and true negatives
    for (true_label, predicted_label) in zip(y_test, y_pred)
        if true_label == positive_class && predicted_label == positive_class
            true_positives += 1
		elseif true_label != positive_class && predicted_label == positive_class
            false_positives += 1
		elseif true_label == positive_class && predicted_label != positive_class
            false_negatives += 1
		elseif true_label != positive_class && predicted_label != positive_class
            true_negatives += 1
        end
    end

	return true_positives, false_positives, true_negatives, false_negatives
end

# ╔═╡ c0d56e33-6dcf-4675-a679-a55e7baaeea1
"""
	Evaluation function for PLA to calculate accuracy

### Fields
- θ: PLA paramters
- pos_data::Matrix{Float64}: Input features for postive class (+1)
- neg_data::Matrix{Float64}: Input features for negative class (-1)
"""
function eval_pla(θ, pos_data, neg_data)
	n = size(pos_data, 2)
	X = vcat(hcat(pos_data, neg_data), ones(n * 2)')
	
	y_test = vcat(ones(n), -ones(n))'
	y_pred = [sign(x) for x ∈ θ' * X]

	# START YOUR CODE
	# TODO: acc, p, r, f1???
	TP, FP, TN, FN = tpfptnfn_cal(y_test, y_pred)
	acc = (TP + TN) / (TP + TN + FP + FN)
	p =  TP / (TP + FP)
	r = TP / (TP + FN)
	f1 = 2 * p * r / (p + r)
	# END YOUR CODE
	
	print(" acc: $acc\n precision: $p\n recall: $r\n f1_score: $f1\n")
	return acc, p, r, f1
end

# ╔═╡ a9d60d10-2d93-4c3e-8720-5534efd646a4
# Uncomment this line below when you finish your implementation
eval_pla(θₘₗ, points1ₜₑₛₜ, points2ₜₑₛₜ)

# ╔═╡ d4709ae3-9de5-4d46-9d95-e15fcf741bc6
md"""

### Convergence Proof

Assume that all the training instances have bounded Euclidean norms), i.e $|| \mathbf{x} || \leq R$ . Assume that exists a linear classifier in class of functions with finite parameter values that correctly classifies all the training instances. For precisely, we assume that there is some $\gamma >0$ such that $y_t(\theta^{*})^\top\mathbf{x}_t \geq \gamma$ for all $t = 1...n$.

The convergence proof is based on combining two results:
- **Result 1**: we will show that the inner product $(\theta^{*})^\top\theta^{(k)}$ increases at least linearly with each update.
"""

# ╔═╡ bd418098-edfb-4989-8bd5-23bca5059c51
md"""
!!! todo
Your task here is show the proof of result 1. (0.25 point)
"""

# ╔═╡ 8e2c8a02-471e-4321-8d8b-d25b224aa0c1
md"""
**START YOUR PROOF**

Để chứng minh $(\theta^{*})^\top\theta^{(k)}$ tăng ít nhất 1 tuyến tính sau mỗi lần cập nhật ta chứng minh rằng:
	
$(\theta^{*})^\top\theta^{(k+1)} - (\theta^{*})^\top\theta^{(k)} \geq a\gamma + b$

- Với $\theta^{(k+1)}$ là giá trị theta cập nhật từ $\theta^{(k)}$.
- a, b là hằng số.

Bước 1: Ta có công thức $\theta^{(k+1)} = \theta^{(k)} + y_t x_t$ (công thức cập nhật theta dựa theo thuật toán PERCEPTON).

Bước 2: Nhân 2 vế của công thức trên cho $(\theta^{*})^\top$ và thực hiện phân phối phép nhân. Ta được:

$(\theta^{*})^\top\theta^{(k+1)} = (\theta^{*})^\top \theta^{(k)} + (\theta^{*})^\top y_t x_t$

Bước 3: Chuyển $(\theta^{*})^\top \theta^{(k)}$ qua bên trái. Ta được: 

$(\theta^{*})^\top\theta^{(k+1)} - (\theta^{*})^\top \theta^{(k)} = (\theta^{*})^\top y_t x_t$

Bước 4: Thực hiện giao hoán vế phải. Ta được:

$(\theta^{*})^\top\theta^{(k+1)} - (\theta^{*})^\top \theta^{(k)} = y_t (\theta^{*})^\top x_t$

Bước 5: Mà $y_t (\theta^{*})^\top x_t \geq \gamma$ luôn đúng với mọi t (giả định từ đề bài). Từ đó, ta có:

$(\theta^{*})^\top\theta^{(k+1)} - (\theta^{*})^\top \theta^{(k)} \geq \gamma$

Bước 6: Vậy kết luận rằng $(\theta^{*})^\top\theta^{(k)}$ tăng ít nhất 1 tuyến tính sau mỗi lần cập nhật với a = 1 và b = 0.

**END YOUR PROOF**
"""

# ╔═╡ fb06ed9a-2b6a-422f-b709-1c2f782da49e
md"""
- **Result 2**: The squared norm $||\theta^{(k)}||^2$ increases at most linearly in the number of updates $k$.
"""

# ╔═╡ 721bc350-c561-4985-b212-17cfd8d11f5a
md"""
!!! todo
Your task here is show the proof of result 2. (0.25 point)
"""

# ╔═╡ 6b4452a1-1cfd-43da-8177-2aee1259bf71
md"""
**START YOUR PROOF**

Để chứng minh $||\theta^{(k)}||^2$ tăng nhiều nhất tuyến tính là dựa theo số lần cập nhật k ta chứng minh rằng:
	
$||\theta^{(k + 1)}||^2 - ||\theta^{(k)}||^2 \leq ag^{(k)} + b$

- Với $\theta^{(k+1)}$ là giá trị theta cập nhật từ $\theta^{(k)}$.
- a, b là hằng số.
- Với $g^{(k)}$ là một con số dựa theo số lần cập nhật k.

Bước 1: Ta có công thức $\theta^{(k+1)} = \theta^{(k)} + y_t x_t$ (công thức cập nhật theta dựa theo thuật toán PERCEPTON).

Bước 2: Bình phương của Euclidean norm cho 2 vế của công thức trên và thực hiện phân phối phép nhân. Ta được:

$||\theta^{(k + 1)}||^2 - ||\theta^{(k)}||^2 = 2(\theta^{(k)})^\top y_tx_t + ||y_tx_t||^2$

Bước 3: Dựa theo bất đẳng thức Cauchy–Schwarz $|<x, y>| \leq ||x||.||y||$, ta được: 

$|2(\theta^{(k)})^\top y_tx_t| \leq ||\theta^{(k)}||.||2y_tx_t||$

Bước 4: Do $2(\theta^{(k)})^\top y_tx_t \geq 0$ (giả thiết đã cho trên đề) nên ta có thể bỏ trị tuyệt đối và ta có:

$2(\theta^{(k)})^\top y_tx_t \leq ||\theta^{(k)}||.||2y_tx_t||$

Bước 5: Cộng 2 vế cho $||y_tx_t||^2$, ta được:

$2(\theta^{(k)})^\top y_tx_t + ||y_tx_t||^2 \leq ||\theta^{(k)}||.||2y_tx_t|| + ||y_tx_t||^2$

Bước 6: Vế trái ở bước 5 cũng bằng vế trái ở bước 2 và ta thay vào, nên ta có:

$||\theta^{(k + 1)}||^2 - ||\theta^{(k)}||^2 \leq ||\theta^{(k)}||.||2y_tx_t|| + ||y_tx_t||^2$

Bước 7: Do $||2y_tx_t||$ và $||y_tx_t||^2$ là các hằng số và luôn đúng với mọi t và $||\theta^{(k)}||$ sẽ là 1 con số thực được tính dựa theo số lần cập nhật k. Từ đó, ta có thể suy ra được:

$||\theta^{(k + 1)}||^2 - ||\theta^{(k)}||^2 \leq ag^{(k)} + b$

- Với $g^{(k)} = ||\theta^{(k)}||$.
- Và $a = ||2y_tx_t||, b = ||y_tx_t||^2$.

Bước 8: Vậy kết luận rằng $||\theta^{(k)}||^2$ tăng nhiều nhất tuyến tính là dựa theo số lần cập nhật k sau mỗi lần cập nhật.

**END YOUR PROOF**
"""

# ╔═╡ e2bde012-e641-4ee6-aaf7-fee91e0626c2
md"""
We can now combine parts 1) and 2) to bound the cosine of the angle between $\theta^{(k)}$ and $\theta^{*}$. Since cosine is bounded by one, thus

$1 \geq \frac{k\gamma}{\sqrt{kR^2}\left \| \theta^{(*)}\right \|} \leftrightarrow k \leq \frac{R^2\left \| \theta^{(*)}\right \|^2}{\gamma^2}$

By combining the two we can show that the cosine of the angle between $\theta^{(k)}$ and $\theta^{*}$ has to increase by a finite increment due to each update. Since cosine is bounded by one, it follows that we can only make a finite number of updates.
"""

# ╔═╡ cd1160d3-4603-4d18-b107-e68355fc0604
md"""
### Geometric margin & SVM Motivation

There is a question? Does $\frac{\left \| \theta^{(*)}\right \|^2}{\gamma^2}$ relate to how difficult the classification problem is? Its inverse, i.e., $\frac{\gamma^2}{\left \| \theta^{(*)}\right \|^2}$ is the smallest distance in the vector space from any samples to the decision boundary specified by $\theta^{(*)$. In other words, it serves as a measure of how well the two classes of data are separated (by a linear boundary). We call this is gemetric margin, donated by $\gamma_{geom}$. As a result, the bound on the number of perceptron updates can be written more succinctly in terms of the geometric margin $\gamma_{geom}$ (You know that man, Vapnik–Chervonenkis Dimension)

![](https://lnhutnam.github.io/assets/images_posts/pla/geometric_margin.png)

$$k \leq \left(\frac{R}{\gamma_{geom}}\right)^2$$. We note some interesting thing about the result:
- Does not depend (directly) on the dimension of the data, nor
- number of training instances

Can’t we find such a large margin classifier directly? YES, in this homework, you will do it with Support Vector Machine :)
"""

# ╔═╡ eb804ff4-806b-4a11-af51-d4c3730c84b0
md"""
## Linear Support Vector Machine (Maximum 6 points)

From the problem statement section, we are given

$\{(x_i, y_i) | x_i \in \mathbb{R}^{d}, y_i \in \{-1, 1\}\}_{i=1}^{n}$

And based on previous section, we want to find the "maximum-geometric margin" that divides the space into two parts so that the distance between the hyperplane and the nearest point from either class is maximized. Any hyperplane can be written as the set of data points $\mathbf{x}$ satisfying

$\mathbf{\theta}^\top\mathbf{x} + b = 0$
"""

# ╔═╡ 4cd4dbad-7583-4dbd-806e-b6279aafc191
md"""
### Hard-margin

The goal of SVM is to choose two parallel hyperplanes that separate the two classes of data in order to maximize the distance between them. The region defined by these two hyperplanes is known as the "margin," and the maximum-margin hyperplane is the one located halfway between them. And these hyperplane can be decribed as

$$\mathbf{\theta}^\top\mathbf{x} + b = 1 \text{(anything on or above this boundary is of one class, with label 1)}$$ and

$$\mathbf{\theta}^\top\mathbf{x} + b = -1 \text{(anything on or below this boundary is of the other class, with label -1)}$$

Geometrically, the distance between these two hyperplanes is $\frac{2}{||\mathbf{\theta}||}$
"""

# ╔═╡ 91e528df-20e4-40b1-8ec0-96b05f59f556
md"""
!!! todo
Your task here is show that the distance between these two hyperplanes is $\frac{2}{||\mathbf{\theta}||}$ (1 point). You can modify your own code in the area bounded by START YOUR PROOF and END YOUR PROOF.
"""

# ╔═╡ e8105abb-6d8b-45ee-aebf-9ccc66b72b23
md"""
**START YOUR PROOF**

Ta áp dụng tính chất tính khoảng cách từ 1 điểm đến một 1 đường thẳng là:

$d = \frac{|A_1X_1 + A_2X_2 + ... + A_nX_n + C|}{\sqrt{A_1^2 + A_2^2 + ... + A_n^2}}$

Ta xét hyperland $\mathbf{\theta}^\top\mathbf{x} + b = 1$, thì khoảng cách từ 1 điểm đến đường thẳng đó sẽ là:

$d = \frac{|\mathbf{\theta}^\top\mathbf{x} + b - 1|}{||\mathbf{\theta}||}$

Ta lấy 1 điểm trên hyperland $\mathbf{\theta}^\top\mathbf{x} + b = -1$. Vậy điểm đó phải thỏa là $\mathbf{\theta}^\top\mathbf{x} = -1 - b$

Ta thay điểm trên vào công thức tính khoảng cách của hyperland $\mathbf{\theta}^\top\mathbf{x} + b = 1$, ta được:

$d = \frac{|-1 - b + b - 1|}{||\mathbf{\theta}||}$

Vậy thực hiện rút gọn, ta được:

$d = \frac{|-2|}{||\mathbf{\theta}||}$

Cuối cùng thực hiện trị tuyệt đối, ta được khoảng cách giữa 2 hyperland là:

$d = \frac{2}{||\mathbf{\theta}||}$

Vậy ta đã chứng mính được khoảng cách giữa 2 hyperland là $\frac{2}{||\mathbf{\theta}||}$

**END YOUR PROOF**
"""

# ╔═╡ aaa8faa8-be04-4886-b336-3b0482a56480
md"""
So we want to maximize the distance betweeen these two hyperplanes? Right? Equivalently, we minimize $||\mathbf{\theta}||$. We also have to prevent data points from falling into the margin, we add the following constraint: for each $i$ either

$$\mathbf{\theta}^\top\mathbf{x}_i + b \geq 1 \text{ if } y_i = 1$$ and

$$\mathbf{\theta}^\top\mathbf{x} + b \leq -1 \text{ if } y_i = -1$$

And, we can rewrite this as

$$y_i(\mathbf{\theta}^\top\mathbf{x}_i + b) \geq 1, \forall i \in \{1...n\}$$

**Finally, the optimization problem is**

$$\begin{gather*}
    \underset{\theta, b}{\text{ min }}\frac{1}{2}\left\| \theta\right\|^2 \\
    \text{s.t.}\quad y_i(\mathbf{\theta}^\top \mathbf{x}_i+b) -1 \geq 0, \forall i = 1...n \\
  \end{gather*}$$

The parameters $\theta$ and $b$ that solve this problem determine the classifier

$$\mathbf{x} \rightarrow \text{sign}(\mathbf{\theta}^\top\mathbf{x}_i + b)$$
"""

# ╔═╡ 9ca8ef1c-cb48-474a-846f-cea211437a6e
md"""
!!! todo
 Your task here is implement the hard-margin SVM solving the primal formulation using gradient descent (3 points). You can modify your own code in the area bounded by START YOUR CODE and END YOUR CODE.
"""

# ╔═╡ 8522e951-c8eb-41b9-9e27-38746934547f
"""
	SVM solving the primal formulation using gradient descent (hard-margin)
### Fields
- pos_data::Matrix{Float64}: Input features for postive class (+1)
- neg_data::Matrix{Float64}: Input features for negative class (-1)
- η::Float64=0.03: Learning rate. Default is 0.03
- n_epochs::Int64=10000: Maximum training epochs. Default is 10000
"""
function hardmargin_svm(pos_data, neg_data, η=0.04, n_epochs=10000)
	# START YOUR CODE

	## Create variables for the separating hyperplane w'*x + b = y.
	N = Base.size(pos_data, 2) + Base.size(neg_data, 2)
	W = zeros(Base.size(pos_data, 1) + 1, 1)
	train_x = vcat(pos_data', neg_data')
	train_x = hcat(train_x, ones(N, 1))
	train_y = vcat(ones(Base.size(pos_data, 2), 1), -1 * ones(Base.size(neg_data, 2), 1))
	# Train using gradient descent
	## For each epoch 
	for epoch in 1:n_epochs
		### For each training instance ∈ D
		y_hat = []
		h = train_x * W
		for i in 1:Base.size(h, 1)
			if h[i] >= 1 && h[i] * train_y[i] >= 1
				push!(y_hat, 1)
			elseif h[i] <= -1 && h[i] * train_y[i] >= 1
				push!(y_hat, -1)
			else
				push!(y_hat, h[i])
			end
		end
		error = y_hat - train_y
		grad = 1/N * (train_x' * error)
		## Update weight
		W -= η * grad
	end
	# END YOUR CODE
	## Return hyperplane parameters
	return W[1:end-1], W[end]
end

# ╔═╡ d9429c3a-04aa-48a7-bd48-07ef9289e907
# Uncomment this line below when you finish your implementation
w, b = hardmargin_svm(points1ₜᵣₐᵢₙ, points2ₜᵣₐᵢₙ)

# ╔═╡ 0eacbb90-e3f2-46e6-a248-5657fbaeaaf3
"""
	Visualization function for SVM solving the primal formulation using gradient descent (hard-margin)

### Fields
- w & b: SVM parameters
- pos_data::Matrix{Float64}: Input features for postive class (+1)
- neg_data::Matrix{Float64}: Input features for negative class (-1)
"""
function draw(w, b, pos_data, neg_data)
  	plt = scatter(pos_data[1, :], pos_data[2, :], label="y = 1")
  	scatter!(plt, neg_data[1, :], neg_data[2, :], label="y = -1")

	hyperplane(x)= w' * x + b

	D = ([
	  tuple.(eachcol(pos_data), 1)
	  tuple.(eachcol(neg_data), -1)
	])

	xₘᵢₙ = minimum(map((p) -> p[1][1], D))
  	yₘᵢₙ = minimum(map((p) -> p[1][2], D))
  	xₘₐₓ = maximum(map((p) -> p[1][1], D))
 	yₘₐₓ = maximum(map((p) -> p[1][2], D))

  	contour!(plt, xₘᵢₙ:0.1:xₘₐₓ, yₘᵢₙ:0.1:yₘₐₓ,
			(x, y) -> hyperplane([x, y]),
			levels=[-1],
			linestyles=:dash,
			colorbar_entry=false, color=:red, label = "Negative points")
  	contour!(plt, xₘᵢₙ:0.1:xₘₐₓ, yₘᵢₙ:0.1:yₘₐₓ,
			(x, y) -> hyperplane([x, y]),
			levels=[0], linestyles=:solid, label="SVM prediction", colorbar_entry=false, color=:green)
  	contour!(plt, xₘᵢₙ:0.1:xₘₐₓ, yₘᵢₙ:0.1:yₘₐₓ,
			(x, y) -> hyperplane([x, y]), levels=[1], linestyles=:dash, colorbar_entry=false, color=:blue, label = "Positive points")
end

# ╔═╡ ed1ae566-46bd-4006-a797-106b2f176623
# Uncomment this line below when you finish your implementation
draw(w, b, points1ₜᵣₐᵢₙ, points2ₜᵣₐᵢₙ)

# ╔═╡ 8ea91cb7-e2b2-4b7a-b6b2-7921c489fb98
"""
	Evaluation function for hard-margin & soft-margin SVM to calculate accuracy

### Fields
- θ: PLA paramters
- pos_data::Matrix{Float64}: Input features for postive class (+1)
- neg_data::Matrix{Float64}: Input features for negative class (-1)
"""
function eval_svm(w, b, pos_data, neg_data)
	n = size(pos_data, 2)
	X = hcat(pos_data, neg_data)

	# Actual labels, and predicted labels
	y_test = vcat(ones(n), -ones(n))'
	y_pred = [sign(x) for x ∈ w' * X .+ b]

	# START YOUR CODE
	# TODO: acc, p, r, f1???
	TP, FP, TN, FN = tpfptnfn_cal(y_test, y_pred)
	acc = (TP + TN) / (TP + TN + FP + FN)
	p =  TP / (TP + FP)
	r = TP / (TP + FN)
	f1 = 2 * p * r / (p + r)
	# END YOUR CODE
	
	print(" acc: $acc\n precision: $p\n recall: $r\n f1_score: $f1\n")
	
	return acc, p, r, f1
end

# ╔═╡ 5c210f2b-910f-46c9-a30e-86d20b744adb
# Uncomment this line below when you finish your implementation
eval_svm(w, b, points1ₜₑₛₜ, points2ₜₑₛₜ)

# ╔═╡ f27aadb8-b2cf-45b9-bf99-c2382d4b2213
md"""
### Soft-margin

The limitation of Hard Margin SVM is that it only works for data that can be separated linearly. In reality, however, this would not be the case. In practice, the data will almost certainly contain noise and may not be linearly separable. In this section, we will talk about soft-margin SVM (an relaxation of the optimization problem).

Basically, the trick here is very simple, we add slack variables ςᵢ to the constraint of the optimization problem.

$$y_i(\mathbf{\theta}^\top \mathbf{x}_i+b) \geq 1 - \varsigma_i, \forall i = 1...n$$

The regularized optimization problem become as

$$\begin{gather*}
    \underset{\theta, b, \varsigma}{\text{ min }}\frac{1}{2}\left\| \theta\right\|^2 + \sum_{i=1}^n\varsigma_i\\
    \text{s.t.}\quad y_i(\mathbf{\theta}^\top \mathbf{x}_i+b) \geq 1 - \varsigma_i, \forall i = 1...n \\
  \end{gather*}$$

Furthermore, we ad a regularization parameter $C$ to determine how important $\varsigma$ should be. And, we got it :)

$$\begin{gather*}
    \underset{\theta, b, \varsigma}{\text{ min }}\frac{1}{2}\left\| \theta\right\|^2 + C\sum_{i=1}^n\varsigma_i\\
    \text{s.t.}\quad y_i(\mathbf{\theta}^\top \mathbf{x}_i+b) \geq 1 - \varsigma_i,\varsigma_i \geq 0, \forall i = 1...n \\
  \end{gather*}$$
"""

# ╔═╡ 3fdaee93-9c4f-441a-9b4a-4c037f101955
md"""
!!! todo
 Your task here is implement the soft-margin SVM solving the primal formulation using gradient descent (3 points). You can modify your own code in the area bounded by START YOUR CODE and END YOUR CODE.
"""

# ╔═╡ 665885b7-9dd7-4ef9-8b5b-948295c20851
"""
	SVM solving the primal formulation using gradient descent (soft-margin)
### Fields
- pos_data::Matrix{Float64}: Input features for postive class (+1)
- neg_data::Matrix{Float64}: Input features for negative class (-1)
- C: relaxation variable control slack variables ς
- η::Float64=0.03: Learning rate. Default is 0.03
- n_epochs::Int64=10000: Maximum training epochs. Default is 10000
"""
function softmargin_svm(pos_data, neg_data, n_epochs=10000, C=0.12, η=0.01)
	# START YOUR CODE

	## Create variables for the separating hyperplane w'*x = b.
	N = Base.size(pos_data, 2) + Base.size(neg_data, 2)
	W = zeros(Base.size(pos_data, 1) + 1, 1)
	train_x = vcat(pos_data', neg_data')
	train_x = hcat(train_x, ones(N, 1))
	train_y = vcat(ones(Base.size(pos_data, 2), 1), -1 * ones(Base.size(neg_data, 2), 1))
	for epoch in 1:n_epochs
		### For each training instance ∈ D
		h = train_x * W
		ς = []
		y_hat = []
		for i in 1:Base.size(h, 1)
			if h[i] >= 1 && h[i] * train_y[i] >= 1
				push!(y_hat, 1)
			elseif h[i] <= -1 && h[i] * train_y[i] >= 1
				push!(y_hat, -1)
			else
				push!(y_hat, h[i])
			end
		end
		#### Calculate slack variables ς
		for i in 1:Base.size(y_hat, 1)
			push!(ς, train_y[i] - y_hat[i])
		end
		## Update weight
		error = (y_hat .+ C .* ς) .- train_y
		grad = 1/N * (train_x' * error)
		W -= η * grad
	end
	# END YOUR CODE
	## Return hyperplane parameters
	return W[1:end-1], W[end]
end

# ╔═╡ eb0f6469-a0dd-4a6b-a3c2-6916c58072a9
# Uncomment this line below when you finish your implementation
sw, sb = softmargin_svm(points1ₜᵣₐᵢₙ, points2ₜᵣₐᵢₙ)

# ╔═╡ d531768a-0aef-43ae-867b-f1670211e06f
# Uncomment this line below when you finish your implementation
draw(sw, sb, points1ₜᵣₐᵢₙ, points2ₜᵣₐᵢₙ)

# ╔═╡ f79e78e5-27d6-43be-bb32-4066dba0d373
# Uncomment this line below when you finish your implementation
eval_svm(sw, sb, points1ₜₑₛₜ, points2ₜₑₛₜ)

# ╔═╡ 547bd5c6-a9a8-472e-87fd-e83ac5aaa0d2
md"""
## Computing the SVM classifier (To get beyond 8.5 points)

We should know about some popular kernel types we could use to classify the data such as linear kernel, polynomial kernel, Gaussian, sigmoid and RBF (radial basis function) kernel.
- Linear Kernel: $K(x_i, x_j) = x_i^\top x_j$
- Polynomial kernel: $K(x_i, x_j) = (1 + x_i^\top x_j)^p$
- Gaussian: $K(x_i, x_j) = \text{exp}\left(-\frac{||x_i - x_j||^2}{2\sigma^2}\right)$
- Sigmoid: $K(x_i, x_j) = \text{tanh}(\beta_0x_i^\top x_j + \beta_1)^p$
- RBF kernel: $K(x_i, x_j) = \text{exp}(-\gamma||x_i - x_j||^2)$
"""

# ╔═╡ 4f882e89-589a-4eb4-a908-e5cb2ef8c829
"""
	Function for creating two spirals dataset.

	You can check the MATLAB implement here: 6 functions for generating artificial datasets, https://www.mathworks.com/matlabcentral/fileexchange/41459-6-functions-for-generating-artificial-datasets
### FIELDS
- nₛₐₘₚₗₑₛ: number of samples you want :)
- noise: noise rate for creating process you want :)
"""
function two_spirals(nₛₐₘₚₗₑₛ, noise::Float64=0.2)
  start_angle = π / 2
  total_angle = 3π

  N₁ = floor(Int, nₛₐₘₚₗₑₛ / 2)
  N₂ = nₛₐₘₚₗₑₛ - N₁

  n = start_angle .+ sqrt.(rand(N₁, 1)) .* total_angle
  d₁ = [-cos.(n) .* n + rand(N₁, 1) .* noise sin.(n) .* n + rand(N₁, 1) .* noise]

  n = start_angle .+ sqrt.(rand(N₂, 1)) .* total_angle
  d₂ = [cos.(n) .* n + rand(N₂, 1) * noise -sin.(n) .* n + rand(N₂, 1) .* noise]

  return d₁', d₂'
end

# ╔═╡ 5784e0c3-4baa-4a55-8e00-6fb501fedee8
# create two spirals which are not linearly seperable
sp_points1, sp_points2 = two_spirals(500)

# ╔═╡ 6e77fe50-767b-48e3-827e-2ed9c7b91b9c
scatter!(scatter(sp_points1[1, :], sp_points1[2, :], label="y = 1"), sp_points2[1, :], sp_points2[2, :], label="y = -1")

# ╔═╡ a7d3fe4a-0367-4ef0-9816-801350fc8534
# Kernel function: in this lab, we use RBF kernel function, you want to do more experiment, please try again at home
γ = 1 / 5

# ╔═╡ 1bc5da97-cb97-4c64-9a32-f9697d6e11fe
K(x, y) = exp(-γ * (x - y)' * (x - y))

# ╔═╡ dc0d267f-4a1e-49e9-8e44-d5674771f193
md"""
### SMO algorithm 

For more detail, you should read: Platt, J. (1998). Sequential minimal optimization: A fast algorithm for training support vector machines.

Wikipedia just quite good for describes this algorithm: MO is an iterative algorithm for solving the optimization problem. MO breaks this problem into a series of smallest possible sub-problems, which are then solved analytically. Because of the linear equality constraint involving the Lagrange multipliers $\lambda_i$, the smallest possible problem involves two such multipliers.

The SMO algorithm proceeds as follows:
- Step 1: Find a Lagrange multiplier $\alpha_1$ that violates the Karush–Kuhn–Tucker (KKT) conditions for the optimization problem.
- Step 2: Pick a second multiplier $\alpha_2$ and optimize the pair ($\alpha_1, \alpha_2$)
- Step 3: Repeat steps 1 and 2 until convergence.
"""

# ╔═╡ 18f39850-c867-4866-9389-13658f71b200
md"""
### Dual SVM - Hard-margin

If you want to find minimum of a function $f$ under the equality constraint $g$, we can use Largrangian function

$$f(x)-\lambda g(x)=0$$
where $\lambda$ is Lagrange multiplier.

In terms of SVM optimization problem

$$\begin{gather*}
    \underset{\theta, b}{\text{ min }}\frac{1}{2}\left\| \theta\right\|^2 \\
    \text{s.t.}\quad y_i(\mathbf{\theta}^\top \mathbf{x}_i+b) -1 \geq 0, \forall i = 1...n \\
  \end{gather*}$$

The equality constraint is $$g(\theta, b) = y_i(\mathbf{\theta}^\top \mathbf{x}_i+b) -1,\forall i = 1...n$$

Then the Lagrangian function is

$$\mathcal{L}(\theta, b, \lambda) = \frac{1}{2}\left\| \theta\right\|^2 + \sum_1^n\lambda_i\left(y_i(\mathbf{\theta}^\top \mathbf{x}_i+b)-1)\right)$$

Equivalently, Lagrangian primal problem is formulated as

$$\begin{gather*}
    \underset{\theta, b}{\text{ min }} {\text{ max }} \mathcal{L}(\theta, b, \lambda)\\
    \text{s.t.}\quad \lambda_i \geq 0, \forall i = 1...n \\
  \end{gather*}$$

!!! note
	We need to MINIMIZE the MAXIMIZATION of $\mathcal{L}(\theta, b, \lambda)$? What we are doing???

!!! danger
	More precisely, $\lambda$ here should be KKT (Karush-Kuhn-Tucker) multipliers

	$$\lambda [-y_i\left(\theta^\top\mathbf{x}_i + b\right) + 1] = 0, \forall i = 1...n$$
"""

# ╔═╡ 730ee186-b178-401c-b274-c72493928e80
md"""
With the Lagrangian function

$$\begin{gather*}
    \underset{\theta, b}{\text{ min }} {\text{ max }} \mathcal{L}(\theta, b, \lambda)= \frac{1}{2}\left\| \theta\right\|^2 + \sum_{i=1}^n\lambda_i\left(y_i(\mathbf{\theta}^\top \mathbf{x}_i+b-1)\right)\\
    \text{s.t.}\quad \lambda_i \geq 0, \forall i = 1...n \\
  \end{gather*}$$

Setting derivatives to 0 yield:

$$\begin{align}
\nabla_{\mathbf{\theta}}\mathcal{L}(\theta, b, \lambda) &= \theta - \sum_{i=1}^n\lambda_iy_i\mathbf{x}_i = 0 \Leftrightarrow \mathbf{\theta}^{*} = \sum_{i=1}^n\lambda_iy_i\mathbf{x}_i \\
\nabla_b \mathcal{L}(\theta, b, \lambda) &= -\sum_{i=1}^n\lambda_iy_i = 0
\end{align}$$

We substitute them into the Lagrangian function, and get

$$W(\lambda, b) = \sum_{i=1}^n\lambda_i -\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\lambda_i\lambda_jy_iy_j\mathbf{x}_i\mathbf{x}_j$$

So, dual problem is stated as

$$\begin{gather*}
    \underset{\lambda}{\text{ max }}\sum_1^n\lambda_i -\frac{1}{2}\sum_i^n\sum_j^n\lambda_i\lambda_jy_iy_j\mathbf{x}_i\mathbf{x}_j\\
    \text{s.t.}\quad \lambda_i \geq 0, \forall i = 1...n, \sum_{i=1}^n\lambda_iyi=0 \\
  \end{gather*}$$

To solve this one has to use quadratic optimization or **sequential minimal optimization**
"""

# ╔═╡ e4a0072e-8920-4005-ba2a-a5e12a9d5f6a
function draw_nl(λ, b, pos_data, neg_data)
  	plt = scatter(pos_data[1, :], pos_data[2, :], label="y = 1")
  	scatter!(plt, neg_data[1, :], neg_data[2, :], label="y = -1")

	D = ([
	  tuple.(eachcol(pos_data), 1)
	  tuple.(eachcol(neg_data), -1)
	])

	X = [x for (x, y) in D]
	Y = [y for (x, y) in D]

	k(x, y) = exp(-1 / 5 * (x - y)' * (x - y))

	hyperplane(x)= (λ .* Y) ⋅ k.(X, Ref(x)) + b
	
	xₘᵢₙ = minimum(map((p) -> p[1][1], D))
  	yₘᵢₙ = minimum(map((p) -> p[1][2], D))
  	xₘₐₓ = maximum(map((p) -> p[1][1], D))
 	yₘₐₓ = maximum(map((p) -> p[1][2], D))

  	contour!(plt, xₘᵢₙ:0.1:xₘₐₓ, yₘᵢₙ:0.1:yₘₐₓ,
			(x, y) -> hyperplane([x, y]),
			levels=[-1],
			linestyles=:dash,
			colorbar_entry=false, color=:red, label = "Negative points")
  	contour!(plt, xₘᵢₙ:0.1:xₘₐₓ, yₘᵢₙ:0.1:yₘₐₓ,
			(x, y) -> hyperplane([x, y]),
			levels=[0], linestyles=:solid, label="SVM prediction", colorbar_entry=false, color=:green)
  	contour!(plt, xₘᵢₙ:0.1:xₘₐₓ, yₘᵢₙ:0.1:yₘₐₓ,
			(x, y) -> hyperplane([x, y]), levels=[1], linestyles=:dash, colorbar_entry=false, color=:blue, label = "Positive points")
end

# ╔═╡ bcc10780-3058-46fa-9123-79b0d0861e0d
md"""
!!! todo
 Your task here is implement the hard-margin SVM solving the dual formulation using sequential minimal optimization (2 points). You can modify your own code in the area bounded by START YOUR CODE and END YOUR CODE.
"""

# ╔═╡ 6b7d6bf7-afcf-4dce-8488-b97509ef8e88
function dualsvm_smo_hard(pos_data, neg_data, n_epochs=100, λₜₒₗ=0.0001, errₜₒₗ=0.0001)
	# You do not need implement kernel, please use the K(.) kernel function in previous cell code.
	
	# START YOUR CODE
	# Step 1: Data preparation
	# First you construct and shuffle to obtain dataset D in a stochastically manner
	D = ([
	  tuple.(eachcol(pos_data), 1)
	  tuple.(eachcol(neg_data), -1)
	])
	shuffle!(D)
	
	# For more easily access to data point
	X = [x for (x, y) ∈ D]
	Y = [y for (x, y) ∈ D]

	# Step 2: Initialization
	# Larangian multipliers, and bias
	λ = zeros(length(D))
	b = 0
	n = length(λ)
	
	# Step 3: Training loop
    for epoch in 1:n_epochs
		changed = false
        for i in 1:n
            alph1 = copy(λ[i])
            y1 = copy(Y[i])
           	E1 = ((λ .* Y) ⋅ K.(Ref(X[i]), X)) + b - y1
			
            # Check KKT condition
            if (y1 * E1 < -errₜₒₗ && alph1 < λₜₒₗ) || (y1 * E1 > errₜₒₗ && alph1 > 0)
                # Select second alpha (α2) heuristically
                j = rand(1:n)
                while j == i
                    j = rand(1:n)
                end

                alph2 = copy(λ[j])
                y2 = copy(Y[j])
                E2 = ((λ .* Y) ⋅ K.(Ref(X[j]), X)) + b - y2

                # Compute the bounds L and H
                if y1 != y2
                    L = max(0, alph2 - alph1)
                    H = min(1, 1 + alph2 - alph1)
                else
                    L = max(0, alph1 + alph2 - 1)
                    H = min(1, alph1 + alph2)
                end

                if L == H
                    continue
                end

				K11 = K(X[i], X[i])
				K12 = K(X[i], X[j])
				K22 = K(X[j], X[j])
                eta = 2 * K12 - K11 - K22

                if eta >= 0
                    continue
                end

                # Compute the unclipped value for alpha2
                alph_unc = alph2 - Y[j] * (E1 - E2) / eta

                # Clip the value of alpha2 using L and H
                alph2 = max(L, min(H, alph_unc))

                # Check if alpha2 has changed significantly
                if abs(alph2 - λ[j]) < λₜₒₗ
                    continue
                end

                # Update alpha1 using the new value of alpha2
                alph1 = alph1 + Y[i] * Y[j] * (λ[j] - alph2)

                # Update the bias term
                b1 = b - E1 - Y[i] * (alph1 - λ[i]) * K11 - Y[j] * (alph2 - λ[j]) 					* K12
                b2 = b - E2 - Y[i] * (alph1 - λ[i]) * K12 - Y[j] * (alph2 - λ[j]) 					* K22
                b = (b1 + b2) / 2
				
                # Update the Lagrangian multipliers
               	λ[i] = copy(alph1)
                λ[j] = copy(alph2)

				changed = true
            end
        end
		if changed == false
			break
		end
    end
	# END YOUR CODE
    # Return hyperplane parameters
    return λ, b
end

# ╔═╡ c5028050-48ac-4e07-9a6c-e836537ff7c7
# Uncomment this line below when you finish your implementation
λₕ, bₕ = dualsvm_smo_hard(points1ₜᵣₐᵢₙ, points2ₜᵣₐᵢₙ)

# ╔═╡ 52128a2f-5a4f-4e11-ad2b-e112098b8b82
# Uncomment this line below when you finish your implementation
draw_nl(λₕ, bₕ, points1ₜᵣₐᵢₙ, points2ₜᵣₐᵢₙ)

# ╔═╡ d14d2d72-8c39-462d-b30f-8e1e4765159e
md"""
### Dual SVM - Soft-margin

As we know that, the regularized optimization problem in the case of soft-margin as

$$\begin{gather*}
    \underset{\theta, b, \varsigma}{\text{ min }}\frac{1}{2}\left\| \theta\right\|^2 + C\sum_{i=1}^n\varsigma_i\\
    \text{s.t.}\quad y_i(\mathbf{\theta}^\top \mathbf{x}_i+b) \geq 1 - \varsigma_i,\varsigma_i \geq 0, \forall i = 1...n \\
  \end{gather*}$$

We use Larangian multipliers, and transform to a dual problem as 

$$\begin{gather*}
    \underset{\lambda}{\text{ max }}\sum_1^n\lambda_i -\frac{1}{2}\sum_i^n\sum_j^n\lambda_i\lambda_jy_iy_j\mathbf{x}_i\mathbf{x}_j\\
    \text{s.t.}\quad  0 \leq \lambda_i \leq C, \forall i = 1...n, \sum_{i=1}^n\lambda_iyi=0 \\
  \end{gather*}$$
"""

# ╔═╡ fbc7b96a-67ae-46b3-b746-4ea50a4455ce
md"""
!!! todo
 Your task here is implement the soft-margin SVM solving the dual formulation using sequential minimal optimization (2 points). You can modify your own code in the area bounded by START YOUR CODE and END YOUR CODE.
"""

# ╔═╡ e75a6b8a-9e34-4b1b-9bd2-7641454f12c0
function dualsvm_smo_soft(pos_data, neg_data, n_epochs=100, C=1000, λₜₒₗ=0.0001, errₜₒₗ=0.0001)
	# START YOUR CODE
	# Step 1: Data preparation
	# First you construct and shuffle to obtain dataset D in a stochastically manner
	D = ([
	  tuple.(eachcol(pos_data), 1)
	  tuple.(eachcol(neg_data), -1)
	])
	shuffle!(D)
	
	# For more easily access to data point
	X = [x for (x, y) ∈ D]
	Y = [y for (x, y) ∈ D]
	
	# Step 2: Initialization
	# Larangian multipliers, and bias
	λ = zeros(length(D))
	b = 0
	n = length(λ)
	
	# Step 3: Training loop
    for epoch in 1:n_epochs
		changed = false
        for i in 1:n
            alph1 = copy(λ[i])
            y1 = copy(Y[i])
           	E1 = sum(λ .* Y .* dot.(Ref(X[i]), X)) + b - y1
			
            # Check KKT condition
            if (y1 * E1 < -errₜₒₗ && alph1 < C) || (y1 * E1 > errₜₒₗ && alph1 > 0)
                # Select second alpha (α2) heuristically
                j = rand(1:n)
                while j == i
                    j = rand(1:n)
                end

                alph2 = copy(λ[j])
                y2 = copy(Y[j])
                E2 = sum(λ .* Y .* dot.(Ref(X[j]), X)) + b - y2

                # Compute the bounds L and H
                if y1 != y2
                    L = max(0, alph2 - alph1)
                    H = min(C, C + alph2 - alph1)
                else
                    L = max(0, alph1 + alph2 - C)
                    H = min(C, alph1 + alph2)
                end

                if L == H
                    continue
                end

				K11 = dot(X[i], X[i])
				K12 = dot(X[i], X[j])
				K22 = dot(X[j], X[j])
                eta = 2 * K12 - K11 - K22

                if eta >= 0
                    continue
                end

                # Compute the unclipped value for alpha2
                alph_unc = alph2 - Y[j] * (E1 - E2) / eta

                # Clip the value of alpha2 using L and H
                alph2 = max(L, min(H, alph_unc))

                # Check if alpha2 has changed significantly
                if abs(alph2 - λ[j]) < λₜₒₗ
                    continue
                end

                # Update alpha1 using the new value of alpha2
                alph1 = alph1 + Y[i] * Y[j] * (λ[j] - alph2)

                # Update the bias term
                b1 = b - E1 - Y[i] * (alph1 - λ[i]) * K11 - Y[j] * (alph2 - λ[j]) 					* K12
                b2 = b - E2 - Y[i] * (alph1 - λ[i]) * K12 - Y[j] * (alph2 - λ[j]) 					* K22
                b = (b1 + b2) / 2
				
                # Update the Lagrangian multipliers
               	λ[i] = alph1
                λ[j] = alph2

				changed = true
            end
        end
		if changed == false
			break
		end
    end
	# END YOUR CODE
    # Return hyperplane parameters
    return λ, b
end

# ╔═╡ 2d29d23f-7463-4d88-8318-fdcb78bacd3f
# Uncomment this line below when you finish your implementation
λₛ, bₛ = dualsvm_smo_soft(sp_points1, sp_points2)

# ╔═╡ 438aea80-21a7-4e56-aaa3-6f8b4dabc976
# Uncomment this line below when you finish your implementation
draw_nl(λₛ, bₛ, sp_points1, sp_points2)

# ╔═╡ 25054281-405d-458f-ab3a-e05f1f956bec
md"""
## Multi-classes classification problem with SVMs (To get beyond 10.0 points)
"""

# ╔═╡ ed31489c-3feb-483d-9787-87df73e116d0
md"""
### Load MNIST dataset
"""

# ╔═╡ 513a10db-cc97-4a6c-b7b3-eee6b6c283f4
begin
	data_dir = joinpath(dirname(@__FILE__), "data")
	train_x_dir = joinpath(data_dir, "train/images/train-images.idx3-ubyte")
	train_y_dir = joinpath(data_dir, "train/labels/train-labels.idx1-ubyte")
	
	test_x_dir = joinpath(data_dir, "test/images/t10k-images.idx3-ubyte")
	test_y_dir = joinpath(data_dir, "test/labels/t10k-labels.idx1-ubyte")
	
	NUMBER_TRAIN_SAMPLES = 60000
	NUMBER_TEST_SAMPLES = 10000
end

# ╔═╡ 8d004f4b-5523-4414-9ca9-a5509d541236
begin
	train_x = Array{Float64}(undef, 28^2, NUMBER_TRAIN_SAMPLES)
	train_y = Array{Int64}(undef, NUMBER_TRAIN_SAMPLES)

	io_images = open(train_x_dir)
	io_labels = open(train_y_dir)

	for i ∈ 1:NUMBER_TRAIN_SAMPLES
		seek(io_images, (i-1)*28^2 + 16) # offset 16 to skip header
		seek(io_labels, (i-1)*1 + 8) # offset 8 to skip header
		train_x[:,i] = convert(Array{Float64}, read(io_images, 28^2))
		train_y[i] = convert(Int, read(io_labels, UInt8))
	end
	close(io_images)
	close(io_labels)

	train_x = train_x
end

# ╔═╡ bead671f-1f61-44ed-ba4c-0b4156757faa
begin
	test_x = Array{Float64}(undef, 28^2, NUMBER_TEST_SAMPLES)
	test_y = Array{Int64}(undef, NUMBER_TEST_SAMPLES)

	io_images_test = open(test_x_dir)
	io_labels_test = open(test_y_dir)

	for i ∈ 1:NUMBER_TEST_SAMPLES
		seek(io_images_test, (i-1)*28^2 + 16) # offset 16 to skip header
		seek(io_labels_test, (i-1)*1 + 8) # offset 8 to skip header
		test_x[:,i] = convert(Array{Float64}, read(io_images_test, 28^2))
		test_y[i] = convert(Int, read(io_labels_test, UInt8))
	end
	close(io_images)
	close(io_labels)

	test_x = test_x
end

# ╔═╡ b949cfb8-c649-46d5-8d9a-47a0a153fe3a
size(train_x), size(train_y), size(test_x), size(test_y)

# ╔═╡ d52bb268-787a-4590-ba7a-699e23a93092
md"""
### Training SVMs
"""

# ╔═╡ bca342c0-135c-45bf-87c1-fd86e3567ae7
begin 
	model = svmtrain(train_x, train_y; svmtype=SVC, kernel=Kernel.Linear)
end
# END YOUR CODE

# ╔═╡ d005dd6f-b5a2-4e59-a453-81d51be5fc8b
md"""
###  Evaluation
"""

# ╔═╡ 954e9da5-021d-4fa9-8b5b-7504d2a31367
begin
	# START YOUR CODE
	# Test model on the other half of the data.
	y_predict, _ = svmpredict(model, test_x)

	# True positives, false positives, false negatives
	acc = sum((test_y .== y_predict))
	
	@printf("Accuracy: %.2f%%\n", acc/size(test_y, 1) * 100)
	# END YOUR CODE
end

# ╔═╡ 6771c4f1-cf02-4a72-8ffc-b78b00514428
md"""
This is the end of Lab 05. However, there still a lot of things that you can learn about SVM. There are many open tasks to do in your sparse time such as how to deal with multi-class, or Bayesian SVM. :) Hope all you will enjoy SVM. Good luck!
"""

# ╔═╡ 488098f8-1881-459f-aaef-df1a59058b73
md"""
## References

[1] Boyd, S. P., & Vandenberghe, L. (2004). Convex optimization. Cambridge university press.

[2] Griva, I., Nash, S. G., & Sofer, A. (2008). Linear and Nonlinear Optimization 2nd Edition. Society for Industrial and Applied Mathematics.

[3] Schölkopf, B., & Smola, A. J. (2002). Learning with kernels: support vector machines, regularization, optimization, and beyond. MIT press.

[4] Lab 3, Logistic Regresion, Introduction to Machine Learning course, Department of Computer Science, Faculty of Information Technology, Ho Chi Minh University of Science, Vietnam National University.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LIBSVM = "b1bec4e5-fd48-53fe-b0cb-9723c09d164b"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
ScikitLearn = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"

[compat]
Distributions = "~0.25.104"
LIBSVM = "~0.8.0"
Plots = "~1.39.0"
PlutoUI = "~0.7.54"
ScikitLearn = "~0.7.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "793501dcd3fa7ce8d375a2c878dca2296232686e"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.2"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e0af648f0692ec1691b5d094b8724ba1346281cf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.18.0"

[[ChangesOfVariables]]
deps = ["InverseFunctions", "LinearAlgebra", "Test"]
git-tree-sha1 = "2fba81a302a7be671aefe194f0525ef231104e7f"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.8"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "886826d76ea9e72b35fcd000e535588f7b60f21d"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.1"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "8cfa272e8bdedfa88b6aefbbca7c19f1befac519"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.0"

[[Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "51cab8e982c5b598eea9c8ceaced4b58d9dd37c9"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.10.0"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

[[Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "9242eec9b7e2e14f9952e8ea1c7e31a50501d587"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.104"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[FillArrays]]
deps = ["LinearAlgebra", "PDMats", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "5b93957f6dcd33fc343044af3d48c215be2562f1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.9.3"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "abbbb9ec3afd783a7cbd82ef01dcd088ea051398"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.1"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "68772f49f54b479fa88ace904f6127f0a3bb2e46"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.12"

[[InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a53ebe394b71470c7f97c2e7e170d51df21b17af"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.7"

[[JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60b1194df0a3298f460063de985eae7b01bc011a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.1+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[LIBLINEAR]]
deps = ["Libdl", "SparseArrays", "liblinear_jll"]
git-tree-sha1 = "81e40115c23acca9dfa30944050096b958271e5a"
uuid = "2d691ee1-e668-5016-a719-b2531b85e0f5"
version = "0.6.0"

[[LIBSVM]]
deps = ["LIBLINEAR", "LinearAlgebra", "ScikitLearnBase", "SparseArrays", "libsvm_jll"]
git-tree-sha1 = "a5e607649aeb9ae3bbde19dc629faaa3b3d8955d"
uuid = "b1bec4e5-fd48-53fe-b0cb-9723c09d164b"
version = "0.8.0"

[[LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc6e1927ac521b659af340e0ca45828a3ffc748f"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.12+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a935806434c9d4c506ba941871b327b96d41f2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.0"

[[Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "bd7c69c7f7173097e7b5e1be07cee2b8b7447f51"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.54"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "88b895d13d53b5577fd53379d913b9ab9ac82660"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "9816a3826b0ebf49ab4926e2b18842ad8b5c8f04"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.96.4"

[[Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9ebcd48c498668c7fa0e97a9cae873fbee7bfee1"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[ScikitLearn]]
deps = ["Compat", "Conda", "DataFrames", "Distributed", "IterTools", "LinearAlgebra", "MacroTools", "Parameters", "Printf", "PyCall", "Random", "ScikitLearnBase", "SparseArrays", "StatsBase", "VersionParsing"]
git-tree-sha1 = "3df098033358431591827bb86cada0bed744105a"
uuid = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
version = "0.7.0"

[[ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "0e7508ff27ba32f26cd459474ca2ede1bc10991f"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.1"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

[[StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "1fbeaaca45801b4ba17c251dd8603ef24801dd84"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.2"

[[Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Unitful]]
deps = ["ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "Random"]
git-tree-sha1 = "3c793be6df9dd77a0cf49d80984ef9ff996948fa"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.19.0"

[[UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "801cbe47eae69adc50f36c3caec4758d2650741b"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.2+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522b8414d40c4cbbab8dee346ac3a09f9768f25d"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.5+0"

[[Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[liblinear_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7f5f1953394b74739eaebd345f4515515a022a5b"
uuid = "275f1f90-abd2-5ca1-9ad8-abd4e3d66eb7"
version = "2.47.0+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "93284c28274d9e75218a416c65ec49d0e0fcdf3d"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.40+0"

[[libsvm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl", "Pkg"]
git-tree-sha1 = "7625dde5e9eab416c1cb791627f065ce55297eff"
uuid = "08558c22-525a-5d2a-acf6-0ac6658ffce4"
version = "3.25.0+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─287aba39-66d9-4ff6-9605-1bca094a1ce5
# ╟─eeae9e2c-aaf8-11ed-32f5-934bc393b3e6
# ╟─86f24f32-d8ee-49f2-b71a-bd1c5cd7a28f
# ╟─cab7c88c-fe3f-40c4-beea-eea924d67975
# ╠═08c5b8cf-be45-4d11-bd24-204b364a8278
# ╠═99329d11-e709-48f0-96b5-32ae0cac1f50
# ╟─bbccfa2d-f5b6-49c7-b11e-53e419808c1b
# ╟─4321f49b-1057-46bc-8d67-be3122be7a68
# ╟─4fdaeeda-beee-41e1-a5f0-3209151a880d
# ╟─ec906d94-8aed-4df1-932c-fa2263e6325d
# ╟─42882ca3-8df6-4fb8-8884-29337373bac5
# ╠═e78094ff-6565-4e9d-812e-3a36f78731ed
# ╠═d0ecb21c-6189-4b10-9162-1b94424f49ce
# ╠═921e8d15-e751-4976-bb80-2cc09e6c950e
# ╠═4048a66b-a89f-4e37-a89f-6fe57519d5d7
# ╠═17663f65-1aa1-44c4-8eae-f4bc6e24fe98
# ╟─16390a59-9ef0-4b05-8412-7eef4dfb13ee
# ╠═43dee3c9-88f7-4c79-b4a3-6ab2cc3bba2e
# ╠═2d7dde2b-59fc-47c0-a2d0-79dcd48d8041
# ╠═8e06040d-512e-4ff6-a035-f121e9d73eb4
# ╠═f40fbc75-2879-4bf8-a2ba-7b9356149dcd
# ╠═da9c313e-4623-4370-9d5f-0560d62deb51
# ╠═c0d56e33-6dcf-4675-a679-a55e7baaeea1
# ╠═a9d60d10-2d93-4c3e-8720-5534efd646a4
# ╟─d4709ae3-9de5-4d46-9d95-e15fcf741bc6
# ╟─bd418098-edfb-4989-8bd5-23bca5059c51
# ╟─8e2c8a02-471e-4321-8d8b-d25b224aa0c1
# ╟─fb06ed9a-2b6a-422f-b709-1c2f782da49e
# ╟─721bc350-c561-4985-b212-17cfd8d11f5a
# ╟─6b4452a1-1cfd-43da-8177-2aee1259bf71
# ╟─e2bde012-e641-4ee6-aaf7-fee91e0626c2
# ╟─cd1160d3-4603-4d18-b107-e68355fc0604
# ╟─eb804ff4-806b-4a11-af51-d4c3730c84b0
# ╟─4cd4dbad-7583-4dbd-806e-b6279aafc191
# ╟─91e528df-20e4-40b1-8ec0-96b05f59f556
# ╟─e8105abb-6d8b-45ee-aebf-9ccc66b72b23
# ╟─aaa8faa8-be04-4886-b336-3b0482a56480
# ╟─9ca8ef1c-cb48-474a-846f-cea211437a6e
# ╠═8522e951-c8eb-41b9-9e27-38746934547f
# ╠═d9429c3a-04aa-48a7-bd48-07ef9289e907
# ╠═0eacbb90-e3f2-46e6-a248-5657fbaeaaf3
# ╠═ed1ae566-46bd-4006-a797-106b2f176623
# ╠═8ea91cb7-e2b2-4b7a-b6b2-7921c489fb98
# ╠═5c210f2b-910f-46c9-a30e-86d20b744adb
# ╟─f27aadb8-b2cf-45b9-bf99-c2382d4b2213
# ╟─3fdaee93-9c4f-441a-9b4a-4c037f101955
# ╠═665885b7-9dd7-4ef9-8b5b-948295c20851
# ╠═eb0f6469-a0dd-4a6b-a3c2-6916c58072a9
# ╠═d531768a-0aef-43ae-867b-f1670211e06f
# ╠═f79e78e5-27d6-43be-bb32-4066dba0d373
# ╟─547bd5c6-a9a8-472e-87fd-e83ac5aaa0d2
# ╠═4f882e89-589a-4eb4-a908-e5cb2ef8c829
# ╠═5784e0c3-4baa-4a55-8e00-6fb501fedee8
# ╠═6e77fe50-767b-48e3-827e-2ed9c7b91b9c
# ╠═a7d3fe4a-0367-4ef0-9816-801350fc8534
# ╠═1bc5da97-cb97-4c64-9a32-f9697d6e11fe
# ╟─dc0d267f-4a1e-49e9-8e44-d5674771f193
# ╟─18f39850-c867-4866-9389-13658f71b200
# ╟─730ee186-b178-401c-b274-c72493928e80
# ╠═e4a0072e-8920-4005-ba2a-a5e12a9d5f6a
# ╟─bcc10780-3058-46fa-9123-79b0d0861e0d
# ╠═6b7d6bf7-afcf-4dce-8488-b97509ef8e88
# ╠═c5028050-48ac-4e07-9a6c-e836537ff7c7
# ╠═52128a2f-5a4f-4e11-ad2b-e112098b8b82
# ╟─d14d2d72-8c39-462d-b30f-8e1e4765159e
# ╟─fbc7b96a-67ae-46b3-b746-4ea50a4455ce
# ╠═e75a6b8a-9e34-4b1b-9bd2-7641454f12c0
# ╠═2d29d23f-7463-4d88-8318-fdcb78bacd3f
# ╠═438aea80-21a7-4e56-aaa3-6f8b4dabc976
# ╠═25054281-405d-458f-ab3a-e05f1f956bec
# ╠═ed31489c-3feb-483d-9787-87df73e116d0
# ╠═513a10db-cc97-4a6c-b7b3-eee6b6c283f4
# ╠═8d004f4b-5523-4414-9ca9-a5509d541236
# ╠═bead671f-1f61-44ed-ba4c-0b4156757faa
# ╠═b949cfb8-c649-46d5-8d9a-47a0a153fe3a
# ╟─d52bb268-787a-4590-ba7a-699e23a93092
# ╠═e48c1538-f9b2-4b7d-b855-1e3cab86567b
# ╠═f1b2170a-9595-4a91-9035-16d7b7f9aeb2
# ╠═dc09fd89-b904-45f6-a5b4-66c148922f8d
# ╠═bca342c0-135c-45bf-87c1-fd86e3567ae7
# ╟─d005dd6f-b5a2-4e59-a453-81d51be5fc8b
# ╠═954e9da5-021d-4fa9-8b5b-7504d2a31367
# ╟─6771c4f1-cf02-4a72-8ffc-b78b00514428
# ╟─488098f8-1881-459f-aaef-df1a59058b73
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
