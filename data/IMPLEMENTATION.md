\section*{IMPLEMENTATION}

There is a rich and mature software ecosystem available for Gaussian process modeling and Bayesian optimization, and it is relatively easy to build sophisticated optimization routines using off-the-shelf libraries. However, successful implementation of the underlying algorithms requires attending to some nitty-gritty details to ensure optimal performance, and what may appear to be simple equations on the page can be challenging to realize in a limited-precision environment. In this chapter we will provide a brief overview of the computational details that practitioners should be aware of when designing Bayesian optimization algorithms, even when availing themselves of existing software libraries.

As is typical with nonparametric models, the computational cost of Gaussian process inference grows (considerably!) with the number of observations, and it is important to understand the nature of this growth and be aware of methods for scaling to large-scale data when necessary.

The primary computational bottleneck in Gaussian process inference is solving systems of linear equations that scale with the number of observed values. Consider the general case of exact inference where we condition a Gaussian process $\mathcal{G P}(f ; \mu, K)$ on the observation of a length$n$ vector of values y with marginal distribution and cross-covariance function (2.6):

$$
p(\mathbf{y})=\mathcal{N}(\mathbf{y} ; \mathbf{m}, \mathbf{C}) ; \quad \kappa(x)=\operatorname{cov}[\mathbf{y}, \phi \mid x] .
$$

The posterior is a Gaussian process with moments (2.10):

$$
\begin{aligned}
\mu_{\mathcal{D}}(x) & =\mu(x)+\kappa(x)^{\top} \mathbf{C}^{-1}(\mathbf{y}-\mathbf{m}) ; \\
K_{\mathcal{D}}\left(x, x^{\prime}\right) & =K\left(x, x^{\prime}\right)-\kappa(x)^{\top} \mathbf{C}^{-1} \kappa\left(x^{\prime}\right) .
\end{aligned}
$$

Evaluating either the posterior mean or the posterior covariance requires solving a linear system with respect to the observation covariance $\mathbf{C}$. Inference with non-Gaussian observations using Monte Carlo sampling or Gaussian approximate inference also entails solving linear systems with respect to this matrix $(2.35,2.39-2.41)$.

\section*{Direct computation via Cholesky decomposition}

NEAL outlined a straightforward implementation based on direct numerical methods that suffices for the small-to-moderate datasets typical in Bayesian optimization. ${ }^{1}$ We take advantage of the fact that $\mathrm{C}$ is symmetric and positive definite and precompute and store its Cholesky factorization: ${ }^{2}$

$$
\mathbf{L}=\operatorname{chol} \mathbf{C} ; \quad \mathbf{L L}^{\top}=\mathbf{C} \text {. }
$$

This computation requires one-time $\mathcal{O}\left(n^{3}\right)$ work and represents the bulk of effort spent in this implementation. Note that despite having the same solving linear systems with respect to the observation covariance $\mathrm{C}$

number of observed values, $n=|\mathrm{y}|$
1 R. M. NEAL (1998). Regression and Classification Using Gaussian Process Priors. In: Bayesian Statistics 6

2 G. H. GOLUB and C. F. VAN LOAN (2013). Matrix Computations. Johns Hopkins University Press. [\$ 4.2] 3 The $2 \sum_{i} \log L_{i i}$ term is the $\log$ determinant of $\mathrm{C}$, where we have exploited the Cholesky factorization and the fact that $\mathrm{L}$ is triangular.

4 That is, a Gaussian process whose hyperparameters are fixed, rather than dependent on the data. asymptotic running time as the general-purpose LU decomposition, the Cholesky factorization exploits symmetry to run twice as fast.

With the Cholesky factor in hand, we may solve an arbitrary linear system $\mathbf{C x}=\mathbf{b}$ in time $\mathcal{O}\left(n^{2}\right)$ via forward-backward substitution by rewriting as $\mathbf{L L}^{\top} \mathbf{x}=\mathbf{b}$ and twice applying a triangular solver. We exploit this fact to additionally precompute the vector

$$
\alpha=\mathrm{C}^{-1}(\mathrm{y}-\mathbf{m})
$$

appearing in the posterior mean. After this initial precomputation of $\mathbf{L}$ and $\boldsymbol{\alpha}$, we may compute the posterior mean

$$
\mu_{\mathcal{D}}(x)=\mu(x)+\kappa(x)^{\top} \boldsymbol{\alpha}
$$

on demand in linear time and the posterior covariance

$$
K_{\mathcal{D}}\left(x, x^{\prime}\right)=K\left(x, x^{\prime}\right)-\left[\mathbf{L}^{-1} \kappa(x)\right]^{\top}\left[\mathbf{L}^{-1} \kappa\left(x^{\prime}\right)\right]
$$

in quadratic time. We may also efficiently compute the log marginal likelihood (4.8) of the data in linear time: ${ }^{3}$

$$
\log p(\mathbf{y} \mid \mathbf{x})=-\frac{1}{2}\left[(\mathbf{y}-\mathbf{m})^{\top} \boldsymbol{\alpha}+2 \sum_{i} \log L_{i i}+n \log 2 \pi\right] .
$$

\section*{Low-rank updates to the Cholesky factorization for sequential inference}

Optimization is an inherently sequential procedure, where our dataset grows incrementally as we gather new observations. In this setting, we can accelerate sequential inference with a fixed Gaussian process ${ }^{4}$ by replacing direct computation of the Cholesky decomposition in favor of fast incremental updates to previously computed Cholesky factors.

For the sake of argument, suppose we have computed the Cholesky factor $\mathbf{L}=\operatorname{chol} \mathbf{C}$ for a set of $n$ observations $\mathbf{y}$ with covariance $\mathbf{C}$. Suppose we then receive $k$ additional observations $\boldsymbol{v}$, resulting in the augmented observation vector

$$
\mathrm{y}^{\prime}=\left[\begin{array}{l}
\mathrm{y} \\
v
\end{array}\right] .
$$

The covariance matrix of $\mathbf{y}^{\prime}$ is formed by appending the previous covariance $\mathrm{C}$ with new rows/columns:

$$
\operatorname{cov}\left[\mathbf{y}^{\prime}\right]=\mathbf{C}^{\prime}=\left[\begin{array}{ll}
\mathbf{C} & \mathbf{X}_{1}^{\top} \\
\mathbf{X}_{1} & \mathbf{X}_{2}
\end{array}\right]
$$

The Cholesky factor of the updated covariance matrix has the form

$$
\operatorname{cholC^{\prime }}=\left[\begin{array}{ll}
\mathrm{L} & 0 \\
\Lambda_{1} & \Lambda_{2}
\end{array}\right]
$$

Note that the upper-left block is simply the previously computed Cholesky factor, which we can reuse. The new blocks may be computed as

$$
\Lambda_{1}=\mathrm{X}_{1} \mathrm{~L}^{-\top} ; \quad \Lambda_{2}=\operatorname{chol}\left[\mathrm{X}_{2}-\Lambda_{1} \Lambda_{1}^{\top}\right]
$$

The first of these blocks can be computed efficiently using a triangular solver with the previous Cholesky factor, and the second block only requires factoring a $(k \times k)$ matrix.

This low-rank update requires $\mathcal{O}\left(k n^{2}\right)$ work to compute, compared to the $\mathcal{O}\left(n^{3}+k n^{2}\right)$ work required to compute the Cholesky decomposition of $\mathrm{C}^{\prime}$ from scratch; asymptotically, the low-rank update is $\mathcal{O}(n / k)$ times faster. In particular, if we begin with an empty dataset and sequentially apply this update for a total of $n$ observations (in any order and with any number of observations at a time) the total cost of inference would be $\mathcal{O}\left(n^{3}\right)$. This is equivalent to the cost of one-time inference with the full dataset, so the update scheme is as efficient as one could hope for.

\section*{Ill-conditioned covariance matrices}

When an optimization policy elects to make an observation in the pursuit of "exploitation," the value observed is (by design!) highly correlated with at least one existing observation. Although these decisions may be well grounded, the resulting highly correlated observations can wreak havoc on numerical linear algebra routines.

To sketch the problems that may arise, consider the following scenario. Suppose the covariance function is stationary, that observations are corrupted by additive Gaussian noise with scale $\sigma_{n}^{2}$, and that the signal-to-noise ratio is large. Now, if two locations in the dataset correspond to highly correlated values, the corresponding rows/columns in the observation covariance matrix $\mathbf{C}$ will be nearly equal, and $\mathbf{C}$ will thus be nearly singular - one-or-more eigenvalues will be near zero, and in extreme cases, some may even become (numerically) negative. This poor conditioning $^{5}$ can cause loss of precision when solving a linear system via the Cholesky decomposition, and a negative eigenvalue will cause the Cholesky routine to fail altogether.

When necessary, we may sidestep these issues with a number of "tricks." One simple solution is to add a small (in terms of $\sigma_{n}^{2}$ ) multiple of the identity to the observation covariance, replacing $\mathbf{C}$ by $\mathbf{C}+\varepsilon \mathbf{I}$. Numerically, this shifts the singular values of $\mathrm{C}$ by $\varepsilon$, improving conditioning; practically, this caps the signal-to-noise ratio by increasing the noise floor. When high correlation is a result of small spatial separation, OSBORNE et al. suggested replacing problematic observations of the objective function with (noisy) observations of directional derivatives, ${ }^{6}$ which are only weakly correlated with the corresponding function values. ${ }^{7}$ Another option is to appeal to iterative numerical methods, discussed below, which actually benefit from having numerous eigenvalues clustered around zero.

\section*{Iterative numerical methods}

Direct methods can handle datasets of perhaps a few tens of thousands of observations before the cubic scaling becomes too much to bear. In most settings where Bayesian optimization would be considered, the amortized analysis

5 Numerical conditioning of $\mathrm{C}$ is usually evaluated by its condition number, defined in terms of its singular values $\sigma=\left\{\sigma_{i}\right\}$ :

$$
\kappa=\frac{\max \sigma}{\min \sigma} .
$$

Given infinite precision, the singular values are equal to the (nonnegative) eigenvalues of $\mathrm{C}$, but very poor conditioning can cause negative (numerical) eigenvalues.

6 That is, we replace observations of $f(x)$ and $f\left(x^{\prime}\right)$ with one of $f(x)$ and one of the directional derivative in the direction of $x-x^{\prime}$, evaluated at the midpoint $\left(x+x^{\prime}\right) / 2$.

7 M. A. Osborne et al. (2009). Gaussian Processes for Global Optimization. LION 3. 8 M. R. HESTENES and E. STIEFEL (1952). Methods of Conjugate Gradients for Solving Linear Systems. Fournal of Research of the National Bureau of Standards 49(6):409-436.

9 If $\mathbf{C x}=\mathbf{b}$, then $|\mathbf{C x}-\mathbf{b}|=0$. As this norm is nonnegative and only vanishes at $\mathbf{x}$, squaring gives

$$
\mathbf{x}=\underset{\mathbf{y}}{\arg \min } \mathbf{y}^{\top} \mathbf{C} \mathbf{y}-2 \mathbf{b}^{\top} \mathbf{y} .
$$

Thus $\mathbf{x}$ is also the solution of an unconstrained quadratic optimization problem, which is convex as $\mathrm{C}$ is symmetric positive definite.

10 G. H. GOLUB and c. F. VAN LOAN (2013). Matrix Computations. Johns Hopkins University Press. [§ 11.5]

11 K. CUtajar et al. (2016). Preconditioning Kernel Matrices. ICML 2016.

12 M. N. GIBbS (1997). Bayesian Gaussian Processes for Regression and Classification. Ph.D. thesis. University of Cambridge.

13 J. R. GARDNER et al. (2018). GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration. NeurIPS 2018.

Gaussian approximate inference: § 2.8, p. 39 cost of observation will preclude obtaining a dataset anywhere near this size, in which case we need not consider the issue further. However, when necessary, we may appeal to more complex approximate inference schemes to scale to larger datasets.

One line of work in this direction is to solve the linear systems arising in the posterior with iterative rather than direct numerical methods. The method of conjugate gradients is especially well suited as it is designed for symmetric positive-definite systems such as appear in the GP posterior. ${ }^{8}$ The main idea behind the conjugate gradient method is to reinterpret the solution of the linear system $\mathbf{C x}=\mathbf{b}$ as the solution of a related and particularly well-behaved convex optimization problem. ${ }^{9}$ With this insight, we may derive a simple procedure to construct a sequence of vectors $\left\{\mathbf{x}_{i}\right\}$ guaranteed to converge in finite time to the desired solution. For a system of size $n$, each iteration of this procedure requires only $\mathcal{O}\left(n^{2}\right)$ work; the most expensive operation is a single matrix-vector multiplication with $\mathrm{C}$.

The method of conjugate gradients is guaranteed to converge (up to round-off error) after $n$ iterations, but this does not offer any speedup over direct methods. However, when $\mathrm{C}$ has a well-behaved spectrum - that is, it is well conditioned and/or has clustered eigenvalues - the sequence converges rapidly. In many cases we may terminate after only $k \ll n$ iterations with an accurate estimate of the solution, for an effectively quadratic running time of $\mathcal{O}\left(k n^{2}\right)$. Although many covariance matrices arising in practice are not necessarily well conditioned, we may use a technique known as preconditioning to transform a poorly conditioned matrix to speed up convergence, with only minor overhead. ${ }^{10,11}$

The use of conjugate gradients for GP inference can be traced back to the doctoral work of GIBBs. ${ }^{12}$ Numerous authors have provided enhancements in the intervening years, and there is a now a substantial body of related work. A good starting point is the work of GARDNER et al., who provide a review of the literature and the key ideas from numerical linear algebra required for large-scale GP inference. ${ }^{13}$ The authors refine these tools to exploit modern massively parallel hardware and build an accompanying software package scaling inference to hundreds of thousands of observations.

\section*{Sparse approximations}

An alternative approach for scaling to large datasets is sparse approximation. Here rather than approximating the linear algebra arising in the exact posterior, we approximate the posterior distribution itself with a Gaussian process admitting tractable computation with direct numerical methods. A large family of sparse approximations have been proposed, which differ in their details but share the same general approach.

As we have seen, specifying an arbitrary Gaussian distribution for a set of values jointly Gaussian distributed with a function of interest induces a GP posterior consistent with that belief (2.39). This is a powerful tool, used in approximate inference to optimize the fit of the induced Gaussian process to a true, intractable posterior. Sparse approximation methods make use of this property as well, but to achieve computational rather than mathematical tractability. The idea is to craft a Gaussian belief for a sufficiently small set of values such that the induced posterior is a faithful, but tractable approximation to the true posterior.

For this discussion it is important that we explicitly account for any observation noise that may be present in the values we wish to condition on, as only independent noise - that is, with diagonal error covariance - is suitable for sparse approximation. Consider conditioning a Gaussian process on a vector of $n$ values z, for large $n$. We will assume the observation model $\mathrm{z}=\mathrm{y}+\varepsilon$, where $\mathrm{y}$ is a vector of jointly Gaussian distributed values as in (9.1), and $\varepsilon$ is a vector of independent, zero-mean Gaussian measurement noise with diagonal covariance matrix N.

The first step of sparse approximation is to identify a set of $m \ll n$ values $\boldsymbol{v}$, called inducing values, whose distribution can in some sense capture most of the information in the full dataset. We will discuss the selection of inducing values shortly; for the moment we assume an arbitrary set has been chosen. The joint prior distribution of the observed and inducing values is Gaussian:

$$
p(\boldsymbol{v}, \mathbf{z})=\mathcal{N}\left(\left[\begin{array}{l}
\boldsymbol{v} \\
\mathbf{z}
\end{array}\right] ;\left[\begin{array}{l}
\boldsymbol{\mu} \\
\mathbf{m}
\end{array}\right],\left[\begin{array}{cc}
\Sigma & \mathbf{K}^{\top} \\
\mathbf{K} & \mathrm{C}+\mathrm{N}
\end{array}\right]\right),
$$

and we will write the cross-covariance function for $v$ as

$$
k(x)=\operatorname{cov}[v, \phi \mid x]
$$

Conditioning (9.3) on $\mathbf{z}$ would yield the true Gaussian posterior on the inducing values, but computing this posterior would be intractable. In a sparse approximation, we instead prescribe a computationally tractable posterior for $\boldsymbol{v}$ informed by the available observations:

$$
p(\boldsymbol{v} \mid \mathbf{z}) \approx q(\boldsymbol{v} \mid \mathbf{z})=\mathcal{N}(\boldsymbol{v} ; \tilde{\boldsymbol{\mu}}, \tilde{\Sigma}) .
$$

This assumed distribution then induces (hence the moniker inducing values) a Gaussian process posterior on $f$ :

$$
p(f \mid \mathbf{z}) \approx \int p(f \mid \boldsymbol{v}) q(\boldsymbol{v} \mid \mathbf{z}) \mathrm{d} \boldsymbol{v}=\mathcal{G P}\left(f ; \mu_{\mathcal{D}}, K_{\mathcal{D}}\right)
$$

which represents the sparse approximation. After transliteration of notation, the posterior moments take the same form as (2.38), and the cost of computation now scales according to the number of inducing values.

To complete this approximation scheme, we must specify a procedure for identifying a set of inducing values $v$ as well as the approximate posterior $q(\boldsymbol{v} \mid \mathbf{z})$, and it is in these details that the various available methods differ. The inducing values are usually taken to be function values at a set of locations $\xi$ called pseudo- or inducing points, taking $v=f(\xi) .{ }^{14}$ These inducing points are fictitious and do not need to coincide with any actual observations. Once a suitable parameterization observed values, $\mathbf{z} ;|\mathbf{z}|=n$

diagonal noise covariance, $\mathrm{N}$

inducing values, $\boldsymbol{v} ;|\boldsymbol{v}|=m \ll n$
14 This is not strictly necessary. We could consider using other values such as derivatives as inducing values for added flexibility.

pseudopoints, inducing points Figure 9.1: Sparse approximation. Top: the exact posterior belief (about noisy observations rather than the latent function) for a Gaussian process conditioned on 200 observations. Bottom: a sparse approximation (9.5-9.7) using ten inducing values corresponding to the indicated inducing points, designed to minimize the KL divergence between the induced and true posterior distributions (9.4).

15 M. TITSIAS (2009). Variational Learning of Inducing Variables in Sparse Gaussian Processes. AISTATS 2009 .

16 For example, we may appeal to the Woodbury identity and recognize that dealing with the diagonal matrix $\mathrm{N}$ is trivial. This is the reason why we restricted the present discussion to diagonal error covariance. posterior mean, exact

posterior $95 \%$ credible interval, exact

![](https://cdn.mathpix.com/cropped/2023_09_22_dccb9fd3992149f3f6bbg-06.jpg?height=200&width=1017&top_left_y=591&top_left_x=771)

posterior mean, approximate

posterior $95 \%$ credible interval, approximate

![](https://cdn.mathpix.com/cropped/2023_09_22_dccb9fd3992149f3f6bbg-06.jpg?height=260&width=1019&top_left_y=932&top_left_x=770)

of the inducing values is chosen, we usually design them - as well as their inducing distribution - by optimizing a measure of fit between the true posterior distribution and the resulting approximation.

TITSIAs introduced a variational approach that has gained prominence. ${ }^{15}$ The idea is to minimize the Kullback-Leibler divergence between the true and induced posteriors on $\mathrm{y}$ and the inducing values $v$ :

$$
D_{\mathrm{KL}}[q(\mathbf{y}, \boldsymbol{v} \mid \mathbf{z}) \| p(\mathbf{y}, \boldsymbol{v} \mid \mathbf{z})],
$$

where

$$
q(\mathbf{y}, \boldsymbol{v} \mid \mathbf{z})=p(\mathbf{y} \mid \boldsymbol{v}) q(\boldsymbol{v} \mid \mathbf{z})
$$

is the implied joint distribution in the approximate posterior. Once optimal inducing values are determined, the optimal inducing distribution is as well, and examining the resulting approximate posterior gives insight into the typical behavior of sparse approximations. The approximate posterior mean is

$$
\mu_{\mathcal{D}}(x)=\mu(x)+\left[\mathrm{K} \Sigma^{-1} k(x)\right]^{\top}\left(\mathrm{K} \Sigma^{-1} \mathbf{K}^{\top}+\mathbf{N}\right)^{-1}(\mathbf{z}-\mathbf{m}) .
$$

Although this expression nominally entails solving a linear system of size $n$, the low-rank-plus-diagonal structure of the matrix $\mathrm{K} \Sigma^{-1} \mathbf{K}^{\top}+\mathrm{N}$ allows the system to be solved in time $\mathcal{O}\left(n m^{2}\right)$, merely linear in the size of the dataset. ${ }^{16}$ With this favorable cost, sparse approximation can scale Gaussian process inference to millions of observations without issue.

Comparing this expression (9.5) with the true posterior mean:

$$
\mu_{\mathcal{D}}(x)=\mu(x)+\kappa(x)^{\top}(\mathbf{C}+\mathbf{N})^{-1}(\mathbf{z}-\mathbf{m}),
$$

we can identify a key approximation to the covariance structure of the GP prior. Namely, we approximate the covariance function with:

$$
K\left(x, x^{\prime}\right) \approx k(x)^{\top} \Sigma^{-1} k\left(x^{\prime}\right)=\operatorname{cov}[x, v] \operatorname{cov}[v, v]^{-1} \operatorname{cov}\left[v, x^{\prime}\right],
$$

that is, we assume that all covariance between function values is moderated through the inducing values. This is a popular approximation scheme known as the Nyström method. ${ }^{17}$ Importantly, however, we note that the posterior mean does still reflect the information contained in the entire dataset through the true residuals $(\mathbf{z}-\mathbf{m})$ and noise $\mathbf{N}$. The approximate posterior covariance also reflects this approximation:

$$
\begin{aligned}
& K\left(x, x^{\prime}\right)-\left[\mathrm{K} \Sigma^{-1} k(x)\right]^{\top}\left(\mathrm{K} \Sigma^{-1} \mathbf{K}^{\top}+\mathrm{N}\right)^{-1}\left[\mathrm{~K} \Sigma^{-1} k\left(x^{\prime}\right)\right] \\
\approx & K\left(x, x^{\prime}\right)-\quad \kappa(x)^{\top} \quad(\mathrm{C}+\mathrm{N})^{-1} \quad \kappa\left(x^{\prime}\right) .
\end{aligned}
$$

Figure 9.1 illustrates a sparse approximation for a toy example following this approach. Here the inducing values were taken to be the function ${ }^{17}$ values at ten inducing points, and both the inducing points and their distribution were designed to minimize the KL divergence between the true and induced posterior distributions (9.4). The approximation is faithful: the posterior mean is nearly identical to the true mean and the posterior credible intervals only display some minor differences from the truth. Increasing the number of inducing values would naturally improve the approximation.

Sparse approximation for Gaussian processes has a long history, beginning in earnest with investigation into the Nyström approximation (9.6).$^{18,19}$ In addition to the variational approach mentioned above, an approximation known as the fully independent training conditional (FITC) approximation has also received significant attention ${ }^{20}$ and gives rise to a similar approximate posterior. HENSMAN et al. provided a variational sparse approximation for non-Gaussian observation models, allowing for scaling general GP latent models to large datasets. ${ }^{21}$

\section*{OPTIMIZING ACQUISITION FUNCTIONS}

In our discussion on computing optimization policies with Gaussian processes, we considered the pointwise evaluation of common acquisition functions and their gradient with respect to the proposed observation location. However, we realize an optimization policy via the global optimization of an acquisition function:

$$
x \in \underset{x^{\prime} \in \mathcal{X}}{\arg \max } \alpha\left(x^{\prime} ; \mathcal{D}\right) .
$$

Every common Bayesian acquisition function is nonconvex in general, so we must resort to some generic global optimization routine for this inner optimization. Some care is required to guarantee success in this optimization, as the behavior of a typical acquisition function can make it a somewhat unusual objective function. In particular, consider a prototypical Gaussian process model combining:

- a constant mean function (3.1),

- a stationary covariance function decaying to zero as $\left|x-x^{\prime}\right| \rightarrow \infty$, and

- independent, homoskedastic observation noise (2.16).
17 C. K. I. WILLIAMS and M. SEEGER (200o). Using the Nyström Method to Speed up Kernel Machines. NeurIPS 2000 .

19 A. J. SMOLA and B. SchölKopf (2000). Sparse Greedy Matrix Approximation for Machine Learning. ICML 2000.

20 E. SNELSON and Z. GHAHRAMANi (2005). Sparse Gaussian Processes Using Pseudo-inputs. NeUrIPS 2005

21 J. HENSMAN et al. (2015). MCMC for Variationally Sparse Gaussian Processes. NeurIPS 2015.

Chapter 8: Computing Policies with Gaussian Processes, p. 157

stationarity: $§ 3.2$, p. 50 curse of dimensionality: §3.5, p. 61

22 D. R. JONES et al. (1993). Lipschitzian Optimization without the Lipschitz Constant. Fournal of Optimization Theory and Application 79(1): 157-181.

23 N. HANSEN (2016). The CMA Evolution Strategy: A Tutorial. arXiv: 1604.00772 [cs.LG].

24 J. KIM and s. CHOI (2020). On Local Optimizers of Acquisition Functions in Bayesian Optimization. 12458:675-69o.

25 M. BALANDAT et al. (2020). BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. NeurIPS 2020 .

26 D. R. JONES (2001). A Taxonomy of Global Optimization Methods Based on Response Surfaces. fournal of Global Optimization 21(4):345383
For such a model, the prior predictive distribution $p(y \mid x)$ is identical regardless of location. However, the posterior predictive distribution $p(y \mid x, \mathcal{D})$ also degenerates to the prior for locations sufficiently far from observed locations, due to the decay of the covariance function. In these regions, the gradients of the posterior predictive parameters effectively vanish:

$$
\frac{\partial \mu}{\partial x} \approx 0 ; \quad \frac{\partial s}{\partial x} \approx 0 .
$$

As a result, the gradient of the acquisition function (8.6) vanishes as well! This vanishing gradient is especially problematic in high-dimensional spaces, where the acquisition function will be flat on an overwhelming fraction of the domain unless the prior encodes absurdly long-scale correlations, a consequence of the unshakable curse of dimensionality. Thus, the acquisition function will only exhibit interesting behavior in the neighborhood of previous observations - where the posterior predictive distribution is nontrivial - and it is here we should spend most of our effort during optimization.

\section*{Optimization approaches}

There are two common lines of attack for optimizing acquisition functions in Bayesian optimization. One approach is to use an off-the-shelf derivative-free global optimization method such as the "dividing rectangles" (DIRECT) algorithm of JONEs et al. ${ }^{22}$ or a member of the covariance matrix adaptation evolution strategy (CMA-ES) family of algorithms. ${ }^{23}$ Although a popular choice in the literature, we argue that neither is a particularly good choice in situations where the acquisition function may devolve into effective flatness as described above. However, an algorithm of this class may be reasonable in modest dimension where optimization can be somewhat exhaustive.

In general, a better alternative is multistart local optimization, making use of the gradients computed in the previous chapter for rapid convergence. KIM and CHOI provided an extensive comparison of the theoretical and empirical performance of global and local optimization routines for acquisition function optimization, and ultimately recommended multistart local optimization as best practice. ${ }^{24}$ This is also the approach used in (at least some) sophisticated modern software packages for Bayesian optimization. ${ }^{25}$

To ensure success, we must carefully select starting points to ensure that the relevant regions of the domain are searched. Jones (the same JONES of the DIRECT algorithm) recognized the problem of vanishing gradients described above and suggested a simple heuristic for selecting local optimization starting points by enumerating and pruning the midpoints between all pairs of observed points. ${ }^{26} \mathrm{~A}$ more brute-force approach is to measure the acquisition function on an exhaustive covering of the domain - generated for example by a low-discrepancy sequence - then begin local searches from the highest values seen. This can be effective if the initial set of points is dense enough to probe the neigh- borhoods of previous observations; otherwise, it would be prudent to augment with a locally motivated approach such as Jones's.

Multistart local optimization has the advantage of being embarrassingly parallel. Further, both global and multistart local optimization of the acquisition function can be treated as anytime algorithms that constantly improve their proposed observations until we are ready to act. $^{27}$

\section*{Optimization in latent spaces}

A common approach for modeling on high-dimensional domains is to apply some mapping from the domain to some lower-dimensional representation space, then construct a Gaussian process on that space. With such a model, it is tempting to optimize an acquisition function on the latent space rather than on the original domain so as to at least partially sidestep the curse of dimensionality. This can be an effective approach when we can faithfully "decode" from the latent space back into the original domain for evaluation, a process that can require careful thought even when the latent embedding is linear (3.30) due to the nonbijective nature of the map. ${ }^{28}$

Fortunately, modern neural embedding techniques such as (variational) autoencoders provide a decoding mechanism as a natural side effect of their construction. When we are fortunate enough to have sufficient unlabeled data to learn a useful unsupervised representation prior to optimization, we may simply optimize in the latent space and feed each chosen observation location through the decoder. GóMEZ-BOMBARELLI et al. for example applied Bayesian optimization to de novo molecular design. ${ }^{29}$ Their model combined a pretrained variational autoencoder for molecular structures with a Gaussian process on the latent embedding space; the autoencoder was trained on a large precompiled database of known molecules. ${ }^{30} \mathrm{~A}$ welcome side effect of this construction was that, by optimizing the acquisition function over the continuous embedding space, the decoding process had the freedom to generate novel structures not seen in the autoencoder's training data.

We may also pursue this approach even when we begin optimization without any data at all. For example, MORICONI et al. demonstrated success jointly learning a nonlinear low-dimensional representation as well as a corresponding decoding stage throughout optimization on the fly. ${ }^{31}$

\section*{Optimization on combinatorial domains}

Combinatorial domains present a challenge for Bayesian optimization as the optimization policy (9.8) requires combinatorial optimization of the acquisition function. It is difficult to provide concrete advice for such a situation, as the details of the domain may in some cases suggest a natural path forward. One potential solution is outlined above: when the domain is a space of discrete structured objects such as graphs (say,
27 E. BRochu et al. (2010). A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning. arXiv: 1012. 2599 [CS.LG].

modeling functions on high-dimensional domains: $\S 3.5$, p. 61

28 Detailed advice for this setting is provided by:

z. WANG et al. (2016b). Bayesian Optimization in a Billion Dimensions via Random Embeddings. Fournal of Artificial Intelligence Research 55:361-387.

linear embeddings: $§ 3.5$, p. 62

neural embeddings: § 3.5, p. 61, § 8.11, p. 199

de novo molecular design: Appendix D, p. 314

29 R. GÓMEZ-BOMBARELLI et al. (2018). Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules. ACS Central Science 4(2):268-276.

30 J. J. IRWIN et al. (2020). ZINC2O - A Free Ultralarge-Scale Chemical Database for Ligand Discovery. Fournal of Chemical Information and Modeling 6o(12):6065-6073.

31 R. MORICONI et al. (2020). High-Dimensional Bayesian Optimization Using LowDimensional Feature Spaces. Machine Learning 109(9-10):1925-1943. gene and protein design: Appendix D, p. 319

32 B. L. HIE and K. K. YANG (2O21). Adaptive Machine Learning for Protein Engineering. arXiv: 2106.05466 [q-bio.QM].

33 H. B. Moss et al. (2020). Boss: Bayesian Optimization over String Spaces. NeurIPS 2020.

34 As evaluated by the covariance function.

35 R. GARnetT et al. (2010). Bayesian Optimization for Sensor Set Selection. IPSN 2010.

36 R. BAPTISTA and M. POLOCZEK (2018). Bayesian Optimization of Combinatorial Structures. ICML 2018.

37 c. oH et al. (2019). Combinatorial Bayesian Optimization Using the Graph Cartesian Product. NeurIPS 2019.

38 J. KIM et al. (2021). Bayesian Optimization with Approximate Set Kernels. Machine Learning $110(5): 857-879$. molecules), we might find a useful continuous embedding of the domain and simply work there instead. ${ }^{29}$

Applications such as gene or protein design entail combinatorial optimization over strings, whose nature presents problems both in modeling and optimizing acquisition functions, although optimizing a neural encoding remains a viable option. ${ }^{32}$ Moss et al. described a more direct approach for Bayesian optimization over strings when this is not possible. ${ }^{33}$ The objective function was modeled by a Gaussian process with a special-purpose string kernel, and the induced acquisition function was then optimized using genetic algorithms, which are well suited for string optimization. The authors also showed how to incorporate a context-free grammar into the optimization policy to impose desired constraints on the structure of the strings investigated.

In the general case, we note that our previous sketch of how a "typical" acquisition function behaves with a "typical" model - with its most interesting behavior near observed data - carries over to combinatorial domains and may lead to useful heuristics. For example, rather than enumerating the domain (presumably impossible) or sampling from the domain (presumably yielding poor coverage), we might instead curate a small list of candidate points offering options for both exploitation and exploration. We could perhaps generate a list of points similar to ${ }^{34}$ the thus-far best-seen points, encouraging exploitation, and augment with a small set of points constructed to cover the domain, encouraging exploration. This approach was used for example by GARNETT et al. for Bayesian set function optimization. ${ }^{35}$

Other approaches are also possible. For example, several authors have constructed GP models for particular combinatorial spaces whose structure simplifies the (usually approximate but near-optimal) optimization of particular acquisition functions induced from the model..$^{36,37,38}$

\section*{The benefit of imperfect optimization?}

We conclude this discussion with one paradoxical remark. In some cases it may actually be advantageous to optimize an acquisition function imperfectly. As many common policies are based on extremely myopic (one-step) reasoning, imperfect optimization may yield a useful exploration benefit as a side effect. This phenomenon, and the effect it may have on empirical performance, deserves thoughtful consideration.

\subsection*{STARTING AND STOPPING OPTIMIZATION}

Finally, we briefly consider the part of optimization that happens outside the application of an optimization policy: initialization and termination.

\section*{Initialization}

Theoretically, one can begin a Bayesian optimization routine with a completely empty dataset $\mathcal{D}=\varnothing$ and then use an optimization policy to design every observation, and indeed this has been our working model of Bayesian optimization since sketching the basic idea in Algorithm 1.1. However, Bayesian optimization policies are informed by an underlying belief about the objective function, which can be significantly misinformed when too little data are available, especially when relying on point estimation for model selection rather than accounting for (significant!) uncertainty in model hyperparameters and/or model structures.

Due to the sequential nature of optimization and the dependence of each decision on the data observed in previous iterations, it can be wise to use a model-independent procedure to design a small number of initial observations before beginning optimization in earnest. This procedure can be as simple as random sampling ${ }^{39}$ or a space-filling design such as a low-discrepancy sequence or Latin hypercube design. ${ }^{40}$ When repeatedly solving related optimization problems, we may even be able to learn how to initialize Bayesian optimization routines from experience. Some authors have proposed sophisticated "warm start" initialization procedures for hyperparameter tuning using so-called metafeatures characterizing the datasets under consideration. ${ }^{41}$

\section*{Termination}

In many applications of Bayesian optimization, we assume a preallocated budget on the number of observations we will make and simply terminate optimization when that budget is expended. However, we may also treat termination as a decision and adaptively determine when to stop based on collected data. Of course, in practice we are free to design a stopping rule however we see fit, but we can outline some possible options.

Especially when using a policy grounded in decision theory, it is natural to terminate optimization when the maximum of our chosen acquisition function drops below some threshold $c$, which may depend on $x$ :

$$
\left[\max _{x \in \mathcal{X}}[\alpha(x ; \mathcal{D})-c(x)]<0\right]
$$

For acquisition functions derived from decision theory, such a stopping rule may be justified theoretically: we stop when the expected gain from the optimal observation is no longer worth the cost of acquisition. ${ }^{42} \mathrm{~A}$ majority of the stopping rules described in the literature assume this form, with the threshold $c$ often being determined dynamically based on the scale of observed data. ${ }^{43}$ DAI et al. combined a stopping rule of this form with an otherwise non-decision-theoretic policy (GP-UCB) and showed that its asymptotic performance in terms of expected regret was not adversely affected despite the mismatch in motivation between the policy and stopping rule. ${ }^{44}$

It may also be prudent to consider purely data-dependent stopping rules in order to avoid undue expense arising from miscalibrated models fruitlessly continuing optimization based on incorrect beliefs. For example, ACERBI and MA proposed augmenting a bound on the total number of model averaging: $§ 4.4$, p. 74

39 These approaches are discussed and evaluated in:

M. W. HOFFMAN and B. SHAHRIARI (2014). Modular Mechanisms for Bayesian Optimization. Bayesian Optimization Workshop, NeurIPS 2014.

4 O D. R. JONES et al. (1998). Efficient Global Optimization of Expensive Black-Box Functions. fournal of Global Optimization 13(4):455-492.

41 M. FEURER et al. (2015). Initializing Bayesian Hyperparameter Optimization via MetaLearning. AAAI 2015.

optimal stopping rules: $§ 5.4$, p. 103

42 See in particular our discussion of the one-step optimal stopping rule, p. 104.

43 Some early (but surely not the earliest!) examples:

D. D. COX and S. JOHN (1992). A Statistical Method for Global Optimization. SMC 1992.

D. R. JONES et al. (1998). Efficient Global Optimization of Expensive Black-Box Functions. fournal of Global Optimization 13(4):455-492.

44 Z. DAI et al. (2019). Bayesian Optimization Meets Bayesian Optimal Stopping. ICML 2019. 45 L. ACERBI and W. J. MA (2017). Practical Bayesian Optimization for Model Fitting with Bayesian Adaptive Direct Search. NeurIPS 2017.

direct computation via Cholesky decomposition: § 9.1, p. 201

iterative numerical methods: § 9.1, p. 203 sparse approximation: $§ 9.1$ p. 204

the potential for vanishing gradients: § 9.2, p. 207

initialization: $§ 9.3$, p. 210

termination: § 9.3, p. 211 observations with an early stopping option if no optimization progress is made over a given number of observations. ${ }^{45}$

\subsection*{SUMMARY OF MAJOR IDEAS}

- Gaussian process inference requires solving a system of linear equations whose size grows with the number of observed values (9.2).

- A direct implementation via the Cholesky decomposition is a straightforward option, but scales cubically with the number of observed values.

- Scaling to larger datasets is possible by appealing to iterative numerical methods such as the method of conjugate gradients, or to sparse approximation, where we approximate an intractable posterior Gaussian process with a Gaussian process conditioned on carefully designed, fictitious observations at locations called inducing points. These methods can scale inference to hundreds of thousands of observations or more.

- When optimizing acquisition functions, it is important to be aware of the potential for vanishing gradients in extrapolatory regions of the domain and plan accordingly.

- It is usually a good idea to begin optimization with a small set of observations designed in a model-agnostic fashion in order to begin with a somewhat informed model of the objective function.

- When dynamic termination is desired, simple schemes based on thresholding the acquisition function can be effective.