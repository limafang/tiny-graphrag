\section*{MODELING WITH GAUSSIAN PROCESSES}

Bayesian optimization relies on a faithful model of the system of interest to make well-informed decisions. In fact, even more so than the details of the optimization policy, the fidelity of the underlying model of the objective function is the most decisive factor determining optimization performance. This has been long acknowledged, with MOcKUs for example commenting in his seminal work that: ${ }^{1}$

The development of some system of a priori distributions suitable for different classes of the function $f$ is probably the most important problem in the application of [the] Bayesian approach to... global optimization.

The importance of careful modeling has not waned in the intervening years, but our capacity for building sophisticated models has improved.

Recall our approach to modeling observations obtained during optimization combines a prior process for a (perhaps not directly observable) objective function (1.8) and an observation model linking the values of the objective to measured values (1.2). Both distributions must be specified before we can derive a posterior belief about the objective function (1.10) and predictive distribution for proposed observations (1.7), which together serve as the key enablers of Bayesian optimization policies.

In practice, the choice of observation model is often noncontrover$\mathrm{sial}^{2}{ }^{2}$ and our running prototypes of exact observation and additive Gaussian noise suffice for many systems. The bulk of modeling effort is thus spent crafting the prior process. Although specifying a Gaussian process is seemingly as simple as choosing a mean and covariance function, it can be difficult to intuit appropriate choices without a great deal of knowledge about the system of interest. As an alternative to prior knowledge, we may appeal to a data-driven approach, where we establish a space of candidate models and search through this space for those offering the best explanation of available data. Almost all Gaussian process models used in practice are designed in this manner, and we will lay the groundwork for this approach in this chapter and the next.

As a Gaussian process is specified by its first two moments, datadriven model design boils down to searching for the prior mean and covariance functions most harmonious with our observations. This can be a daunting task as the space of possibilities is limitless. However, we do not need to begin from scratch: there are mean and covariance functions available off-the-shelf for modeling a range of behavioral archetypes, and by systematically combining these components we may model functions with a rich variety of behavior. We will explore the world of possibilities in this chapter, while addressing details important to optimization.

Once we have established a space of candidate models, we will require some mechanism to differentiate possible choices based on their merits, a process known as model assessment that we will explore at length in the next chapter. We will begin the present discussion by revisiting the topic
1 J. Mockus (1974). On Bayesian Methods for Seeking the Extremum. Optimization Techniques: IFIP Technical Conference.

Bayesian inference of the objective function: $\S 1.2$, p. 8

2 However, we may not be certain about some details, such as the scale of observation noise, an issue we will address in the next chapter.

Chapter 4: Model Assessment, Selection, and Averaging, p. 67 
![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-02.jpg?height=284&width=1626&top_left_y=468&top_left_x=160)

Figure 3.1: The importance of the prior mean function in determining sample path behavior. The models in the first two panels differ in their mean function but share the same covariance function. Sample path behavior is identical up to translation. The model in the third panel features the same mean function as the first panel but a different covariance function. Samples exhibit dramatically different behavior.

impact of prior mean on sample paths

impact of prior mean on posterior mean of prior mean and covariance functions with an eye toward practical utility.

THE PRIOR MEAN FUNCTION

Recall the mean function of a Gaussian process specifies the expected value of an arbitrary function value $\phi=f(x)$ :

$$
\mu(x)=\mathbb{E}[\phi \mid x] .
$$

Although this is obviously a fundamental concern, the choice of prior mean function has received relatively little consideration in the Bayesian optimization literature.

There are several reasons for this. To begin, it is actually the covariance function rather than the mean function that largely determines the behavior of sample paths. This should not be surprising: the mean function only affects the marginal distribution of function values, whereas the covariance function can further modify the joint distribution of function values. To elaborate, consider an arbitrary Gaussian process $\mathcal{G P}(f ; \mu, K)$. Its sample paths are distributed identically to those from the corresponding centered process $f-\mu$, after shifting pointwise by $\mu$. Therefore the sample paths of any Gaussian process with the same covariance function are effectively the same up to translation, and it is the covariance function determining their behavior otherwise; see the demonstration in Figure 3.1.

It is also important to understand the role of the prior mean function in the posterior process. Suppose we condition a Gaussian process $\mathcal{G P}(f ; \mu, K)$ on the observation of a vector $\mathbf{y}$ with marginal distribution (2.7) and cross-covariance function (2.24)

$$
p(\mathbf{y})=\mathcal{N}(\mathbf{y} ; \mathbf{m}, \mathbf{C}) ; \quad \kappa(x)=\operatorname{cov}[\mathbf{y}, \phi \mid x] .
$$

The prior mean influences the posterior process only through the posterior mean (2.10):

$$
\mu_{\mathcal{D}}(x)=\mu(x)+\kappa(x)^{\top} \mathbf{C}^{-1}(\mathbf{y}-\mathbf{m}) .
$$



![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-03.jpg?height=357&width=1014&top_left_y=504&top_left_x=270)

We can roughly understand the behavior of the posterior mean by identifying two regimes determined by the strength of correlation between a given function value and the observations. In "interpolatory" regions, where function values have significant correlation with one-or-more observed value, the posterior mean is mostly determined by the data rather than the prior mean. On the other hand, in "extrapolatory" regions, where $\kappa(x) \approx 0$, the data have little influence and the posterior mean effectively equals the prior mean. Figure 3.2 demonstrates this effect.

\section*{Constant mean function}

The primary impact of the prior mean on our predictions - and on an optimization policy informed by these predictions - is in the extrapolatory regime. However, extrapolation without strong prior knowledge can be a dangerous business. As a result, in Bayesian optimization, the prior mean is often taken to be a constant:

$$
\mu(x ; c) \equiv c,
$$

in order to avoid any unwanted bias on our decisions caused by spurious structure in the prior process. This simple choice is supported empirically by a study comparing optimization performance across a range of problems as a function of the choice of prior mean. ${ }^{3}$

When adopting a constant mean, the value of the constant $c$ is usually treated as a parameter to be estimated or (approximately) marginalized, as we will discuss in the next chapter. However, we can actually do better in some cases. Consider a parametric Gaussian process prior with constant mean (3.1) and arbitrary covariance function:

$$
p(f \mid c)=\mathcal{G P}(f ; \mu \equiv c, K)
$$

and suppose we place a normal prior on $c$ :

$$
p(c)=\mathcal{N}\left(c ; a, b^{2}\right) .
$$

Then we can marginalize the unknown constant mean exactly to derive the marginal Gaussian process

$$
p(f)=\int p(f \mid c) p(c) \mathrm{d} c=\mathcal{G P}\left(f ; \mu \equiv a, K+b^{2}\right),
$$

Figure 3.2: The influence of the prior mean on the posterior mean. We show two Gaussian process posteriors differing only in their prior mean functions, shown as dashed lines. In the "interpolatory" region between the observations, the posterior means are mostly determined by the data, but devolve to the respective prior means when extrapolating outside this region.

behavior in interpolatory and extrapolatory regions

3 G. DE ATH et al. (2020). What Do You Mean? The Role of the Mean Function in Bayesian Optimization. GECCO 2020.

marginalizing constant prior mean

model selection and averaging: §§ 4.3-4.4, p. 73 4 Noting that $c$ and $f$ form a joint Gaussian process, we may perform inference as described in $\S 2.4$, p. 26 to reveal their joint posterior.

5 The basis functions can be arbitrarily complex, such as the output layer of a deep neural network:

J. SNOEK et al. (2015). Scalable Bayesian Optimization Using Deep Neural Networks. ICML 2015 .

basis functions, $\psi$ weight vector, $\boldsymbol{\beta}$

6 A. o'Hagan (1978). Curve Fitting and Optimal Design for Prediction. Fournal of the Royal Statistical Society Series B (Methodological) 40(1): $1-42$.

7 C. E. RASMUSSEN and C. K. I. WILLIAMS (2006). Gaussian Processes for Machine Learning. MIT Press. [§ 2.7]

8 For an example of such modeling in physics, where the mean function was taken to be the output of a physically informed model, see:

M. A. ZIATDINOV et al. (2021). Physics Makes the Difference: Bayesian Optimization and Active Learning via Augmented Gaussian Process. arXiv: 2108.10280 [physics.comp-ph].

9 This mean function was proposed in the context of Bayesian optimization (with diagonal A) by

J. SNOEK et al. (2015). Scalable Bayesian Optimization Using Deep Neural Networks. ICML 2015,

who also proposed appropriate priors for A and $\mathbf{b}$. The mean was also proposed in the related context of Bayesian quadrature (see $\S 2.6$, p. 33) by

L. ACERBI (2018). Variational Bayesian Monte Carlo. NeurIPS 2018 where the uncertainty in the mean has been absorbed into the prior covariance function. We may now use this prior directly, avoiding any estimation of $c$. The unknown mean will be automatically marginalized in both the prior and posterior process, and we may additionally derive the posterior belief over $c$ given data if it is of interest. ${ }^{4}$

\section*{Linear combination of basis functions}

We may extend the above result to marginalize the weights of an arbitrary linear combination of basis functions under a normal prior, making this a particularly convenient class of mean functions. Namely, consider a parametric mean function of the form

$$
\mu(x ; \boldsymbol{\beta})=\boldsymbol{\beta}^{\top} \boldsymbol{\psi}(x),
$$

where the vector-valued function $\psi: \mathcal{X} \rightarrow \mathbb{R}^{n}$ defines the basis functions and $\boldsymbol{\beta}$ is a vector of weights. ${ }^{5}$

Now consider a parametric Gaussian process prior with a mean function of this form (3.4) and arbitrary covariance function $K$. Placing a multivariate normal prior on $\boldsymbol{\beta}$,

$$
p(\boldsymbol{\beta})=\mathcal{N}(\boldsymbol{\beta} ; \mathbf{a}, \mathbf{B}),
$$

and marginalizing yields the marginal prior, ${ }^{6,7}$

$$
p(f)=\mathcal{G P}(f ; m, C),
$$

where

$$
m(x)=\mathbf{a}^{\top} \boldsymbol{\psi}(x) ; \quad C\left(x, x^{\prime}\right)=K\left(x, x^{\prime}\right)+\boldsymbol{\psi}(x)^{\top} \mathbf{B} \boldsymbol{\psi}\left(x^{\prime}\right) .
$$

We may recover the constant mean case above by taking $\psi(x) \equiv 1$.

\section*{Other options}

We stress that a constant or linear mean function is by no means necessary, and when a system is understood sufficiently well to suggest a plausible alternative - perhaps the output of a baseline predictive model - it should be strongly considered. However, it is hard to provide general advice, as this modeling will be situation dependent. ${ }^{8}$

One option that might be a reasonable choice in some optimization contexts is a concave quadratic mean:

$$
\mu(\mathbf{x} ; \mathbf{A}, \mathbf{b}, c)=(\mathbf{x}-\mathbf{b})^{\top} \mathbf{A}^{-1}(\mathbf{x}-\mathbf{b})+c,
$$

where $\mathbf{A}<\mathbf{0}$. ${ }^{9}$ This mean encodes that values near $\mathbf{b}$ (according to the Mahalanobis distance (A.8)) are expected to be higher than those farther away and could reasonably model an objective function expected to be "bowl-shaped" to a first approximation. The middle panel of Figure 3.1 incorporates a mean of this form; note that the maxima of sample paths are of course not constrained to agree with that of the prior mean. The prior covariance function determines the covariance between the function values corresponding to a pair of input locations $x$ and $x^{\prime}$ :

$$
K\left(x, x^{\prime}\right)=\operatorname{cov}\left[\phi, \phi^{\prime} \mid x, x^{\prime}\right] .
$$

The covariance function determines fundamental properties of sample path behavior, including continuity, differentiability, and aspects of the global optima, as we have already seen. Perhaps more so than the mean function, careful design of the covariance function is critical to ensure fidelity in modeling. We will devote considerable discussion to this topic, beginning with some important properties and moving on to useful examples and mechanisms for systematically modifying and composing multiple covariance functions together to model complex behavior.

After appropriate normalization, a covariance function $K$ may be loosely interpreted as a measure of similarity between points in the domain. Namely, given $x, x^{\prime} \in \mathcal{X}$, the correlation between the corresponding function values is

$$
\rho=\operatorname{corr}\left[\phi, \phi^{\prime} \mid x, x^{\prime}\right]=\frac{K\left(x, x^{\prime}\right)}{\sqrt{K(x, x) K\left(x^{\prime}, x^{\prime}\right)}},
$$

and we may interpret the strength of this dependence as a measure of similarity between the input locations. This intuition can be useful, but some caveats are in order. To begin, note that correlation may be negative, which might be interpreted as indicating dis-similarity as the function values react to information with opposite sign.

Further, for a proposed covariance function $K$ to be admissible, it must satisfy two global consistency properties ensuring that the collection of random variables comprising $f$ are able to satisfy the purported relationships. First, we can immediately deduce from its definition (3.8) that $K$ must be symmetric in its inputs. Second, the covariance function must be positive semidefinite; that is, given any finite set of points $\mathbf{x} \subset \mathcal{X}$, the Gram matrix $K(\mathbf{x}, \mathbf{x})$ must have only nonnegative eigenvalues. ${ }^{10}$

To illustrate how positive semidefiniteness ensures statistical validity, note that a direct consequence is that $K(x, x)=\operatorname{var}[\phi \mid x] \geq 0$, and thus marginal variance is always nonnegative. On a slightly less trivial level, consider a pair of points $\mathbf{x}=\left(x, x^{\prime}\right)$ and normalize the corresponding Gram matrix $\Sigma=K(\mathbf{x}, \mathbf{x})$ to yield the correlation matrix:

$$
\mathbf{P}=\operatorname{corr}[\boldsymbol{\phi} \mid \mathbf{x}]=\left[\begin{array}{ll}
1 & \rho \\
\rho & 1
\end{array}\right],
$$

where $\rho$ is given by (3.9). For this matrix to be valid, we must have $\rho \in[-1,1]$. This happens precisely when $\mathbf{P}$ is positive semidefinite, as its eigenvalues are $1 \pm \rho$. Finally, noting that $\mathbf{P}$ is congruent to $\Sigma,{ }^{11}$ we conclude the implied correlations are consistent if and only if $\Sigma$ is positive semidefinite. With more than two points, the positive semidefiniteness of $K$ ensures similar consistency at higher orders. sample path continuity: $§ 2.5$, p. 28 sample path differentiability: § 2.6, p. 30 existence and uniqueness of global maxima: $\S 2.7$, p. 33

correlation between function values, $\rho$

symmetry and positive semidefiniteness

10 Symmetry guarantees the eigenvalues are real. consequences of positive semidefiniteness

11 We have $\Sigma=$ SPS, where $\mathrm{S}$ is diagonal with $S_{i i}=\sqrt{\Sigma_{i i}}$ Figure 3.3: Left: a sample from a stationary Gaussian process in two dimensions. The joint distribution of function values is translation- but not rotationinvariant, as the function tends to vary faster in some directions than others. Right: a sample from an isotropic process. The joint distribution of function values is both translation- and rotationinvariant.

stationary covariance function, $K\left(x-x^{\prime}\right)$

12 Of course this definition requires $x-x^{\prime}$ to be well defined. This is trivial in Euclidean spaces; a fairly general treatment for more exotic spaces would assume an abelian group structure on $\mathcal{X}$ with binary operation + and inverse - and define $x-x^{\prime}=x+\left(-x^{\prime}\right)$.

isotropic covariance function, $K(d)$

13 S. BOCHNER (1933). Monotone Funktionen, Stieltjessche Integrale und harmonische Analyse. Mathematische Annalen 108:378-410.

14 We do not quote the most general version of the theorem here; the result can be extended to complex-valued covariance functions on arbitrary locally compact abelian groups if necessary. It is remarkably universal.

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-06.jpg?height=474&width=462&top_left_y=457&top_left_x=774)

stationary and anisotropic

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-06.jpg?height=474&width=531&top_left_y=457&top_left_x=1254)

stationary and isotropic

\section*{Stationarity, isotropy, and Bochner's theorem}

Some covariance functions exhibit structure giving rise to certain computational benefits. Namely, a covariance function $K\left(x, x^{\prime}\right)$ that only depends on the difference $x-x^{\prime}$ is called stationary. ${ }^{12}$ When convenient, we will abuse notation and write a stationary covariance function in terms of a single input, writing $K\left(x-x^{\prime}\right)$ for $K\left(x, x^{\prime}\right)=K\left(x-x^{\prime}, 0\right)$. If a GP has a stationary covariance function and constant mean function (3.1), then the process itself is also called stationary. A consequence of stationarity is that the distribution of any set of function values is invariant under translation; that is, the function "acts the same" everywhere from a statistical viewpoint. The left panel of Figure 3.3 shows a sample from a $2 d$ stationary GP, demonstrating this translation-invariant behavior.

Stationarity is a convenient assumption when modeling, as defining the local behavior around a single point suffices to specify the global behavior of an entire function. Many common covariance functions have this property as a result. However, this may not always be a valid assumption in the context of optimization, as an objective function may for example exhibit markedly different behavior near the optimum than elsewhere. We will shortly see some general approaches for addressing nonstationarity when appropriate.

If $\mathcal{X} \subset \mathbb{R}^{n}$, a covariance function $K\left(x, x^{\prime}\right)$ only depending on the Euclidean distance $d=\left|x-x^{\prime}\right|$ is called isotropic. Again, when convenient, we will notate such a covariance with $K(d)$. Isotropy is a more restrictive assumption than stationarity - indeed it trivially implies stationarity as it implies the covariance is invariant to both translation and rotation, and thus the function has identical behavior in every direction from every point. An example sample from a $2 d$ isotropic GP is shown in the right panel of Figure 3.3. Many of the standard covariance functions we will define shortly will be isotropic on first definition, but we will again develop generic mechanisms to modify them in order to induce anisotropic behavior when desired.

BOCHNER's theorem is an landmark result characterizing stationary covariance functions in terms of their Fourier transforms: $:^{13,14}$ Theorem (Bochner, 1933). A continuous function $K: \mathbb{R}^{n} \rightarrow \mathbb{R}$ is positive semidefinite (that is, represents a stationary covariance function) if and only if we have

$$
K(\mathbf{x})=\int \exp \left(2 \pi i \mathbf{x}^{\top} \boldsymbol{\xi}\right) \mathrm{d} v,
$$

where $v$ is a finite, positive Borel measure on $\mathbb{R}^{n}$. Further, this measure is symmetric around the origin; that is, $v(A)=v(-A)$ for any Borel set $A \subset \mathbb{R}^{n}$, where $-A$ is the "negation" of $A:-A=\{-a \mid a \in A\}$.

To summarize, BOCHNER's theorem states that the Fourier transform of any stationary covariance function on $\mathbb{R}^{n}$ is proportional to a probability measure and vice versa; the constant of proportionality is $K(\mathbf{0})$. The measure $v$ corresponding to $K$ is called the spectral measure of $K$. When a corresponding density function $\kappa$ exists, it is called the spectral density of $K$ and forms a Fourier pair with $K$ :

$$
K(\mathbf{x})=\int \exp \left(2 \pi i \mathbf{x}^{\top} \boldsymbol{\xi}\right) \kappa(\xi) \mathrm{d} \boldsymbol{\xi} ; \quad \kappa(\boldsymbol{\xi})=\int \exp \left(-2 \pi i \mathbf{x}^{\top} \boldsymbol{\xi}\right) K(\mathbf{x}) \mathrm{d} \mathbf{x} .
$$

The symmetry of the spectral measure implies a similar symmetry in the spectral density: $\kappa(\xi)=\kappa(-\xi)$ for all $\xi \in \mathbb{R}^{n}$.

BOCHNER's theorem is surprisingly useful in practice, allowing us to approximate an arbitrary stationary covariance function by approximating (e.g., by modeling or sampling from) its spectral density. This is the basis of the spectral mixture covariance described in the next section, as well as the sparse spectrum approximation scheme, which facilitates the computation of some Bayesian optimization policies.

It can be difficult to define a new covariance function for a given scenario de novo, as the positive-semidefinite criterion can be nontrivial to guarantee for what might otherwise be an intuitive notion of similarity. In practice, it is common to instead construct covariance functions by combining and transforming established "building blocks" modeling various atomic behaviors while following rules guaranteeing the result will be valid. We describe several useful examples below. ${ }^{15}$

Our presentation will depart from most in that several of the covariance functions below will initially be defined without parameters that some readers may be expecting. We will shortly demonstrate how coupling these covariance functions with particular transformations of the function domain and output gives rise to common covariance function parameters such as characteristic length scales and output scales.

\section*{The Matérn family}

If there is one class of covariance functions to be familiar with, it is the Matern family. This is a versatile family of covariance functions for modeling isotropic behavior on Euclidean domains $\mathcal{X} \subset \mathbb{R}^{n}$ of any spectral measure, $v$ spectral density, $\kappa$

symmetry of spectral density

sparse spectrum approximation: $§ 8.7$, p. 178

15 For a more complete survey, see

C. E. RASMUSSEN and C. K. I. WILLIAMS (2006). Gaussian Processes for Machine Learning. MIT Press. [chapter 4] 

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-08.jpg?height=206&width=525&top_left_y=454&top_left_x=160)

$v=1 / 2,(3.11)$

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-08.jpg?height=208&width=528&top_left_y=453&top_left_x=707)

$v=3 / 2,(3.13)$

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-08.jpg?height=206&width=511&top_left_y=454&top_left_x=1272)

$v=5 / 2,(3.14)$

Figure 3.4: Samples from centered Gaussian processes with the Matern covariance function with different values of the smoothness parameter $v$. Sample paths with $v=1 / 2$ are continuous but not differentiable; incrementing this parameter by one unit increases the number of continuous derivatives by one.

sample path differentiability: $\$ 2.6$, p. 30

16 In theoretical contexts, general values for the smoothness parameter $v \in R_{>0}$ are considered, but lead to unwieldy expressions (10.12).

exponential covariance function

Ornstein-Uhlenbeck (ou) process

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-08.jpg?height=220&width=529&top_left_y=1820&top_left_x=158)

Samples from a centered Gaussian process with squared exponential covariance $K_{\mathrm{SE}}$.

17 M. L. STEIN (1999). Interpolation of Spatial Data: Some Theory for Kriging. Springer-Verlag. [§ 1.7$]$ desired degree of smoothness, in terms of the differentiability of sample paths. The Matérn covariance $K_{\mathrm{M}(v)}$ depends on a parameter $v \in \mathbb{R}_{>0}$ determining this smoothness; sample paths from a centered Gaussian process with this covariance are $\lceil v\rceil-1$ times continuously differentiable. In practice $v$ is almost always taken to be a half-integer, ${ }^{16}$ in which case the expression for the covariance assumes a simple form as a function of the Euclidean distance $d=\left|x-x^{\prime}\right|$.

To begin with the extremes, the case $v=1 / 2$ yields the so-called exponential covariance:

$$
K_{\mathrm{M} 1 / 2}\left(x, x^{\prime}\right)=\exp (-d) .
$$

Sample paths from a centered Gaussian process with exponential covariance are continuous but nowhere differentiable, which is perhaps too rough to be interesting in most optimization contexts. However, this covariance is often encountered in historical literature. In the onedimensional case $\mathcal{X} \subset \mathbb{R}$, a Gaussian process with this covariance is known as a Ornstein-Uhlenbeck (ou) process and satisfies a continuoustime Markov property that renders its posterior moments particularly convenient.

Taking the limit of increasing smoothness $v \rightarrow \infty$ yields the squared exponential covariance from the previous chapter:

$$
K_{\mathrm{SE}}\left(x, x^{\prime}\right)=\exp \left(-\frac{1}{2} d^{2}\right) .
$$

We will refer to the Matérn and the limiting case of the squared exponential covariance functions together as the Matérn family. The squared exponential covariance is without a doubt the most prevalent covariance function in the statistical and machine learning literature. However, it may not always be a good choice in practice. Sample paths from a centered Gaussian process with squared exponential covariance are infinitely differentiable, which has been ridiculed as an absurd assumption for most physical processes. ${ }^{17}$ sTEIN does not mince words on this, starting off a three-sentence "summary of practical suggestions" with "use the Matérn model" and devoting significant effort to discouraging the use of the squared exponential in the context of geostatistics. Between these extremes are the cases $v=3 / 2$ and $v=5 / 2$, which respectively model once- and twice-differentiable functions:

$$
\begin{aligned}
& K_{\mathrm{M}^{3} / 2}\left(x, x^{\prime}\right)=(1+\sqrt{3} d) \exp (-\sqrt{3} d) ; \\
& K_{\mathrm{M} 5 / 2}\left(x, x^{\prime}\right)=\left(1+\sqrt{5} d+\frac{5}{3} d^{2}\right) \exp (-\sqrt{5} d) .
\end{aligned}
$$

Figure 3.4 illustrates samples from centered Gaussian processes with different values of the smoothness parameters $v$. The $v=5 / 2$ case in particular has been singled out as a prudent off-the-shelf choice for Bayesian optimization when no better alternative is obvious. ${ }^{18}$

\section*{The spectral mixture covariance}

Covariance functions in the Matérn family express fairly simple correlation structure, with the covariance dropping monotonically to zero as the distance $d=\left|x-x^{\prime}\right|$ increases. All differences in sample path behavior such as differentiability, etc. are expressed entirely through nuances in the tail behavior of the covariance functions; see the figure in the margin.

The Fourier transforms of these covariances are also broadly comparable: all are proportional to unimodal distributions centered on the origin. However, BOchNeR's theorem indicates that there is a vast world of stationary covariance functions indexed by the entire space of symmetric spectral measures, which may have considerably more complex structure. Several authors have sought to exploit this characterization to build stationary covariance functions with virtually unlimited flexibility.

A notable contribution in this direction is the spectral mixture covariance function proposed by wILSON and ADAMs. ${ }^{19}$ The idea is simple but powerful: we parameterize a space of stationary covariance functions by some suitable family of mixture distributions in the Fourier domain representing their spectral density. The parameters of this spectral mixture distribution specify a covariance function via the correspondence in (3.10), and we can make the resulting family as rich as desired by adjusting the number of components in the mixture. WILSON and ADAMS proposed Gaussian mixtures for the spectral density, which are universal approximators and have a convenient Fourier transform. We define a Gaussian mixture spectral density $\kappa$ as

$$
k(\xi)=\sum_{i} w_{i} \mathcal{N}\left(\xi ; \boldsymbol{\mu}_{i}, \Sigma_{i}\right) ; \quad \kappa(\xi)=\frac{1}{2}[k(\xi)+k(-\xi)],
$$

where the indirect construction via $k$ ensures the required symmetry. Note that the weights $\left\{w_{i}\right\}$ must be positive but need not sum to unity. Taking the inverse Fourier transform (3.10), the corresponding covariance function is

$$
\begin{aligned}
& K_{\mathrm{SM}}\left(\mathbf{x}, \mathbf{x}^{\prime} ;\left\{w_{i}\right\},\left\{\boldsymbol{\mu}_{i}\right\},\left\{\Sigma_{i}\right\}\right)= \\
& \quad \sum_{i} w_{i} \exp \left(-2 \pi^{2}\left(\mathbf{x}-\mathbf{x}^{\prime}\right)^{\top} \Sigma_{i}\left(\mathbf{x}-\mathbf{x}^{\prime}\right)\right) \cos \left(2 \pi\left(\mathbf{x}-\mathbf{x}^{\prime}\right)^{\top} \boldsymbol{\mu}_{i}\right)
\end{aligned}
$$

18 J. SNOEK et al. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. NeUrIPS 2012.

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-09.jpg?height=319&width=531&top_left_y=1037&top_left_x=1368)

Some members of the Matern family and the squared exponential covariance as a function of the distance between inputs. All decay to zero correlation as distance increases.

19 A. G. WILSON and R. P. ADAMS (2013). Gaussian Process Kernels for Pattern Discovery and Extrapolation. ICML 2013.

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-09.jpg?height=434&width=531&top_left_y=1907&top_left_x=1368)

Samples from centered Gaussian processes with two realizations of a Gaussian spectral mixture covariance function, offering a glimpse into the flexibility of this class. 2o Independence is usual but not necessary; an arbitrary joint prior would add a term of $2 \mathbf{b}^{\top} \mathbf{x}$ to (3.16), where $\mathbf{b}=\operatorname{cov}[\boldsymbol{\beta}, \beta]$.

linear basis functions: $§ 3.1$, p. 48

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-10.jpg?height=212&width=523&top_left_y=1393&top_left_x=161)

Samples from a centered Gaussian process with linear covariance $K_{\text {LIN }}$.
21 Although an important concept, there is no clear-cut definition of characteristic length scale. It is simply a convenient separation distance for which correlation remains appreciable, but beyond which correlation begins to noticeably decay.
Inspecting this expression, we can see that every covariance function induced by a Gaussian mixture spectral density is infinitely differentiable, and one might object to this choice on the grounds of overly smooth sample paths. This can be mitigated by using enough mixture components to induce sufficiently complex structure in the covariance (on the order of $\sim 5$ is common). Another option would be to use a different family of spectral distributions; for example, a mixture of Cauchy distributions would induce a family of continuous but nondifferentiable covariance functions analogous to the exponential covariance (3.11), but this idea has not been explored.

\section*{Linear covariance function}

Another useful covariance function arises from a Bayesian realization of linear regression. Let the domain be Euclidean, $\mathcal{X} \subset \mathbb{R}^{n}$ and consider the model

$$
f(\mathbf{x})=\beta+\boldsymbol{\beta}^{\top} \mathbf{x},
$$

where we have abused notation slightly to distinguish the constant term from the remaining coefficients. Following our discussion on linear basis functions, if we take independent ${ }^{20}$ normal priors on $\beta$ and $\boldsymbol{\beta}$ :

$$
p(\beta)=\mathcal{N}\left(\beta ; a, b^{2}\right) ; \quad p(\boldsymbol{\beta})=\mathcal{N}(\boldsymbol{\beta} ; \mathbf{a}, \mathbf{B}),
$$

we arrive at the so-called linear covariance:

$$
K_{\mathrm{LIN}}\left(\mathbf{x}, \mathbf{x}^{\prime} ; b, \mathbf{B}\right)=b^{2}+\mathbf{x}^{\top} \mathbf{B} \mathbf{x} .
$$

Although this covariance is unlikely to be of any direct use in Bayesian optimization (linear programming is much simpler!), it can be a useful component of more complex composite covariance structures.

\subsection*{MODIFYING AND COMBINING COVARIANCE FUNCTIONS}

With the notable exception of the spectral mixture covariance, which can approximate any stationary covariance function, several of the covariances introduced in the last section are still too rigid to be useful.

In particular, consider any of the Matérn family (3.11-3.14). Each of these covariances encodes several explicit and possibly dubious assumptions about the function of interest. To begin, each prescribes unit variance for every function value:

$$
\operatorname{var}[\phi \mid x]=K(x, x)=1,
$$

which is an arbitrary, possibly inappropriate choice of scale. Further, each of these covariance functions fixes an isotropic characteristic length scale of correlation of approximately one unit: ${ }^{21}$ at a separation of $\left|x-x^{\prime}\right|=1$, the correlation between the corresponding function values drops to roughly

$$
\operatorname{corr}\left[\phi, \phi^{\prime} \mid x, x^{\prime}\right] \approx 0.5
$$

— prior mean prior $95 \%$ credible interval

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-11.jpg?height=223&width=1031&top_left_y=545&top_left_x=267)

— scaling function, $a$
![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-11.jpg?height=364&width=1056&top_left_y=841&top_left_x=226)

and this correlation continues to drop effectively to zero at a separation of approximately five units. Again, this choice of scale is arbitrary, and the assumption of isotropy is particularly restrictive.

In general, a Gaussian process encodes strong assumptions regarding the joint distribution of function values (2.5), which may not be compatible with a given function "out of the box." However, we can often improve model fit by appropriate transformations of the objective. In fact, linear transformations of function inputs and outputs are almost universally considered, although only implicitly by introducing parameters conveying the effects of these transformations. We will show how both linear and nonlinear transformations of function input and output lead to more expressive models and give rise to common model parameters.

\section*{Scaling function outputs}

We first address the issue of scale in function output (3.17) by considering the statistical effects of arbitrary scaling. Consider a random function $f: \mathcal{X} \rightarrow \mathbb{R}$ with covariance function $K$ and let $a: \mathcal{X} \rightarrow \mathbb{R}$ be a known scaling function. ${ }^{22}$ Then the pointwise product af $: x \mapsto a(x) f(x)$ has covariance function

$$
\operatorname{cov}[a f \mid a]=a(x) K\left(x, x^{\prime}\right) a\left(x^{\prime}\right),
$$

by the bilinearity of covariance. If the scaling function is constant, $a \equiv \lambda$, then we have

$$
\operatorname{cov}[\lambda f \mid \lambda]=\lambda^{2} K
$$

This simple result allows us to extend a "base" covariance $K$ with fixed scale, as in (3.17), to a parametric family with arbitrary scale:

$$
K^{\prime}\left(x, x^{\prime} ; \lambda\right)=\lambda^{2} K\left(x, x^{\prime}\right) .
$$

Figure 3.5: Scaling a stationary covariance by a nonconstant function (here, a smooth bump function of compact support) to yield a nonstationary covariance. 22 For this result $f$ need not have a GP distribu-
tion. 

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-12.jpg?height=328&width=529&top_left_y=464&top_left_x=158)

The squared exponential covariance $K_{\mathrm{SE}}$ scaled by a range of output scales $\lambda(3.20)$

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-12.jpg?height=436&width=531&top_left_y=907&top_left_x=157)

Sample paths from centered GPs with smaller (top) and larger (bottom) output scales.

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-12.jpg?height=340&width=531&top_left_y=1458&top_left_x=157)

The squared exponential covariance $K_{\mathrm{SE}}$ dilated by a range of length scales $\ell$ (3.22).

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-12.jpg?height=437&width=534&top_left_y=1929&top_left_x=156)

Sample paths from centered GPs with shorter (top) and longer (bottom) characteristic length scales.
In this context the parameter $\lambda$ is known as an output scale, or when the base covariance is stationary with $K(x, x)=1$, the signal variance, as it determines the variance of any function value: $\operatorname{var}[\phi \mid x, \lambda]=\lambda^{2}$. The illustration in the margin shows the effect of scaling the squared exponential covariance function by a series of increasing output scales.

We can also of course consider nonlinear transformations of the function output as well. This can be useful for modeling constraints such as nonegativity or boundedness - that are not compatible with the Gaussian assumption. However, a nonlinear transformation of a Gaussian process is no longer Gaussian, so it is often more convenient to model the transformed function after "removing the constraint."

We may use the general form of this scaling result (3.19) to transform a stationary covariance into a nonstationary one, as any nonconstant scaling is sufficient to break translation invariance. We show an example of such a transformation in Figure 3.5 , where we have scaled a stationary covariance by a bump function to create a prior on smooth functions with compact support.

\section*{Transforming the domain and length scale parameters}

We now address the issue of the scaling of correlation as a function of distance (3.18) by introducing a powerful tool: transforming the domain of the function of interest into a more convenient space for modeling.

Namely, suppose we wish to reason about a function $f: \mathcal{X} \rightarrow \mathbb{R}$, and let $g: \mathcal{X} \rightarrow \mathcal{Z}$ be a map from the domain to some arbitrary space $\mathcal{Z}$, which might also be $\mathcal{X}$. If $K_{\mathcal{Z}}$ is a covariance function on $\mathcal{Z}$, then the composition

$$
K_{\mathcal{X}}\left(x, x^{\prime}\right)=K_{\mathcal{Z}}\left(g(x), g\left(x^{\prime}\right)\right)
$$

is trivially a covariance function on $\mathcal{X}$. This allows us to define a covariance for $f$ indirectly by jointly designing a map $g$ to another space and a corresponding covariance $K_{\mathcal{Z}}$ (and mean $\mu_{\mathcal{Z}}$ ) on that space. This approach offers a lot of flexibility, as we are free to design these components as we see fit to impose any desired structure.

We will spend some time exploring this idea, beginning with the relatively simple but immensely useful case of combining a linear transformation on a Euclidean domain $\mathcal{X} \subset \mathbb{R}^{n}$ with an isotropic covariance on the output. Perhaps the simplest example is the dilation $\mathbf{x} \mapsto \mathbf{x} / \ell$, which simply scales distance by $\ell^{-1}$ Incorporating this transformation into an isotropic base covariance $K(d)$ on $\mathcal{X}$ yields a parametric family of dilated versions:

$$
K^{\prime}\left(x, x^{\prime} ; \ell\right)=K(d / \ell) .
$$

If the base covariance has a characteristic length scale of one unit, the length scale of the dilated version will be $\ell$; for this reason, this parameter is simply called the characteristic length scale of the parameterized covariance (3.22). Adjusting the length scale allows us to model functions with a range of "wiggliness," where shorter length scale implies more wiggly behavior; see the margin for examples. 
![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-13.jpg?height=454&width=1022&top_left_y=453&top_left_x=276)

Taking this one step further, we may consider dilating each axis by a separate factor:

$$
x_{i} \mapsto x_{i} / \ell_{i} ; \quad \mathbf{x} \mapsto[\operatorname{diag} \ell]^{-1} \mathbf{x}
$$

which induces the weighted Euclidean distance

$$
d_{\ell}=\sqrt{\sum_{i} \frac{\left(x_{i}-x_{i}^{\prime}\right)^{2}}{\ell_{i}}} .
$$

Geometrically, the effect of this map is to transform surfaces of equal distance around each point - which represent curves of constant covariance for an isotropic covariance - from spheres into axis-aligned ellipsoids; see the figure in the margin. Incorporating into an isotropic base covariance $K(d)$ produces a parametric family of anisotropic covariances with different characteristic length scales along each axis, corresponding to the parameters $\boldsymbol{\ell}$ :

$$
K^{\prime}\left(x, x^{\prime} ; \boldsymbol{\ell}\right)=K\left(d_{\ell}\right) .
$$

When the length scale parameters are inferred from data, this construction is known as automatic relevance determination (ARD). The motivation for the name is that if the function has only weak dependence on some mostly irrelevant dimension of the input, we could hope to infer a very long length scale for that dimension. The contribution to the weighted distance (3.24) for that dimension would then be effectively nullified, and the resulting covariance would effectively "ignore" that dimension.

Figure 3.6 shows samples from $2 d$ centered Gaussian processes, comparing behavior with an isotropic covariance and an ARD modified version that contracts the horizontal and expands the vertical axis (see curves of constant covariance in the margin). The result is anisotropic behavior with a longer characteristic length scale in the vertical direction than in the horizontal direction, but with the behavior of local features remaining aligned with the axes overall.

Finally, we may also consider an arbitrary linear transformation $g: \mathbf{x} \mapsto \mathrm{Ax}$, which induces the Mahalanobis distance (A.8)

$$
d_{\mathrm{A}}=\left|\mathbf{A x}-\mathbf{A x}^{\prime}\right|
$$

Figure 3.6: Left: a sample from a centered Gaussian process in two dimensions with isotropic squared exponential covariance. Right: a sample from a centered Gaussian process with an ARD squared exponential covariance. The length of the lines on each axis are proportional to the length scale along that axis.

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-13.jpg?height=476&width=485&top_left_y=1047&top_left_x=1391)

Possible surfaces of equal covariance with the center when combining separate dilation of each axis with an isotropic covariance.

automatic relevance determination, ARD

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-13.jpg?height=297&width=165&top_left_y=1896&top_left_x=1551)

Surfaces of equal covariance with the center for the examples in Figure 3.6: the isotropic covariance in the left panel (the smaller circle), and the ARD covariance in the right panel (the elongated ellipse). 

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-14.jpg?height=483&width=483&top_left_y=478&top_left_x=181)

Possible surfaces of equal covariance with the center when combining an arbitrary linear transformation with an isotropic covariance.

high-dimensional domains: $§ 3.5$, p. 61

23 We did see a periodic GP in the previous chapter (2.30); however, that model only had support on perfectly sinusoidal functions.

24 D. J. C. MACKAY (1998). Introduction to Gaussian Processes. In: Neural Networks and Machine Learning. [\$ 5.2]

25 The covariance on the circle is usually inherited from a covariance on $\mathbb{R}^{2}$. The result of composing with the squared exponential covariance in particular is often called "the" periodic covariance, but we stress that any other covariance on $\mathbb{R}^{2}$ could be used instead.

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-14.jpg?height=222&width=531&top_left_y=1965&top_left_x=157)

A sample path of a centered GP with Matérn covariance with $v=5 / 2(3.14)$ after applying the periodic warping function (3.27).

26 J. SNOEK et al. (2014). Input Warping for Bayesian Optimization of Non-Stationary Functions. ICML 2014.
As before, we may incorporate this map into an isotropic base covariance $K$ to realize a family of anisotropic covariance functions:

$$
K^{\prime}\left(x, x^{\prime} ; \mathbf{A}\right)=K\left(d_{\mathrm{A}}\right) .
$$

Geometrically, an arbitrary linear map can transform surfaces of constant covariance from spheres into arbitrary ellipsoids; see the figure in the margin. The sample from the left-hand side of Figure 3.3 was generated by composing an isotropic covariance with a map inducing both anisotropic scaling and rotation. The effect of the underlying transformation can be seen in the shapes of local features, which are not aligned with the axes.

Due to the inherent number of parameters required to specify a general transformation, this construction is perhaps most useful when the map is to a much lower-dimensional space: $\mathbb{R}^{n} \rightarrow \mathbb{R}^{k} k \ll n$. This has been promoted as one strategy for modeling functions on highdimensional domains suspected of having hidden low-dimensional structure - if this low-dimensional structure is along a linear subspace of the domain, we could capture it by an appropriately designed projection A. We will discuss this idea further in the next section.

\section*{Nonlinear warping}

When using a covariance function with an inherent length scale, such as a Matérn or squared exponential covariance, some linear transformation of the domain is almost always considered, whether it be simple dilation (3.22), anisotropic scaling (3.25), or a general transformation (3.26). However, nonlinear transformations can also be useful for imposing structure on the domain, a process commonly referred to as warping.

To provide an example that may not often be useful in optimization but is illustrative nonetheless, suppose we wish to model a function $f: \mathbb{R} \rightarrow \mathbb{R}$ that we believe to be smooth and periodic with period $p$. None of the covariance functions introduced thus far would be able to induce the periodic correlations that this assumption would entail. ${ }^{23} \mathrm{~A}$ construction due to MACKAY is to compose a map onto a circle of radius $r=p /(2 \pi):^{24}$

$$
x \mapsto\left[\begin{array}{l}
r \cos x \\
r \sin x
\end{array}\right]
$$

with a covariance function on that space reflecting any desired properties of $f .{ }^{25}$ As this map identifies points separated by any multiple of the period, the corresponding function values are perfectly correlated, as desired. A sample from a Gaussian process employing this construction with a Matérn covariance after warping is shown in the margin.

A compelling use of warping is to build nonstationary models by composing a nonlinear map with a stationary covariance, an idea SNOEK et al. explored in the context of Bayesian optimization. ${ }^{26}$ Many objective functions exhibit different behavior depending on the proximity to the optimum, suggesting that nonstationary models may sometimes be worth exploring. SNOEK et al. proposed a flexible family of warping functions for optimization problems with box-bounded constraints, where 

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-15.jpg?height=289&width=1622&top_left_y=461&top_left_x=274)

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-15.jpg?height=345&width=756&top_left_y=821&top_left_x=273)

we may take the domain to be the unit cube by scaling and translating as necessary: $\mathcal{X}=[0,1]$. The idea is to warp each coordinate of the input via the cumulative distribution function of a beta distribution:

$$
x_{i} \mapsto I\left(x_{i} ; \alpha_{i}, \beta_{i}\right)
$$

where $\left(\alpha_{i}, \beta_{i}\right)$ are shape parameters and $I$ is the regularized beta function. This represents a monotonic bijection on the unit interval that can assume several shapes; see the marginal figure for examples. The map may contract portions of the domain and expand others, effectively decreasing and increasing the length scale in those regions. Finally, taking $\alpha=\beta=1$ recovers the identity map, allowing us to degrade gracefully to the unwarped case if desired.

In Figure 3.7 we combine a beta warping on a one-dimensional domain with a stationary covariance on the output. The chosen warping shortens the length scale near the center of the domain and extends it near the boundary, which might be reasonable for an objective expected to exhibit the most "interesting" behavior on the interior of its domain.

A recent innovation is to use sophisticated artificial neural networks as warping maps for modeling functions of high-dimensional data with complex structure. Notable examples of this approach include the families of manifold Gaussian processes introduced by CALANDRA et al. ${ }^{27}$ and deep kernels introduced contemporaneously by wiLson et al. ${ }^{28}$ Here the warping function was taken to be an arbitrary neural network, the output layer of which was fed into a suitable stationary covariance function. This gives a highly parameterized covariance function where the parameters of the base covariance and the neural map become parameters of the resulting model. In the context of Bayesian optimization, this can be especially useful when there is sufficient data to learn a useful representation of the domain via unsupervised methods.

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-15.jpg?height=366&width=528&top_left_y=1322&top_left_x=1369)

Some examples of beta CDF warping functions (3.28).

27 R. CALANDRA et al. (2016). Manifold Gaussian Processes for Regression. IJCNN 2016.

28 A. G. WILson et al. (2016). Deep Kernel Learning. AISTATS 2016.

neural representation learning: § 8.11, p. 198 

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-16.jpg?height=209&width=523&top_left_y=455&top_left_x=161)

$K_{1}$

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-16.jpg?height=64&width=523&top_left_y=553&top_left_x=709)

$K_{2}$

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-16.jpg?height=205&width=511&top_left_y=457&top_left_x=1275)

$K_{1}+K_{2}$

Figure 3.8: Samples from centered Gaussian processes with different covariance functions: (left) a squared exponential covariance, (middle) a squared exponential covariance with smaller output scale and shorter length scale, and (right) the sum of the two. Samples from the process with the sum covariance show smooth variation on two different scales.

29 The assumption of the processes being centered is needed for the product result only; otherwise, there would be additional terms involving scaled versions of each individual covariance as in (3.19). The sum result does not depend on any assumptions regarding the mean functions.

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-16.jpg?height=229&width=531&top_left_y=2127&top_left_x=157)

A sample from a centered Gaussian process with an "almost periodic" covariance function.

\section*{Combining covariance functions}

In addition to modifying covariance functions via scaling the output and/or transforming the domain, we may also combine multiple covariance functions together to model functions influenced by multiple random processes.

Let $f, g: \mathcal{X} \rightarrow \mathbb{R}$ be two centered, independent (not necessarily Gaussian) random functions with covariance functions $K_{f}$ and $K_{g}$, respectively. By the properties of covariance, the sum and pointwise product of these functions have covariance functions with the same structure:

$$
\operatorname{cov}[f+g]=K_{f}+K_{g} ; \quad \operatorname{cov}[f g]=K_{f} K_{g}
$$

and thus covariance functions are closed under addition and pointwise multiplication. ${ }^{29}$ Combining this result with (3.20), we have that any polynomial of covariance functions with nonnegative coefficients forms a valid covariance. This enables us to construct infinite families of increasingly complex covariance functions from simple components.

We may use a sum of covariance functions to model a function with independent additive contributions, such as random behavior on several length scales. Precisely such a construction is illustrated in Figure 3.8. If the covariance functions are nonnegative and have roughly the same scale, the effect of addition is roughly one of logical disjunction: the sum will assume nontrivial values whenever any one of its constituents does.

Meanwhile, a product of covariance functions can loosely be interpreted in terms of logical conjunction, with function values having appreciable covariance only when every individual covariance function does. A prototypical example of this effect is a covariance function modeling functions that are "almost periodic", formed by the product of a bump-shaped isotropic covariance function such as a squared exponential (3.12) with a warped version modeling perfectly periodic functions (3.27). The former moderates the influence of the latter by driving the correlation between function values to zero for inputs that are sufficiently separated, regardless of their positions in the periodic cycle. We show a sample from such a covariance in the margin, where the length scale of the modulation term is three times the period. Optimization on a high-dimensional domain can be challenging, as we can succumb to the curse of dimensionality if we are not careful. As an example, consider optimizing an objective function on the unit cube $[0,1]^{n}$. Suppose we model this function with an isotropic covariance from the Matern family, taking the length scale to be $\ell=1 / 10$ so that ten length scales span the domain along each axis. ${ }^{30}$ This choice implies that function values on the corners of the domain would be effectively independent, as $\exp (-10)<10^{-4}(3.11)$ and $\exp (-50)$ is smaller still (3.12). If we were to demand even a modicum of confidence in these regions at termination, say by having a measurement within one length scale of every corner, we would need $2^{n}$ observations! This exponential growth in the number of observations required to cover the domain is the tyrannical curse of dimensionality.

However, compelling objectives do not tend to have so many degrees of freedom; if they did, we should perhaps give up on the idea of global optimization altogether. Rather, many authors have noted a tendency toward low intrinsic dimensionality in real-world problems: that is, most of the variation in the objective is confined to a low-dimensional subspace of the domain. This phenomenon has been noted for example in hyperparameter optimization ${ }^{31}$ and optimizing the parameters of neural networks. ${ }^{32}$ LEVINA and BICKEL suggested that "hidden" low-dimensional structure is actually a universal requirement for success on any task: ${ }^{33}$

There is a consensus in the high-dimensional data analysis community that the only reason any methods work in very high dimensions is that, in fact, the data are not truly high dimensional.

The global optimization community shares a similar consensus: typical high-dimensional objectives are not "truly" high dimensional. This intuition presents us with an opportunity: if we could only identify inherent low-dimensional structure during optimization, we could sidestep the curse of dimensionality by restricting our search accordingly.

Several strategies are available for capturing low intrinsic dimension with Gaussian process models. The general approach closely follows our discussion from the previous section: we identify some appropriate mapping from the high-dimensional domain to a lower-dimensional space, then model the objective function after composing with this embedding (3.21). This is one realization of the general class of manifold Gaussian processes,${ }^{34}$ where the sought-after manifold is low dimensional. Adopting this approach then raises the issue of identifying useful families of mappings that can suitably reduce dimension while preserving enough structure of the objective to keep optimization feasible.

\section*{Neural embeddings}

Given the success of deep learning in designing feature representations for complex, high-dimensional objects, neural embeddings - as used in curse of dimensionality

30 This is far from excessive: the domain for the marginal sampling examples in this chapter spans 15 length scales and there's just enough room for interesting behavior to emerge.

31 J. BERgSTRA and y. BENGIO (2012). Random Search for Hyper-Parameter Optimization. fournal of Machine Learning Research 13:281305 .

32 c. LI et al. (2018a). Measuring the Intrinsic Dimension of Objective Landscapes. ICLR 2018. arXiv: 1804.08838 [cs.LG].

33 E. LEVINA and P. J. BICKEL (2004). Maximum Likelihood Estimation of Intrinsic Dimension. NeurIPS 2004

34 R. CAlANDra et al. (2016). Manifold Gaussian Processes for Regression. IJCNN 2016. Figure 3.9: An objective function on a two-dimensional domain (left) with intrinsic dimension 1. The entire variation of the objective is determined on the one-dimensional linear subspace $\mathcal{Z}$ corresponding to the diagonal black line, which we can model in its inherent dimension (right).

35 A. G. WiLson et al. (2016). Deep Kernel Learning. AISTATS 2016.

neural representation learning: § 8.11, p. 198

36 J. SNOEK et al. (2015). Scalable Bayesian Optimization Using Deep Neural Networks. ICML 2015 .

cost of Gaussian process inference: § 9.1, p. 201

37 J. BERGSTRA and y. BENGIO (2012). Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research 13:281305.

38 F. VIVARELli and C. K. I. WILliams (1998). Discovering Hidden Features with Gaussian Process Regression. NeurIPS 1998.

39 z. WANG et al. (2016b). Bayesian Optimization in a Billion Dimensions via Random Embeddings. Journal of Artificial Intelligence Research $55: 361-387$.

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-18.jpg?height=483&width=1014&top_left_y=455&top_left_x=772)

the family of deep kernels ${ }^{35}$ - present a tantalizing option. Neural embeddings have shown some success in Bayesian optimization, where they can facilitate optimization over complex structured objects by providing a nice continuous latent space to work in.

SNOEK et al. demonstrated excellent performance on hyperparameter tuning tasks by interpreting the output layer of a deep neural network as a set of custom nonlinear basis functions for Bayesian linear regression, as in (3.6). ${ }^{36}$ An advantage of this particular construction is that Gaussian process inference and prediction is accelerated dramatically by adopting the linear covariance (3.6) - the cost of inference scales linearly with the number of observations, rather than cubically as in the general case.

\section*{Linear embeddings}

Another line of attack is to search for a low-dimensional linear subspace of the domain encompassing the relevant variation in inputs and model the function after projection onto that space. For an objective $f$ on a high-dimensional domain $\mathcal{X} \subset \mathbb{R}$, ${ }^{n}$ we consider models of the form

$$
f(\mathbf{x})=g(\mathbf{A x}) ; \quad \mathbf{A} \in \mathbb{R}^{k \times n}
$$

where $g: \mathbb{R}^{k} \rightarrow \mathbb{R}$ is a $(k \ll n)$-dimensional surrogate for $f$.

The simplest such approach is automatic relevance determination (3.25), where we learn separate length scales along each dimension. ${ }^{37}$ Although the corresponding linear transformation (3.23) does not reduce dimension, axes with sufficiently long length scales are effectively eliminated, as they do not have strong influence on the covariance. This can be effective when some dimensions are likely to be irrelevant, but limits us to axis-aligned subspaces only.

A more flexible option is to consider arbitrary linear transformations in the model $(3.26,3.30)$, an idea that has seen significant attention for Gaussian process modeling in general ${ }^{38}$ and for Bayesian optimization in particular. ${ }^{39}$ Figure 3.9 illustrates a simple example where a onedimensional objective function is embedded in two dimensions in a nonaxis-aligned manner. Both axes would appear important for explaining the function when using ARD, but a one-dimensional subspace suffices 

![](https://cdn.mathpix.com/cropped/2023_09_22_d531b2fdc1d079f611bbg-19.jpg?height=748&width=756&top_left_y=454&top_left_x=273)

if chosen carefully. This approach offers considerably more modeling flexibility than ARD at the expense of a $k$-fold increase in the number of parameters that must be specified. However, several algorithms have been proposed for efficiently identifying a suitable map A, ${ }^{40,41}$ and WANG et al. demonstrated success in optimizing objectives in extremely high dimension by simply searching along a random low-dimensional subspace. The authors also provided theoretical guarantees regarding the recoverability of the global optimum with this approach, assuming the hypothesis of low intrinsic dimensionality holds.

If more flexibility is desired, we may represent an objective function as a sum of contributions on multiple relevant linear subspaces:

$$
f(\mathbf{x})=\sum_{i} g_{i}\left(\mathbf{A}_{i} \mathbf{x}\right)
$$

This decomposition is similar in spirit to the classical family of generalized additive models, ${ }^{42}$ where the linear maps can be arbitrary and of variable dimension. If we assume the additive components in (3.31) are independent, each with Gaussian process prior $\mathcal{G} \mathcal{P}\left(g_{i} ; \mu_{i}, K_{i}\right)$, then the resulting model for $f$ is a Gaussian process with additive moments (3.29):

$$
\mu(\mathbf{x})=\sum_{i} \mu_{i}\left(\mathbf{A}_{i} \mathbf{x}\right) ; \quad K\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\sum_{i} K_{i}\left(\mathbf{A}_{i} \mathbf{x}, \mathbf{A}_{i} \mathbf{x}^{\prime}\right) .
$$

Several specific schemes have been proposed for building such decompositions. One convenient approach is to partition the coordinates of the input into disjoint groups and add a contribution defined on each subset. ${ }^{434}$ Figure 3.10 shows an example, where a two-dimensional objective is the sum of independent axis-aligned components. We might use such a model when every feature of the input is likely to be relevant but only through interaction with a limited number of additional variables.
40 J. DJOLONGA et al. (2013). High-Dimensional Gaussian Process Bandits. NeurIPS 2013.

41 R. GARNETT et al. (2014). Active Learning of Linear Embeddings for Gaussian Processes. UAI 2014 .

42 T. HASTIE and R. TIBSHIRANI (1986). Generalized Additive Models. Statistical Science 1(3): 297-318.

43 K. KANDASAmy et al. (2015). High Dimensional Bayesian Optimisation and Bandits via Additive Models. ICML 2015.

44 J. R. GARDNER et al. (2017). Discovering and Exploiting Additive Structure for Bayesian Optimization. AISTATS 2017. 45 P. Rolland et al. (2018). High-Dimensional Bayesian Optimization via Additive Models with Overlapping Groups. AISTATS 2018.

46 M. MUTNÝ and A. KRAUSE (2018). Efficient High Dimensional Bayesian Optimization with Additivity and Quadrature Fourier Features. NeurIPS 2018

47 T. N. HOANG et al. (2018). Decentralized HighDimensional Bayesian Optimization with Factor Graphs. AAAI 2018.

48 E. GILBOA et al. (2013). Scaling Multidimensional Gaussian Processes Using Projected Additive Approximations. ICML 2013.

49 C.--L. LI et al. (2016). High Dimensional Bayesian Optimization via Restricted Projection Pursuit Models. AISTATS 2016.

prior mean function: $\S 3.1$, p. 46

impact on sample path behavior: Figure 3.1, p. 46 and surrounding discussion

impact on extrapolation: Figure 3.2, p. 47 and surrounding discussion

prior covariance function: $§ 3.2$, p. 49
An advantage of a disjoint partition is that we may reduce optimization of the high-dimensional objective to separate optimization of each of its lower-dimensional components (3.31). Several other additive schemes have been proposed as well, including partitions with (perhaps sparsely) overlapping groups ${ }^{45,46,47}$ and decompositions of the general form (3.31) with arbitrary projection matrices. ${ }^{48,49}$

\subsection*{SUMMARY OF MAJOR IDEAS}

Specifying a Gaussian process entails choosing a mean and covariance function for the function of interest. As we saw in the previous chapter, the structure of these functions has important implications regarding sample path behavior, and as we will see in the next chapter, important implications regarding its ability to explain a given set of data.

In practice, the design of a Gaussian process model is usually datadriven: we establish some space of candidate models to consider, then search this space for the models providing the best explanation of available data. In this chapter we offered some guidance for the construction of models - or parametric spaces of models - as possible explanations of a given system. We will continue the discussion in the next chapter by taking up the question of assessing model quality in light of data. Below we summarize the important ideas arising in the present discussion.

- The mean function of a Gaussian process determines the expected value of function values. Although an important concern, the mean function can only affect sample path behavior through pointwise translation, and most interesting properties of sample paths are determined by the covariance function instead.

- Nonetheless, the mean function has important implications for prediction, namely, in extrapolation. When making predictions in locations poorly explained by available data - that is, locations where function value are not strongly correlated with any observation - the prior mean function effectively determines the posterior predictive mean.

- There are no restrictions on the mean function of a Gaussian process, and we are free to use any sensible choice in a given scenario. In practice, unless a better option is apparent, the mean function is usually taken to have some relatively simple parametric form, such as a constant (3.1) or a low-order polynomial (3.7). Such choices are both simple and unlikely to cause grossly undesirable extrapolatory behavior.

- When the mean function includes a linear combination of basis functions, we may exactly marginalize the coefficients under a multivariate normal prior (3.5). The result is a marginal Gaussian process where uncertainty in the linear terms of the mean is absorbed into the covariance function (3.6). As an important special case, we may marginalize the value of a constant mean (3.3) under a normal prior (3.2).

- The covariance function of a Gaussian process is critical to determining the behavior of its sample paths. To be valid, a covariance function must be symmetric and positive semidefinite. The latter condition can be difficult to guarantee for arbitrary "similarity measures," but covariance functions are closed under several natural operations, allowing us to build complex covariance functions from simple building blocks.

- In particular, sums and pointwise products of covariance functions are valid covariance functions, and by extension any polynomial expression of covariance functions with positive coefficients.

- Many common covariance functions are invariant to translation of their inputs, a property known as stationarity. An important result known as BOCHNER's theorem provides a useful representation for the space of stationary covariance functions: their Fourier transforms are symmetric, finite measures, and vice versa. This result has important implications for modeling and computation, as the Fourier representation can be much easier to work with than the covariance function itself.

- Numerous useful covariance functions are available "off-the-shelf." The family of Matérn covariances - and its limiting case the squared exponential covariance - can model functions with any desired degree of smoothness (3.11-3.14). A notable special case is the Matern covariance with $v=5 / 2(3.14)$, which has been promoted as a reasonable default

- The spectral mixture covariance (3.15) appeals to BOcHNER's theorem to provide a parametric family of covariance functions able to approximate any stationary covariance.

- Covariance functions can be modified by arbitrary scaling of function outputs (3.19) and/or arbitrary transformation of function inputs (3.21) This ability allows us to create parametric families of covariance functions with tunable behavior.

- Considering arbitrary constant scaling of function outputs gives rise to parameters known as output scales (3.20).

- Considering arbitrary dilations of function inputs gives rise to parameters known as characteristic length scales (3.22). Taking the dilation to be anisotropic introduces a characteristic length scale for each input dimension, a construction known as automatic relevance determination (ARD). With an ARD covariance, setting a given dimension's length scale very high effectively "turns off” its influence on the model.

- Nonlinear warping of function inputs is also possible. This enables us to easily build custom nonstationary covariance functions by combining a nonlinear warping with a stationary base covariance.

- Optimization can be especially challenging in high dimensions due to the curse of dimensionality. However, if an objective function has intrinsic low-dimensional structure, we can avoid some of the challenges by finding a structure-preserving mapping to a lower-dimensional space and modeling the function on the "smaller" space. This idea has repeatedly proven successful, and several general-purpose constructions are available. sums and products of covariance functions: $\S 3.4$, p. 55

stationarity: $§ 3.2$, p. 50

BOCHNER's theorem: $§ 3.2$, p. 51

the Matérn family and squared exponential covariance: $\S 3.3$, p. 51

spectral mixture covariance: $\S 3 \cdot 3$, p. 53

scaling function outputs: $§ 3.4$, p. 55 transforming function inputs: $\S 3.4$, p. 56

nonlinear warping: Figure 3.7, p. 59 and surrounding discussion

modeling functions on high-dimensional domains: $\S 3.5$, p. 61 