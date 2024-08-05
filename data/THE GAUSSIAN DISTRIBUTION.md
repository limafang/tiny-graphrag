\section*{THE GAUSSIAN DISTRIBUTION}

The Gaussian (or normal) distribution is a fundamental probability distribution in probabilistic modeling and inference. Gaussian distributions are frequently encountered in Bayesian optimization as they serve as the foundation of Gaussian processes, an infinite-dimensional extension appropriate for reasoning about unknown objective functions. In this chapter we provide a brief introduction to finite-dimensional Gaussian distributions and establish important properties referenced throughout this book. We will begin with the univariate (one-dimensional) case, then construct the multivariate (vector-valued) case via linear transformations of univariate Gaussians.

\section*{A.1 UNIVARIATE GAUSSIAN DISTRIBUTION}

The univariate Gaussian distribution on a random variable $x \in \mathbb{R}$ has two scalar parameters corresponding to its first two moments: $\mu=\mathbb{E}[x]$ specifies the mean (also median and mode) and serves as a location parameter, and $\sigma^{2}=\operatorname{var}[x]$ specifies the variance and serves as a scale parameter.

\section*{Probability density function and degenerate case}

When the variance is nonzero, the distribution has the probability density function

$$
\mathcal{N}\left(x ; \mu, \sigma^{2}>0\right)=Z^{-1} \exp \left(-\frac{1}{2} z^{2}\right),
$$

where $Z=\sqrt{2 \pi} \sigma$ is a normalization constant and $z$ is the familiar $z$-score of $x$ :

$$
z=\frac{x-\mu}{\sigma} .
$$

This PDF is illustrated in the margin. The probability density is rapidly decreasing with the magnitude of the $z$-score, with for example $99.7 \%$ of the density lying in the interval $|z| \leq 3$, or $x \in(\mu \pm 3 \sigma)$.

In the degenerate case $\sigma^{2}=0$, the distribution collapses to a point mass at the mean and a probability density function does not exist. We can express this case with the Dirac delta distribution:

$$
\mathcal{N}\left(x ; \mu, \sigma^{2}=0\right)=\delta(x-\mu) .
$$

\section*{Standard normal distribution}

The special case of zero mean and unit variance is called the standard normal distribution and enjoys a privileged role. We notate its density with the compact notation $\phi(x)=\mathcal{N}\left(x ; 0,1^{2}\right)$. The cumulative density function of the standard normal cannot be expressed in terms of elementary functions, but is such an important quantity that it also merits its

![](https://cdn.mathpix.com/cropped/2023_09_22_83e34a2b754beb037b10g-01.jpg?height=197&width=214&top_left_y=461&top_left_x=1595)

Gaussian processes: Chapters 2-4, p. 15

mean, $\mu$

variance, $\sigma^{2}$

![](https://cdn.mathpix.com/cropped/2023_09_22_83e34a2b754beb037b10g-01.jpg?height=257&width=525&top_left_y=1448&top_left_x=1371)

A univariate Gaussian probability density function $\mathcal{N}\left(x ; \mu, \sigma^{2}\right)$ as a function of the $z$-score.

degenerate case, $\sigma^{2}=0$

standard normal PDF, $\phi$ standard normal CDF, $\Phi$

expressing arbitrary PDFs and CDFs in terms of the standard normal

![](https://cdn.mathpix.com/cropped/2023_09_22_83e34a2b754beb037b10g-02.jpg?height=306&width=534&top_left_y=1212&top_left_x=153)

A standard normal PDF (dashed) and the PDF after applying the transformation $x \mapsto 1+x / \sqrt{2}$ (solid). own special notation:

$$
\Phi(y)=\operatorname{Pr}(x<y)=\int_{-\infty}^{y} \phi(x) \mathrm{d} x .
$$

We can write the PDF and CDF of an arbitrary nondegenerate Gaussian distribution $\mathcal{N}\left(x ; \mu, \sigma^{2}\right)$ in terms of the standard normal PDF and CDF by appropriately rescaling and translating arguments to their $z$-scores (A.1):

$$
p(x)=\frac{1}{\sigma} \phi\left(\frac{x-\mu}{\sigma}\right) ; \quad \operatorname{Pr}(x<y)=\Phi\left(\frac{y-\mu}{\sigma}\right),
$$

where the multiplicative factor in the PDF guarantees normalization.

\section*{Affine transformations}

The family of univariate Gaussian distributions is closed under affine transformations, which simply translate and rescale the distribution and adjust its moments accordingly. If $x$ has distribution $\mathcal{N}\left(x ; \mu, \sigma^{2}\right)$, then the transformation $\xi=a x+b$ has distribution

$$
p(\xi)=\mathcal{N}\left(\xi ; a \mu+b, a^{2} \sigma^{2}\right) .
$$

The above results allow us to interpret any univariate Gaussian distribution as a translated and scaled copy of the standard normal after applying the transformation $x \mapsto \mu+\sigma x$. This process is illustrated in the margin, where a standard normal distribution is transformed via the mapping $x \mapsto 1+x / \sqrt{2}$, resulting in a new Gaussian distribution with increased mean $\mu=1$ and decreased variance $\sigma^{2}=1 / 2$. The PDF is appropriately translated and rescaled.

\section*{A.2 MULTIVARIATE GAUSSIAN DISTRIBUTION}

The multivariate Gaussian distribution extends the univariate case to an arbitrary random vector $\mathbf{x} \in \mathbb{R}^{d}$. We will provide an explicit construction of the multivariate Gaussian distribution as the result of applying an affine transformation to independent univariate standard normal random variables, extending the properties noted at the end of the previous section.

\section*{Standard multivariate normal and construction of general case}

First, we construct the standard multivariate normal distribution, represented by a random vector $\mathrm{z} \in \mathbb{R}^{d}$ whose entries are independent standard univariate normal random variables: $p(\mathbf{z})=\prod_{i} \phi\left(z_{i}\right)$. It is clear from construction that the mean of this distribution is the zero vector and its covariance is the identity matrix:

$$
\mathbb{E}[\mathrm{z}]=\mathbf{0} ; \quad \operatorname{cov}[\mathrm{z}]=\mathbf{I},
$$

and we will denote its density with $p(\mathbf{z})=\mathcal{N}(\mathbf{z} ; \mathbf{0}, \mathbf{I})$. As before, this will serve as the basis of the general multivariate case by considering arbitrary affine transformations of this "standard" example. Suppose $\mathbf{x} \in \mathbb{R}^{d}$ is a vector-valued random variable and $\mathbf{z} \in \mathbb{R}$, $k \leq d$, is a $k$-dimensional standard multivariate normal vector. If we can write

$$
\mathbf{x}=\boldsymbol{\mu}+\Lambda \mathrm{z}
$$

for some vector $\boldsymbol{\mu} \in \mathbb{R}^{d}$ and $d \times k$ matrix $\Lambda$, then $\mathbf{x}$ has a multivariate normal distribution. We can compute its mean and covariance directly from (A.4-A.5):

$$
\mathbb{E}[\mathbf{x}]=\mu ; \quad \operatorname{cov}[\mathbf{x}]=\Lambda \Lambda^{\top}=\Sigma
$$

This property completely characterizes the distribution. As in the univariate case, we can interpret every multivariate normal distribution as an affine transformation of a (possibly lower-dimensional) standard normal vector. We again parameterize this family by its first two moments: the mean vector $\boldsymbol{\mu}$ and the covariance matrix $\Sigma$. This covariance matrix is necessarily symmetric and positive semidefinite, which means all its eigenvalues are nonnegative. We can factor any such matrix as $\Sigma=\Lambda \Lambda$, allowing us to recover the underlying transformation (A.5), although $\Lambda$ need not be unique.

\section*{Probability density function and degenerate case}

If $\Lambda$ has full rank $d$, then the range of (A.5) is all of $\mathbb{R}^{d}$ and a probability density function exists. This condition further implies that the covariance matrix $\Sigma$ is positive definite; that is, its eigenvalues are strictly positive, implying its determinant is positive and the matrix is invertible. The distribution has a probability density function in this case analogous to the univariate PDF (A.1):

$$
\mathcal{N}(\mathbf{x} ; \boldsymbol{\mu}, \Sigma)=Z^{-1} \exp \left(-\frac{1}{2} \Delta^{2}\right) .
$$

Here $Z$ again represents a normalization constant:

$$
Z=\sqrt{|2 \pi \Sigma|}=(2 \pi)^{\frac{d}{2}}|\Sigma|^{\frac{1}{2}},
$$

and $\Delta$ represents the Mahalanobis distance, a multivariate analog of the (absolute) $z$-score (A.2):

$$
\Delta^{2}=(\mathbf{x}-\boldsymbol{\mu})^{\top} \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}) .
$$

It is easy to verify that these definitions are compatible with the univariate case when $d=1$. Note in that case the condition of $\Sigma$ being positive definite reduces to the previous condition for nondegeneracy, $\sigma^{2}>0$.

The dependence of the multivariate Gaussian density on $\mathbf{x}$ is entirely through the value of the Mahalanobis distance $\Delta$. To gain some geometric insight into the probability density, we can set this value to a constant and compute isoprobability contours. In the case of the standard multivariate constructing general case via affine transformations of standard normal vectors

parameterization in terms of first two moments

mean vector and covariance matrix, $(\mu, \Sigma)$ positive semidefinite

positive definite

normalization constant, $Z$

Mahalanobis distance, $\Delta$

compatibility with univariate case Figure A.1: Isoprobability contours $\Delta=1$ for the standard bivariate normal distribution (left), a distribution with diagonal covariance, scaling the probability along the axes (middle), and a distribution with nonzero off-diagonal covariance, tilting the probability (right). The standard normal contour is shown for reference on the latter two examples.

![](https://cdn.mathpix.com/cropped/2023_09_22_83e34a2b754beb037b10g-04.jpg?height=292&width=300&top_left_y=1002&top_left_x=273)

Probability density and circular isoprobability contours of a standard bivariate normal distribution.

degenerate case, $|\Sigma|=0$

1 On this space, the PDF is similar to (A.6-A.8) but replaces the determinant with the pseudodeterminant and the inverse with the pseudoinverse. If $\Sigma=0$, the distribution is a Dirac delta on $\boldsymbol{\mu}$.

distribution of affine transformation $\mathbf{A x}+\mathbf{b}$
![](https://cdn.mathpix.com/cropped/2023_09_22_83e34a2b754beb037b10g-04.jpg?height=314&width=664&top_left_y=460&top_left_x=1121)

$\Sigma=\mathbf{I}$

![](https://cdn.mathpix.com/cropped/2023_09_22_83e34a2b754beb037b10g-04.jpg?height=321&width=318&top_left_y=456&top_left_x=763)

$\Sigma=\left[\begin{array}{ll}0.5 & 0 \\ 0 & 2\end{array}\right]$

$\Sigma=\left[\begin{array}{cc}1 & -0.9 \\ -0.9 & 1\end{array}\right]$
Gaussian distribution, the Mahalanobis distance reduces to the normal Euclidean distance, and the set of points satisfying $\Delta=c>0$ is then a sphere of radius $c$ centered at the origin - see the illustration in the margin.

We can now understand the geometry of the general case $\mathcal{N}(\mathbf{x} ; \boldsymbol{\mu}, \Sigma)$ purely in terms of the affine transformation (A.5) translating and warping the spherical distribution of the standard normal. Taking this view, the action of multiplying by $\Lambda$ warps the isoprobability contours into ellipsoids, which are then translated to the new center $\boldsymbol{\mu}$. The standard theory of linear maps then gives further insight: the principal axes of the ellipsoids are given by the eigenvectors of $\Lambda$, and their axis semilengths are given by the eigenvalues scaled by $c$. See Figure A.1 for an illustration.

The probability density function does not exist when $\Lambda$ (and thus $\Sigma$ ) is rank-deficient, as the range of $\mathbf{x}$ would then be restricted to the lower-dimensional affine subspace $\left\{\boldsymbol{\mu}+\Lambda \mathbf{z} \mid \mathbf{z} \in \mathbb{R}^{d}\right\}$ (A.5). However, it is still possible to define a probability density function in this degenerate case by restricting the support to this subspace. ${ }^{1}$ This is analogous to the degenerate univariate case (A.3), where probability was restricted to the zero-dimensional subspace containing the mean only, $\{\mu\}$.

\section*{Affine transformations}

The multivariate Gaussian distribution has a number of convenient mathematical properties, many of which follow immediately from the characterization in (A.5). First, it is obvious that any affine transformation of a multivariate normal distributed vector is also multivariate normal, as affine transformations are closed under composition. If $p(\mathbf{x})=\mathcal{N}(\mathbf{x} ; \boldsymbol{\mu}, \Sigma)$, then $\xi=\mathbf{A x}+\mathbf{b}$ has distribution

$$
p(\xi)=\mathcal{N}\left(\xi ; \mathrm{A} \boldsymbol{\mu}+\mathrm{b}, \mathrm{A} \Sigma \mathbf{A}^{\top}\right)
$$

Further, if we apply this result with the transformation

$$
\mathrm{x} \mapsto\left[\begin{array}{l}
\mathrm{I} \\
\mathrm{A}
\end{array}\right] \mathrm{x}+\left[\begin{array}{l}
\mathbf{0} \\
\mathrm{b}
\end{array}\right]=\left[\begin{array}{l}
\mathbf{x} \\
\xi
\end{array}\right]
$$

we can see that $\mathbf{x}$ and $\xi$ in fact have a joint Gaussian distribution:

$$
p(\mathbf{x}, \xi)=\mathcal{N}\left(\left[\begin{array}{l}
\mathbf{x} \\
\xi
\end{array}\right] ;\left[\begin{array}{c}
\boldsymbol{\mu} \\
\mathbf{A} \boldsymbol{\mu}+\mathbf{b}
\end{array}\right],\left[\begin{array}{rr}
\Sigma & \Sigma \mathbf{A}^{\top} \\
\mathrm{A} \Sigma & \mathrm{A} \Sigma \mathbf{A}^{\top}
\end{array}\right]\right) .
$$

\section*{Sampling}

The characterization of the multivariate normal in terms of affine transformations of standard normal random variables (A.5) also suggests a simple algorithm for drawing samples from the distribution. Given an arbitrary multivariate normal distribution $\mathcal{N}(\mathbf{x} ; \mu, \Sigma)$, we first factor the covariance as $\Sigma=\Lambda \Lambda$, where $\Lambda$ has size $d \times k$; when $\Sigma$ is positive definite, the Cholesky decomposition is the canonical choice. We now sample a $k$-dimensional standard normal vector $\mathbf{z}$ by sampling each entry independently from a univariate standard normal; routines for this task are readily available. Finally, we transform this vector appropriately to provide a sample of $\mathbf{x}: \mathbf{z} \mapsto \boldsymbol{\mu}+\Lambda \mathbf{z}=\mathbf{x}$. This procedure entails one-time $\mathcal{O}\left(d^{3}\right)$ work to compute $\Lambda$ (which can be reused), followed by $\mathcal{O}\left(d^{2}\right)$ work to produce each sample.

\section*{Marginalization}

Often we will have a vector $\mathbf{x}$ with a multivariate Gaussian distribution, but only be interested in reasoning about a subset of its entries. Suppose $p(\mathbf{x})=\mathcal{N}(\mathbf{x} ; \boldsymbol{\mu}, \Sigma)$, and partition the vector into two components: ${ }^{2}$

$$
\mathbf{x}=\left[\begin{array}{l}
\mathbf{x}_{1} \\
\mathbf{x}_{2}
\end{array}\right] .
$$

We partition the mean vector and covariance matrix in the same way:

$$
p(\mathbf{x})=\mathcal{N}\left(\left[\begin{array}{l}
\mathbf{x}_{1} \\
\mathbf{x}_{2}
\end{array}\right] ;\left[\begin{array}{l}
\boldsymbol{\mu}_{1} \\
\boldsymbol{\mu}_{2}
\end{array}\right],\left[\begin{array}{ll}
\Sigma_{11} & \Sigma_{12} \\
\Sigma_{21} & \Sigma_{22}
\end{array}\right]\right) .
$$

Now writing the subvector $\mathbf{x}_{1}$ as $\mathbf{x}_{1}=[\mathbf{I}, 0] \mathbf{x}$ and applying the affine property (A.9), we have:

$$
p\left(\mathbf{x}_{1}\right)=\mathcal{N}\left(\mathbf{x}_{1} ; \boldsymbol{\mu}_{1}, \Sigma_{11}\right) .
$$

That is, to derive the marginal distribution of $\mathbf{x}_{1}$ we simply pick out the corresponding entries of $\boldsymbol{\mu}$ and $\Sigma$.

\section*{Conditioning}

Multivariate Gaussian distributions are also closed under conditioning on the values of given entries. Suppose again that $p(\mathbf{x})=\mathcal{N}(\mathbf{x} ; \boldsymbol{\mu}, \Sigma)$ and partition $\mathbf{x}, \boldsymbol{\mu}$, and $\Sigma$ as before (A.11-A.12). Suppose now that we learn the exact value of the subvector $x_{2}$. The posterior on the remaining entries $p\left(\mathbf{x}_{1} \mid \mathbf{x}_{2}\right)$ remains Gaussian, with distribution

$$
p\left(\mathbf{x}_{1} \mid \mathbf{x}_{2}\right)=\mathcal{N}\left(\mathbf{x}_{1} ; \boldsymbol{\mu}_{1 \mid 2}, \Sigma_{11 \mid 2}\right) .
$$

joint distribution with affine transformations

2 We can first permute $\mathbf{x}$ if required; as a linear transformation, this will simply permute the entries of $\boldsymbol{\mu}$ and the rows/columns of $\Sigma$.
![](https://cdn.mathpix.com/cropped/2023_09_22_83e34a2b754beb037b10g-05.jpg?height=650&width=528&top_left_y=1536&top_left_x=1369)

A bivariate Gaussian PDF $p\left(x_{1}, x_{2}\right)$ (top) and the Gaussian marginal $p\left(x_{1}\right)$ (bottom) (A.13). 
![](https://cdn.mathpix.com/cropped/2023_09_22_83e34a2b754beb037b10g-06.jpg?height=784&width=530&top_left_y=474&top_left_x=160)

The PDFs of a bivariate Gaussian $p\left(x_{1}, x_{2}\right)$ (top) and the conditional distribution $p\left(x_{1} \mid x_{2}\right)$ given the value of $x_{2}$ marked by the dashed line (bottom) (A.14). The prior marginal distribution $p\left(x_{1}\right)$ is shown for reference; the observation decreased both the mean and standard deviation.

independent case
The posterior mean and covariance take the form of updates to the prior moments in light of the revealed information:

$$
\boldsymbol{\mu}_{1 \mid 2}=\boldsymbol{\mu}_{1}+\Sigma_{12} \Sigma_{22}^{-1}\left(\mathbf{x}_{2}-\boldsymbol{\mu}_{2}\right) ; \quad \Sigma_{11 \mid 2}=\Sigma_{11}-\Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21} .
$$

The mean is adjusted by an amount dependent on

1. the covariance between $\mathbf{x}_{1}$ and $\mathbf{x}_{2}, \Sigma_{12}$,

2. the uncertainty in $\mathrm{x}_{2}, \Sigma_{22}$, and

3. the deviation of the observed values from the prior mean, $\left(\mathbf{x}_{2}-\boldsymbol{\mu}_{2}\right)$.

Similarly, the uncertainty in $\mathbf{x}_{1}, \Sigma_{11}$, is reduced by an amount dependent on factors 1-2. Notably, the correction to the covariance matrix does not depend on the observed values. Note that if $\mathbf{x}_{1}$ and $\mathbf{x}_{2}$ are independent, then $\Sigma_{12}=\mathbf{0}$, and conditioning does not alter the distribution of $\mathbf{x}_{1}$.

\section*{Sums of normal vectors}

Suppose $\mathbf{x}$ and $\mathbf{y}$ are $d$-dimensional random vectors with joint multivariate normal distribution

$$
p(\mathbf{x}, \mathbf{y})=\mathcal{N}\left(\left[\begin{array}{l}
\mathbf{x} \\
\mathbf{y}
\end{array}\right] ;\left[\begin{array}{l}
\boldsymbol{\mu} \\
\boldsymbol{v}
\end{array}\right],\left[\begin{array}{ll}
\Sigma & \mathbf{P} \\
\mathbf{P} & \mathbf{T}
\end{array}\right]\right) .
$$

Then recognizing their $\operatorname{sum} \mathbf{z}=\mathbf{x}+\mathbf{y}=[\mathbf{I}, \mathbf{I}][\mathbf{x}, \mathbf{y}]^{\top}$ as a linear transformation and applying (A.9), we have:

$$
p(\mathbf{z})=\mathcal{N}(\mathbf{z} ; \boldsymbol{\mu}+\boldsymbol{v}, \Sigma+2 \mathbf{P}+\mathbf{T}) .
$$

When $\mathbf{x}$ and $\mathbf{y}$ are independent, $\mathbf{P}=\mathbf{0}$, and this simplifies to

$$
p(\mathbf{z})=\mathcal{N}(\mathbf{z} ; \boldsymbol{\mu}+\boldsymbol{v}, \Sigma+\mathbf{T}),
$$

where the moments simply add.

\section*{Differential entropy}

The differential entropy of a multivariate normal random variable $\mathbf{x}$ with distribution $p(\mathbf{x})=\mathcal{N}(\mathbf{x} ; \boldsymbol{\mu}, \Sigma)$, expressed in nats, is

$$
H[\mathbf{x}]=\frac{1}{2} \log |2 \pi e \Sigma| .
$$

In the univariate case $p(x)=\mathcal{N}\left(x ; \mu, \sigma^{2}\right)$, this reduces to

$$
H[x]=\frac{1}{2} \log 2 \pi e \sigma^{2}
$$

Sequences of normal random variables

If $\left\{\mathbf{x}_{i}\right\}$ is a sequence of normal random variables with means $\left\{\boldsymbol{\mu}_{i}\right\}$ and covariances $\left\{\Sigma_{i}\right\}$ converging respectively to finite limits $\boldsymbol{\mu}_{i} \rightarrow \boldsymbol{\mu}$ and $\Sigma_{i} \rightarrow \Sigma$, then the sequence converges in distribution to a normal random variable $\mathbf{x}$ with mean $\boldsymbol{\mu}$ and covariance $\Sigma$. 

\section*{METHODS FOR APPROXIMATE BAYESIAN INFERENCE}

In Bayesian optimization we occasionally face intractable posterior distributions that must be approximated before we can proceed. The Laplace approximation and Gaussian expectation propagation are two workhorses of approximate Bayesian inference, and at least one will suffice in most scenarios. Both result in Gaussian approximations to the posterior, especially convenient when working with Gaussian processes.

Consider a vector-valued random variable $\mathrm{x} \in \mathbb{R}^{d}$ with arbitrary prior distribution $p(\mathbf{x})$. Suppose we obtain information $\mathcal{D}$, yielding an intractable posterior

$$
p(\mathbf{x} \mid \mathcal{D})=Z^{-1} p(\mathbf{x}) p(\mathcal{D} \mid \mathbf{x})
$$

that we wish to approximate. The Laplace approximation is based on approximating the logarithm of the unnormalized posterior:

$$
\Psi(\mathbf{x})=\log p(\mathbf{x})+\log p(\mathcal{D} \mid \mathbf{x})
$$

with a Taylor expansion around its maximum:

$$
\hat{\mathbf{x}}=\arg \max \Psi(\mathbf{x}) .
$$

Taking a second-order Taylor expansion around this point yields

$$
\Psi(\mathbf{x}) \approx \Psi(\hat{\mathbf{x}})-\frac{1}{2}(\mathbf{x}-\hat{\mathbf{x}})^{\top} \mathbf{H}(\mathbf{x}-\hat{\mathbf{x}}),
$$

where $\mathbf{H}$ is the Hessian of the negative log posterior evaluated at $\hat{\mathbf{x}}$ :

$$
\mathbf{H}=-\frac{\partial^{2} \Psi}{\partial \mathbf{x} \partial \mathbf{x}^{\top}}(\hat{\mathbf{x}}) .
$$

Note the first-order term vanishes as we are expanding around a maximum. Exponentiating, we derive an approximation to the unnormalized posterior:

$$
p(\mathbf{x} \mid \mathcal{D}) \propto \exp \Psi(\mathbf{x}) \approx \exp \Psi(\hat{\mathbf{x}}) \exp \left(-\frac{1}{2}(\mathbf{x}-\hat{\mathbf{x}})^{\top} \mathbf{H}(\mathbf{x}-\hat{\mathbf{x}})\right) .
$$

We recognize this as proportional to a Gaussian distribution, yielding a normal approximate posterior:

$$
p(\mathbf{x} \mid \mathcal{D}) \approx q(\mathbf{x})=\mathcal{N}\left(\mathbf{x} ; \hat{\mathbf{x}}, \mathbf{H}^{-1}\right) .
$$

Through some accounting when normalizing (B.1), the Laplace approximation also gives an approximation to the normalizing constant $Z$ :

$$
Z \approx \hat{Z}_{\mathrm{LA}}=(2 \pi)^{\frac{d}{2}}|\mathbf{H}|^{-\frac{1}{2}} \exp \Psi(\hat{\mathbf{x}}) .
$$

This material has been published by Cambridge University Press as Bayesian Optimization This version is free to view and download for personal use only. Not for redistribution, resale, or use in derivative works. (CRoman Garnett 2023. https: //bayesoptbook. com/ unnormalized $\log$ posterior, $\Psi$

maximum a posteriori point, $\hat{\mathbf{x}}$

Hessian of negative log posterior, $\mathbf{H}$

Laplace approximation to posterior

Laplace approximation to normalizing constant, $\hat{Z}_{\mathrm{LA}}$ true distribution

unnormalized Laplace approximation

![](https://cdn.mathpix.com/cropped/2023_09_22_83e34a2b754beb037b10g-08.jpg?height=405&width=788&top_left_y=460&top_left_x=997)

Figure B.1: A Laplace approximation to a one-dimensional posterior distribution. The left panel shows the Laplace approximation before normalization, and the right panel afterwards.

1 T. P. MINKA (2001). A Family of Algorithms for Approximate Bayesian Inference. Ph.D. thesis. Massachusetts Institute of Technology.

2 J. P. cunningham et al. (2011). Gaussian Probabilities and Expectation Propagation. arXiv: 1111.6832 [stat.ML].

3 These scalar products often simply pick out single entries of $\xi$, in which case we can slightly simplify notation. However, we sometimes require this more general formulation.
The Laplace approximation procedure is illustrated in Figure B.1, where we show the approximate posterior both before and after normalization. The posterior density is an excellent local approximation around the maximum but is not a great global fit as a significant fraction of the true posterior mass is ignored. However, the Laplace approximation is remarkably simple and general and is sometimes the only viable approximation scheme.

\section*{B.2 GAUSSIAN EXPECTATION PROPAGATION}

Expectation propagation (EP) is a technique for approximate Bayesian inference that enjoys some use in Bayesian optimization. We will give a brief and incomplete introduction that should nonetheless suffice for common applications in this context. A complete introduction can be found in MINKA's thesis, ${ }^{1}$ and CUNNINGHAM et al. provide in-depth advice regarding efficient and stable computation for the rank-one case we consider here. ${ }^{2}$

Consider a multivariate Gaussian random variable $\xi$ with distribution

$$
p(\xi)=\mathcal{N}\left(\xi ; \mu_{0}, \Sigma_{0}\right)
$$

Suppose we obtain information $\mathcal{D}$ about $\xi$ in the form of a collection of factors, each of which specifies the likelihood of a scalar product $x=\mathbf{a}^{\top} \xi$ associated with that factor. ${ }^{3}$ We consider the posterior distribution

$$
p(\xi \mid \mathcal{D})=Z^{-1} p(\xi) \prod_{i} t_{i}\left(x_{i}\right)
$$

where $i$ th factor $t_{i}$ informs our belief about $x_{i}=\mathbf{a}_{i}^{\top} \xi$. Unfortunately, this posterior is intractable except the notable case when all factors are Gaussian.

Gaussian expectation propagation proceeds by replacing each of the factors with an unnormalized Gaussian distribution:

$$
t_{i}\left(x_{i}\right) \approx \tilde{t}_{i}\left(x_{i}\right)=\tilde{Z}_{i} \mathcal{N}\left(x_{i} ; \tilde{\mu}_{i}, \tilde{\sigma}_{i}^{2}\right) .
$$

Here $\left(\tilde{Z}_{i}, \tilde{\mu}_{i}, \tilde{\sigma}_{i}^{2}\right)$ are called site parameters for the approximate factor $\tilde{t}_{i}$, which we may design to optimize the fit. We will consider this issue further shortly. Given arbitrary site parameters for each factor, the resulting approximation to (B.3) is

$$
p(\xi \mid \mathcal{D}) \approx q(\xi)=\hat{Z}_{\mathrm{EP}}^{-1} p(\xi) \prod_{i} \tilde{t}_{i}\left(x_{i}\right) .
$$

As a product of Gaussians, the approximate posterior is also Gaussian:

$$
q(\xi)=\mathcal{N}(\xi ; \mu, \Sigma)
$$

with parameters: ${ }^{4}$

$$
\boldsymbol{\mu}=\Sigma\left(\Sigma_{0}^{-1} \boldsymbol{\mu}_{0}+\sum_{i} \frac{\tilde{\mu}_{i}}{\tilde{\sigma}_{i}^{2}} \mathbf{a}_{i}\right) ; \quad \Sigma=\left(\Sigma_{0}^{-1}+\sum_{i} \frac{1}{\tilde{\sigma}_{i}^{2}} \mathbf{a}_{i} \mathbf{a}_{i}^{\top}\right)^{-1}
$$

Gaussian EP also yields an approximation of the normalizing constant $Z$, if desired:

$$
Z \approx \hat{Z}_{\mathrm{EP}}=\frac{\mathcal{N}(\mathbf{0} ; \boldsymbol{\mu}, \Sigma)}{\mathcal{N}\left(\mathbf{0} ; \boldsymbol{\mu}_{0}, \Sigma_{0}\right)} \prod_{i} \tilde{Z}_{i} \mathcal{N}\left(0 ; \tilde{\mu}_{i}, \tilde{\sigma}_{i}^{2}\right)
$$

What remains to be determined is an effective means of choosing the site parameters to maximize the approximation fidelity. One reasonable goal would be to minimize the Kullback-Leibler (KL) divergence between the true and approximate distributions; for our Gaussian approximation (B.5), this is achieved through moment matching. ${ }^{2}$ Unfortunately, determining the moments of the true posterior (B.3) may be difficult, so expectation propagation instead matches the marginal moments for each of the $\left\{x_{i}\right\}$, approximately minimizing the KL divergence. This is accomplished through an iterative procedure where we repeatedly sweep over each of the approximate factors and refine its parameters until convergence.

We initialize all site parameters to $\left(\tilde{Z}, \tilde{\mu}, \tilde{\sigma}^{2}\right)=(1,0, \infty)$; with these choices the approximate factors drop away, and our initial approximation is simply the prior: $(\boldsymbol{\mu}, \Sigma)=\left(\boldsymbol{\mu}_{0}, \Sigma_{0}\right)$. Now we perform a series of updates for each of the approximate factors in turn. These updates take a convenient general form, and we will drop factor index subscripts below to simplify notation.

Let $\tilde{t}(x)=\tilde{Z} \mathcal{N}\left(x ; \tilde{\mu}, \tilde{\sigma}^{2}\right)$ be an arbitrary factor in our approximation (B.4). The idea behind expectation propagation is to drop this factor from the approximation entirely, forming the cavity distribution:

$$
\bar{q}(\xi)=\frac{q(\xi)}{\tilde{t}(x)}
$$

and replace it with the true factor $t(x)$, forming the tilted distribution $\bar{q}(\xi) t(x)$. The tilted distribution is closer to the true posterior (в.3) as the factor in question is no longer approximated. We now adjust the site site parameters, $\left(\tilde{Z}, \tilde{\mu}, \tilde{\sigma}^{2}\right)$

Gaussian EP approximate posterior, $q(\xi)$

4 The updated covariance incorporates only a series of rank-one updates, which can be applied using the Sherman-Morrison formula.

Gaussian EP parameters, $(\boldsymbol{\mu}, \Sigma)$

Gaussian EP approximation to normalizing constant, $\hat{Z}_{\mathrm{EP}}$

setting site parameters

expectation propagation approximately minimizes KL divergence

site parameter initialization

cavity distribution, $\bar{q}$

tilted distribution Figure B.2: A Gaussian EP approximation to the distribution in Figure B.1.

5 M. SEeger (2008). Expectation Propagation for Exponential Families. Technical report. University of California, Berkeley.
6 т. MINKA (2008). EP: A Quick Reference.

![](https://cdn.mathpix.com/cropped/2023_09_22_83e34a2b754beb037b10g-10.jpg?height=312&width=795&top_left_y=478&top_left_x=993)

parameters to minimize the KL divergence between the tilted distribution and the new approximation $q^{\prime}(\xi)=\bar{q}(\xi) \tilde{t}^{\prime}(x)$ :

$$
\left(\tilde{Z}, \tilde{\mu}, \tilde{\sigma}^{2}\right)=\arg \min D_{\mathrm{KL}}\left[\bar{q}(\xi) t(x) \| q^{\prime}(\xi)\right]
$$

by matching zeroth, first, and second moments.

Because Gaussian distributions are closed under marginalization, we can simplify this procedure by manipulating only marginal distributions for $x$ rather than the full joint distribution. ${ }^{5}$ The marginal belief about $x=\mathbf{a}^{\top} \xi$ in our current approximation (B.5) is:

$$
q(x)=\mathcal{N}\left(x ; \mu, \sigma^{2}\right) ; \quad \mu=\mathbf{a}_{i}^{\top} \boldsymbol{\mu} ; \quad \sigma^{2}=\mathbf{a}_{i}^{\top} \Sigma \mathbf{a}_{i} .
$$

By dividing by the approximate factor $\tilde{t}(\xi)$, we arrive at the marginal cavity distribution, which is Gaussian:

$$
\bar{q}(x)=\mathcal{N}\left(x ; \bar{\mu}, \bar{\sigma}^{2}\right) ; \quad \bar{\mu}=\bar{\sigma}^{2}\left(\mu \sigma^{-2}-\tilde{\mu} \tilde{\sigma}^{-2}\right) ; \quad \bar{\sigma}^{2}=\left(\sigma^{-2}-\tilde{\sigma}^{-2}\right)^{-1}(\text { в.7 })
$$

Consider the zeroth moment of the marginal tilted distribution:

$$
Z=\int t(x) \mathcal{N}\left(x ; \bar{\mu}, \bar{\sigma}^{2}\right) \mathrm{d} x
$$

this quantity clearly depends on the cavity parameters $\left(\bar{\mu}, \bar{\sigma}^{2}\right)$. If we define

$$
\alpha=\frac{\partial \log Z}{\partial \bar{\mu}} ; \quad \beta=\frac{\partial \log Z}{\partial \bar{\sigma}^{2}} ;
$$

and an auxiliary variable $\gamma=\left(\alpha^{2}-2 \beta\right)^{-1}$ then we may achieve the desired moment matching by updating the site parameters to: ${ }^{6}$

$$
\tilde{\mu}=\bar{\mu}+\alpha \gamma ; \quad \tilde{\sigma}^{2}=\gamma-\bar{\sigma}^{2} ; \quad \tilde{Z}=Z \sqrt{2 \pi} \sqrt{\tilde{\sigma}^{2}+\bar{\sigma}^{2}} \exp \left(\frac{1}{2} \alpha^{2} \gamma\right) .
$$

This completes our update for the chosen factor; the full EP procedure repeatedly updates each factor in this manner until convergence. The result of Gaussian expectation propagation for the distribution from Figure B.1 is shown in Figure B.2. The fit is good and reflects the more global nature of the expectation propagation scheme achieved through moment matching rather than merely maximizing the posterior.

A convenient aspect of expectation propagation is that incorporating a new factor only requires computing the zeroth moment against an arbitrary normal distribution (в.8) and the partial derivatives in (в.9). We provide these computations for several useful factor types below. factor, $t(x)$

tilted distribution

![](https://cdn.mathpix.com/cropped/2023_09_22_83e34a2b754beb037b10g-11.jpg?height=263&width=802&top_left_y=545&top_left_x=267)

Gaussian EP approximation

![](https://cdn.mathpix.com/cropped/2023_09_22_83e34a2b754beb037b10g-11.jpg?height=277&width=802&top_left_y=535&top_left_x=1095)

Figure B.3: A Gaussian EP approximation to a one-dimensional normal distribution truncated at $a$.
![](https://cdn.mathpix.com/cropped/2023_09_22_83e34a2b754beb037b10g-11.jpg?height=428&width=856&top_left_y=917&top_left_x=268)

Truncating a variable

A common use of expectation propagation in Bayesian optimization is to approximately constrain a Gaussian random variable $x$ to be less than a threshold $a$. We may capture this information by a single factor $t(x)=[x<a]$. In the context of expectation propagation, we must consider the normalizing constant of the tilted distribution, which is a truncated normal:

$$
Z=\int[x<a] \mathcal{N}\left(x ; \bar{\mu}, \bar{\sigma}^{2}\right) \mathrm{d} x=\Phi(z) ; \quad z=\frac{a-\bar{\mu}}{\bar{\sigma}} .
$$

The required quantities for an expectation propagation update are now (B.10):

$$
\alpha=-\frac{\phi(z)}{\Phi(z) \bar{\sigma}} ; \quad \beta=\frac{z \alpha}{2 \bar{\sigma}} ; \quad \gamma=-\frac{\bar{\sigma}}{\alpha}\left(\frac{\phi(z)}{\Phi(z)}+z\right)^{-1}
$$

A Gaussian EP approximation to a truncated normal distribution is illustrated in Figure (в.3). The fit is good, but not perfect: approximately $5 \%$ of its mass exceeds the threshold. This inaccuracy is the price of approximation.

We may also apply this approach to approximately condition our belief on $\xi$ on one entry being dominated by another: $\xi_{i}<\xi_{j}$. Consider the vector a where $a_{i}=1, a_{j}=-1$ and all other entries are zero. Then $x=\mathbf{a}^{\top} \xi=\xi_{i}-\xi_{j}$. The condition is now equivalent to $[x<0]$, and we can proceed as outlined above. This approximation is illustrated for a bivariate normal in Figure B.4; again the fit appears reasonable. conditioning on one variable being greater than another factor, $t(x)$

tilted distribution

![](https://cdn.mathpix.com/cropped/2023_09_22_83e34a2b754beb037b10g-12.jpg?height=265&width=797&top_left_y=547&top_left_x=161)

Gaussian EP approximation

![](https://cdn.mathpix.com/cropped/2023_09_22_83e34a2b754beb037b10g-12.jpg?height=280&width=802&top_left_y=537&top_left_x=981)

Figure B.5: A Gaussian EP approximation to a one-dimensional normal distribution truncated at an unknown threshold $a$ with the marked $95 \%$ credible interval.

\section*{Truncation at an uncertain threshold}

A sometimes useful extension of the above is to consider truncation at an uncertain threshold $a$. Suppose we have a Gaussian belief about $a$ :

$$
p(a)=\mathcal{N}\left(a ; \mu, \sigma^{2}\right) .
$$

Integrating the hard truncation factor $[x<a]$ against this belief yields the following "soft truncation" factor:

$$
t(x)=\int[x<a] p(a) \mathrm{d} a=\Phi\left(\frac{\mu-x}{\sigma}\right) .
$$

We consider again the normalizing constant of the tilted distribution:

$$
Z=\int \Phi\left(\frac{\mu-x}{\sigma}\right) \mathcal{N}\left(x ; \bar{\mu}, \bar{\sigma}^{2}\right) \mathrm{d} x=\Phi(z) ; \quad z=\frac{\mu-\bar{\mu}}{\sqrt{\sigma^{2}+\bar{\sigma}^{2}}} .
$$

Defining $s=\sqrt{\sigma^{2}+\bar{\sigma}^{2}}$, we may compute:

$$
\alpha=-\frac{\phi(z)}{\Phi(z) s} ; \quad \beta=\frac{z \alpha}{2 s} ; \quad \gamma=-\frac{s}{\alpha}\left(\frac{\phi(z)}{\Phi(z)}+z\right)^{-1}
$$

The hard truncation formulas above may be interpreted as a special case of this result by setting $\left(\mu, \sigma^{2}\right)=(a, 0)$. This procedure is illustrated in Figure B.5, where we softly truncate a one-dimensional Gaussian distribution. 

\section*{GRADIENTS}

Under mild continuity assumptions, for Gaussian processes conditioned with exact inference, we may compute the gradient of both the log marginal likelihood with respect to model hyperparameters and of the posterior predictive moments with respect to observation location. The former aids in maximizing or sampling from the model posterior, and the latter in maximizing acquisition functions derived from the predictive distribution. In modern software these gradients are often computed via automatic differentiation, but we present their functional forms here to offer insight into their behavior.

We will consider a function $f: \mathbb{R}^{d} \rightarrow \mathbb{R}$ with distribution $\mathcal{G} \mathcal{P}(f ; m, K)$, observed with independent (but possibly hetereoskedastic) additive Gaussian noise:

$$
p\left(y \mid x, \phi, \sigma_{n}\right)=\mathcal{N}\left(y ; \phi, \sigma_{n}^{2}\right),
$$

where the noise scale $\sigma_{n}$ may optionally depend on $x$. We will notate the prior moment and noise scale functions with:

$$
m(x ; \boldsymbol{\theta}) ; \quad K\left(x, x^{\prime} ; \boldsymbol{\theta}\right) ; \quad \sigma_{n}(x ; \boldsymbol{\theta}),
$$

and assume these are differentiable with respect to observation location and any parameters they may have. In a slight abuse of notation, we will use $\boldsymbol{\theta}$ to indicate a vector collating the values of all hyperparameters of this model.

We will consider an arbitrary set of observations $\mathcal{D}=(\mathbf{x}, \mathbf{y})$. We will write the prior moments of $\phi=f(\mathrm{x})$ and the noise covariance associated with these observations as:

$$
\boldsymbol{\mu}=m(\mathbf{x} ; \boldsymbol{\theta}) ; \quad \Sigma=K(\mathbf{x}, \mathbf{x} ; \boldsymbol{\theta}) ; \quad \mathrm{N}=\operatorname{diag} \sigma_{n}^{2}(\mathbf{x} ; \boldsymbol{\theta}),
$$

all of which have implicit dependence on the hyperparameters. It will also be useful to introduce notation for two repeating quantities:

$$
\mathrm{V}=\operatorname{cov}[\mathbf{y} \mid \mathbf{x}, \boldsymbol{\theta}]=\Sigma+\mathrm{N} ; \quad \boldsymbol{\alpha}=\mathrm{V}^{-1}(\mathbf{y}-\boldsymbol{\mu}) .
$$

\section*{C.1 GRADIENT OF LOG MARGINAL LIKELIHOOD}

The log marginal likelihood of the data is (4.8):

$$
\mathcal{L}(\boldsymbol{\theta})=\log p(\mathbf{y} \mid \mathbf{x}, \boldsymbol{\theta})=-\frac{1}{2}\left[\boldsymbol{\alpha}^{\top}(\mathbf{y}-\boldsymbol{\mu})+\log |\mathbf{V}|+n \log 2 \pi\right] .
$$

The partial derivatives with respect to mean function parameters have the form:

$$
\frac{\partial \mathcal{L}}{\partial \theta}=\boldsymbol{\alpha}^{\top} \frac{\partial \boldsymbol{\mu}}{\partial \theta},
$$

and partial derivatives with respect to covariance function and likelihood parameters (that is, the parameters of $\mathrm{V}$ ) take the form:

$$
\frac{\partial \mathcal{L}}{\partial \theta}=\frac{1}{2}\left[\boldsymbol{\alpha}^{\top} \frac{\partial \mathbf{V}}{\partial \theta} \boldsymbol{\alpha}-\operatorname{tr}\left[\mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \theta}\right]\right] .
$$

This material has been published by Cambridge University Press as Bayesian Optimization This version is free to view and download for personal use only. Not for redistribution, this picks up the discussion from $\S 8.2$, p. 163

1 w. ScotT et al. (2011). The Correlated Knowledge Gradient for Simulation Optimization of Continuous Parameters Using Gaussian Process Regression. SIAM fournal on Optimization 21(3):996-1026.

gradient of endpoints with respect to $\mathbf{a}, \mathbf{b}$

gradient of $g$ with respect to $\mathbf{a}$ and $\mathbf{b}$
C.2 GRADIENT OF PREDICTIVE DISTRIBUTION WITH RESPECT TO LOCATION

For a given observation location $x$, let us define the vectors

$$
\mathbf{k}=K(\mathbf{x}, x) ; \quad \boldsymbol{\beta}=\mathbf{V}^{-1} \mathbf{k}
$$

The posterior moments of $\phi=f(x)$ are (2.19):

$$
\mu=m(x)+\boldsymbol{\alpha}^{\top} \mathbf{k} ; \quad \sigma^{2}=K(x, x)-\boldsymbol{\beta}^{\top} \mathbf{k},
$$

and the partial derivatives of these moments with respect to observation location are:

$$
\frac{\partial \mu}{\partial x}=\frac{\partial m}{\partial x}-\boldsymbol{\alpha}^{\top} \frac{\partial \mathbf{k}}{\partial x} ; \quad \frac{\partial \sigma^{2}}{\partial x}=\frac{\partial K(x, x)}{\partial x}-2 \boldsymbol{\beta}^{\top} \frac{\partial \mathbf{k}}{\partial x} .
$$

The predictive distribution for a noisy observation $y$ at $x$ is (8.5):

$$
p\left(y \mid x, \mathcal{D}, \sigma_{n}^{2}\right)=\mathcal{N}\left(y ; \mu, s^{2}\right) ; \quad s^{2}=\sigma^{2}+\sigma_{n}^{2},
$$

and the partial derivative of the predictive variance and standard deviation with respect to $x$ are:

$$
\frac{\partial s^{2}}{\partial x}=\frac{\partial \sigma^{2}}{\partial x}+\frac{\partial \sigma_{n}^{2}}{\partial x} ; \quad \frac{\partial s}{\partial x}=\frac{1}{2 s} \frac{\partial s^{2}}{\partial x} .
$$

C.3 GRADIENTS OF COMMON ACQUISITION FUNCTIONS

Gradient of noisy expected improvement

To aid in the optimization of expected improvement, we may also compute the gradient of $g,{ }^{1}$ and thus the gradient of expected improvement with respect to the proposed observation location $x$. This will require several applications of the chain rule due to numerous dependencies among the variables involved.

First, we note that the $\left\{c_{i}\right\}$ values defining the intervals of dominance (8.15) depend on the vectors $\mathbf{a}$ and $\mathbf{b}$. In particular, for $2 \leq i \leq n, c_{i}$ is the $z$-value where the $(i-1)$ th and $i$ th lines intersect, which occurs at

$$
c_{i}=\frac{a_{i}-a_{i-1}}{b_{i-1}-b_{i}}
$$

making the dependence explicit. For $2 \leq i \leq n$, we have:

$\frac{\partial c_{i}}{\partial a_{i}}=\frac{1}{b_{i-1}-b_{i}} ; \quad \frac{\partial c_{i}}{\partial a_{i-1}}=-\frac{\partial c_{i}}{\partial a_{i}} \quad \frac{\partial c_{i}}{\partial b_{i}}=\frac{a_{i}-a_{i-1}}{\left(b_{i}-b_{i-1}\right)^{2}} ; \quad \frac{\partial c_{i}}{\partial b_{i-1}}=-\frac{\partial c_{i}}{\partial b_{i}}$,

and at the fixed endpoints at infinity $i \in\{1, n+1\}$, we have

$$
\frac{\partial c_{i}}{\partial \mathbf{a}}=\frac{\partial c_{i}}{\partial \mathbf{b}}=\mathbf{0}{ }^{\top}
$$

Now we may compute the gradient of $g(\mathbf{a}, \mathbf{b})$ with respect to its inputs, accounting for the implicit dependence of interval endpoints on these values:

$$
\begin{aligned}
\frac{\partial g}{\partial a_{i}}=[\Phi( & \left.\left.c_{i+1}\right)-\Phi\left(c_{i}\right)\right] \\
& +\frac{\partial c_{i+1}}{\partial a_{i}}\left[a_{i}+b_{i} c_{i+1}-[i \leq n]\left(a_{i+1}+b_{i+1} c_{i+1}\right)\right] \phi\left(c_{i+1}\right) \\
& -\frac{\partial c_{i}}{\partial a_{i}}\left[a_{i}+b_{i} c_{i}-[i>1]\left(a_{i-1}+b_{i-1} c_{i}\right)\right] \phi\left(c_{i}\right) ; \\
\frac{\partial g}{\partial b_{i}}=\left[\phi\left(c_{i}\right)-\phi\left(c_{i+1}\right)\right] & \\
& +\frac{\partial c_{i+1}}{\partial b_{i}}\left[a_{i}+b_{i} c_{i+1}-[i \leq n]\left(a_{i+1}+b_{i+1} c_{i+1}\right)\right] \phi\left(c_{i+1}\right) \\
& -\frac{\partial c_{i}}{\partial b_{i}}\left[a_{i}+b_{i} c_{i}-[i>1]\left(a_{i-1}+b_{i-1} c_{i}\right)\right] \phi\left(c_{i}\right) .
\end{aligned}
$$

Here $[i>1]$ and $[i \leq n]$ represent the Iverson bracket. We will also require the gradient of the $\mathbf{a}$ and $\mathbf{b}$ vectors with respect to $x$ :

$$
\frac{\partial \mathbf{a}}{\partial x}=\left[\begin{array}{c}
0 \\
\frac{\partial \mu}{\partial x}
\end{array}\right] ; \quad \frac{\partial \mathbf{b}}{\partial x}=\frac{1}{s}\left[\frac{\partial K_{\mathcal{D}}\left(\mathbf{x}^{\prime}, x\right)}{\partial x}-\mathbf{b} \frac{\partial s}{\partial x}\right] .
$$

Note the dependence on the gradient of the predictive parameters. Finally, if we preprocess the inputs by identifying an appropriate transformation matrix $\mathbf{P}$ such that $g(\mathbf{a}, \mathbf{b})=g(\mathbf{P a}, \mathbf{P b})=g(\boldsymbol{\alpha}, \boldsymbol{\beta})(8.14)$, then the desired gradient of expected improvement is:

$$
\frac{\partial \alpha_{\mathrm{EI}}}{\partial x}=\frac{\partial g}{\partial \boldsymbol{\alpha}} \mathbf{P} \frac{\partial \mathbf{a}}{\partial x}+\frac{\partial g}{\partial \boldsymbol{\beta}} \mathbf{P} \frac{\partial \mathbf{b}}{\partial x} .
$$

\section*{Gradient of noisy probability of improvement}

We may compute the gradient of (8.23) via several applications of the chain rule, under the (mild) assumption that the endpoints $\ell$ and $u$ correspond to unique lines $\left(a_{\ell}, b_{\ell}\right)$ and $\left(a_{u}, b_{u}\right){ }^{2}$ Then we have

$$
\frac{\partial \alpha_{\mathrm{PI}}}{\partial a_{\ell}}=-\frac{\phi(\ell)}{b_{\ell}} ; \quad \frac{\partial \alpha_{\mathrm{PI}}}{\partial b_{\ell}}=-\frac{\ell \phi(\ell)}{b_{\ell}} ; \quad \frac{\partial \alpha_{\mathrm{PI}}}{\partial a_{u}}=\frac{\phi(-u)}{b_{u}} ; \quad \frac{\partial \alpha_{\mathrm{PI}}}{\partial b_{u}}=\frac{u \phi(-u)}{b_{u}},
$$

this picks up the discussion from $\S 8.3$, p. 170

2 If not, probability of improvement is not differentiable as moving one of the lines at the shared intersection will alter the probability of improvement only in favorable directions.

and we may compute the gradient with respect to the proposed observation location as

$$
\frac{\partial \alpha_{\mathrm{PI}}}{\partial x}=\frac{\partial \alpha_{\mathrm{PI}}}{\partial \mathbf{a}} \frac{\partial \mathbf{a}}{\partial x}+\frac{\partial \alpha_{\mathrm{PI}}}{\partial \mathbf{b}} \frac{\partial \mathbf{b}}{\partial x} .
$$

The gradient with respect to $\mathbf{a}$ and $\mathbf{b}$ was computed previously (c.2).

\section*{Approximating gradient via Gauss-Hermite quadrature}

We may also use numerical techniques to estimate the gradient of the acquisition function with respect to the proposed observation location:

$$
\frac{\partial \alpha}{\partial x}=\frac{\partial}{\partial x} \int \Delta(x, y) \mathcal{N}\left(y ; \mu, s^{2}\right) \mathrm{d} y .
$$

3 It is sufficient for example that both $\Delta$ and $\frac{\partial \Delta}{\partial x}$ be continuous in $x$ and $y$, and that the moments of the predictive distribution of $y$ be continuous with respect to $x$.

approximation with Gauss-Hermite quadrature

accounting for dependence of $\left\{y_{i}\right\}$ on predictive parameters

this picks up the discussion from $\S 8.6$, p. 175

4 P. Milgrom and I. SEgAL (2002). Envelope Theorems for Arbitrary Choice Sets. Econometrica $70(2): 583-601$.

5 We assume $x$ is in the interior of $\mathcal{X}$ to avoid issues with defining differentiability on the boundary. We can avoid this assumption if the Gaussian process can be extended to all of $\mathbb{R}^{d}$ This is usually possible trivially unless the prior mean or covariance function of the Gaussian process is exotic.

6 Sufficient conditions are that the prior mean and covariance functions be differentiable.
If we assume sufficient regularity, ${ }^{3}$ then we may swap the order of expectation and differentiation:

$$
\frac{\partial \alpha}{\partial x}=\int \frac{\partial}{\partial x} \Delta(x, y) \mathcal{N}\left(y ; \mu, s^{2}\right) \mathrm{d} y,
$$

reducing the computation of the gradient to Gaussian expectation.

Gauss-Hermite quadrature remains a natural choice, resulting in the approximation

$$
\frac{\partial \alpha}{\partial x} \approx \sum_{i=1}^{n} \bar{w}_{i} \frac{\partial \Delta}{\partial x}\left(x, y_{i}\right)
$$

which is simply the derivative of the estimator in (8.29). When using this estimate we must remember that the $\left\{y_{i}\right\}$ samples depend on $x$ through the parameters of predictive distribution (8.29). Accounting for this dependence, the required gradient of the marginal gain at the $i$ th integration node $z_{i}$ is:

$$
\frac{\partial \Delta}{\partial x}\left(x, y_{i}\right)=\left[\frac{\partial \Delta}{\partial x}\right]_{y}+\frac{\partial \Delta}{\partial y}\left[\frac{\partial \mu}{\partial x}+\frac{y_{i}-\mu}{s} \frac{\partial s}{\partial x}\right],
$$

where $\left[\frac{\partial \Delta}{\partial x}\right]_{y}$ indicates the partial derivative of the updated utility when the observed value $y$ is held constant.

\section*{Approximating the gradient of knowledge gradient}

Special care is needed to estimate the gradient of the knowledge gradient using numerical techniques. Inspecting (c.4), we must compute the gradient of the marginal gain

$$
\frac{\partial \Delta}{\partial x}=\frac{\partial}{\partial x} \max _{x^{\prime} \in \mathcal{X}} \mu_{\mathcal{D}^{\prime}}\left(x^{\prime}\right) .
$$

It is not clear how we can differentiate through the max operator in this expression. However, under mild assumptions we may appeal to the envelope theorem for arbitrary choice sets to proceed. ${ }^{4}$ Consider a point $x$ in the interior of the domain $\mathcal{X} \subset \mathbb{R}^{d}$, and fix an arbitrary associated observation value $y .{ }^{5}$ Take any point maximizing the updated posterior mean:

$$
x^{*} \in \underset{x^{\prime} \in \mathcal{X}}{\arg \max } \mu_{\mathcal{D}^{\prime}}\left(x^{\prime}\right) ;
$$

this point need not be unique. Assuming the updated posterior mean is differentiable, ${ }^{6}$ the envelope theorem states that the gradient of the global reward utility is equal to the gradient of the updated posterior mean evaluated at $x^{*}$ :

$$
\frac{\partial \Delta}{\partial x}=\frac{\partial}{\partial x} \mu_{\mathcal{D}^{\prime}}\left(x^{*}\right)
$$

For the knowledge gradient, and accounting for the dependence on $x$ in the locations of the $y$ samples (c.5), we may compute:

$$
\frac{\partial \Delta}{\partial x}\left(x, y_{i}\right)=\frac{y_{i}-\mu}{s^{3}}\left[s \frac{\partial K_{\mathcal{D}}}{\partial x}\left(x, x_{i}^{*}\right)-\frac{\partial s}{\partial x} K_{\mathcal{D}}\left(x, x_{i}^{*}\right)\right],
$$

where $x_{i}^{*}$ maximizes the updated posterior mean corresponding to $y_{i}$. Combining this result with (c.4) yields a Gauss-Hermite approximation to the gradient of the acquisition function. This approach requires maximizing the updated posterior mean for each sample $y_{i}$. We may appeal to standard techniques for this task, making use of the gradient (c.1)

$$
\frac{\partial \mu_{\mathcal{D}^{\prime}}}{\partial x}
$$

Gradient of predictive entropy search acquisition function

The explicit formula (8.52) and results above allow us to compute - after investing considerable tedium - the gradient of the predictive entropy search approximation to the mutual information (8.41). We begin by differentiating that estimate:

$$
\frac{\partial \alpha_{\mathrm{PES}}}{\partial x}=\frac{1}{s} \frac{\partial s}{\partial x}+\frac{1}{2 n} \sum_{i=1}^{n} \frac{1}{s_{*_{i}}^{2}} \frac{\partial s_{*_{i}}^{2}}{\partial x} .
$$

Given a fixed sample $x^{*}$ and dropping the subscript, we may compute (8.39):

$$
\frac{\partial s_{*}^{2}}{\partial x}=\frac{\partial \sigma_{*}^{2}}{\partial x}+\frac{\partial \sigma_{n}^{2}}{\partial x} .
$$

We proceed by differentiating the approximate latent predictive variance (8.52):

$$
\frac{\partial \sigma_{*}^{2}}{\partial x}=\frac{\partial \varsigma^{2}}{\partial x}-\frac{2 \varsigma^{2}-2 \rho}{\gamma}\left[\frac{\partial \varsigma^{2}}{\partial x}-\frac{\partial \rho}{\partial x}\right]+\frac{\left(\varsigma^{2}-\rho\right)^{2}}{\gamma^{2}} \frac{\partial \gamma}{\partial x},
$$

which depends on the derivatives of the expectation propagation update terms:

$$
\begin{aligned}
& \frac{\partial \gamma}{\partial x}=\frac{1}{\alpha}\left[1-\frac{(\Phi(z)-\phi(z))^{2}}{(z \Phi(z)+\phi(z))^{2}}\right]\left[\frac{\partial m}{\partial x}+z \frac{\partial \bar{\sigma}}{\partial x}\right]+\frac{2 \gamma}{\bar{\sigma}} \frac{\partial \bar{\sigma}}{\partial x} \\
& \frac{\partial \bar{\sigma}}{\partial x}=\frac{1}{2 \bar{\sigma}}\left[\frac{\partial \varsigma^{2}}{\partial x}-2 \frac{\partial \rho}{\partial x}\right]
\end{aligned}
$$

and the predictive parameters (8.51):

$$
\begin{array}{ll}
\frac{\partial m}{\partial x}=\frac{\partial \mu}{\partial x}+\left(\boldsymbol{\alpha}^{*}\right)^{\top} \frac{\partial \mathbf{k}}{\partial x} ; & \\
\frac{\partial \varsigma^{2}}{\partial x}=\frac{\partial \sigma^{2}}{\partial x}-2 \mathbf{k}^{\top} \mathbf{V}_{*}^{-1} \frac{\partial \mathbf{k}}{\partial x} ; & \frac{\partial \rho}{\partial x}=\frac{\partial k_{1}}{\partial x}-\mathbf{k}_{*}^{\top} \mathbf{V}_{*}^{-1} \frac{\partial \mathbf{k}}{\partial x}
\end{array}
$$

\section*{Gradient of OPES acquisition function}

To compute the gradient of the opes approximation, we differentiate (8.63):

$$
\frac{\partial \alpha_{\mathrm{OPES}}}{\partial x}=\frac{1}{s} \frac{\partial s}{\partial x}+\frac{1}{2} \sum_{i} \frac{w_{i}}{s_{*_{i}}^{2}} \frac{\partial s_{*_{i}}^{2}}{\partial x} .
$$

this picks up the discussion from $\S 8.8$, p. 187

this picks up the discussion from $\S 8.9$, p. 192 Given a fixed sample $f^{*}$ and dropping the subscript, we may compute (8.63):

$$
\frac{\partial s_{*}^{2}}{\partial x}=\frac{\partial \sigma_{*}^{2}}{\partial x}+\frac{\partial \sigma_{n}^{2}}{\partial x} .
$$

Finally, we differentiate (8.61):

$$
\begin{aligned}
\frac{\partial \sigma_{*}^{2}}{\partial x}= & \sigma \frac{\phi(z)}{\Phi(z)}\left[\frac{\partial \mu}{\partial x}+z \frac{\partial \sigma}{\partial x}\right]\left[1-2 \frac{\phi(z)^{2}}{\Phi(z)^{2}}-3 z \frac{\phi(z)}{\Phi(z)}-z^{2}\right] \\
& +\frac{\partial \sigma^{2}}{\partial x}\left[1-z \frac{\phi(z)}{\Phi(z)}-\frac{\phi(z)^{2}}{\Phi(z)^{2}}\right] .
\end{aligned}
$$



\section*{ANNOTATED BIBLIOGRAPHY OF APPLICATIONS}

Countless settings across science, engineering, and beyond involve free parameters that can be tuned at will to achieve some objective. However, in many cases the evaluation of a given parameter setting can be extremely costly, stymieing exhaustive exploration of the design space.

Of course, such a situation is a perfect use case for Bayesian optimization. Through careful modeling and intelligent policy design, Bayesian optimization algorithms can deliver impressive optimization performance even with small observation budgets. This capability has been demonstrated in hundreds of studies across a wide range of domains.

Here we provide a brief survey of some notable applications of Bayesian optimization. The selected references are not intended to be exhaustive (that would be impossible given the size of the source material!), but rather to be representative, diverse, and good starting points for further investigation.

\section*{CHEMISTRY AND MATERIALS SCIENCE}

At a high level, the synthesis of everything from small molecules to bulk materials proceeds in the same manner: initial raw materials are combined and subjected to suitable conditions such that they transform into a final product. Although this seems like a simple recipe, we face massive challenges in its realization:

- We usually wish that the resulting products be useful, that is, that they exhibit desirable properties. In drug discovery, for example, we seek molecules exhibiting binding activity against an identified biological target. In other settings we might seek products exhibiting favorable optical, electronic, mechanical, thermal, and/or other properties.

- The space of possible products can be enormous and the sought after properties exceptionally rare. For example, there are an estimated $10^{60}$ pharmacologically active molecules, only a tiny fraction of which might exhibit binding activity against any given biological target.

- Determining the properties of a candidate molecule or material ultimately requires synthesis and characterization in a laboratory, which can be complicated, costly, and slow.

For these reasons, exploration of molecular or material spaces can devolve into a cumbersome trial-and-error search for "needles in a haystack."

Over the past few decades, the fields of computational chemistry and computational materials science have developed sophisticated techniques for estimating chemical and material properties from simulation. As accurate simulation often requires consideration of quantum mechanical interactions, these surrogates can still be quite costly, but are nonetheless cheaper and more easily parallelized than laboratory experiments. This has enabled computer-guided exploration of large molecular and materials spaces at relatively little cost, but in many cases exhaustive

![](https://cdn.mathpix.com/cropped/2023_09_22_83e34a2b754beb037b10g-19.jpg?height=202&width=220&top_left_y=453&top_left_x=1575)

computational chemistry, computational materials science chemoinformatics molecular fingerprints

neural fingerprints

1 See $\S 8.11$, p. 199 and $\S 9.2$, p. 209 for related discussion. exploration remains untenable, and we must appeal to methods such as Bayesian optimization for guidance.

\section*{Virtual screening of molecular spaces}

The ability to estimate molecular properties using computational methods has enabled so-called virtual screening, where early stages of discovery can be performed in silico. Bayesian optimization and active search can dramatically increase the rate of discovery in this setting.

One challenge here is constructing predictive models for the properties of molecules, which are complex structured objects. The field of chemoinformatics has developed a number of molecular fingerprints intended to serve as useful feature representations for molecules when predicting chemical properties. Traditionally, these fingerprints were developed based on chemical intuition; however, numerous neural fingerprints have emerged in recent years, designed via deep representation learning on huge molecular databases. ${ }^{1}$

GRAFF, DAVID E. et al. (2021). Accelerating High-Throughput Virtual Screening through Molecular Pool-Based Active Learning. Chemical Science 12(22):7866-7881.

HERNÁNDEZ-LOBATO, JosÉ Miguel et al. (2017). Parallel and Distributed Thompson Sampling for Large-Scale Accelerated Exploration of Chemical Space. Proceedings of the 34th International Conference on Machine Learning (ICML 2017).

JiANG, SHAli et al. (2017). Efficient Nonmyopic Active Search. Proceedings of the 34 th International Conference on Machine Learning (ICML 2017).

\section*{De novo design}

A recent trend in molecular discovery has been to learn deep generative models for molecular structures and perform optimization in the (continuous) latent space of such a model. This enables de novo design, where the optimization routine can propose entirely novel structures for evaluation by feeding selected points in the latent space through the generational procedure. ${ }^{1}$

Although appealing, a major complication with this scheme is identifying a synthetic route - if one even exists! - for proposed structures. This issue deserves careful consideration when designing a system that can effectively transition from virtual screening to the laboratory.

GAO, WenhaO et al. (2020). The Synthesizability of Molecules Proposed by Generative Models. Journal of Chemical Information and Modeling 6o(12):5714-5723.

GÓMEZ-BOMBARELLI, RAFAEL et al. (2018). Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules. ACS Central Science 4(2):268-276. GRIFFITHS, RYAN-RHYS et al. (2020). Constrained Bayesian Optimization for Automatic Chemical Design Using Variational Autoencoders. Chemical Science 11(2):577-586.

KOROVINA, KSENIA et al. (2020). ChembO: Bayesian Optimization of Small Organic Molecules with Synthesizable Recommendations. Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS 202O).

\section*{Reaction optimization}

Not all applications of Bayesian optimization in chemistry take the form of optimizing chemical properties as a function of molecular structure. For example, even once a useful molecule has been identified, there may remain numerous parameters of its synthesis - reaction environment, processing parameters, etc. - that can be further optimized, seeking for example to reduce the cost and/or increase the yield of production.

SHIELDS, BENJAMIN J. et al. (2021). Bayesian Reaction Optimization as a Tool for Chemical Synthesis. Nature 590:89-96.

\section*{Conformational search}

Conformational search seeks to identify the configuration(s) of a molecule with the lowest potential energy. Even for molecules of moderate size, the space of possible configurations can be enormous and the potential energy surface can exhibit numerous local minima. Interaction with other structures such as a surface (adsorption) can further complicate the computation of potential energy, rendering the search even more difficult.

CARR, SHANE F. et al. (2017). Accelerating the Search for Global Minima on Potential Energy Surfaces Using Machine Learning. Fournal of Chemical Physics 145(15):154106.

FANG, LINCAN et al. (2021). Efficient Amino Acid Conformer Search with Bayesian Optimization. fournal of Chemical Theory and Computation 17(3):1955-1966.

PACKWOOD, DANIEL (2017). Bayesian Optimization for Materials Science. Springer-Verlag.

We may also use related Bayesian methods to map out minimalenergy pathways between neighboring minima on a potential energy surface in order to understand the intermediate geometry of a molecule as it transforms from one energetically favorable state to another.

KOISTINEN, OLLI-PEKKA et al. (2017). Nudged Elastic Band Calculations Accelerated with Gaussian Process Regression. Journal of Chemical Physics 147(15):152720. adsorption

mapping minimal-energy pathways Optimization of material properties and performance

Bayesian optimization has demonstrated remarkable success in accelerating materials design. As in molecular design, in many cases we can perform early screening in silico, using computational methods to approximate properties of interest. The literature in this space is vast, and the studies below are representative examples targeting a range of material types and properties.

ATtiA, PETER M. et al. (2020). Closed-Loop Optimization of Fast-Charging Protocols for Batteries with Machine Learning. Nature 578:397402.

FUKAZAWA, TARO et al. (2019). Bayesian Optimization of Chemical Composition: A Comprehensive Framework and Its Application to $R \mathrm{Fe}_{12}$-Type Magnet Compounds. Physical Review Materials 3(5): 053807 .

HAGHANIFAR, SAJAD et al. (2020). Discovering High-Performance Broadband and Broad Angle Antireflection Surfaces by Machine Learning. Optica $7(7): 784^{-789}$.

HERBOL, HENRY C. et al. (2018). Efficient Search of Compositional Space for Hybrid Organic-Inorganic Perovskites via Bayesian Optimization. npj Computational Materials 4(51).

Ju, SHENGHONG et al. (2017). Designing Nanostructures for Phonon Transport via Bayesian Optimization. Physical Review X 7:021024.

MIYAGAWA, SHINSUKE et al. (2021). Application of Bayesian Optimization for Improved Passivation Performance in $\mathrm{TiO}_{x} / \mathrm{SiO}_{y} / \mathrm{c}-\mathrm{Si}$ Heterostructure by Hydrogen Plasma Treatment. Applied Physics Express 14(2):025503.

NAKAMURA, KENSAKU et al. (2021). Multi-Objective Bayesian Optimization of Optical Glass Compositions. Ceramics International 47(11): $15819-15824$.

NUGRAHA, ASEP SUgIH et al. (2020). Mesoporous Trimetallic PtPdAu Alloy Films toward Enhanced Electrocatalytic Activity in Methanol Oxidation: Unexpected Chemical Compositions Discovered by Bayesian Optimization. Journal of Materials Chemistry A 8(27): 13532-13540.

OSADA, KEIICHI et al. (2020). Adaptive Bayesian Optimization for Epitaxial Growth of Si Thin Films under Various Constraints. Materials Today Communications 25:1015382.

SEKo, Atsuto et al. (2015). Prediction of Low-Thermal-Conductivity Compounds with First-Principles Anharmonic Lattice-Dynamics Calculations and Bayesian Optimization. Physical Review Letters 115(20):205901.

Structural search

A fundamental question in materials science is how the structure of a material gives rise to its material properties. However, predicting the likely structure of a material - for example by evaluating the potential energy of plausible structures and minimizing - can be extraordinarily difficult, as the number of possible configurations can be astronomical.

KIYOHARA, SHIN et al. (2016). Acceleration of Stable Interface Structure Searching Using a Kriging Approach. Fapanese fournal of Applied Physics 55(4):045502.

окAмOTO, YASUHARU (2017). Applying Bayesian Approach to Combinatorial Problem in Chemistry. fournal of Physical Chemistry $A$ 121(17):3299-3304.

TODOROVIĆ, MILICA et al. (2019). Bayesian Inference of Atomistic Structure in Functional Materials. npj Computational Materials 5(35).

Software for Bayesian optimization in chemistry and materials science

HÄSE, FLORIAN et al. (2018). Phoenics: A Bayesian Optimizer for Chemistry. ACS Central Science 4(9):1134-1145.

UENO, TSUYOSHI et al. (2016). сомBO: An Efficient Bayesian Optimization Library for Materials Science. Materials Discovery 4:18-21.

\section*{PHYSICS}

Modern physics is driven by experiments of massive scope, which have enabled the refinement of theories of the Universe on all scales. The complexity of these experiments - from data acquisition to the following analysis and inference - offers many opportunities for optimization, but the same complexity often renders optimization difficult due to large parameter spaces and/or the expense of evaluating a particular setting.

\section*{Experimental physics}

Complex physical instruments such as particle accelerators offer the capability of extraordinarily fine tuning through careful setting of their control parameters. In some cases, these parameters can be altered on-thefly during operation, resulting in a huge space of possible configurations. Bayesian optimization can help accelerate the tuning process.

DURIS, J. et al. (2020). Bayesian Optimization of a Free-Electron Laser. Physical Review Letters 124(12):124801.

ROUSSEL, RYAN et al. (2021). Multiobjective Bayesian Optimization for Online Accelerator Tuning. Physical Review Accelerators and Beams 24(6):062801.

SHALLOO, R. J. et al. (2020). Automation and Control of Laser Wakefield Accelerators Using Bayesian Optimization. Nature Communications 11:6355.

Wigley, P. B. et al. (2016). Fast Machine-Learning Online Optimization of Ultra-Cold-Atom Experiments. Scientific Reports 6:2589o. 2 This model may not be probabilistic but rather a complex simulation procedure.

\section*{forward map}

Bayesian analog

\section*{Inverse problems in physics}

The goal of an inverse problem is to determine the free parameters of a generative model ${ }^{2}$ from observations of its output. Inverse problems are pervasive in physics - many physical models feature parameters (physical constants) that cannot be determined from first principles. However, we can infer these parameters experimentally by comparing the predictions of the model with the behavior of relevant observations.

In this context, determining the predictions of the model given a setting of its parameters is called the forward map. The inverse problem is then the task of "inverting" this map: searching the parameter space for the settings in the greatest accord with observed data.

We can draw parallels here with Bayesian inference, where the forward map is characterized by the likelihood, which determines the distribution of observed data given the parameters. The Bayesian answer to the inverse problem is then encapsulated by the posterior distribution of the parameters given the data, which identifies the most plausible parameters in light of the observations.

For complex physical models, even the forward map can be exceptionally expensive to compute, making it difficult to completely explore its parameter space. For example, the forward map of a cosmological model may require simulating the evolution of an entire universe at a fine enough resolution to observe its large scale structure. Exhaustively traversing the space of cosmological parameters and comparing with observed structure would be infeasible, but progress may be possible with careful guidance. Bayesian optimization has proven useful to this end on a range of difficult inverse problems.

ILTEN, P. et al. (2017). Event Generator Tuning Using Bayesian Optimization. Fournal of Instrumentation 12(4):Po4028.

LECLERCQ, FLORENT (2018). Bayesian Optimization for Likelihood-Free Cosmological Inference. Physical Review D 98(6):063511.

ROGERS, KEIR K. et al. (2019). Bayesian Emulator Optimisation for Cosmology: Application to the Lyman-Alpha Forest. fournal of Cosmology and Astroparticle Physics 2019(2):031.

VARGAS-HERNÁNDEZ, R. A. et al. (2019). Bayesian Optimization for the Inverse Scattering Problem in Quantum Reaction Dynamics. New fournal of Physics 21(2):022001.

\section*{BIOLOGICAL SCIENCES AND ENGINEERING}

Biological systems are extraordinarily complex. Obtaining experimental measurements to shed light on the behavior of these systems can be difficult, slow, and expensive, and the resulting data can be corrupted with significant noise. Efficient experimental design is thus critical to make progress, and Bayesian optimization is a natural tool to consider.

LI, YAN et al. (2018). A Knowledge Gradient Policy for Sequencing Experiments to Identify the Structure of RNA Molecules Using a Sparse Additive Belief Model. INFORMS fournal on Computing 30(4):625786.

LORENZ, ROMY et al. (2018). Dissociating frontoparietal brain networks with neuroadaptive Bayesian optimization. Nature Communications 9:1227.

NIKITIN, ARTYOM et al. (2019). Bayesian optimization for seed germination. Plant Methods 15:43.

Inverse problems in the biological sciences

Challenging inverse problems (see above) are pervasive in biology. Effective modeling of biological systems often requires complicated, nonlinear models with numerous free parameters. Bayesian optimization can help guide the search for the parameters offering the best explanation of observed data.

DOKOOHAKI, HAMZE et al. (2018). Use of Inverse Modelling and Bayesian Optimization for Investigating the $\mathrm{T}$ Effect of Biochar on Soil Hydrological Properties. Agricultural Water Management 2018: 268-274.

THOMAS, MARCus et al. (2018). A Method for Efficient Bayesian Optimization of Self-Assembly Systems from Scattering Data. BMC Systems Biology 12:65.

ULMASOV, DONIYOR et al. (2016). Bayesian Optimization with Dimension Scheduling: Application to Biological Systems. Computer Aided Chemical Engineering 38:1051-1056.

\section*{Gene and protein design}

The tools of modern biology enable the custom design of genetic sequences and even entire proteins, which we can - at least theoretically - tailor as we see fit. It is natural to pose both gene and protein design in terms of optimizing some figure of merit over a space of alternatives. However, in either case we face an immediate combinatorial explosion in the number of possible genetic or amino acid sequences we might consider. Bayesian optimization has shown promise for overcoming this obstacle through careful experimental design. A running theme throughout recent efforts is the training of generative models for gene/protein sequences, after which we may optimize in the continuous latent space in order to sidestep the need for combinatorial optimization. ${ }^{3}$

GONZÁLEZ, JAVIER et al. (2015). Bayesian Optimization for Synthetic Gene Design. Bayesian Optimization: Scalability and Flexibility Workshop (BayesOpt 2015), Conference on Neural Information Processing Systems (NeurIPS 2015).

HIE, BRIAN L. et al. (2021). Adaptive Machine Learning for Protein Engineering. arXiv: 2106.05466 [q-bio. QM].
3 See $\S 8.11$, p. 199 and $\S 9.2$, p. 209 for related discussion. ROMERo, PHILIP A. et al. (2013). Navigating the Protein Fitness Landscape with Gaussian Processes. Proceedings of the National Academy of Sciences 110(3):E193-E2O1.

SINAI, SAM et al. (2020). A Primer on Model-Guided Exploration of Fitness Landscapes for Biological Sequence Design. arXiv: 2010.10614 [q-bio.QM].

YANG, KEVIN K. et al. (2019). Machine-Learning-Guided Directed Evolution for Protein Engineering. Nature Methods 16:687-694.

A related problem is faced in modern plant breeding, where the goal is to develop plant varieties with desirable characteristics: yield, drought or pest resistance, etc. Plant phenotyping is an inherently slow process, as we must wait for planted seeds to germinate and grow until sufficiently mature that traits can be measured. This slow turnover rate makes it impossible to fully explore the space of possible genotypes, the genetic information that (in combination with other factors including environment and management) gives rise to the phenotypes we seek to optimize. Modern plant breeding uses genetic sequencing to guide the breeding process by building models to predict phenotype from genotype and using these models in combination with large gene banks to inform the breeding process. A challenge in this approach is that the space of possible genotypes is huge, but Bayesian optimization can help accelerate the search.

TANAKA, RYOKEI et al. (2018). Bayesian Optimization for Genomic Selection: A Method for Discovering the Best Genotype among a Large Number of Candidates. Theoretical and Applied Genetics 131(1):93-105.

\section*{Biomedical engineering}

Difficult optimization problems are pervasive in biomedical engineering due to the complexity of the systems involved and the often considerable cost of gathering data, whether through studies with human subjects, complicated laboratory testing, and/or nontrivial simulation.

COLOPY, GLEN WRIGHT et al. (2018). Bayesian Optimization of Personalized Models for Patient Vital-Sign Monitoring. IEEE fournal of Biomedical and Health Informatics 22(2):301-310.

GHASSEMI, MOHAMмAD et al. (2014). Global Optimization Approaches for Parameter Tuning in Biomedical Signal Processing: A Focus of Multi-scale Entropy. Computing in Cardiology 41(12-2):993-996.

KIM, GILHWAN et al. (2021). Using Bayesian Optimization to Identify Optimal Exoskeleton Parameters Targeting Propulsion Mechanics: A Simulation Study. bioRxiv: 2021.01.14.426703.

KIM, MYUNGHEE et al. (2017). Human-in-the-loop Bayesian Optimization of Wearable Device Parameters. PLOS ONE 12(9):eo184054. OLOFSSON, SIMON et al. (2019). Bayesian Multiobjective Optimisation with Mixed Analytical and Black-Box Functions: Application to Tissue Engineering. IEEE Transactions on Biomedical Engineering 66(3): 727-739.

ROBOTICS

Robotics is fraught with difficult optimization problems. A robotic platform may have numerous tunable parameters influencing its behavior in a highly complex manner. Further, empirical evaluation can be difficult real-world experiments must proceed in real time, and there may be a considerable setup time between experiments. However, in some cases we may be able to accelerate optimization by augmenting real-world evaluation with simulation in a multifidelity setup.

BANSAL, SOMIL et al. (2017). Goal-Driven Dynamics Learning via Bayesian Optimization. Proceedings of the 56th Annual IEEE Conference on Decision and Control (CDC 2017).

CALANDRA, ROBERTO et al. (2016). Bayesian Optimization for Learning Gaits under Uncertainty: An Experimental Comparison on a Dynamic Bipedal Walker. Annals of Mathematics and Artificial Intelligence 76(1-2):5-23.

JUNGE, KAI et al. (2020). Improving Robotic Cooking Using Batch Bayesian Optimization. IEEE Robotics and Automation Letters 5(2):760765 .

MARCO, ALONSO et al. (2017). Virtual vs. Real: Trading off Simulations and Physical Experiments in Reinforcement Learning with Bayesian Optimization. Proceedings of the 2017 IEEE International Conference on Robotics and Automation (ICRA 2017).

MARTINEZ-CANTIN, RUBEN et al. (2009). A Bayesian Exploration-Exploitation Approach for Optimal Online Sensing and Planning with a Visually Guided Mobile Robot. Autonomous Robotics 27(2):93-103. RAI, AKSHARA et al. (n.d.). Using Simulation to Improve Sample-Efficiency of Bayesian Optimization for Bipedal Robots. Journal of Machine Learning Research 20(49).

\section*{Modeling for robotics}

Modeling robot performance as a function of its parameters can be difficult due to nominally high-dimensional parameter spaces and the potential for context-dependent nonstationarity.

JAQUIER, NOÉMIE et al. (2019). Bayesian Optimization Meets Riemannian Manifolds in Robot Learning. Proceedings of the 3 rd Conference on Robot Learning (CORL 2019).

MARTINEZ-CANTIN, RUBEN (2017). Bayesian Optimization with Adaptive Kernels for Robot Control. Proceedings of the 2017 IEEE International Conference on Robotics and Automation (ICRA 2017). multifidelity optimization: $\S 11.5$, p. 263 robust optimization: § 11.9, p. 277

Chapter 6: Utility Functions for Optimization,

$$
\text { p. } 109
$$

constrained optimization: $§ 11.2$, p. 249
YUAN, KAI et al. (2019). Bayesian Optimization for Whole-Body Control of High-Degree-of-Freedom Robots through Reduction of Dimensionality. IEEE Robotics and Automation Letters 4(3):22682275 .

\section*{Safe and robust optimization}

A complication faced in some robotic optimization settings is in ensuring that the evaluated parameters are both robust (that is, that performance is not overly sensitive to minor perturbations in the parameters) and safe (that is, that there is no chance of catastrophic failure in the robotic platform). We may address these concerns in the design of the optimization policy. For example, we might realize robustness by redefining the utility function in terms of the expected performance of perturbed parameters, and we might realize safety by incorporating appropriate constraints.

BERKENKAMP, FELIX et al. (2021). Bayesian Optimization with Safety Constraints: Safe and Automatic Parameter Tuning in Robotics. Machine Learning Special Issue on Robust Machine Learning. GARcíA-BARcos, JAVIER et al. (2021). Robust Policy Search for Robot Navigation. IEEE Robotics and Automation Letters 6(2):2389-2396. NOGUEIRA, JOSÉ et al. (2016). Unscented Bayesian Optimization for Safe Robot Grasping. Proceedings of the 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2016).

\section*{Adversarial attacks}

A specific safety issue to consider in robotic platforms incorporating deep neural networks is the possibility of adversarial attacks that may be able to alter the environment so as to induce unsafe behavior. Bayesian optimization has been applied to the efficient construction of adversarial attacks; this capability can in turn be used during the design phase seeking to build robotic controllers that are robust to such attack.

BOLOOR, ADITH et al. (2020). Attacking Vision-Based Perception in Endto-end Autonomous Driving Models. Journal of Systems Architecture 110:101766.

GHOSH, SHROMONA et al. (2018). Verifying Controllers against Adversarial Examples with Bayesian Optimization. Proceedings of the 2018 IEEE International Conference on Robotics and Automation (ICRA 2018).

\section*{REINFORCEMENT LEARNING}

The main challenge in reinforcement learning is that agents can only evaluate decision policies through trial-and-error: by following a proposed policy and observing the resulting reward. When the system is complex, evaluating a given policy may be costly in terms of time or other resources, and when the space of potential policies is large, careful exploration is necessary.

Policy search is one approach to reinforcement learning that models the problem of policy design in terms of optimization. We enumerate a space of potential policies (for example, via parameterization) and seek to maximize the expected cumulative reward over the policy space. This abstraction allows the straightforward use of Bayesian optimization to construct a series of policies seeking to balance exploration and exploitation of the policy space. A recurring theme in this research is the careful use of simulators, when available, to help guide the search via techniques from multifidelity optimization.

Reinforcement learning is a central concept in robotics, and many of the papers cited in the previous section represent applications of policy search to particular robotic systems. The citations below concern reinforcement learning in a broader context.

FENG, QING et al. (2020). High-Dimensional Contextual Policy Search with Unknown Context Rewards Using Bayesian Optimization. Advances in Neural Information Processing Systems 33 (NeurIPS 2020).

Letham, Benjamin et al. (2019). Bayesian Optimization for Policy Search via Online-Offline Experimentation. Journal of Machine Learning Research 20(145):1-30.

WILSON, AARON et al. (2014). Using Trajectory Data to Improve Bayesian Optimization for Reinforcement Learning. Fournal of Machine Learning Research 15(8):253-282.

Bayesian optimization has also proven useful for variations on the classical reinforcement learning problem, such as the multiagent setting or the so-called inverse reinforcement learning setting, where an agent must reconstruct an unknown reward function from observing demonstrations from other agents.

BALAKRISHNAN, SREEJITH et al. (2020). Efficient Exploration of Reward Functions in Inverse Reinforcement Learning via Bayesian Optimizaiton. Advances in Neural Information Processing Systems 33 (NeurIPS 2020).

DAI, ZHONGXIANG et al. (2020). R2-B2: Recursive Reasoning-Based Bayesian Optimization for No-Regret Learning in Games. Proceedings of the 37th International Conference on Machine Learning (ICML 2020).

IMANI, MAHDI et al. (2021). Scalable Inverse Reinforcement Learning through Multifidelity Bayesian Optimization. IEEE Transactions on Neural Networks and Learning Systems.

\section*{CIVIL ENGINEERING}

The optimization of large-scale systems such as power, transportation, water distribution, and sensor networks can be difficult due to complex multifidelity optimization: $\S 11.5$, p. 263 dynamics and the considerable expense of reconfiguration. Optimization can sometimes be aided via computer simulation, but the design spaces involved can nonetheless be huge, precluding exhaustive search.

BAHERI, ALI et al. (2017). Altitude Optimization of Airborne Wind Energy Systems: A Bayesian Optimization Approach. Proceedings of the 2017 American Control Conference (ACC 2017).

CORNEJO-BUENO, L. et al. (2018). Bayesian Optimization of a Hybrid System for Robust Ocean Wave Features Prediction. Neurocomputing $275: 818-828$.

GARNETT, R. et al. (2010). Bayesian Optimization for Sensor Set Selection. Proceedings of the gth ACM/IEEE International Conference on Information Processing in Sensor Networks (IPSN 2010).

GRAMACY, ROBERT B. et al. (2016). Modeling an Augmented Lagrangian for Blackbox Constrained Optimization. Technometrics 58(1):1-11. HICKISH, вОВ et al. (2020). Investigating Bayesian Optimization for Rail Network Optimization. International fournal of Rail Transporation $8(4): 307-323$.

KOPSIAfTIS, GEORGE et al. (2019). Gaussian Process Regression Tuned by Bayesian Optimization for Seawater Intrusion Prediction. Computational Intelligence and Neuroscience 2019:2859429.

MARCHANT, ROMAN et al. (2012). Bayesian Optimisation for Intelligent Environmental Monitoring. Proceedings of the 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2012).

\section*{Structural engineering}

The following study applied Bayesian optimization in a structural engineering setting: tuning the hyperparameters of a real-time predictive model for the typhoon-induced response of a $\sim 1000 \mathrm{~m}$ real-world bridge.

ZHANG, YI-MING et al. (2021). Probabilistic Framework with Bayesian Optimization for Predicting Typhoon-Induced Dynamic Responses of a Long-Span Bridge. Journal of Structural Engineering 147(1): 04020297.

\section*{ELECTRICAL ENGINEERING}

Over the past few decades, sophisticated tools have enabled the automation of many aspects of digital circuit design, even for extremely large circuits. However, the design of analog circuits remains largely a manual process. As circuits grow increasingly complex, the optimization of analog circuits (for example, to minimize power consumption subject to performance constraints) is becoming increasingly more difficult. Even computer simulation of complex analog circuits can entail significant cost, so careful experimental design is imperative for exploring the design space. Bayesian optimization has proven effective in this regard. CHEN, PENG (2015). Bayesian Optimization for Broadband High-Efficiency Power Amplifier Designs. IEEE Transactions on Microwave Theory and Techniques 63(12):4263-4272.

FANG, YAORAN et al. (2018). A Bayesian Optimization and Partial Element Equivalent Circuit Approach to Coil Design in Inductive Power Transfer Systems. Proceedings of the 2018 IEEE PELS Workshop on Emerging Technologies: Wireless Power Transfer (WoW 2018).

LIU, MINGJIE et al. (2020). Closing the Design Loop: Bayesian Optimization Assisted Hierarchical Analog Layout Synthesis. Proceedings of the 57th ACM/IEEE Design Automation Conference (DAC 2O2O).

LYU, WENLONG et al. (2018). An Efficient Bayesian Optimization Approach for Automated Optimization of Analog Circuits. IEEE Transactions on Circuits and Systems-I: Regular Papers 65(6):1954-1967.

TORUN, HAKKI MERT et al. (2018). A Global Bayesian Optimization Algorithm and Its Application to Integrated System Design. IEEE Transactions on Very Large Scale Integration (VLSI) Systems 26(4): 792-802.

\section*{MECHANICAL ENGINEERING}

The following study applied Bayesian optimization in a mechanical engineering setting: tuning the parameters of a welding process (via slow and expensive real-world experiments) to maximize weld quality.

STERLING, DILlON et al. (2015). Welding Parameter Optimization Based on Gaussian Process Regression Bayesian Optimization Algorithm. Proceedings of the 2015 IEEE International Conference on Automation Science and Engineering (CASE 2015).

Aerospace engineering

Nuances in airfoil design can have significant impact on aerodynamic performance, improvements in which can in turn lead to considerable cost savings from increased fuel efficiency. However, airfoil optimization is challenging due to the large design spaces involved and the nontrivial cost of evaluating a proposed configuration. Empirical measurement requires constructing an airfoil and testing its performance in a wind chamber. This process is too slow to explore the configuration space effectively, but we can simulate the process via computational fluid dynamics. These computational surrogates are still fairly costly due to the need to numerically solve nonlinear partial differential equations (the Navier-Stokes equations) at a sufficiently fine resolution, but they are nonetheless cheaper and more easily parallelized than experiments. ${ }^{4}$

Bayesian optimization can accelerate airfoil optimization via careful and cost-aware experimental design. One important idea here that can lead to significant computational savings is multifidelity modeling and optimization. It is relatively easy to control the cost-fidelity tradeoff in computational fluid dynamics simulations by altering its resolution
4 There are many parallels with this situation and that faced in (computational) chemistry and materials science, where quantummechanical simulation also requires numerically solving a partial differential equation (the Schrödinger equation).

multifidelity optimization: $§ 11.5$, p. 263 accordingly. This allows to rapidly explore the design space with cheapbut-rough simulations, then refine the most promising regions.

CHAITANYA, PARUCHURI et al. (2020). Bayesian Optimisation for LowNoise Aerofoil Design with Aerodynamic Constraints. International fournal of Aeroacoustics 20(1-2):109-129.

FORRESTER ALEXANDER, I. J. et al. (2009). Recent Advances in SurrogateBased Optimization. Progress in Aerospace Sciences 45(1-3):5079 .

HEBBAL, ALI et al. (2019). Multi-Objective Optimization Using Deep Gaussian Processes: Application to Aerospace Vehicle Design. Proceedings of the 2019 AIAA Scitech Forum.

LAM, REMI R. et al. (2018). Advances in Bayesian Optimization with Applications in Aerospace Engineering. Proceedings of the 2018 AIAA Scitech Forum.

PRIEM, RÉMY et al. (2020). An Efficient Application of Bayesian Optimization to an Industrial MDO Framework for Aircraft Design. Proceedings of the 2020 AIAA Aviation Forum.

REISENTHEL, PATRICK H. et al. (2011). A Numerical Experiment on Allocating Resources between Design of Experiment Samples and Surrogate-Based Optimization Infills. Proceedings of the 2011 AIAA/ASME/ASCE/AHS/ ASC Structures, Structural Dynamics and Materials Conference.

ZHENG, HONGyu et al. (2020). Multifidelity Kinematic Parameter Optimization of a Flapping Airfoil. Physical Review E 101(1):013107.

\section*{Automobile engineering}

Bayesian optimization has also proven useful in automobile engineering. Automobile components and subsystems can have numerous tunable parameters affecting performance, and evaluating a given configuration can be complicated due to complex interactions among vehicle components. Bayesian optimization has shown success in this setting.

LIESSNER, ROMAN et al. (2019). Simultaneous Electric Powertrain Hardware and Energy Management Optimization of a Hybrid Electric Vehicle Using Deep Reinforcement Learning and Bayesian Optimization. Proceedings of the 2019 IEEE Vehicle Power and Propulsion Conference (VPPC 2019).

NEUMANN-Brosig, MATthias et al. (2020). Data-Efficient Autotuning with Bayesian Optimization: An Industrial Control Study. IEEE Transactions on Control Systems Technology 28(3):730-740.

THOMAS, SINNU SUSAN et al. (2019). Designing MacPherson Suspension Architectures Using Bayesian Optimization. Proceedings of the 28th Belgian Dutch Conference on Machine Learning (Benelearn 2019). 

\section*{ALGORITHM CONFIGURATION, HYPERPARAMETER TUNING, AND AUTOML}

Complex algorithms and software pipelines can have numerous parameters influencing their performance, and determining the optimal configuration for a given situation can require significant trial-and-error Bayesian optimization has demonstrated remarkable success on a range of problems under the umbrella of algorithm configuration.

In the most general setting, we may simply model algorithm performance as a black box. A challenge here is dealing with high-dimensional parameter spaces that may have complex structure, such as conditional parameters that only become relevant depending on the settings of other parameters. This can pose a challenge for Gaussian process models, but alternatives such as random forests can perform admirably.

DALIBARD, VALENTIN et al. (2017). BOAT: Building Auto-Tuners with Structured Bayesian Optimization. Proceedings of the 26th International Conference on World Wide Web (WWW 2017).

GonZAlvez, JOAN et al. (2019). Financial Applications of Gaussian Processes and Bayesian Optimization. arXiv: 1903.04841 [q-fin.PM].

HOOS, HOlger H. (2012). Programming by Optimization. Communications of the ACM 55(2):70-80.

HUTTER, FRANK et al. (2011). Sequential Model-Based Optimization for General Algorithm Configuration. Proceedings of the 5 th Learning and Intelligent Optimization Conference (LION 5).

KunJIR, MAYURESH (2019). Guided Bayesian Optimization to AutoTune Memory-Based Analytics. Proceedings of the 35th IEEE International Conference on Data Engineering Workshops (ICDEW 2019).

SŁowik, AGNieszKa et al. (2019). Bayesian Optimisation for Heuristic Configuration in Automated Theorem Proving. Proceedings of the $5^{\text {th }}$ and 6th Vampire Workshops (Vampire 2019).

VARGAS-HERnÁNDEZ, R. A. (2020). Bayesian Optimization for Calibrating and Selecting Hybrid-Density Functional Models. fournal of Physical Chemistry A 124(20):4053-4061.

\section*{Hyperparameter tuning of machine learning algorithms}

The task of configuring machine learning algorithms in particular is known as hyperparameter tuning. Hyperparameter tuning is especially challenging in the era of deep learning, due to the often considerable cost of training and validating a proposed configuration. However, Bayesian optimization has proven effective at this task, and the results of a recent black-box optimization competition (run by TURNER et al. below) focusing on optimization problems from machine learning soundly established its superiority to alternatives such as random search.

QUITADAMO, ANDREw et al. (2017). Bayesian Hyperparameter Optimization for Machine Learning Based eQTL Analysis. Proceedings of the 8th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics (BCB 2017). alternatives to GPs, including random forests: $\S 8.11$, p. 196 tuning samplers, etc.

Chapter 4: Model Assessment, Selection, and Averaging, p. 67

neural architecture search
SNOEK, JASPER et al. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. Advances in Neural Information Processing Systems 25 (NeurIPS 2012).

TURNER, RYAN et al. (2021). Bayesian Optimization Is Superior to Random Search for Machine Learning Hyperparameter Tuning: Analysis of the Black-Box Optimization Challenge 2020. Proceedings of the NeurIPS 2020 Competition and Demonstration Track.

YOGATAMA, DANI et al. (2015). Bayesian Optimization of Text Representations. Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP 2015).

In addition to tuning machine learning algorithms to maximize predictive performance, Bayesian optimization can also tune the parameters of other algorithms for learning and inference such as Monte Carlo samplers.

HAMZE, FIRAS et al. (2013). Self-Avoiding Random Dynamics on Integer Complex Systems. ACM Transactions on Modeling and Computer Simulation 23(1):9.

MAHENDRAN, NIMALAN et al. (2010). Adaptive MCMC with Bayesian Optimization. Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS 2010).

One recent high-profile use of Bayesian optimization was in tuning the hyperparameters of DeepMind's AlphaGo agent. Bayesian optimization was able to improve AlphaGo's self-play win rate from one-half of games to nearly two-thirds, and the version tuned with Bayesian optimization was used in the final match against Lee Sedol.

CHEN, yUtiAn et al. (2018). Bayesian Optimization in AlphaGo. arXiv: 1812.06855 [cs.LG].

\section*{Automated machine learning}

The goal of automated machine learning (automL) is to develop automated procedures for machine learning tasks in order to boost efficiency and open the power of machine learning to a wider audience. Hyperparameter tuning is one particular instance of this overall vision, but we can also consider the automation of other aspects of machine learning.

In model selection, for example, we seek to tune not only the hyperparameters of a machine learning model but also the structure of the model itself. One notable special case is neural architecture search, the optimization of neural network architectures. Such problems are particularly difficult due to the discrete, structured nature of the search spaces involved. However, with careful modeling and acquisition strategies, Bayesian optimization becomes a feasible solution.

JIN, HAIFENG et al. (2019). Auto-Keras: An Efficient Neural Architecture Search System. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2019). KANDASAMY, KIRTHEVASAN et al. (2018). Neural Architecture Search with Bayesian Optimisation and Optimal Transport. Advances in Neural Information Processing Systems 31 (NeurIPS 2018).

MALKOMES, GUSTAVo et al. (2016). Bayesian Optimization for Automated Model Selection. Advances in Neural Information Processing Systems 29 (NeurIPS 2016).

WHITE, COLIN et al. (2021). BANANAs: Bayesian Optimization with Neural Architectures for Neural Architecture Search. Proceedings of 35 th AAAI Conference on Artificial Intelligence (AAAI 2021).

Researchers have even considered using Bayesian optimization for dynamic model selection during Bayesian optimization itself to realize fully autonomous and robust optimization routines.

MALKOMES, GUSTAVo et al. (2018). Automating Bayesian Optimization with Bayesian Optimization. Advances in Neural Information Processing Systems 31 (NeurIPS 2018).

\section*{Optimization of entire machine learning pipelines}

We may also consider the joint optimization of entire machine learning pipelines, starting from raw data and automatically constructing a bespoke machine learning system. The space of possible pipelines is absolutely enormous, as we must consider preprocessing steps such as feature selection and engineering in addition to model selection and hyperparameter tuning. Nonetheless, Bayesian optimization has proven up to the task.

Building a working automL system of this scope entails a number of subtle design questions: the space of pipelines to consider, the allocation of training resources to proposed pipelines (which may vary enormously in their complexity), and the modeling of performance as a function of dataset and pipeline, to name a few. All of these questions have received careful consideration in the literature, and the following reference is an excellent entry point to that body of work.

FEURER, MATthiAs et al. (2020). Auto-Sklearn 2.0: The Next Generation. arXiv: 2007.04074 [cs.LG].

\section*{ADAPTIVE HUMAN-COMPUTER INTERFACES}

Adaptive human-computer interfaces seek to tailor themselves on-thefly to suit the preferences of the user and/or the system provider. For example:

- a content provider may wish to learn user preferences to ensure recommended content is relevant,

- a website depending on ad revenue may wish to tune their interface and advertising placement algorithms to maximize ad revenue, - a computer game may seek to adjust its difficulty dynamically to keep the player engaged, or

- a data-visualization system may seek to infer the user's goals in interacting with a dataset and customize the presentation of data accordingly.

The design of such a system can be challenging, as the space of possible user preferences and/or the space of possible algorithmic settings can be large, and the optimal configuration may change over time or depend on other context. Further, we may only assess the utility of a given interface configuration through user interaction, which is a slow, cumbersome, and noisy channel from which to glean information. Finally, we face the additional challenge that if we are not careful, the user may become annoyed and simply abandon the platform altogether! Nonetheless, Bayesian optimization has shown success in tuning adaptive interfaces and in related problems such as preference optimization, A/B testing, etc.

BROCHU, ERIC et al. (2010). A Bayesian Interactive Optimization Approach to Procedural Animation Design. Proceedings of the 2010 ACM SIGGRAPH/Eurographics Symposium on Computer Animiation (SCA 2010).

BROCHU, ERIC et al. (2015). Active Preference Learning with Discrete Choice Data. Advances in Neural Information Processing Systems 20 (NeUrIPS 2007).

GONZÁlez, JAVIER et al. (2017). Preferential Bayesian Optimization. Proceedings of the 34th International Conference on Machine Learning (ICML 2017).

KADNER, FLORIAN et al. (2021). AdaptiFont: Increasing Individuals' Reading Speed with a Generative Font Model and Bayesian Optimization. Proceedings of the $2021 \mathrm{CHI}$ Conference on Human Factors in Computing Systems (CHI 2021).

KHAJAH, MOHAMмAD M. et al. (2016). Designing Engaging Games Using Bayesian Optimization. Proceedings of the $2016 \mathrm{CHI}$ Conference on Human Factors in Computing Systems (CHI 2016).

LETHAM, BENJAmin et al. (2019). Constrained Bayesian Optimization with Noisy Experiments. Bayesian Analysis 14(2):495-519.

monAdJEmi, ShaYAn et al. (2020). Active Visual Analytics: Assisted Data Discovery in Interactive Visualizations via Active Search. arXiv: 2010.08155 [cs.HC].