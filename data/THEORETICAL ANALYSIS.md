\section*{THEORETICAL ANALYSIS}

The Bayesian optimization procedures we have developed throughout this book have demonstrated remarkable empirical performance in a huge swath of practical settings, many of which are outlined in Appendix D. However, good empirical performance may not be enough to satisfy those who value rigor over results. Fortunately, many Bayesian optimization algorithms are also backed by strong theoretical guarantees on their performance. The literature on this topic is now quite vast, and convergence has been studied by different authors in different ways, sometimes involving slight nuances in approach and interpretation. We will take an in-depth look at this topic in this chapter, covering the most common lines of attack and outlining the state-of-the-art in results.

A running theme throughout this chapter will be understanding how various measures of optimization error decrease asymptotically as an optimization policy is repeatedly executed. To facilitate this discussion, we will universally use $\tau$ in this chapter to indicate dataset size, or equivalently to indicate the number of steps an optimization procedure is assumed to have run. As we will primarily be interested in asymptotic results as $\tau \rightarrow \infty$, this convention does not rule out the possibility of starting optimization with some arbitrary dataset of fixed size. We will use subscripts to indicate dataset size when necessary, notating the dataset comprising the first $\tau$ observations with:

$$
\mathcal{D}_{\tau}=\left(\mathbf{x}_{\tau}, \mathbf{y}_{\tau}\right)=\left\{\left(x_{i}, y_{i}\right)\right\}_{i=1}^{\tau} .
$$

When studying the convergence of a global optimization algorithm, we must be careful to define exactly what we mean by "convergence." In general, a convergence argument entails:

- choosing some measure of optimization error,

- choosing some space of possible objective functions, and

- establishing some guarantee for the chosen error on the chosen function space, such as an asymptotic bound on the worst- or average-case error in the large-sample limit $\tau \rightarrow \infty$.

There is a great deal of freedom in the last of these steps, and we will discuss several important results and proof strategies later in this chapter. However, there are well-established conventions for the first two of these steps, which we will introduce in the following two sections. We begin with the notion of regret, which provides a natural measure of optimization error.

\section*{REGRET}

Regret is a core concept in the analysis of optimization algorithms, Bayesian or otherwise. The role of regret is to quantify optimization progress in a manner suitable for establishing convergence to the global optimum and studying the rate of this convergence. There are several definitions

\section*{0}

Appendix D, p. 313: Annotated Bibliography of Applications

size of dataset, $\tau$

dataset after $\tau$ observations, $\mathcal{D}_{\tau}$

what does it mean to converge?

regret: below

useful spaces of objective functions: § 10.2, p. 215 1 One might propose an alternative definition of error by measuring how closely the observed locations approach a global maximum $x^{*}$ rather than by how closely the observed values approach the value of the global optimum $f^{*}$. However, it turns out this is both less convenient for analysis and harder to motivate: in practice it is the value of the objective that we care about the most, and discovering a near-optimal value is a success regardless of its distance to the global optimum.

2 As outlined in Chapter 5 (p. 87), optimal actions maximize expected utility in the face of uncertainty. Many observations may not result in progress, even though designed with the best of intentions.

3 For example, it is easy to develop a spacefilling design that will eventually locate the global optimum of any continuous function through sheer dumb luck - but it might not do so very quickly!

4 The measures of regret introduced here do not depend on the observed values y but only on the underlying objective values $\phi$. We are making a tacit assumption that $\mathrm{y}$ is sufficiently informative about $\phi$ for this to be sensible in the large-sample limit.

5 Occasionally a slightly different definition of simple regret is used, analogous to the global reward (6.5), where we measure regret with respect to the maximum of the posterior mean:

$$
r_{\tau}=f^{*}-\max _{x \in \mathcal{X}} \mu_{\mathcal{D}_{\tau}}(x)
$$

convergence goal: show $r_{\tau} \rightarrow 0$

instantaneous regret, $\rho$

cumulative regret of $\mathcal{D}_{\tau}, R_{\tau}$ of regret used in different contexts, all based on the same idea: comparing the objective function values visited during optimization to the globally optimal value, $f *$ The larger this gap, the more "regret" we incur in retrospect for having invested in observations at suboptimal locations. ${ }^{1}$

Regret is an unavoidable consequence of decision making under uncertainty. Without foreknowledge of the global optimum, we must of course spend some time searching for it, and even what may be optimal actions in the face of uncertainty may seem disappointing in retrospect. ${ }^{2}$ However, such actions are necessary in order to learn about the environment and inform future decisions. This reasoning gives rise to the classic tension between exploration and exploitation in policy design: although exploration may not yield immediate progress, it enables future success, and if we are careful, reduces future regret. Of course, exploration alone is not sufficient to realize a compelling optimization strategy, ${ }^{3}$ as we must also exploit what we have learned and adapt our behavior accordingly. An ideal algorithm thus explores efficiently enough that its regret can at least be limited in the long run.

Most analysis is performed in terms of one of two closely related notions of regret: simple or cumulative regret, defined below.

\section*{Simple regret}

Let $\mathcal{D}_{\tau}$ represent some set of (potentially noisy) observations gathered during optimization, ${ }^{4}$ and let $\boldsymbol{\phi}_{\tau}=f\left(\mathbf{x}_{\tau}\right)$ represent the objective function values at the observed locations. The simple regret associated with this data is the difference between the global maximum of the objective and the maximum restricted to the observed locations: ${ }^{5}$

$$
r_{\tau}=f^{*}-\max \phi_{\tau}
$$

It is immediate from its definition that simple regret is nonnegative and vanishes only if the data contain a global optimum. With this in mind, a common goal is to show that the simple regret of data obtained by some policy approaches zero, implying the policy will eventually (and perhaps efficiently) identify the global optimum, up to vanishing error.

\section*{Cumulative regret}

To define cumulative regret, we first introduce the instantaneous regret $\rho$ corresponding to an observation at some point $x$, which is the difference between the global maximum of the objective and the function value $\phi$ :

$$
\rho=f^{*}-\phi
$$

The cumulative regret for a dataset $\mathcal{D}_{\tau}$ is then the total instantaneous regret incurred:

$$
R_{\tau}=\sum_{i} \rho_{i}=\tau f^{*}-\sum_{i} \phi_{i}
$$



\section*{Relationship between simple and cumulative regret}

Simple and cumulative regret are analogous to the simple (6.3) and cumulative (6.7) reward utility functions, and any intuition regarding these utilities transfers to their regret counterparts. ${ }^{6}$

However, these two definitions of regret are not directly comparable, even on the same data - for starters, simple regret is nonincreasing as more data are collected, whereas cumulative regret is nondecreasing. However, suitable normalization allows some useful comparison. ${ }^{7}$ Namely, consider the average, rather than cumulative, regret:

$$
\frac{R_{\tau}}{\tau}=f^{*}-\frac{1}{\tau} \sum_{i} \phi_{i} .
$$

As the mean of a vector is a lower bound on its maximum, we may derive an upper bound on simple regret in terms of cumulative regret:

$$
r_{\tau} \leq \frac{R_{\tau}}{\tau}
$$

In this light, a common goal is to show that an optimization algorithm has the so-called no-regret property, which means that its average regret vanishes with increasing data:

$$
\lim _{\tau \rightarrow \infty} \frac{R_{\tau}}{\tau}=0
$$

Equivalently, a policy achieves no regret if its cumulative regret grows sublinearly with the dataset size $\tau$. This is sufficient to prove convergence in terms of simple regret as well by appealing to the squeeze theorem:

$$
0 \leq r_{\tau} \leq \frac{R_{\tau}}{\tau} \rightarrow 0
$$

Although no regret guarantees convergence in simple regret, the reverse is not necessarily the case. It is easy to find counterexamples - consider, for example, modifying a no-regret policy to select a fixed suboptimal point every other observation. The simple regret would still vanish (half as fast), but constant instantaneous regret on alternating iterations would prevent sublinear cumulative regret. Thus the no-regret property is somewhat stronger than convergence in simple regret alone. From another perspective, simple regret is more tolerant of exploration: we only need to visit the global optimum once to converge in terms of simple regret, whereas effectively all observations must eventually become effectively optimal to achieve the no-regret property.

\section*{USEFUL FUNCTION SPACES FOR STUDYING CONVERGENCE}

Identifying the "right" function space to consider when studying convergence is a subtle decision, as we must strike a balance between generality and practicality. One obvious choice would be the space of all continuous
6 Note that neither notion of regret represents a valid utility function itself, as in general both $f^{*}$ and $\phi$ would be random variables in that context.

7 For a deeper discussion on the connections between simple and cumulative regret, see:

S. BUBECK et al. (2009). Pure Exploration in Multi-Armed Bandits Problems. ALT 2009.

convergence goal: show $R_{\tau} / \tau \rightarrow 0$

no-regret property implies convergence in simple regret

convergence in simple regret does not imply the no-regret property 
![](https://cdn.mathpix.com/cropped/2023_09_22_751c1e67194d77e6639bg-04.jpg?height=366&width=1632&top_left_y=455&top_left_x=156)

Figure 10.1: Modifying a continuous function $f$ (left) to feature a "needle" on some ball (here an interval) $B$. We construct a continuous function $g$ vanishing on the complement of $B$ (middle), then add this "correction" to $f$ (right).

convergence on "nice" continuous functions: p. 218

8 For example by taking $x_{i}$ within a ball of radius $1 / i$ around $x^{*}$.

9 Some care would be required to make the following argument rigorous for stochastic observations and/or policies, but the spirit would remain intact. functions. However, it turns out this space is far too large to be of much theoretical interest, as it contains functions that are arbitrarily hard to optimize. Nonetheless, it is easy to characterize convergence (in terms of simple regret) on this space, and some convergence guarantees of this type are known for select Bayesian optimization algorithms. We will begin our discussion here.

We may gain some traction by considering a family of more plausible, "nice" functions whose complexity can be controlled enough to guarantee rapid convergence. In particular, choosing a Gaussian process prior for the objective function implies strong correlation structure, and this insight leads to natural function spaces to study. This has become the standard approach in modern analysis, and we will consider it shortly.

\section*{Convergence on all continuous functions}

The largest reasonable space we might want to consider is the space of all continuous functions. However, continuous functions can be poorly behaved from the point of view of optimization, and we cannot hope for strong convergence guarantees as a result. There is simply too much freedom for functions to "hide" their optima in inconvenient places.

To begin, we establish a simple characterization of convergence on all continuous functions in terms of eventual density of observation.

Theorem. Let $\mathcal{X}$ be a compact metric space. An optimization policy converges in terms of simple regret on all continuous functions $f: \mathcal{X} \rightarrow \mathbb{R}$ if and only if the set of eventually observed points $\bigcup_{i=1}^{\infty}\left\{x_{i}\right\}$ is always dense.

The proof is instructive as it shows what can go wrong with general continuous functions. First, if the set of eventually observed points is dense in $\mathcal{X}$, we may construct a sequence of observations converging to a global maximum $x^{* 8}$ The associated function values $\left\{\phi_{i}\right\}$ then converge to $f^{*}$ by continuity, and thus $r_{\tau} \rightarrow 0$.

If density fails ${ }^{9}$ for some continuous function $f-$ and thus there is some ball $B \subset \mathcal{X}$ that will never contain an observation - then we may foil the policy with a "needle in a haystack." We construct a continuous function $g$ vanishing on the complement of $B$ and achieving arbitrarily high values on $B .{ }^{10}$ Adding the needle to $f$ creates a continuous function $f+g$ with arbitrary - and, once the needle is sufficiently tall, never observed - maximum; see Figure 10.1. As $f$ and $f+g$ agree outside $B$, the policy cannot distinguish between these functions, and thus the policy can have arbitrarily high simple regret.

We can use this strategy of building adversarial "needles in haystacks" to further show that, even if we can guarantee convergence on all continuous functions, we cannot hope to demonstrate rapid convergence in simple regret, at least not in the worst case. Unless the domain is finite, running any policy for any finite number of iterations will leave unobserved "holes" that we can fill in with arbitrarily tall "needles," and thus the worst-case regret will be unbounded at every stage of the algorithm.

\section*{Convergence results for all continuous functions on the unit interval}

Establishing the density criterion above has proven difficult for Bayesian optimization algorithms in general spaces with arbitrary models. However, restricting the domain to the unit interval $\mathcal{X}=[0,1]$ and limiting the objective function and observation models to certain well-behaved combinations has yielded universal convergence guarantees for some Bayesian procedures.

For example, KUSHNER sketched a proof of convergence in simple regret when maximizing probability of improvement on the unit interval, when the objective function model is the Wiener process and observations are exact or corrupted by additive Gaussian noise. ${ }^{11}$ žILINSKAs later provided a proof in the noiseless case. ${ }^{12}$ This particular model exhibits a Markov property that enables relatively straightforward analysis by characterizing its behavior on each of the subintervals subdivided by the data. This structure enables a simple proof strategy by assuming some subinterval is never subdivided and arriving at a contradiction.

Convergence guarantees for this policy (under the name the " $p$ algorithm") on the unit interval have also been established for smoother models of the objective function - that is, with differentiable rather than merely continuous sample paths ${ }^{13}$ - including the once-integrated Wiener process. ${ }^{14}$ Convergence rates for these algorithms for general continuous functions have also been derived. ${ }^{13,14,15}$ In light of the above discussion, these convergence rates are not in terms of simple regret but rather in terms of shrinkage in the size of the subinterval containing the global optimum, and thus the distance from the closest observed location to the global optimum. The proofs of these results all relied on exact observation of the objective function.

Convergence on all continuous functions on the unit interval has also been established for maximizing expected improvement, in the special case of exact observation and a Wiener process model on the objective. ${ }^{16}$ The proof strategy again relied on the special structure of the posterior to show that the observed locations will always subdivide the domain such that no "hole" is left behind.
10 For example, we may scale the distance to the complement of $B$.

worst-case simple regret unbounded
11 H. J. KUSHNER (1964). A New Method of Locating the Maximum Point of an Arbitrary Multipeak Curve in the Presence of Noise. Fournal of Basic Engineering 86(1):97-106.

12 A. G. ŽILINSKAS (1975). Single-Step Bayesian Search Method for an Extremum of Functions of a Single Variable. Kibernetika (Cybernetics) 11(1):16o-166.

13 J. CALVIN and A. ŽIlinskas (1999). On the Convergence of the P-Algorithm for OneDimensional Global Optimization of Smooth Functions. fournal of Optimization Theory and Applications 102(3):479-495.

14 J. M. CALVIN and A. ŽILINSKAS (2001). On Convergence of a P-Algorithm Based on a Statistical Model of Continuously Differentiable Functions. Fournal of Global Optimization 19(3):229245 .

15 J. M. CALvin (200o). Convergence Rate of the P-Algorithm for Optimization of Continuous Functions. In: Approximation and Complexity in Numerical Optimization: Continuous and Discrete Problems.

16 M. LOCATELli (1997). Bayesian Algorithms for One-Dimensional Global Optimization. fournal of Global Optimization 10(1):57-76. 17 All of the algorithms we will study for the remainder of the chapter will use a Gaussian process belief on the objective function, where the theory is mature.

sample path continuity: $§ 2.5$, p. 28

![](https://cdn.mathpix.com/cropped/2023_09_22_751c1e67194d77e6639bg-06.jpg?height=231&width=534&top_left_y=1152&top_left_x=156)

Sample paths of a stationary Gaussian process with Matérn $v=5 / 2$ covariance (3.14) show more regularity than arbitrary continuous functions.

Bayesian (Bayes) regret

18 In some analyses, we may also seek bounds on the regret that hold with high probability with respect to these random variables rather than bounds on the expected regret.

worst-case regret on $\mathcal{H}$ after $\tau$ steps: $\bar{r}(\tau, \mathcal{H})$, $\bar{R}(\tau, \mathcal{H})$

\section*{Convergence on "nice" continuous functions}

In the pursuit of stronger convergence results, we may abandon the space of all continuous functions and focus on some space of suitably well-behaved objectives. By limiting the complexity of the functions we consider, we can avoid complications arising from adversarial examples like "needles in haystacks." This can be motivated from the basic assumption that optimization is to be feasible at all, which is not the case when facing an adversary creating arbitrarily difficult problems. Instead, we may seek strong performance guarantees when optimizing "plausible" objective functions. Conveniently, a Gaussian process model on the objective $\mathcal{G P}(f ; \mu, K)$ gives rise to several paths forward. ${ }^{17}$

\section*{Sample paths of a Gaussian process}

In a Bayesian analysis, it is natural to assume that the objective function is a sample path from the Gaussian process used to model it. Assuming sample path continuity, sample paths of a Gaussian process are much better behaved than general continuous functions. The covariance function provides regularization on the behavior of sample paths via the induced correlations among function values; see the figure in the margin. As a result, we can ensure that functions with exceptionally bad behavior (such as hidden "needles") are also exceptionally rare.

The sample path assumption provides a known distribution for the function values at observed locations (2.2-2.3), allowing us to derive average-case results. An important concept here is the Bayesian (or simply Bayes) regret of a policy, which is the expected value of the (simple or cumulative) regret incurred when following the policy, say for $\tau$ steps:

$$
\mathbb{E}\left[r_{\tau}\right] ; \quad \mathbb{E}\left[R_{\tau}\right]
$$

This expectation is taken with respect to uncertainty in the objective function $f$, the observation locations $\mathbf{x}$, and the observed values $\mathbf{y} .^{18}$

\section*{The worst-case alternative}

The GP sample path assumption is not always desirable, for example in the context of a frequentist (that is, worst-case) analysis of a Bayesian optimization algorithm. This is not as contradictory as it may seem, as Bayesian analyses can lack robustness to model misspecification - a certainty in practice. An alternative is to assume that the objective function lies in some explicit space of "nice" functions $\mathcal{H}$, then find worst-case convergence guarantees for the Bayesian algorithm on inputs satisfying this regularity assumption. For example, we might seek to bound the worst-case expected (simple or cumulative) regret for a function in this space after $\tau$ decisions:

$$
\bar{r}_{\tau}[\mathcal{H}]=\sup _{f \in \mathcal{H}} \mathbb{E}\left[r_{\tau}\right] ; \quad \bar{R}_{\tau}[\mathcal{H}]=\sup _{f \in \mathcal{H}} \mathbb{E}\left[R_{\tau}\right]
$$


![](https://cdn.mathpix.com/cropped/2023_09_22_751c1e67194d77e6639bg-07.jpg?height=210&width=1012&top_left_y=454&top_left_x=272)

Here the expectation is over with respect to the observed locations $\mathbf{x}$ and the observed values $y,{ }^{19}$ but uncertainty in the objective function presumably the most troubling factor in a Bayesian analysis - is replaced by a pessimistic bound on the functions in $\mathcal{H}$. Note that the algorithms we will analyze still use a Gaussian process belief on $f$ in making decisions, but the objective function is not generated according to that model.

\section*{Reproducing kernel Hilbert spaces}

Corresponding to every covariance function $K$ is a natural companion function space, its reproducing kernel Hilbert space (RKHS) $\mathcal{H}_{K}$. There is a strong connection between this space and a centered Gaussian process with the same covariance function, $\mathcal{G P}(f ; \mu \equiv 0, K)$. Namely, consider the set of functions of the form

$$
x \mapsto \sum_{i=1}^{n} \alpha_{i} K\left(x_{i}, x\right),
$$

where $\left\{x_{i}\right\} \subset \mathcal{X}$ is an arbitrary finite set of input locations with corresponding real-valued weights $\left\{\alpha_{i}\right\} .{ }^{20}$ Note this is precisely the set of all possible posterior mean functions for the Gaussian process arising from exact inference! ${ }^{21}$ The RKHs $\mathcal{H}_{K}$ is then the completion of this space endowed with the inner product

$$
\left\langle\sum_{i=1}^{n} \alpha_{i} K\left(x_{i}, x\right), \sum_{j=1}^{m} \beta_{j} K\left(x_{j}^{\prime}, x\right)\right\rangle=\sum_{i=1}^{n} \sum_{j=1}^{m} \alpha_{i} \beta_{j} K\left(x_{i}, x_{j}^{\prime}\right) .
$$

That is, the RKHS is roughly the set of functions "as smooth as" a posterior mean function of the corresponding GP, according to a notion of "explainability" by the covariance function.

It turns out that belonging to the RKHS $\mathcal{H}_{K}$ is a stronger regularity assumption than being a sample path of the corresponding GP. In fact, unless the RKHs is finite-dimensional (which is not normally the case), sample paths from a Gaussian process almost surely do not lie in the corresponding RKHS: ${ }^{22,23} \operatorname{Pr}\left(f \in \mathcal{H}_{K}\right)=0$. However, the posterior mean function of the same process always lies in the RKHS by the above construction. Figure 10.2 illustrates a striking example of this phenomenon: sample paths from a stationary GP with Matérn covariance function with $v=1 / 2$ (3.11) are nowhere differentiable, whereas members of the corresponding RKHS are almost everywhere differentiable. Effectively, the process of averaging over sample paths "smooths out" their erratic behavior in the posterior mean, and elements of the RKHS exhibit similar smoothness.
Figure 10.2: Left: functions in the RKHS corresponding to the Matérn $v=1 / 2$ covariance function (3.11). Right: sample paths from a GP with the same covariance.

19 In the special case of exact observation and a deterministic policy, such a bound would entail no probabilistic elements.

reproducing kernel Hilbert space corresponding to $K, \mathcal{H}_{K}$

20 Equivalently, this is the span of the set of covariance functions with one input held fixed: $\operatorname{span}\left\{x \mapsto K\left(x, x^{\prime}\right) \mid x^{\prime} \in \mathcal{X}\right\}$.

21 Inspection of the general posterior mean functions in $(2.14,2.19)$ reveals they can always be (and can only be) written in this form.

22 M. N. LUKIĆ and J. H. BEDER (2001). Stochastic Processes with Sample Paths in Reproducing Kernel Hilbert Spaces. Transactions of the American Mathematical Society 353(10):39453969.

23 However, for the Matérn family, sample paths do lie in a "larger" RKHS we can determine from the covariance function. Namely, sample paths of processes with a Matérn covariance with finite $v$ are (almost everywhere) one-time less differentiable than the functions in the corresponding RKHS; see Figure 10.2 for an illustration of this phenomenon with $v=1 / 2$. Sample paths for the squared exponential covariance, meanwhile, lie "just barely" outside the corresponding RHKs. For more details, see:

M. KANAGAWA et al. (2018). Gaussian Processes and Kernel Methods: A Review on Connections and Equivalences. arXiv: 1807 . 02582 [stat.ML]. [theorem 4.12] RKHS norm, $\|f\|_{\mathcal{H}_{K}}$

RKHS norm of posterior mean function

24 For this example we have $\alpha=\Sigma^{-1} \mathrm{y}$ in (10.8).

connection between RKHS norm and GP marginal likelihood

25 The remaining terms are independent of data.

RKHs ball of radius $B, \mathcal{H}_{K}[B]$

\section*{Reproducing kernel Hilbert space norm}

Associated with an RKHS $\mathcal{H}_{K}$ is a norm $\|f\|_{\mathcal{H}_{K}}$ that can be interpreted as a measure of function complexity with respect to the covariance function $K$. The RKHS norm derives from the pre-completion inner product (10.9), and we can build intuition for the norm by drawing a connection between that inner product and familiar concepts from Gaussian process regression. The key is the characterization of the pre-completion function space (10.8) as the space of all possible posterior mean functions for the corresponding centered Gaussian process $\mathcal{G P}(f ; \mu \equiv 0, K)$.

To be more explicit, consider the posterior mean after observing a dataset of exact observations $\mathcal{D}=(\mathbf{x}, \mathbf{y})$, inducing the posterior mean (2.14):

$$
\mu_{\mathcal{D}}(x)=K(x, \mathbf{x}) \Sigma^{-1} \mathbf{y},
$$

where $\Sigma=K(\mathbf{x}, \mathbf{x})$. Then the (squared) RKHs norm of the posterior mean is: ${ }^{24}$

$$
\left\|\mu_{\mathcal{D}}\right\|_{\mathcal{H}_{K}}^{2}=\left\langle\mu_{\mathcal{D}}, \mu_{\mathcal{D}}\right\rangle=\mathbf{y}^{\top} \Sigma^{-1} \mathbf{y}
$$

That is, the RKHS norm of the posterior mean is the Mahalanobis norm of the observed data $y$ under their Gaussian prior distribution (2.2-2.3).

We have actually seen this score before: it appears in the log marginal likelihood of the data under the Gaussian process (4.8), where we interpreted it as a score of data fit. ${ }^{25}$ This reveals a deep connection between the RKHS norm and the associated Gaussian process: posterior mean functions arising from "more unusual" observations from the point of view of the GP have higher complexity from the point of view of the RKHS. Whereas the Gaussian process judges the data y to be complex via the marginal likelihood, the RKHs judges the resulting posterior mean $\mu_{\mathcal{D}}$ to be complex via the RKHs norm - but these are simply two ways of interpreting the same scenario. ${ }^{26}$

The role of the RKHS norm in quantifying function complexity suggests a natural space of objective functions to work with when seeking worst-case results (10.7). We take the RKHS ball of radius $B$, the space of functions with complexity bounded by $B$ in the RKHS:

$$
\mathcal{H}_{K}[B]=\left\{f \mid\|f\|_{\mathcal{H}_{K}} \leq B\right\} .
$$

The radius $B$ is left as a parameter that is absorbed into derived bounds.

\subsection*{RELEVANT PROPERTIES OF COVARIANCE FUNCTIONS}

A sizable majority of the results discussed in the remainder of this chapter concern optimization performance on one of the function spaces discussed in the previous section: sample paths of a centered Gaussian process with covariance function $K$ or functions in the corresponding RKHS $\mathcal{H}_{K}$. Given the fundamental role the covariance function plays in determining sample path behavior, it should not be surprising that the nature of the covariance function also has profound influence on optimization performance. 

![](https://cdn.mathpix.com/cropped/2023_09_22_751c1e67194d77e6639bg-09.jpg?height=223&width=528&top_left_y=454&top_left_x=267)

$v=1.01$

![](https://cdn.mathpix.com/cropped/2023_09_22_751c1e67194d77e6639bg-09.jpg?height=219&width=534&top_left_y=453&top_left_x=818)

$v=3 / 2$

![](https://cdn.mathpix.com/cropped/2023_09_22_751c1e67194d77e6639bg-09.jpg?height=209&width=528&top_left_y=455&top_left_x=1369)

$v=2$

Figure 10.3: Sample paths from a centered GP with Matérn covariance with smoothness parameter $v$ ranging from just over 1 (left) to 2 (right). The random seed is shared so that corresponding paths are comparable across the panels. All samples are once differentiable, but some are smoother than others.

One might intuitively expect that optimizing smoother functions should be easier than optimizing rougher functions, as a rougher function gives the optimum more places to "hide"; see the figure in the margin. This intuition turns out to be correct. The key insight is that rougher functions require more information to describe than smoother ones. As each observation we make is limited in how much information it can reveal regarding the objective function, rougher objectives require more observations to learn with a similar level of confidence, and to optimize with a similar level of success. We can make this intuition precise through the concept of information capacity, which bounds the rate at which we can learn about the objective through noisy observations and serves as a fundamental measure of function complexity appearing in numerous analyses.

\section*{Smoothness of sample paths}

The connection between sample path smoothness and the inherent difficulty of learning is best understood for the Matérn covariance family and the limiting case of the squared exponential covariance. Combined, these covariance functions allow us to model functions with a continuum of smoothness. To this end, there is a general form of the Matérn covariance function modeling functions of any finite smoothness, controlled by a parameter $v>0::^{27}$

$$
K_{\mathrm{M}}(d ; v)=\frac{2^{1-v}}{\Gamma(v)}(\sqrt{2 v} d)^{v} K_{v}(\sqrt{2 v} d),
$$

where $d=\left|x-x^{\prime}\right|$ and $K_{v}$ is the modified Bessel function of the second kind. Sample paths from a centered Gaussian process with this covariance are $\lceil v\rceil-1$ times continuously differentiable, but the smoothness of sample paths is not as granular as a simple count of derivatives. Rather, the parameter $v$ allows us to fine-tune sample path smoothness as desired. ${ }^{28}$ Figure 10.3 illustrates sample paths generated from a Matérn covariance with a range of smoothness from $v=1.01$ to $v=2$. All of these samples are exactly once differentiable, but we might say that the $v=1.01$ samples are "just barely" so, and that the $v=2$ samples are "very nearly" twice differentiable.
![](https://cdn.mathpix.com/cropped/2023_09_22_751c1e67194d77e6639bg-09.jpg?height=422&width=528&top_left_y=958&top_left_x=1369)

Sample paths from a smooth GP (above) and a rough one (below). The rough samples have more degrees of freedom - and far more local maxima - and we might conclude they are harder to optimize as a result.

Matérn and squared exponential covariance functions: $\S 3.3, p \cdot 51$

27 The given expression has unit length and output scale; if desired, we can introduce parameters for these following $\S 3.4$.

smoothness parameter, $v$

28 This can be made precise through the coefficient of Hölder continuity in the "final" derivative of $K_{\mathrm{M}}$, which controls the smoothness of the final derivative of sample paths. Except when $v$ is an integer, the Matern covariance with parameter $v$ belongs to the Hölder space

$$
\mathcal{C}^{\alpha, \beta} ; \quad \alpha=\lfloor 2 v\rfloor ; \quad \beta=2 v-\lfloor 2 v\rfloor,
$$

and we can expect the final derivative of sample paths to be $(v-\lfloor v\rfloor)$-Hölder continuous. 29 We have $H[\mathbf{y} \mid f]=H[\mathbf{y} \mid \phi]$ by conditional independence (1.3).

30 For more on information gain, see $\S 6.3$, p. 115 . Thus far we have primarily concerned ourselves with information gain regarding $x^{*}$ or $f^{*}$; here we are reasoning about the function $f$ itself.

information capacity, $\gamma_{\tau}$
Taking the limit $v \rightarrow \infty$ recovers the squared exponential covariance $K_{\mathrm{SE}}$. This serves as the extreme end of the continuum, modeling functions with infinitely many continuous derivatives. Together, the Matérn and squared exponential covariances allow us to model functions with any smoothness $v \in(0, \infty]$; we will call this collection the Matérn family.

\section*{Information capacity}

We now require some way to relate the complexity of sample path behavior to our ability to learn about an unknown function. Information theory provides an answer through the concept of information capacity, the maximum rate of information transfer through a noisy observation mechanism.

In the analysis of Bayesian optimization algorithms, the central concern is how efficiently we can learn about a GP-distributed objective function $\mathcal{G P}(f ; \mu, K)$ through a set of $\tau$ noisy observations $\mathcal{D}_{\tau}$. The information capacity of this observation process, as a function of the number of observations $\tau$, provides a fundamental bound on our ability to learn about $f$. For this discussion, let us adopt the common observation model of independent and homoskedastic additive Gaussian noise with scale $\sigma_{n}>0(2.16)$. In this case, information capacity is a function of:

- the covariance $K$, which determines the information content of $f$, and

- the noise scale $\sigma_{n}$, which limits the amount of information obtainable through a single observation.

The information regarding $f$ contained in an arbitrary set of observations $\mathcal{D}=(\mathbf{x}, \mathbf{y})$ can be quantified by the mutual information (A.16): ${ }^{29}$

$$
I(\mathbf{y} ; f)=H[\mathbf{y}]-H[\mathbf{y} \mid \boldsymbol{\phi}]=\frac{1}{2} \log \left|\mathbf{I}+\sigma_{n}^{-2} \Sigma\right|,
$$

where $\Sigma=K(\mathbf{x}, \mathbf{x})$. Note that the entropy of $\mathbf{y}$ given $\boldsymbol{\phi}$ does not depend on the actual value of $\boldsymbol{\phi}$, and thus the mutual information $I(\mathrm{y} ; f)$ is also the information gain about $f$ provided by the data. ${ }^{30}$ The information capacity (also known as the maximum information gain) of this observation process is now the maximum amount of information about $f$ obtainable through any set of $\tau$ observations:

$$
\gamma_{\tau}=\sup _{|\mathbf{x}|=\tau} I(\mathbf{y} ; f) .
$$

\section*{Known bounds on information capacity}

The information capacity of a GP observation process (10.14) is commonly invoked in theoretical analyses of algorithms making use of this model class. Unfortunately, working with information capacity is somewhat unwieldy for two reasons. First, as mentioned above, information capacity is a function of the covariance function $K$, which can be verified through the explicit formula in (10.13). Thus performance guarantees in terms of information capacity require further analysis to derive explicit results for a particular choice of model. Second, information capacity of any given model is in general NP-hard to compute due to the difficulty of the set function maximization in (10.14). ${ }^{31}$

For these reasons, the typical strategy is to derive agnostic convergence results in terms of the information capacity of an arbitrary observation process, then seek to derive bounds on the information capacity for notable covariance functions such as those in the Matérn family. A common proof strategy for bounding the information gain is to relate the information capacity to the spectrum of the covariance function, with faster spectral decay yielding stronger bounds on information capacity. The first explicit bounds on information capacity for the Matérn and squared exponential covariances were provided by SRINIVAs et al., ${ }^{32}$ and the bounds for the Matérn covariance have since been sharpened using similar techniques. ${ }^{33,34}$

For a compact domain $\mathcal{X} \subset \mathbb{R}^{d}$ and fixed noise scale $\sigma_{n}$, we have the following asymptotic bounds on the information capacity. For the Matérn covariance function with smoothness $v$, we have: ${ }^{34}$

$$
\gamma_{\tau}=\mathcal{O}\left(\tau^{\alpha}(\log \tau)^{1-\alpha}\right), \quad \alpha=\frac{d}{2 v+d}
$$

and for the squared exponential covariance, we have: ${ }^{32}$

$$
\gamma_{\tau}=\mathcal{O}\left((\log \tau)^{d+1}\right) .
$$

These results embody our stated goal of characterizing smoother sample paths (as measured by $v$ ) as being inherently less complex than rougher sample paths. The information capacity decreases steadily as $v \rightarrow \infty$, eventually dropping to only logarithmic growth in $\tau$ for the squared exponential covariance. The correct interpretation of this result is that the smoother sample paths require less information to describe, and thus the maximum amount of information one could learn is limited compared to rougher sample paths.

\section*{Bounding the sum of predictive variances}

A key result linking information capacity to an optimization policy is the following. ${ }^{32}$ Suppose some optimization policy selected an arbitrary sequence of $\tau$ observation locations $\left\{x_{i}\right\}$, and let $\left\{\sigma_{i}\right\}$ be the corresponding predictive standard deviations at the time of selection. By applying the chain rule for mutual information, we can rewrite the information gain (10.13) from observing $y$ in terms of the marginal predictive variances:

$$
I(\mathbf{y} ; f)=\frac{1}{2} \sum_{i=1}^{\tau} \log \left(1+\frac{\sigma_{i}^{2}}{\sigma_{n}^{2}}\right) .
$$

Now assume that the prior covariance function is bounded: $K(x, x) \leq M$. Noting that $z^{2} / \log \left(1+z^{2}\right)$ is increasing for $z>0$ and that $\sigma_{i}^{2} \leq M$, the following inequality holds for every observation:

$$
\sigma_{i}^{2} \leq \frac{M}{\log \left(1+\sigma_{n}^{-2} M\right)} \log \left(1+\frac{\sigma_{i}^{2}}{\sigma_{n}^{2}}\right) .
$$

31 C.-W. ko et al. (1995). An Exact Algorithm for Maximum Entropy Sampling. Operations Research $43(4): 684^{-691 .}$

32 N. SRINIVAS et al. (2010). Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design. ICML 2010.

33 D. JANZ et al. (2020). Bandit Optimisation of Functions in the Matérn Kernel RKHS. AISTATS 2020

34 s. vAKILI et al. (2021b). On Information Gain and Regret Bounds in Gaussian Process Bandits. AISTATS 2021

predictive standard deviation of $\phi_{i}, \sigma_{i}$

bound on prior variance $K(x, x), M$ posterior variance is nonincreasing: $\S 2.2$, p. 22 35 If an explicit leading constant is desired, we have $\sum_{i} \sigma_{i}^{2} \leq c \gamma_{\tau}$ with

$$
c=\frac{2 M}{\log \left(1+\sigma_{n}^{-2} M\right)} .
$$

analysis: frequentist or Bayesian?

observations: noisy or exact?

asymptotic behavior with logarithmic factors suppressed, $\mathcal{O}^{*}$

assumption: $f$ is a sample path from $\mathcal{G} \mathcal{P}(f ; \mu \equiv 0, K)$

assumption: observations are corrupted by iid Gaussian noise with scale $\sigma_{n}$
We can now bound the sum of the predictive variances in terms of the information capacity: ${ }^{35}$

$$
\sum_{i=1}^{\tau} \sigma_{i}^{2}=\mathcal{O}\left(\gamma_{\tau}\right)
$$

This bound will repeatedly prove useful below.

\subsection*{BAYESIAN REGRET WITH OBSERVATION NOISE}

We have now covered the background required to understand - or at least to meaningfully interpret - most of the theoretical convergence results appearing in the literature. In the remainder of the chapter we will summarize some notable results built upon these ideas. The literature is expansive, and navigation can be challenging. Broadly, we can categorize these results according to certain dichotomies in their approach and focus, listed below.

- The first is whether the result is regarding the average-case (Bayesian) or worst-case (frequentist) regret. In the former case, we assume the objective function is a sample path from a Gaussian process $\mathcal{G P}(f ; \mu, K)$ and seek to bound the expected regret (10.6). In the latter case, we assume the objective function lies in some RKHS $\mathcal{H}_{K}$ with bounded norm (10.11) and seek to bound the worst-case regret on this space (10.7).

- The second is the assumption and treatment of observation noise. Stronger guarantees can often be derived in the noiseless setting, as we can learn much faster about the objective function. Observation noise is also modeled somewhat differently in the Bayesian and frequentist settings.

In this and the following sections, we will provide an overview of convergence results for all combinations of these choices: frequentist and Bayesian guarantees, with and without noise. For each of these cases, we will discuss both upper bounds on regret, which provide guarantees for the performance of specific Bayesian optimization algorithms, and also lower bounds on regret, which provide algorithm-agnostic bounds on the best possible performance. Here we will begin with results for Bayesian regret in the noisy setting.

To facilitate the discussion, we will adopt the notation $\mathcal{O}^{*}$ (sometimes written $\tilde{\mathcal{O}}$ in other texts) to describe asymptotic bounds in which dimension-independent $\operatorname{logarithmic}$ factors of $\tau$ are suppressed:

$$
f(\tau)=\mathcal{O}\left(g(\tau)(\log \tau)^{k}\right) ; \Longrightarrow f(\tau)=\mathcal{O}^{*}(g(\tau)) .
$$

\section*{Common assumptions}

In this section, we will assume that the objective function $f: \mathcal{X} \rightarrow \mathbb{R}$ is a sample path from a centered Gaussian process $\mathcal{G P}(f ; \mu \equiv 0, K)$, and that observation noise is independent, homoskedastic Gaussian noise with scale $\sigma_{n}>0(2.16)$. The domain $\mathcal{X}$ will at various times be either a finite set (as a stepping stone toward the continuous case) or a compact and convex subset of a $d$-dimensional cube: $\mathcal{X} \subset[0, m]$. We will also be assuming that the covariance function is continuous and bounded on $\mathcal{X}: K(x, x) \leq 1$. Since the covariance function is guaranteed to be bounded anyway (as $\mathcal{X}$ is compact) this simply fixes the scale without loss of generality.

\section*{Upper confidence bound}

In a landmark paper, SRINIVAs et al. derived sublinear cumulative regret bounds for the Gaussian process upper confidence bound (GP-UCB) policy in the Bayesian setting with noise. ${ }^{36}$ The authors considered policies of the form (8.25):

$$
x_{i}=\underset{x \in \mathcal{X}}{\arg \max } \mu+\beta_{i} \sigma,
$$

where $x_{i}$ is the point chosen in the $i$ th iteration of the policy, $\mu$ and $\sigma$ are shorthand for the posterior mean and standard deviation of $\phi=f(x)$ given the data available at time $i, \mathcal{D}_{i-1}$, and $\beta_{i}$ is a time-dependent exploration parameter. The authors were able to demonstrate that if this exploration parameter is carefully tuned over the course of optimization, then the cumulative regret of the policy can be asymptotically bounded, with high probability, in terms of the information capacity.

We will discuss this result and its derivation in some detail below, as it demonstrates important proof strategies that will be repeated throughout this section and the next.

\section*{Regret bound on finite domains}

To proceed, we first assume the domain $\mathcal{X}$ is finite; we will lift this to the continuous case shortly via a secondary argument. ${ }^{37}$

The construction of the UCB policy suggests the following confidence interval condition is likely to hold for any point $x \in \mathcal{X}$ at any time $i$ :

$$
\phi \in\left[\mu-\beta_{i} \sigma, \mu+\beta_{i} \sigma\right]=\mathcal{B}_{i}(x) .
$$

In fact, we can show that for appropriately chosen confidence parameters $\left\{\beta_{i}\right\}$, every such confidence interval is always valid, with high probability.

This may seem like a strong claim, but it is simply a consequence of the exponentially decreasing tails of the Gaussian distribution. At time $i$, we can use tail bounds on the Gaussian CDF to bound the probability of a given confidence interval failing in terms of $\beta_{i},{ }^{38}$ then use the union bound to bound the probability of (10.20) failing anywhere at time $i{ }^{39}$ Finally, we show that by increasing the confidence parameter $\beta_{i}$ over time - so that the probability of failure decreases suitably quickly - the probability of failure anywhere and at any time is small. SRINIVAS et al showed in particular that for any $\delta \in(0,1)$, taking ${ }^{40}$

$$
\beta_{i}^{2}=2 \log \left(\frac{i^{2} \pi^{2}|\mathcal{X}|}{6 \delta}\right)
$$

assumption: $K(x, x) \leq 1$ is bounded

upper confidence bound: § 7.8, p. 145, § 8.4, p. 170

36 N. SRINIVAs et al. (2010). Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design. ICML 2010.

information capacity: § 10.3, p. 222

37 The general strategy for this case was established in the linear bandit setting in:

v. DANI et al. (2008). Stochastic Linear Optimization under Bandit Feedback. COLT 2008.

step 1: show confidence intervals (10.20) are universally valid with high probability

38 SRINIVAs et al. use

$$
\operatorname{Pr}\left(\phi \notin \mathcal{B}_{i}(x)\right) \leq \exp \left(-\beta_{i}^{2} / 2\right) .
$$

39 Using the above bound, the probability of failure anywhere at time $i$ is at most

$$
|\mathcal{X}| \exp \left(-\beta_{i}^{2} / 2\right) \text {. }
$$

40 The mysterious appearance of $\pi^{2} / 6$ comes from

$$
\sum_{i=1}^{\infty} \frac{1}{i^{2}}=\frac{\pi^{2}}{6} .
$$

step 2: bound instantaneous regret by width of chosen confidence interval

the lower bound of every point holds for $f^{*}$

the upper bound of the chosen point holds for every function value

41 D. RUSSO and B. VAN ROY (2014). Learning to Optimize via Posterior Sampling. Mathematics of Operations Research 39(4):1221-1243.

step 3: bound cumulative regret via bound on instantaneous regret

42 We have $M=1$ as a bound on $K(x, x)$ according to our common assumptions (p. 224).

assumption: $\mathcal{X}$ is convex and compact assumption: $f$ is Lipschitz continuous guarantees that the confidence intervals (10.20) are universally valid with probability at least $1-\delta$.

With this, we are actually almost finished. The next key insight is that if the confidence intervals in (10.20) are universally valid, then we can bound the instantaneous regret of the policy in every iteration by noting that the confidence interval of the chosen point always contains the global optimum:

$$
f^{*} \in \mathcal{B}_{i}\left(x_{i}\right) .
$$

To show this, we first note that the lower bound of every confidence interval applies to $f^{*}$, as all intervals are valid and $f^{*}$ is the global maximum. Further, the upper bound of the chosen confidence interval is valid for $f^{*}$, as it is the maximal upper bound by definition (and thus is valid for every function value) (10.19).

This result allows us to bound, with high probability, the instantaneous regret (10.2) in every iteration by the width of the confidence interval of the chosen point:

$$
\rho_{i} \leq 2 \beta_{i} \sigma_{i}
$$

RUSSO and VAN ROY interpret this bound on the instantaneous regret as guaranteeing that regret can only be high when we also learn a great deal about the objective function to compensate (10.17). ${ }^{41}$

Finally, we bound the cumulative regret. Assuming the confidence intervals (10.20) are universally valid, we may bound the sum of the squared instantaneous regret up to time $\tau$ by

$$
\sum_{i=1}^{\tau} \rho_{i}^{2} \leq 4 \sum_{i=1}^{\tau} \beta_{i}^{2} \sigma_{i}^{2} \leq 4 \beta_{\tau}^{2} \sum_{i=1}^{\tau} \sigma_{i}^{2}=\mathcal{O}\left(\beta_{\tau}^{2} \gamma_{\tau}\right)
$$

From left-to-right, we plug in (10.23), note that $\left\{\beta_{i}\right\}$ is nondecreasing (10.21), and appeal to the information capacity bound on the sum of predictive variances (10.18). ${ }^{42}$ Plugging in $\beta_{\tau}$ and appealing to the CauchySchwartz inequality gives

$$
R_{\tau}=\mathcal{O}^{*}\left(\sqrt{\tau \gamma_{\tau} \log |\mathcal{X}|}\right)
$$

with probability at least $1-\delta$.

\section*{Extending to continuous domains}

The above analysis can be extended to continuous domains via a discretization argument. The proof is technical, but the technique is often useful in other settings, so we provide a sketch of a general argument:

- We assume that the domain $\mathcal{X} \subset[0, m]^{d}$ is convex and compact.

- We assume that the objective function is continuously differentiable. As $\mathcal{X}$ is compact, this implies the objective is in fact Lipschitz continuous with some Lipschitz constant $L$. - Purely for the sake of analysis, in each iteration $i$, we discretize the domain with a grid $\mathcal{X}_{i} \subset \mathcal{X}$; these grids become finer over time and eventually dense. The exact details vary, but it is typical to take a regular grid in $[0, m]^{d}$ with spacing on the order of $\mathcal{O}\left(1 / i^{c}\right)$ (for some constant $c$ not depending on $d$ ) and take the intersection with $\mathcal{X}$. The resulting discretizations have size $\log \left|\mathcal{X}_{i}\right|=\mathcal{O}(d \log i)$.

- We note that Lipschitz continuity of $f$ allows us to extend valid confidence intervals for the function values at $\mathcal{X}_{i}$ to all of $\mathcal{X}$ with only slight inflation. Namely, for any $x \in \mathcal{X}$, let $[x]_{i}$ denote the closest point to $x$ in $\mathcal{X}_{i}$. By Lipschitz continuity, a valid confidence interval at $[x]_{i}$ can be extended to one at $x$ with inflation on the order of $\mathcal{O}\left(L / i^{c}\right)$ due to the discretizations becoming finer over time.

- With this intuition, we design a confidence parameter sequence $\left\{\beta_{i}\right\}$ guaranteeing that the function values on both the grids $\left\{\mathcal{X}_{i}\right\}$ and at the points chosen by the algorithm $\left\{x_{i}\right\}$ always lie in their respective confidence intervals with high probability. As everything is discrete, we can generally start from a guarantee such as (10.21), replace $|\mathcal{X}|$ with $\left|\mathcal{X}_{i}\right|$, and fiddle with constants as necessary.

- Finally, we proceed as in the finite case. We bound the instantaneous regret in each iteration in terms of the width of the confidence intervals of the selected points, noting that any extra regret due to discretization shrinks rapidly as $\mathcal{O}\left(L / i^{c}\right)$ and, if we are careful, does not affect the asymptotic regret. Generally the resulting bound simply replaces any factors of $\log |\mathcal{X}|$ with factors of $\log \left|\mathcal{X}_{i}\right|=\mathcal{O}(d \log i)$.

For the particular case of bounding the Bayesian regret of GP-UCB, we can effectively follow the above argument but must deal with some nuance regarding the Lipschitz constant of the objective function. First, note that if the covariance function of our Gaussian process is smooth enough, its sample paths will be continuously differentiable ${ }^{43}$ and Lipschitz continuous - but with random Lipschitz constant. SRINIVAs et al. proceed by assuming that the covariance function is sufficiently smooth to ensure that the Lipschitz constant has an exponential tail bound of the following form:

$$
\forall \lambda>0: \operatorname{Pr}(L>\lambda) \leq d a \exp \left(-\lambda^{2} / b^{2}\right),
$$

where $a, b>0$; this allows us to derive a high-probability upper bound on $L$. We will say more about this assumption shortly.

SRINIVAs et al. showed that, under this assumption, a slight modification of the confidence parameter sequence from the finite case ${ }^{44}$ (10.21) is sufficient to ensure

$$
R_{\tau}=\mathcal{O}^{*}\left(\sqrt{d \tau \gamma_{\tau}}\right)
$$

with high probability. Thus, assuming the Gaussian process sample paths are smooth enough, the cumulative regret of GP-UCB on continuous domains grows at rate comparable to the discrete case.

Plugging in the information capacity bounds in (10.15-10.16) and dropping the $\sqrt{d}$ factor, we have the following high-probability regret step 1: discretize the domain with a sequence of increasingly fine grids $\left\{X_{i}\right\}$

size of discretizations: $\log \left|\mathcal{X}_{i}\right|=\mathcal{O}(d \log i)$

step 2: valid confidence intervals on $\mathcal{X}_{i}$ can be extended to all of $\mathcal{X}$ with slight inflation

inflation in iteration $i: \mathcal{O}\left(L / i^{c}\right)$

step 3: find a confidence parameter sequence $\left\{\beta_{i}\right\}$ guaranteeing validity on $\left\{x_{i}\right\}$ and $\left\{\mathcal{X}_{i}\right\}$

step 4: proceed as in the discrete case

sample path differentiability: § 2.6, p. 30

43 Hölder continuity of the derivative process covariance is sufficient; this holds for the Matérn family with $v>1$.

44 Specifically, SRINIVAS et al. took:

$$
\begin{aligned}
\beta_{i}^{2}=2 & \log \left(\frac{2 i^{2} \pi^{2}}{3 \delta}\right) \\
+ & 2 d \log \left(i^{2} d b m \sqrt{\log (4 d a / \delta)}\right) .
\end{aligned}
$$

The resulting bound holds with probability at least $1-\delta$. 45 A. W. VAN DER VAART and J. A. WELLNER (1996). Weak Convergence and Empirical Processes with Applications to Statistics. Springer-Verlag. [proposition A.2.1]
46 s. GHOSAL and A. Roy (2006). Posterior Consistency of Gaussian Process Prior for Nonparametric Binary Regression. The Annals of Statistics 34(5):2413-2429. [theorem 5]

47 N. DE fREITAs et al. (2012a). Regret Bounds for Deterministic Gaussian Process Bandits. arXiv: 1203.2177 [cs.LG]. [lemma 1] bounds for specific covariance functions. For the Matérn covariance, we have

$$
R_{\tau}=\mathcal{O}^{*}\left(\tau^{\alpha}(\log \tau)^{\beta}\right), \quad \alpha=\frac{v+d}{2 v+d}, \quad \beta=\frac{2 v}{2 v+d},
$$

and for the squared exponential we have

$$
R_{\tau}=\mathcal{O}^{*}\left(\sqrt{\tau(\log \tau)^{d}}\right)
$$

The growth is sublinear for all values of the smoothness parameter $v$, so GP-UCB achieves no regret with high probability for the Matérn family.

\section*{The Lipschitz tail bound condition}

Finally, we address the exponential tail bound assumption on the Lipschitz constant (10.26). Despite its seeming strength in controlling the behavior of sample paths, it is actually a fairly weak assumption on the Gaussian process. In fact, all that is needed is that sample paths be continuously differentiable. In this case, each coordinate of the gradient, $f_{i}=\partial f / \partial x_{i}$, is a sample path continuous Gaussian process. By compactness, each $f_{i}$ is then almost surely bounded on $\mathcal{X}$. Now the Borell-Tis inequality ensures that for any $\lambda>0$ we have: ${ }^{45}$

$$
\operatorname{Pr}\left(\max \left|f_{i}\right|>\lambda\right) \leq 4 \exp \left(-\lambda^{2} / b_{i}^{2}\right), \quad b_{i}=2 \sqrt{2} \mathbb{E}\left[\max \left|f_{i}\right|\right] . \quad \text { (10.30) }
$$

Taking a union bound then establishes a bound of the desired form (10.26). In particular, this argument applies to the Matérn covariance with $v>1$, which has continuously differentiable sample paths. Thus GP-UCB can achieve sublinear cumulative regret for all but the roughest of sample paths.

However, the bound in (10.30) does not immediately lead to an algorithm we can realize in practice, as the expected extremum $\mathbb{E}\left[\max \left|f_{i}\right|\right]$ is difficult to compute (or even bound). Things simplify considerably if sample paths are twice differentiable, in which case GHOSAL and ROY showed how to derive bounds of the above form (10.30) for each of the coordinates of the gradient process with explicitly computable constants, ${ }^{46}$ yielding a practical algorithm for the Matern covariance with $v>2$.

Fortunately, intricate arguments regarding the objective function's Lipschitz constant are only necessary due to its randomness in the Bayesian setting. In the frequentist setting, where $f$ is in some RKHs ball $\mathcal{H}_{K}[B]$, we can immediately derive a hard upper bound $L$ in terms of $K$ and $B{ }^{47}$

\section*{Bounding the expected regret}

Without too much effort, we may augment the high-probability bound presented above with a corresponding bound on the expected regret (10.6). However, note that the high-probability regret bound is stronger in the sense that we can guarantee good performance not only on average but also in extremely "unlucky" scenarios as well. For simplicity, let us begin again with a finite domain $\mathcal{X}$. Consider a run of the GP-UCB algorithm in (10.19) for some confidence parameter sequence $\left\{\beta_{i}\right\}$ to be determined shortly. Let $\left\{x_{i}\right\}$ be the sequence of selected points and let $x^{*}$ be a global optimum of $f$. Define

$$
u_{i}=\mu_{i}+\beta_{i} \sigma_{i} ; \quad u_{i}^{*}=\mu_{i}^{*}+\beta_{i} \sigma_{i}^{*}
$$

to be, respectively, the upper confidence bound associated with $x_{i}$ at the time of its selection and the upper confidence bound associated with $x^{*}$ at the same time. Since the algorithm always maximizes the upper confidence bound in each iteration, we always have $u_{i} \leq u_{i}^{*}$. This allows us to bound the expected regret as follows:

$$
\mathbb{E}\left[R_{\tau}\right]=\sum_{i=1}^{\tau} \mathbb{E}\left[f^{*}-\phi_{i}\right] \leq \sum_{i=1}^{\tau} \mathbb{E}\left[f^{*}-u_{i}^{*}\right]+\sum_{i=1}^{\tau} \mathbb{E}\left[u_{i}-\phi_{i}\right], \quad
$$

where we have added $u_{i}-u_{i}^{*} \geq 0$ to each term and rearranged.

In their analysis of the Gaussian process Thompson sampling algorithm (discussed below), RUSSO and VAN ROY showed how to bound each of the terms on the right-hand side. First, they show that the confidence parameter sequence (compare with the analogous and spiritually equivalent sequence used above (10.21))

$$
\beta_{i}^{2}=2 \log \left(\frac{\left(i^{2}+1\right)|\mathcal{X}|}{\sqrt{2 \pi}}\right)
$$

is sufficient to bound the first term by a constant: $\sum_{i} \mathbb{E}\left[f^{*}-u_{i}^{*}\right] \leq 1{ }^{48}$ The second term can then be bounded in terms of the information capacity following our previous discussion $(10.18,10.24):{ }^{49}$

$$
\sum_{i=1}^{\tau} \mathbb{E}\left[u_{i}-\phi_{i}\right]=\sum_{i=1}^{\tau} \beta_{i} \sigma_{i} \leq \beta_{\tau} \sum_{i=1}^{\tau} \sigma_{i}=\mathcal{O}^{*}\left(\sqrt{\tau \gamma_{\tau} \log |\mathcal{X}|}\right) .
$$

As before, we may extend this result to the continuous case via a discretization argument to prove $\mathbb{E}\left[R_{\tau}\right]=\mathcal{O}^{*}\left(\sqrt{d \tau \gamma_{\tau}}\right)$, matching the high-probability bound (10.27). ${ }^{50}$

\section*{Thompson sampling}

RUSSO and VAN ROY developed a general approach for transforming Bayesian regret bounds for a wide class of UCB-style algorithms into regret bounds for analogous Thompson sampling algorithms. ${ }^{51}$

Namely, consider a UCB policy selecting a sequence of points $\left\{x_{i}\right\}$ for observation by maximizing a sequence of "upper confidence bounds," which here can be any deterministic functions of the observed data, regardless of their statistical validity. Let $\left\{u_{i}\right\}$ be the sequence of upper confidence bounds associated with the selected points at the time of their selection, and let $\left\{u_{i}^{*}\right\}$ be the sequence of upper confidence bounds associated with a given global optimum, $x$.
48 This is again a consequence of the rapidly decaying tails of the Gaussian distribution; note that $f^{*}-u_{i}^{*}$ has distribution

$$
\mathcal{N}\left(-\beta_{i} \sigma_{i}^{*},\left(\sigma_{i}^{*}\right)^{2}\right) .
$$

As $\beta_{i}$ increases without bound, this term will eventually be negative with overwhelming probability.

$49 u_{i}-\phi_{i}$ has distribution $\mathcal{N}\left(\beta_{i} \sigma_{i}, \sigma_{i}^{2}\right)$.

50 Although used for Thompson sampling, the discretization argument used in the below reference suffices here as well:

K. KANDASAmY et al. (2018). Parallelised Bayesian Optimisation via Thompson Sampling. AISTATS 2018. [appendix, theorem 11]

Thompson sampling: § 7.9, p. 148, §8.7, p. 176

51 D. RUSSO and B. VAN ROY (2014). Learning to Optimize via Posterior Sampling. Mathematics of Operations Research 39(4):1221-1243. 52 K. KANDASAmy et al. (2018). Parallelised Bayesian Optimisation via Thompson Sampling. AISTATS 2018 .

$$
r_{\tau} \leq \frac{R_{\tau}}{\tau}
$$

53 There is an additional multiplicative factor of $\sqrt{\log |\mathcal{X}|}$ in the finite case and $\sqrt{d}$ in the continuous case.
Now consider a corresponding Thompson sampling policy selecting $x_{i} \sim p\left(x^{*} \mid \mathcal{D}_{i-1}\right)$. Note that, given the observed data, $x_{i}$ and $x^{*}$ are identically distributed by design; because the upper confidence bounds are deterministic functions of the observed data, $u_{i}$ and $u_{i}^{*}$ are thus identically distributed as well. This allows us to express the expected cumulative regret of the Thompson sampling policy entirely in terms of the upper confidence bounds:

$$
\mathbb{E}\left[R_{\tau}\right]=\sum_{i=1}^{\tau} \mathbb{E}\left[f^{*}-u_{i}^{*}\right]+\sum_{i=1}^{\tau} \mathbb{E}\left[u_{i}-\phi_{i}\right] .
$$

This is of the exactly same form as the bound derived above for GPUCB (10.31), leading immediately to a regret bound for GP-TS matching that for GP-UCB (10.32)

$$
\mathbb{E}\left[R_{\tau}\right]=\mathcal{O}^{*}\left(\sqrt{\tau \gamma_{\tau} \log |\mathcal{X}|}\right) ;
$$

once again, we may extend this result to the continuous case to derive a bound matching that for GP-UCB $(10.27) .{ }^{52}$

\section*{Upper bounds on simple regret}

We may use the bound on simple regret in terms of the average regret in (10.4) to derive bounds on the simple regret of GP-UCB and GP-Ts policies. Dropping dependence on the domain size,${ }^{53}$ we have

$$
r_{\tau}=\mathcal{O}^{*}\left(\sqrt{\gamma_{\tau} / \tau}\right)
$$

for both algorithms, where this is to be understood as either a highprobability (GP-UCB) or expected-case (GP-UCB, GP-TS) result.

Plugging in the information capacity bounds (10.15-10.16), we have the following bounds for the simple regret for specific covariance functions. For the Matérn covariance in dimension $d$, we have

$$
r_{\tau}=\mathcal{O}^{*}\left(\tau^{\alpha}(\log \tau)^{\beta}\right), \quad \alpha=-\frac{v}{2 v+d}, \quad \beta=\frac{2 v}{2 v+d},
$$

and for the squared exponential we have

$$
r_{\tau}=\mathcal{O}^{*}\left(\sqrt{(\log \tau)^{d} / \tau}\right)
$$

\section*{Lower bounds and tightness of existing algorithms}

We have now derived upper bounds on the regret of particular Bayesian optimization algorithms in the Bayesian setting with noise. A natural question is whether we can derive corresponding algorithm-agnostic lower regret bounds establishing the fundamental difficulty of optimization in this setting, which might hint at how much room for improvement there may be.

Lower bounds on Bayesian regret are not easy to come by. As we will see, the frequentist setting offers considerable flexibility in deriving 

![](https://cdn.mathpix.com/cropped/2023_09_22_751c1e67194d77e6639bg-19.jpg?height=340&width=1017&top_left_y=458&top_left_x=268)

lower bounds, as we can construct explicit objective functions in a given RKHS, then prove that they are difficult to optimize. In the Bayesian setting, the objective function is random, so we must instead seek explicit distributions over objective functions with enough structure that we can bound the expected regret, a much more challenging task.

That said, nontrivial lower bounds have been derived in this setting, most notably on the unit interval $\mathcal{X}=[0,1]$. Under the assumption of a stationary Gaussian process with twice differentiable sample paths, SCARLETT demonstrated the following high-probability lower bound on the expected cumulative regret of any optimization algorithm: ${ }^{54}$

$$
\mathbb{E}\left[R_{\tau}\right]=\Omega(\sqrt{\tau}) .
$$

This result is as about as good as we could hope for, as it matches the lower bound for optimizing convex functions. ${ }^{55}$

The key idea behind this result is to identify multiple plausible objective functions with identical distributions such that an optimization algorithm is prone to "prefer the wrong function," and in doing so, incur high regret. To illustrate the construction, we imagine an objective function on the unit interval is generated indirectly by the following process. We first realize an initial sample path $f$ on the larger domain $[-\Delta, 1+\Delta]$, for some $\Delta>0 . .^{56}$ We then take the objective function to be one of the following translations of $f$ with equal probability:

$$
f^{+}: x \mapsto f(x+\Delta) ; \quad f^{-}: x \mapsto f(x-\Delta) .
$$

As the prior process is assumed to be stationary, this procedure does not change the distribution of the objective function.

To proceed, we consider optimization of the generated objective given access to the initial reference function $f .{ }^{57}$ We now frame optimization in terms of using a sequence of noisy observations to determine which of the two possible translations (10.36) represents the objective function; see Figure 10.4. Fano's inequality, ${ }^{58}$ an information-theoretic lower bound on error probability in adaptive hypothesis testing, then allows us to show that there is a significant probability that the data collected by any algorithm have better cumulative regret for the wrong translation. Finally, we may use assumed statistical properties of the objective function to show that when this happens, the cumulative regret is significant. ${ }^{59}$

On the rougher side of the spectrum, wANG et al. used the same proof strategy to bound the expected regret - both simple and cumulative - of
Figure 10.4: A sketch of SCARLETT's proof strategy. Given access to a reference function $f$, which of its translations by $\Delta, f^{-}$or $f^{+}$is the objective function?

54 J. SCARLETt (2018). Tight Regret Bounds for Bayesian Optimization in One Dimension. ICML 2018.

55 O. SHAMIR (2013). On the Complexity of Bandit and Derivative-Free Stochastic Convex Optimization. COLT 2013.

56 This introduces the additional minor requirement that the covariance function be defined on this larger domain.

57 This is a so-called genie argument: as the extra information (provided by a "genie") can be ignored, it cannot possibly impede optimization.

58 J. SCARLETt and V. CEVHAR (2021). An Introductory Guide to Fano's Inequality with Applications in Statistical Estimation. In: InformationTheoretic Methods in Data Science.

59 Specifically, we need that the global maximum of the reference function is unlikely to be "pushed off the edge" of the domain during translation, and that sample paths are likely to have locally quadratic behavior in a neighborhood of the global optimum. The lower bound on regret then holds with probability depending on these events. For more discussion, see:

N. DE Freitas et al. (2012b). Exponential Regret Bounds for Gaussian Process Bandits with Deterministic Observations. ICML 2012. $60 \mathrm{z}$. WANG et al. (2020b). Tight Regret Bounds for Noisy Optimization of a Brownian Motion. arXiv: 2001.09327 [cs.LG].

61 Again $v>2$ is sufficient for the Matérn family.
62 Recall that sample paths of a centered Gaussian process with covariance $K$ do not lie inside $\mathcal{H}_{K}$; see p. 219 any algorithm maximizing a (nondifferentiable) sample path of Brownian motion (from the Wiener process) on the unit interval: ${ }^{60}$

$$
\mathbb{E}\left[r_{\tau}\right]=\Omega(1 / \sqrt{\tau \log \tau}) ; \quad \mathbb{E}\left[R_{\tau}\right]=\Omega(\sqrt{\tau / \log \tau})
$$

Both wANG et al. and SCARLETT also described straightforward, but not necessarily practical, optimization algorithms to provide corresponding upper bounds on the unit interval. The algorithms are based on a simple branch-and-bound scheme whereby the domain is adaptively partitioned based on confidence bounds computed from available data. For relatively smooth sample paths, ${ }^{61}$ SCARLETT's algorithm achieves cumulative regret

$$
\mathbb{E}\left[R_{\tau}\right]=\mathcal{O}(\sqrt{\tau \log \tau})
$$

with high probability. This is within a factor of $\sqrt{\log \tau}$ of the corresponding lower bound (10.35), so there is not too much room for improvement on the unit interval. Note that both GP-UCB and GP-TS with the squared exponential covariance match this rate up to logarithmic factors (10.29), but SCARLETT's algorithm is better for the Matérn covariance for $v>2$ (10.28) (as assumed in the analysis). For Brownian motion sample paths, WANG et al. established the following upper bounds:

$$
\mathbb{E}\left[r_{\tau}\right]=\mathcal{O}(\log \tau / \sqrt{\tau}) ; \quad \mathbb{E}\left[R_{\tau}\right]=\mathcal{O}(\sqrt{\tau} \log \tau),
$$

which are within a factor of $(\log \tau)^{3 / 2}$ of the corresponding lower bounds (10.37), again fairly tight.

\section*{WORST-CASE REGRET WITH OBSERVATION NOISE}

We now turn our attention to worst-case (frequentist) results analogous to the expected-case (Bayesian) results discussed in the previous section. Here we no longer reason about the objective function as a sample path from a Gaussian process, but rather as a fixed, but unknown function lying in some reproducing kernel Hilbert space $\mathcal{H}_{K}$ with norm bounded by some constant $B$ (10.11). We also replace the assumption of independent Gaussian observation errors with somewhat more flexible and agnostic conditions. The goal is then to bound the worst-case (simple or cumulative) regret (10.7) of a given algorithm when applied to such a function, which we will notate with $\bar{r}_{\tau}[B]$ and $\bar{R}_{\tau}[B]$, respectively.

Compared with the analysis of Bayesian regret, the analysis of worstcase regret is complicated by model misspecification, as the model assumptions underlying the Bayesian optimization algorithms are no longer valid. ${ }^{62}$ We must therefore seek other methods of analysis.

\section*{Common assumptions}

In this section, we will assume that the objective function $f: \mathcal{X} \rightarrow \mathbb{R}$, where $\mathcal{X}$ is compact, lies in a reproducing kernel Hilbert space corresponding to covariance function $K$ and has bounded norm: $f \in \mathcal{H}_{K}[B]$ (10.11). As in the previous section, we will also assume that the covariance function is continuous and bounded on $\mathcal{X}: K(x, x) \leq 1$.

To model observations, we will assume that a sequence of observations $\left\{y_{i}\right\}$ at $\left\{x_{i}\right\}$ are corrupted by additive noise, $y_{i}=\phi_{i}+\varepsilon_{i}$, and that the distribution of the errors satisfies mild regularity conditions. First, we will assume each $\varepsilon_{i}$ has mean zero conditioned on its history:

$$
\mathbb{E}\left[\varepsilon_{i} \mid \varepsilon_{<i}\right]=0,
$$

where $\varepsilon_{<i}$ is the vector of errors occurring before time $i$. We will also make assumptions regarding the scale of the errors. The most typical assumption is that the distribution of each $\varepsilon_{i}$ is $\sigma_{n}-s u b$-Gaussian conditioned on its history, that is, the tail of the conditional distribution shrinks at least as quickly as a Gaussian distribution with variance $\sigma_{n}^{2}$ :

$$
\forall c>0: \operatorname{Pr}\left(\left|\varepsilon_{i}\right|>c \mid \varepsilon_{<i}\right) \leq 2 \exp \left(-\frac{1}{2} c^{2} / \sigma_{n}^{2}\right) .
$$

This condition is satisfied, for example, by a distribution bounded on the interval $\left[-\sigma_{n}, \sigma_{n}\right]$ and by any Gaussian distribution with standard deviation of at most $\sigma_{n}$.

Complementary with the above assumptions, the Bayesian optimization algorithms that we will analyze model the function with the centered Gaussian process $\mathcal{G P}(f ; \mu \equiv 0, K)$ and assume independent Gaussian observation noise with scale $\sigma_{n}$.

\section*{Upper confidence bound and Thompson sampling}

Sublinear bounds on cumulative regret have been established for both the Gaussian process upper confidence bound (GP-UCB) and Thompson sampling (GP-TS) algorithms in this setting, albeit with somewhat slower convergence rates that in the Bayesian setting. However, more complex algorithms built on similar ideas are able to close this gap, as we will discuss shortly.

The primary proof strategy in this setting is to bound the deviation of the Gaussian process posterior mean used in the algorithm from the objective function in the assumed RKHs. A prototypical result is to show that for some sequence of confidence parameters $\left\{\beta_{i}\right\}$, the confidence interval assumption (10.20) is universally valid with high probability:

$$
\phi \in\left[\mu-\beta_{i} \sigma, \mu+\beta_{i} \sigma\right] .
$$

This would then allow us to bound the cumulative regret in terms of the information capacity following our previous analysis.

The strongest known result of this form was derived by ABBASIYADKORI in the context of regression ${ }^{63}$ and later applied in the context of optimization by CHOWDHURY and GOPALAN. ${ }^{64}$ Namely, under our common assumptions for this section and the assumption of $\sigma_{n}$-sub-Gaussian errors, the following confidence parameter sequence yields universally valid confidence intervals with probability at least $1-\delta$ :

$$
\beta_{i}=B+\sigma_{n} \sqrt{2 \gamma_{i-1}+2 \log (1 / \delta)} .
$$

assumption: $K(x, x) \leq 1$ is bounded

assumption: errors have zero mean conditioned on their history

assumption: scale of errors is limited

sub-Gaussian distribution finite-domain Bayesian analysis of GP-UCB: p. 225

63 Y. ABBASI-YADKORI (2012). Online Learning for Linearly Parameterized Control Problems. Ph.D. thesis. University of Alberta. [theorem 3.11, remark 3.13]

64 S. R. CHOWDHURY and A. GOPALAN (2017). On Kernelized Multi-Armed Bandits. ICML 2017. [theorem 2] 65 If we wish to retain explicit dependence on the RKHS norm $B$, we have

$$
\bar{R}_{\tau}[B]=\mathcal{O}^{*}\left(\sqrt{\tau \gamma_{\tau}}\left(B+\sqrt{\gamma_{\tau}}\right)\right) .
$$

extending to continuous domains with a discretization argument: p. 226

66 s. R. CHOWDHURY and A. GOPALAN (2017). On Kernelized Multi-Armed Bandits. ICML 2017.

67 s. vaKiLI et al. (2021c). Open Problem: Tight Online Confidence Intervals for RKHs Elements. COLT 2021
68 A wider family of light-tailed noise distributions was also considered.

69 s. VAKILI et al. (2021a). Optimal Order Simple Regret for Gaussian Process Bandits. NeurIPS 2021. [theorem 1]

70 If $\mathcal{X}$ is finite, we can ensure universally valid confidence intervals at all times with probability $1-\delta$ as in (10.21):

$$
\beta_{i}=B+\sigma_{n} \sqrt{2 \log \left(i^{2} \pi^{2}|\mathcal{X}| / 6 \delta\right)} .
$$

For continuous domains, we can again rely on discretization arguments. For a convex and compact domain $\mathcal{X} \subset R^{d}$ coupled with a covariance function in the Matern family with $v>1$, we may take:

$$
\beta_{i}=B+\sigma_{n} \sqrt{d \log (i / \delta)} .
$$

Unfortunately, these confidence parameters are much larger than needed in the Bayesian setting (see (10.21) and footnote 44), and the resulting regret bounds suffer as a result. Following the same argument leading up to (10.24), we may show that, with high probability, the worstcase cumulative regret of GP-UCB is ${ }^{65}$

$$
\bar{R}_{\tau}[B]=\mathcal{O}^{*}\left(\sqrt{\tau} \gamma_{\tau}\right) ;
$$

the same bound holds for GP-TS with an additional factor of $\sqrt{d}$ stemming from a discretization argument. ${ }^{66}$

There is a gap on the order of $\sqrt{\gamma_{\tau}}$ between these bounds and the analogous bounds in the Bayesian setting (10.27), and for the Matérn family, the best known bounds on the information capacity (10.15-10.16) only guarantee sublinear cumulative regret when $d<2 v$. It is unclear whether this gap stems from inherent weakness of the GP-UCB and GP-TS algorithms in this setting or merely from weakness in their analysis to date, and there may be room for improvement by proving that a tighter confidence parameter sequence than above (10.40) would suffice for universally valid confidence intervals. ${ }^{67}$

\section*{Simple regret bounds in the nonadaptive setting}

Although the best (yet) known worst-case regret bounds for the vanilla GP-UCB and GP-TS algorithms do not grow favorably compared with their Bayesian regret counterparts, several authors have been able to construct spiritually related algorithms that close this gap.

A common idea in the development of these algorithms is to carefully sever some of the dependence between the sequence of points observed throughout optimization $\left\{x_{i}\right\}$ and the corresponding observed values $\left\{y_{i}\right\}$. This may seem counterintuitive, as the power of Bayesian optimization algorithms surely stems from their adaptivity! However, injecting some independence allows us to appeal to significantly more powerful concentration results in their analysis.

The following result illustrates the power of nonadaptivity in this setting. Consider conditioning on a an arbitrary sequence of observation locations $\left\{x_{i}\right\}$ chosen independently of the corresponding values $\left\{y_{i}\right\}$. Under our common assumptions for this section and the assumption of $\sigma_{n}$-sub-Gaussian errors, ${ }^{68}$ VAKILI et al. showed that for any $x \in \mathcal{X}$, the following confidence interval condition holds with probability at least $1-\delta:^{69}$

$$
\phi \in[\mu-\beta \sigma, \mu+\beta \sigma] ; \quad \beta=B+\sigma_{n} \sqrt{2 \log (1 / \delta)} .
$$

Although this bound only holds for a fixed point at a fixed time, we may extend this result to hold at all points and at all times with only logarithmic inflation of the confidence parameter.$^{70}$ Comparing to CHOWDHURY and GOPALAN's corresponding result in the adaptive setting (10.40), the confidence parameter sequence here does not depend on the information capacity. Thus we can ensure dramatically tighter confidence intervals when observations are designed nonadaptively. This result has enabled several strong convergence guarantees in the worst-case setting with noise. VAKILI et al., for example, were able to show that the simple policy of uncertainty sampling (also known as maximum variance reduction) achieves near-optimal simple regret in the worst case. ${ }^{71}$ Uncertainty sampling designs each observation by maximizing the posterior predictive variance, which for a Gaussian process does not depend on the observed values (2.19). One insightful interpretation of uncertainty sampling is that, from the perspective of the Gaussian process used in the algorithm, it acts to greedily maximize the information about the objective function revealed by the observations (10.17) such a policy is sometimes described as performing "pure exploration."

For typical (stationary) processes on typical (convex and compact) domains, uncertainty sampling designs observations seeking to, in a rough sense, cover the domain as evenly as possible; an example run is shown in the marginal figure. In light of the concentration result above, we might expect this space-filling behavior to guarantee that the maximum of the posterior mean cannot stray too far from the maximum of the objective function. VAKILI et al. showed that this is indeed the case for the Matérn family. Namely, under our shared assumptions for this section and the minor additional assumption that $\mathcal{X}$ is convex, the simple regret of uncertainty sampling is, with high probability, bounded by $^{72}$

$$
\bar{r}_{\tau}[B]=\mathcal{O}^{*}\left(\sqrt{d \gamma_{\tau} / \tau}\right) .
$$

This rate is within logarithmic factors of the best-known lower bounds for convergence in simple regret, as discussed later in this section.

\section*{Closing the cumulative regret bound gap}

Although uncertainty sampling achieves near-optimal simple regret in the worst case, we cannot expect to meaningfully bound its cumulative regret due to its nonadaptive nature. However, the results discussed above have proven useful for building algorithms that do achieve nearoptimal cumulative regret in the worst case. To make the best use of results such as VAKILI et al.'s toward this end, algorithm development becomes a careful balancing act of including enough adaptivity to ensure low cumulative regret while including enough independence to ensure that we can prove low cumulative regret.

LI and SCARLETT proposed and analyzed one relatively simple algorithm in this direction. ${ }^{73}$ Given an observation budget of $\tau$, we proceed by constructing a sequence of batch observations in a nonadaptive fashion. After each batch observation is resolved, we pause to derive the resulting confidence intervals from the observed data and eliminate any points from consideration whose upper confidence bound is dominated by the maximum lower confidence bound among the points remaining under consideration; see the marginal figure. This occasional elimination step is the only time the algorithm exhibits adaptive behavior - in fact, after every elimination step, the GP used in the algorithm is "reset" to the prior
71 This result uses the alternative definition of simple regret mentioned in footnote 5 : the difference between the global maximum and the maximum of the posterior mean.

![](https://cdn.mathpix.com/cropped/2023_09_22_751c1e67194d77e6639bg-23.jpg?height=349&width=548&top_left_y=922&top_left_x=1368)

A sequence of 20 observations designed via a nonadaptive uncertainty sampling policy for the running example from Chapter 7 .

72 s. vakili et al. (2021a). Optimal Order Simple Regret for Gaussian Process Bandits. NeurIPS 2021. [theorem 3, remark 2]

73 Z. LI and J. SCARLETT (2021). Gaussian Process Bandit Optimization with Few Batches. arXiv: 2110.07788 [stat.ML].

![](https://cdn.mathpix.com/cropped/2023_09_22_751c1e67194d77e6639bg-23.jpg?height=206&width=527&top_left_y=2053&top_left_x=1370)

The lighter region of this Gaussian process has upper confidence bounds dominated by the maximum lower confidence bound (dashed line). Assuming the confidence bounds are valid with high probability, points in this region are unlikely to be optimal. 74 X. CAI et al. (2021). Lenient Regret and GoodAction Identification in Gaussian Process Bandits. ICML 2021. [§B.4]

75

The pruning scheme ensures that (if the last batch concluded after $i$ steps) the global optimum is very likely among the remaining candidates, and thus the instantaneous regret incurred in each stage of the next batch will be bounded by $\mathcal{O}\left(\beta_{i} \sqrt{\gamma_{i} / i}\right)$ with high probability.

76 м. valko et al. (2013). Finite-Time Analysis of Kernelised Contextual Bandits. UAI 2013

77 S. SALGIA et al. (2020). A Computationally Efficient Approach to Black-Box Optimization Using Gaussian Process Models. arXiv: 2010. 13997 [stat.ML]

78 R. CAMILlERI et al. (2021). High-Dimensional Experimental Design and Kernel Bandits. ICML 2021.

needles in haystacks: see p. 216

79 J. SCARLETt et al. (2017). Lower Bounds on Regret for Noisy Gaussian Process Bandit Optimization. COLT 2017.

![](https://cdn.mathpix.com/cropped/2023_09_22_751c1e67194d77e6639bg-24.jpg?height=177&width=531&top_left_y=1873&top_left_x=157)

The bump function used in SCARLETT et al.'s lower bound analysis, the Fourier transform of a smooth function with compact support as in Figure 10.5.

8o This particular bump function is the prototypical smooth function with compact support:

$$
x \mapsto \begin{cases}\exp \left(-\frac{1}{1-|x|^{2}}\right) & |x|<1 ; \\ 0 & \text { otherwise }\end{cases}
$$

(on a smaller domain) by discarding all observed data. This ensures that every observation is always gathered nonadaptively.

The intervening batches are constructed by uncertainty sampling. The motivation behind this strategy is that we can bound the width of confidence intervals after $i$ rounds of uncertainty sampling in terms of the information capacity; in particular, we can bound the maximum posterior standard deviation after $i$ rounds of uncertainty sampling with: ${ }^{74}$

$$
\max _{x \in \mathcal{X}} \sigma=\mathcal{O}\left(\sqrt{\gamma_{i} / i}\right) .
$$

By combining this result with the concentration inequality in (10.42), and by periodically eliminating regions that are very likely to be suboptimal throughout optimization, ${ }^{75}$ the authors were able to show that the resulting algorithm has worst-case regret

$$
\bar{R}_{\tau}[B]=\mathcal{O}^{*}\left(\sqrt{d \tau \gamma_{\tau}}\right)
$$

with high probability, matching the regret bounds for GP-UCB and GP-TS in the Bayesian setting (10.27).

This is not the only result of this type; several other authors have been able to construct algorithms achieving similar worst-case regret bounds through other means. ${ }^{76,77,78}$

\section*{Lower bounds and tightness of existing algorithms}

Lower bounds on regret are easier to come by in the frequentist setting than in the Bayesian setting, as we have considerable freedom to construct explicit objective functions that are provably difficult to optimize.

A common strategy to this end is to take inspiration from the "needle in a haystack" trick discussed earlier. We construct a large set of suitably well-behaved "needles" that have little overlap, then argue that there will always be some needle "missed" by an algorithm with insufficient budget to distinguish all the functions. Figure 10.5 shows a motivating example with four translations of a smooth bump function with height $\varepsilon$ and mutually disjoint support. Given any set of three observations regardless of how they were chosen - the cumulative regret for at least one of these functions would be $3 \varepsilon$.

This construction embodies the spirit of most of the lower bound arguments appearing in the literature. To yield a full proof, we must show how to construct a large number of suitable needles with bounded RKHS norm. For a stationary process, this can usually be accomplished by scaling and translating a suitable bump-shaped function to cover the domain. We also need to bound the regret of an algorithm given an input chosen from this set; here, we can keep the set of potential objectives larger than the optimization budget and appeal to pigeonhole arguments.

In the frequentist setting with noise, the strongest known lower bounds are due to SCARLETT et al. ${ }^{79}$ The function class considered in the analysis was scaled and translated versions of a function similar to (in fact, 

![](https://cdn.mathpix.com/cropped/2023_09_22_751c1e67194d77e6639bg-25.jpg?height=300&width=1073&top_left_y=455&top_left_x=226)

precisely the Fourier transform of) the bump function in Figure 10.5; ${ }^{80}$ this function has the advantage of having "nearly compact" support while having finite RKHS norm in the entire Matérn family. For optimization on the unit cube $\mathcal{X}=[0,1]^{d}$ with the Matern covariance function, the authors were able to establish a lower bound on the cumulative regret of any algorithm of

$$
\bar{R}_{\tau}[B]=\Omega\left(\tau^{\alpha}\right) ; \quad \alpha=\frac{v+d}{2 v+d},
$$

and for the squared exponential covariance, a lower bound of

$$
\bar{R}_{\tau}[B]=\Omega\left(\sqrt{\tau(\log \tau)^{d}}\right) .
$$

These bounds are fairly tight - they are within logarithmic factors of the best-known upper bounds in the worst-case setting (10.15-10.16, 10.44); see also the corresponding Bayesian bounds (10.28-10.29).

SCARLETT et al. also provided lower bounds on the worst-case simple regret $\bar{r}_{\tau}[B]$, in terms of the expected time required to reach a given level of regret. Inverting these bounds in terms of the simple regret at a given time yields rates that are as expected in light of the relation in (10.4).

\section*{THE EXACT OBSERVATION CASE}

The arguments outlined in the previous sections all ultimately depend on the information capacity of noisy observations of a Gaussian process. An upper bound on this quantity allows us to control the width of posterior confidence intervals through a pivotal result (10.18). After noting that the instantaneous regret of an upper confidence bound algorithm is in turn bounded by the width of these confidence intervals (10.23), we may piece together a bound on its cumulative regret (10.24).

Unfortunately, this line of attack breaks down with exact observations, as the information capacity of the now deterministic observation process diverges to infinity. ${ }^{81}$ However, all is not lost - we can appeal to different techniques to find much stronger bounds on the width of confidence intervals induced by exact observations, and thereby establish much faster rates of convergence than in the noisy case.

Throughout this section, as in the previous two sections, we will assume that the domain $\mathcal{X} \subset \mathbb{R}^{d}$ is compact and that the covariance function is bounded by unity: $K(x, x) \leq 1$.
Figure 10.5: Four smooth objective functions with disjoint support.

$$
r_{\tau} \leq \frac{R_{\tau}}{\tau}
$$

81 This can be seen by taking $\sigma_{n} \rightarrow 0$ in (10.17)

assumptions: $\mathcal{X}$ is compact, $K(x, x) \leq 1$ is bounded maximum posterior standard deviation given exact observations at $\mathbf{x}, \bar{\sigma}_{\mathbf{x}}$

82 We previously derived a bound of this form for uncertainty sampling in the noisy regime (10.43).

83 The literature on this topic is substantial, but the following references provide good entry points:

M. KANAGAWA et al. (2018). Gaussian Processes and Kernel Methods: A Review on Connections and Equivalences. arXiv: 1807.02582 [stat.ML]. [§5.2]

H. WENDLAND (2004). Scattered Data Approximation. Cambridge University Press. [chapter 11]

fill distance of observation locations $\mathbf{x}, \delta_{\mathbf{x}}$

84 Carefully taking this limit yields the following bound for the squared exponential covariance. For sufficiently small $\delta_{\mathbf{x}}$, we have

$$
\bar{\sigma}_{\mathrm{x}} \leq \delta_{\mathrm{x}}{ }^{c / \delta_{\mathrm{x}}}
$$

for some $c$ not depending on $\mathbf{x}$. Thus the rate of convergence increases as $\delta_{\mathbf{x}} \rightarrow 0$ :

H. WENDLAND (2004). Scattered Data Approximation. Cambridge University Press. [theorem 11.22]

85 For an excellent survey on both maximin (sphere packing) and minimax (fill distance) optimal designs, see:

L. PRonZATo (2017). Minimax and Maximin Space-Filling Designs: Some Properties and Methods for Construction. Fournal de la Société Française de Statistique 158(1):7-36.

86 We successively exhaust all combinations of dyadic rationals $\left\{i / 2^{n}\right\}$ with increasing height $n=0,1, \ldots$.

\section*{Bounding the posterior standard deviation}

As in the noisy setting, a key idea in deriving regret bounds with exact observations is bounding the width of posterior confidence intervals (10.39) resulting from observations. In this light, let us consider the behavior of a Gaussian process $\mathcal{G P}(f ; \mu, K)$ on an objective $f: \mathcal{X} \rightarrow \mathbb{R}$ as it is conditioned on a set of exact observations. Given some set of observation locations $\mathbf{x} \subset \mathcal{X}$, we seek to bound the maximum posterior standard deviation of the process:

$$
\bar{\sigma}_{\mathbf{x}}=\max _{x \in \mathcal{X}} \sigma,
$$

in terms of properties of $\mathbf{x}^{82}$ This would serve as a universal bound on the width of all credible intervals, allowing us to bound the cumulative regret of a UCB-style algorithm via previous arguments.

Thankfully, bounds of this type have enjoyed a great deal of attention, and strong results are available. ${ }^{83}$ Most of these results bound the posterior standard deviation of a Gaussian process in terms of the fill distance, a measure of how densely a given set of observations fills the domain. For a set of observation locations $\mathbf{x} \subset \mathcal{X}$, its fill distance is defined to be

$$
\delta_{\mathrm{x}}=\max _{x \in \mathcal{X}} \min _{x_{i} \in \mathrm{x}}\left|x-x_{i}\right|,
$$

the largest distance from a point in the domain to the closest observation.

Intuitively, we should expect the maximum posterior standard deviation induced by a set of observations to shrink with its fill distance. This is indeed the case, and particularly nice results for the rate of this shrinkage are available for the Matérn family. Namely, for finite $v$, once the fill distance is sufficiently small (below a constant depending on the covariance but not $\mathbf{x}$ ), we have:

$$
\bar{\sigma}_{\mathrm{x}} \leq c \delta_{\mathrm{x}}^{v},
$$

where the constant $c$ does not depend on $\mathbf{x}$. That is, once the observations achieve some critical density, the posterior standard deviation of the process shrinks rapidly with the fill distance, especially as sample paths become increasingly smooth in the limit $v \rightarrow \infty .{ }^{84}$ This result will be instrumental in several results discussed below.

\section*{Optimizing fill distance}

The question of optimizing the fill distance of a given number of points in compact subsets of $\mathbb{R}^{d}$ - so as to maximally strengthen the bound on prediction error above - is a major open question in geometry. Unfortunately, this problem is exceptionally difficult due to its close connection to the notoriously difficult problem of sphere packing. ${ }^{85}$

In the unit cube $\mathcal{X}=[0,1]{ }^{d}$ a simple grid strategy ${ }^{86}$ achieves asymptotic fill distance $\mathcal{O}\left(\tau^{-1 / d}\right)$, and thus for a Matérn covariance, we can guarantee (10.46):

$$
\bar{\sigma}_{\mathrm{x}}=\mathcal{O}\left(\tau^{-v / d}\right) .
$$

This will suffice for the arguments to follow. ${ }^{87}$ However, this strategy may not be the best choice in practical settings, as the fill distance decreases sporadically by large jumps each time the grid is refined. An alternative with superior "anytime" performance would be a low-discrepancy sequence such as a Sobol or Halton sequence, which would achieve slightly larger (by a factor of $\log \tau$ ), but more smoothly decreasing, asymptotic fill distance.

Several sophisticated algorithms are also available for (approximately) optimizing the fill distance from a given fixed number of points. ${ }^{85,88}$

\section*{Worst-case regret with deterministic observations}

We now turn our attention to specific results, beginning with bounds on the worst-case regret, where the analysis is somewhat simpler. As in the noisy setting, we assume that the objective function lies in the RKHS ball of radius $B$ corresponding to the covariance function $K$ used to model the objective function during optimization, $\mathcal{H}_{K}[B]$.

In this setting, we have the remarkable result that posterior confidence intervals with fixed confidence parameter $B$ are always valid: ${ }^{89}$

$$
\forall x \in \mathcal{X}: \phi \in[\mu-B \sigma, \mu+B \sigma]
$$

Combining this with the bound on standard deviation in (10.47) immediately shows that a simple grid strategy achieves worst-case simple regret $^{90}$

$$
\bar{r}_{\tau}[B]=\mathcal{O}\left(\tau^{-v / d}\right)
$$

for the Matérn covariance with smoothness $v$. BULL provided a corresponding lower bound establishing that this rate is actually optimal: $\bar{r}_{\tau}[B]=\Theta\left(\tau^{-v / d}\right) .{ }^{91}$ As with previous results, the lower bound derives from an adversarial "needle in haystack" construction as in Figure 10.5.

This result is perhaps not as exciting as it could be, as it demonstrates that no adaptive algorithm can perform (asymptotically) better than grid search in the worst case. However, we may reasonably seek similar guarantees for algorithms that are also effective in practice. BULL was able to show that maximizing expected improvement yields worst-case simple regret

$$
\bar{r}_{\tau}[B]=\mathcal{O}^{*}\left(\tau^{-\min (v, 1) / d}\right),
$$

which is near optimal for $v \leq 1$. BuLL also showed that augmenting expected improvement with occasional random exploration akin to an $\varepsilon$-greedy policy improves its performance to near optimal for any finite smoothness:

$$
\bar{r}_{\tau}[B]=\mathcal{O}^{*}\left(\tau^{-v / d}\right) .
$$

The added randomness effectively guarantees the fill distance of observations shrinks quickly enough that we can still rely on posterior contraction arguments, and this strategy could be useful for analyzing other policies.
87 We can extend this argument to any arbitrary compact domain $\mathcal{X} \subset \mathbb{R}^{d}$ by enclosing it in a cube.

88 y. LYU et al. (2019). Efficient Batch BlackBox Optimization with Deterministic Regret Bounds. arXiv: 1905.10041 [cs.LG].

worst-case regret with noise: p. 232

89 M. KANAgawA et al. (2018). Gaussian Processes and Kernel Methods: A Review on Connections and Equivalences. arXiv: 1807.02582 [stat.ML]. [corollary 3.11]

90 Here we use the alternative definition in footnote 5 . As the confidence intervals (10.48) are always valid, $f^{*}$ must always be in the (rapidly shrinking) confidence interval of whatever point maximizes the posterior mean.

91 A. D. BULL (2011). Convergence Rates of Efficient Global Optimization Algorithms. Journal of Machine Learning Research 12(88):28792904

convergence rates for expected improvement 92 S. GRÜNEWÄLDER et al. (2010). Regret Bounds for Gaussian Process Bandit Problems. AISTATS 2010 .

93 The upper bound is within a factor of $\sqrt{\log \tau}$ of wANG et al.'s upper bound for the noisy optimization of Brownian motion sample paths, where $\alpha=d=1$ (10.38). The lower bound is identical to that case (10.37)

![](https://cdn.mathpix.com/cropped/2023_09_22_751c1e67194d77e6639bg-28.jpg?height=180&width=525&top_left_y=1241&top_left_x=160)

Samples from an example Gaussian process used in deriving GRÜNEWÄLDER et al.'s lower bound. Here 10 (1/2)-Hölder continuous "needles" are scaled and translated to cover the unit interval, with one centered on each tick mark. Each needle is then scaled by an independent normal random variable and summed, yielding a Gaussian process with the desired properties.

94 P. MASSART (2007). Concentration Inequalities and Model Selection: Ecole d'Eté de Probabilités de Saint-Flour XXXIII - 2003. Vol. 1896. Springer-Verlag.

uniqueness of global maxima: § 2.7, p. 34

95 N. DE FREITAS et al. (2012b). Exponential Regret Bounds for Gaussian Process Bandits with Deterministic Observations. ICML 2012.

\section*{Bayesian regret with deterministic observations}

GRÜNEWÄLDER et al. proved bounds on the Bayesian regret in the exact observation case, although their results are only relevant for particularly rough objectives. ${ }^{92}$ The authors considered sample paths of a Gaussian process on the unit cube $[0,1]^{d}$ with Lipschitz continuous mean function and $\alpha$-Hölder continuous covariance function. The latter is an exceptionally weak smoothness assumption: the Matérn covariance with parameter $v$ is Hölder continuous with $\alpha=\min (2 v, 1)$, and any distinction in smoothness is lost beyond $v>1 / 2$, where sample paths are not yet even differentiable. Under these assumptions, GRÜNEWÄLDER et al. proved the following bounds on the Bayesian simple regret, where the lower bound is not universal but rather achieved by a specific construction satisfying the assumptions: $:^{93}$

$$
\mathbb{E}\left[r_{\tau}\right]=\Omega\left(\tau^{-c} / \sqrt{\log \tau}\right) ; \quad \mathbb{E}\left[r_{\tau}\right]=\mathcal{O}\left(\tau^{-c} \sqrt{\log \tau}\right) ; \quad c=-\frac{\alpha}{2 d} .
$$

The lower bound is achieved by an explicit Gaussian process with Hölder continuous covariance function whose "needle in a haystack" nature makes it difficult to optimize by any strategy. Specifically, the authors show how to cover the unit cube with $2 \tau$ disjoint and $\alpha$-Hölder continuous "needles" of compact support. We may then create a Gaussian process by scaling each of these needles by an independent normal random variable and summing. This construction is perhaps most clearly demonstrated visually, and the marginal figure shows an example on the unit interval with $\tau=5$ and $\alpha=1 / 2$. As the needles are disjoint with independent heights, no policy with a budget of $\tau$ can determine more than half of the weights, and we must therefore pay a penalty in terms of expected simple regret.

The upper bound derives from analyzing the performance of a simple grid strategy via classical concentration inequalities. ${ }^{94}$ It is remarkable that a nonadaptive strategy would yield such a small gap in regret compared to the corresponding lower bound, but this result is probably best understood as illustrating the inherent difficulty of optimizing rough functions rather than any inherent aptitude of grid search.

To underscore this remark, we may turn to a result of DE FREITAS et al., who showed that we may optimize sample paths of sufficiently smooth Gaussian processes with exponentially decreasing expected simple regret in the deterministic setting. ${ }^{95}$ This result relies on a few technical properties of the Gaussian process in question, in particular that it has a unique global optimum and that it exhibit "nice" behavior in the neighborhood of the global optimum. A centered Gaussian process with covariance function in the Matern family satisfies the required assumptions if $v>2$.

The algorithm analyzed by the authors was a simple branch-andbound policy based on GP-UCB, wherein the domain is recursively subdivided into finer and finer divisions. After each division, we identify the regions that could still contain the global optimum based on the current confidence intervals, then evaluate on these regions such that the fill distance (10.45) is sufficiently small before the next round of subdivision. In this step, DE FREITAs et al. relied on the bound in (10.46) (with $v=2$ ) to ensure that the confidence intervals induced by observed data shrink rapidly throughout this procedure.

At some point in this procedure, we will (with high probability) find ourselves having rejected all but the local neighborhood of the global optimum, at which point the assumed "nice" behavior of the sample path guarantees rapid convergence thereafter. Specifically, the authors demonstrate that at some point the instantaneous regret of this algorithm will converge exponentially:

$$
\mathbb{E}\left[\rho_{\tau}\right]=\mathcal{O}\left(\exp \left(-\frac{c \tau}{(\log \tau)^{d / 4}}\right)\right),
$$

for some constant $c>0$ depending on the process but not on $\tau$. This condition implies rapid convergence in terms of simple regret as well (as we obviously have $r_{\tau} \leq \rho_{\tau}$ ) and bounded cumulative regret after we have entered the converged regime. ${ }^{96}$

Evidently optimization is much easier with deterministic observations than noisy ones, at least in terms of Bayesian regret. Intuitively, the reason for this discrepancy is that noisy observations may compel us to make repeated measurements in the same region in order to shore up our understanding of the objective function, whereas this is never necessary with exact observations.

\section*{THE EFFECT OF UNKNOWN HYPERPARAMETERS}

We have now outlined a plethora of convergence results: upper bounds on the regret of specific algorithms and algorithm-agnostic lower bounds, in the expected and worst case, with and without observation noise. However, all of these results assumed intimate knowledge about the objective function being optimized: in Bayesian analysis, we assumed the objective is sampled from the prior used to model it, and for worstcase analysis, we assumed the objective lay in the corresponding RKHS. Of course, neither of these assumptions (especially the former!) may hold in practice. Further, in practice the model used to reason about the objective is typically inferred from observed data and constantly updated throughout optimization, but the analysis discussed thus far has assumed the model is not only perfectly informed, but also fixed.

It turns out that violations to these (rather implausible!) assumptions can be disastrous for convergence. For example, consider this prototypical Bayesian optimization algorithm: we model the objective function with an automatic relevance determination (ARD) version of a Matérn covariance function, learn its output (3.20) and length scales (3.25) via maximum likelihood estimation, and design observation locations by maximizing expected improvement. Although this setup has proven remarkably effective in practice, ${ }^{97}$ BULL proved that it can actually fail miserably in the frequentist setting: we may construct functions in the RKHS of any such covariance function (that is, with any desired param-
96 This is not inconsistent with GRÜNEWÄLDER et al.'s lower bound: here we assume smoothness consistent with $v>2$ in the Matern family, whereas the adversarial Gaussian process constructed in the lower bound is only as smooth as $v=1 / 2$.

Chapter 3: Modeling with Gaussian Processes, p. 45

automatic relevance determination: $\S 3.4$, p. 56 ML estimation of hyperparameters: $§ 4.3$, p. 73

97 J. SNOEK et al. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. NeurIPS 2012. Figure 10.6: The problem with estimating hyperparameters. Above: a function in the RKHS for the $v=5 / 2$ Matern covariance with unit norm. Below: a sample path from a Gaussian process with the same covariance. Unless we're lucky, we may never find the "hump" in the former function and thus may never build a reasonable belief regarding the objective.

98 A. D. BULL (2011). Convergence Rates of Efficient Global Optimization Algorithms. Fournal of Machine Learning Research 12(88):28792904. [theorem 3]

99 M. LOCATELli (1997). Bayesian Algorithms for One-Dimensional Global Optimization. fournal of Global Optimization 10(1):57-76.

![](https://cdn.mathpix.com/cropped/2023_09_22_751c1e67194d77e6639bg-30.jpg?height=262&width=534&top_left_y=1351&top_left_x=156)

A function given by LOCATELLI for which expected improvement combined with a Wiener process prior will not converge if its output scale is marginalized. The first three observations are at the indicated locations; after iteration 3, the left-hand side of the domain is forever ignored.

100 Recall that sample paths almost surely do not lie in $\mathcal{H}_{K}$, assuming it is infinite dimensional, and see Figure 10.2 and the surrounding text.

101 BULL simply inflated the output scale estimated by maximum likelihood by a factor of $\sqrt{\tau}$, and LOCATELLI chose a prior on the output scale that had no support below some minimum threshold.

![](https://cdn.mathpix.com/cropped/2023_09_22_751c1e67194d77e6639bg-30.jpg?height=320&width=1031&top_left_y=528&top_left_x=752)

eters) that will "fool" this algorithm into having high regret with high probability. ${ }^{98}$

Estimation of hyperparameters can also cause problems with convergence in the Bayesian setting. For example, LOCATELLI considered optimization on the unit interval $\mathcal{X}=[0,1]$ from exact observations using a Wiener process prior and maximizing expected improvement. In this relatively straightforward model, the only hyperparameter under consideration was an output scale (3.20). LOCATELLI showed that for a fixed output scale, this procedure converges for all continuous functions. ${ }^{99}$ However, when the output scale is learned from data - even in a fully Bayesian manner where it is endowed with a prior distribution and marginalized in the predictive distribution - there are extremely simple (piecewise linear!) functions for which this algorithm does not recover the optimum. An example is shown in the margin.

In both of these examples, the roadblock to convergence when estimating hyperparameters is a mismatch between the objective function model used by the Bayesian optimization algorithm and the true objective. In the frequentist setting, there is an inherent tension as functions lying in the RKHS $\mathcal{H}_{K}$ are not representative of sample paths from the Gaussian process $\mathcal{G P}(f ; \mu, K){ }^{100}$

Functions in the RKHS are much smoother than sample paths and may feature relatively "flat" regions of little variation, which may in turn lead to poorly fit models. See Figure 10.6 for a striking example. LOCATELLI's piecewise linear counterexample also reflects extreme model misspecification - the function is far too smooth to resemble Brownian motion in any statistical sense, and we should not be surprised that maximum likelihood estimation leads to an inconsistent algorithm.

A line of continuing work has been to derive robust algorithms that do not require perfect prior knowledge regarding the objective function in order to provide strong convergence guarantees. There are several ideas in this direction, all of which employ some mechanism to ensure that the space of objective functions considered by the algorithm does not ever become too "small." For example, both BULL and LOCATELLI addressed their counterexamples with schemes wherein the output scale used during optimization is never allowed to shrink so rapidly that true objective function is left behind. ${ }^{101}$ In the frequentist setting, one common strategy is to replace the assumption that the objective function lay in some particular RKHS with the assumption that it lay in some parametric family of RKHses indexed by a set of hyperparameters. We then slowly expand the space of functions considered in the algorithm over the course of the algorithm such that the objective function is guaranteed to eventually - and forever thereafter - be well explained. In particular, consider augmenting an isotropic covariance function from the Matérn family $K$ with an output scale (3.20) $\lambda$ and a vector of length scales $\boldsymbol{\ell}$ (3.25). We have the remarkable property that if a function $f$ is in the RKHs ball of radius $B$ (10.11) for some setting of these hyperparameters:

$$
f \in \mathcal{H}_{K}[B ; \lambda, \boldsymbol{\ell}],
$$

then the same function is also in the RKHs ball for any larger output scale and for any vector of shorter (as in the lexicographic order) length scales:

$$
f \in \mathcal{H}_{K}\left[B ; \lambda^{\prime}, \boldsymbol{\ell}^{\prime}\right] ; \quad \lambda^{\prime} \geq \lambda ; \quad \boldsymbol{\ell}^{\prime} \leq \boldsymbol{\ell} .
$$

With this in mind, a natural idea for deriving theoretical convergence results (but not necessarily for realizing a practical algorithm!) is to ignore any data-dependent scheme for setting hyperparameters and instead simply slowly increase the output scale and slowly decrease the length scales over the course of optimization, so that at some point any function lying in any such RKHS will eventually be captured. This scheme has been used to provide convergence guarantees for both expected improvement ${ }^{102}$ and $\mathrm{GP}-\mathrm{UCB}^{103}$ with unknown hyperparameters.

\section*{O.8 SUMMARY OF MAJOR IDEAS}

- Convergence analysis for Bayesian optimization algorithms entails selecting a measure of optimization performance, a space of objective functions to consider, and deriving bounds on the asymptotic growth of the chosen performance measure on the chosen function space.

- Optimization performance is almost always assessed via one of two related notations of regret, both of which depend on the difference between the function value at measured locations and the global optimum.

- Although tempting, the space of all continuous functions is too large to be of much interest in analysis: we may construct "needles in haystacks" to foil any algorithm by any amount. Instead, we may study spaces of objective functions with more plausible behavior. A Gaussian process model suggests two natural possibilities: sample paths from the process and the reproducing kernel Hilbert space (RKHs) associated with the process, the closure of the space of all possible posterior mean functions.

- Putting these pieces together, a Bayesian analysis of regret assumes the objective is a sample path from a Gaussian process and seeks asymptotic bounds on the expected regret (10.6). A frequentist analysis, on the other hand, assumes the objective function is a fixed, but unknown function
102 Z. WANG and N. DE FREITAS (2014). Theoretical Analysis of Bayesian Optimization with Unknown Gaussian Process Hyper-Parameters. arXiv: 1406.7758 [stat.ML]

103 F. BERKENKAMP et al. (2019). No-Regret Bayesian Optimization with Unknown Hyperparameters. Journal of Machine Learning Research $20(50): 1-24$.

simple and cumulative regret: § 10.1, p. 213

useful function spaces for studying convergence: $\S 10.2$, p. 215

upper bounds:

Bayesian regret with noise: § 10.4, p. 224 worst-case regret with noise: $\S 10.5$, p. 232 Bayesian regret without noise: $\S 10.6$, p. 240 worst-case regret without noise: § 10.6, p. 239

this argument is carefully laid out for Bayesian regret with noise in $\S 10.4$, p. 225

information capacity: §10.3, p. 222

bounding the posterior standard deviation with exact observations: $§ 10.6$, p. 238

lower bounds:

Bayesian regret with noise: $§ 10.4$, p. 230 worst-case regret with noise, § 10.5, p. 236 Bayesian regret without noise: $\S 10.6$, p. 240 worst-case regret without noise: § 10.6, p. 239 in a given RKHS and seeks asymptotic bounds on the worst-case regret (10.6).

- Most upper bounds on cumulative regret derive from a proof strategy where we identify some suitable set of predictive credible intervals that are universally valid with high probability. We may then argue that the cumulative regret is bounded in terms of the total width of these intervals. This strategy lends itself most naturally to analyzing the Gaussian process upper confidence bound (GP-UCB) policy, but also yields bounds for Gaussian process Thompson sampling (GP-TS) due to strong theoretical connections between these algorithms (10.33).

- In the presence of noise, a key quantity is the information capacity, the maximum information about the objective that can be revealed by a set of noisy observations. Bounds on this quantity yield bounds on the sum of predictive variances (10.18) and thus cumulative regret. With exact observations, we may derive bounds on credible intervals by relating the fill distance (10.45) of the observations to the maximum standard deviation of the process, as in (10.46).

- To derive lower bounds on Bayesian regret, we seek arguments limiting the rate at which Gaussian process sample paths can be optimized. For worst-case regret, we may construct explicit objective functions in a given RKHS and prove that they are difficult to optimize; here the "needles in haystacks" idea again proves useful.