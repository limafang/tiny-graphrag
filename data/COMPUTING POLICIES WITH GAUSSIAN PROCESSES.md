\section*{COMPUTING POLICIES WITH GAUSSIAN PROCESSES}

In the last chapter we introduced several notable Bayesian optimization policies in a model-agnostic setting, concentrating on their motivation and behavior while ignoring computational details. In this chapter we will provide further information for effectively implementing these policies. We will focus on Gaussian process models of the objective function, combined with either an exact or additive Gaussian noise observation model; this family accounts for the vast majority of models encountered in practice. However, we will discuss computation for alternative model classes such as Bayesian neural networks at the end of the chapter.

Implementing each policy in the previous chapter ultimately requires optimizing some acquisition function over the domain: ${ }^{1}$

$$
x \in \underset{x^{\prime} \in \mathcal{X}}{\arg \max } \alpha\left(x^{\prime} ; \mathcal{D}\right) .
$$

We will demonstrate how to compute (or approximate) each of these acquisition functions with respect to Gaussian process models. Some will admit exact analytical expressions; when this is not possible, we will describe effective approximation schemes. In Euclidean domains, we will also show how to compute the gradient of these acquisition functions with respect to the proposed observation location, allowing efficient optimization via gradient methods. These gradient computations will sometimes be somewhat involved (but not difficult), and we will defer some details to an appendix for the sake of brevity.

The order of our presentation will differ from that in the previous chapter. Here we will begin with the acquisition functions for which exact computation is possible, then develop approximation techniques for those that remain. First, we pause to establish notation for important reoccurring quantities.

\subsection*{NOTATION FOR OBJECTIVE FUNCTION MODEL}

As usual, let us consider an objective function $f: \mathcal{X} \rightarrow \mathbb{R}$, with a Gaussian process belief conditioned on arbitrary observations $\mathcal{D}=(\mathbf{x}, \mathbf{y})$ :

$$
p(f \mid \mathcal{D})=\mathcal{G} \mathcal{P}\left(f ; \mu_{\mathcal{D}}, K_{\mathcal{D}}\right)
$$

Our main task in this chapter will be to compute a given acquisition function at an arbitrary location $x \in \mathcal{X}$ with respect to this belief. Given a proposed location $x$, we will write the predictive distribution for $\phi=f(x)$ as:

$$
p(\phi \mid x, \mathcal{D})=\mathcal{N}\left(\phi ; \mu, \sigma^{2}\right),
$$

where the predictive mean and variance

$$
\mu=\mu_{\mathcal{D}}(x) ; \quad \sigma^{2}=K_{\mathcal{D}}(x, x)
$$

depend implicitly on $x$. We will always treat $x$ as given and fixed, so this convention will not lead to ambiguity.

This material has been published by Cambridge University Press as Bayesian Optimization This version is free to view and download for personal use only. Not for redistribution, resale, or use in derivative works. (CRoman Garnett 2023. https://bayesoptbook. com/

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-01.jpg?height=358&width=556&top_left_y=717&top_left_x=1345)

1 Even the nondeterministic Thompson sampling policy, which may be realized by optimizing a random acquisition function: $§ 7.9$, p. 148 .

gradients of common acquisition functions: § . 3, p. 308

Gaussian process on $f, \mathcal{G P}\left(f ; \mu_{\mathcal{D}}, K_{\mathcal{D}}\right)$

predictive distribution for $\phi, \mathcal{N}\left(\phi ; \mu, \sigma^{2}\right)$

predictive mean and variance for $\phi: \mu, \sigma^{2}$ observation noise scale, $\sigma_{n}$ predictive distribution for $y, \mathcal{N}\left(y ; \mu, s^{2}\right)$

predictive variance for $y, s^{2}$

prime notation for post-observation quantities

general form of gradient and dependence on parameter gradients

gradients of GP predictive distribution: § C.2, p. 308

expected improvement: § 7.3, p. 127 simple reward: $\S 6.1$, p. 112
We will also require the predictive distribution for the observed value $y$ resulting from a measurement at $x$. In addition to the straightforward case of exact measurements, where $y=\phi$ and the predictive distribution is given above (8.2), we will also consider corruption by independent, zero-mean additive Gaussian noise:

$$
p\left(y \mid \phi, \sigma_{n}\right)=\mathcal{N}\left(y ; \phi, \sigma_{n}^{2}\right) .
$$

Again we allow the noise scale $\sigma_{n}$ to depend on $x$ if desired. We will notate the resulting predictive distribution for $y$ with:

$$
p\left(y \mid x, \mathcal{D}, \sigma_{n}\right)=\mathcal{N}\left(y ; \mu, \sigma^{2}+\sigma_{n}^{2}\right)=\mathcal{N}\left(y ; \mu, s^{2}\right)
$$

where $\mu$ and $\sigma^{2}$ are the predictive moments of $\phi(8.3)$ and $s^{2}=\sigma^{2}+\sigma_{n}^{2}$ is the predictive variance for $y$, which again depends implicitly on $x$. When no distinction between exact and noisy observations is necessary, we will use the above general notation (8.5). With exact observations, the observation noise scale is identically zero, and we have $s^{2}=\sigma^{2}$.

We will retain our convention from the last chapter of indicating quantities available after acquiring a proposed observation with a prime symbol. For example $\mathbf{x}^{\prime}=\mathbf{x} \cup\{x\}$ represents the updated set of observation locations after adding an observation at $x$, and $\mathcal{D}^{\prime}=\left(\mathbf{x}^{\prime}, \mathbf{y}^{\prime}\right)$ represents the current data augmented with the observation $(x, y)-\mathrm{a}$ random variable.

The value of an acquisition function $\alpha$ at a point $x$ will naturally depend on the distribution of the corresponding observation $y$, and its gradient must reflect this dependence. Applying the chain rule, the general form of the gradient will be written in terms of the gradient of the predictive parameters:

$$
\frac{\partial \alpha}{\partial x}=\frac{\partial \alpha}{\partial \mu} \frac{\partial \mu}{\partial x}+\frac{\partial \alpha}{\partial s} \frac{\partial s}{\partial x}
$$

The gradients of the predictive mean and standard deviation for a Gaussian process are easily computable assuming the prior mean and covariance functions and the observation noise scale are differentiable, and general expressions are provided in an appendix.

\subsection*{EXPECTED IMPROVEMENT}

We first consider the expected improvement acquisition function (7.2), the expected marginal gain in simple reward (6.3):

$$
\alpha_{\mathrm{EI}}(x ; \mathcal{D})=\mathbb{E}\left[\max \mu_{\mathcal{D}^{\prime}}\left(\mathbf{x}^{\prime}\right) \mid x, \mathcal{D}\right]-\max \mu_{\mathcal{D}}(\mathbf{x}) .
$$

Remarkably, this expectation can be computed analytically for Gaussian processes with both exact and noisy observations. We will consider each case separately, as the former is considerably simpler and the latter involves minor controversy. 

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-03.jpg?height=297&width=806&top_left_y=460&top_left_x=268)

Expected improvement without noise

Expected improvement assumes a convenient form when measurements are exact $(7 \cdot 3)$ :

$$
\alpha_{\mathrm{EI}}(x ; \mathcal{D})=\int \max (\phi-\phi, 0) \mathcal{N}\left(\phi ; \mu, \sigma^{2}\right) \mathrm{d} \phi .
$$

Here $\phi^{*}$ is the previously best-seen, incumbent objective function value, and $\max (\phi-\phi, 0)$ measures the improvement offered by observing a value of $\phi$. Figure 8.1 illustrates this integral.

To proceed, we resolve the max operator to yield two integrals:

$$
\alpha_{\mathrm{EI}}(x ; \mathcal{D})=\int_{\phi^{*}}^{\infty} \phi \mathcal{N}\left(\phi ; \mu, \sigma^{2}\right) \mathrm{d} \phi-\phi^{*} \int_{\phi^{*}}^{\infty} \mathcal{N}\left(\phi ; \mu, \sigma^{2}\right) \mathrm{d} \phi,
$$

both of which can be computed easily assuming $\sigma>0 .^{2}$ The first term is proportional to the expected value of a normal distribution truncated at $\phi$, and the second term is the complementary normal cDF scaled by $\phi *$ The resulting acquisition function can be written conveniently in terms of the standard normal PDF and CDF:

$$
\alpha_{\mathrm{EI}}(x ; \mathcal{D})=\left(\mu-\phi^{*}\right) \Phi\left(\frac{\mu-\phi^{*}}{\sigma}\right)+\sigma \phi\left(\frac{\mu-\phi^{*}}{\sigma}\right) .
$$

Examining this expression, it is tempting to interpret its two terms as respectively encouraging exploitation (favoring points with high expected value $\mu$ ) and exploration (favoring points with high uncertainty $\sigma)$. Indeed, taking partial derivatives with respect to $\mu$ and $\sigma$, we have:

$$
\frac{\partial \alpha_{\mathrm{EI}}}{\partial \mu}=\Phi\left(\frac{\mu-\phi^{*}}{\sigma}\right)>0 ; \quad \frac{\partial \alpha_{\mathrm{EI}}}{\partial \sigma}=\phi\left(\frac{\mu-\phi^{*}}{\sigma}\right)>0 .
$$

Expected improvement is thus monotonically increasing in both $\mu$ and $\sigma$. Increasing a point's expected value naturally makes the point more favorable for exploitation, and increasing its uncertainty makes it more favorable for exploration. Either action would increase the expected improvement. The tradeoff between these two concerns is considered automatically and is reflected in the magnitude of the derivatives above.

Maximization of expected improvement in Euclidean domains may be guided by its gradient with respect to the proposed evaluation location $x$. Using the results above and applying the chain rule, we have (8.6):

$$
\frac{\partial \alpha_{\mathrm{EI}}}{\partial x}=\Phi\left(\frac{\mu-\phi^{*}}{\sigma}\right) \frac{\partial \mu}{\partial x}+\phi\left(\frac{\mu-\phi^{*}}{\sigma}\right) \frac{\partial \sigma}{\partial x},
$$

incumbent function value, $\phi^{*}$

\footnotetext{
2 In the degenerate case $\sigma=0$, we simply have $\alpha_{\mathrm{EI}}(x ; \mathcal{D})=\max \left(\mu-\phi^{*} 0\right)$.
}

exploitation and exploration

partial derivatives with respect to predictive distribution parameters

gradient of expected improvement without noise 3 D. R. JONEs et al. (1998). Efficient Global Optimization of Expensive Black-Box Functions. Fournal of Global Optimization 13(4):455-492.

behavior of posterior moments: § 2.2, p. 21

any point may maximize the posterior mean

4 P. FRAZier et al. (2009). The KnowledgeGradient Policy for Correlated Normal Beliefs. INFORMS fournal on Computing 21(4):599-613. which expresses the gradient in terms of the implicit exploration-exploitation tradeoff parameter and the change in the predictive distribution.

\section*{Expected improvement with noise}

Computing expected improvement with noisy observations is somewhat more complicated than in the noiseless case. In fact, there is not even universal agreement regarding what the definition of noisy expected improvement should be! The situation is so nuanced that Jones et al. sidestepped the issue entirely in their landmark paper: ${ }^{3}$

Unfortunately it is not immediately clear how to extend our optimization algorithm to the case of noisy functions. With noisy data, we really want to find the point where the signal is optimized. Similarly, our expected improvement criterion should be defined in terms of the signal component. [emphasis added by JonEs et al.]

This summarizes the main challenge to optimization with noisy observations: how can we determine whether a particularly high observed value reflects a true underlying effect or is merely an artifact of noise? Several alternative definitions and heuristics have been proposed to address this question, which we will discuss further shortly.

We will first argue that seeking to maximize the simple reward utility (6.3) is precisely aligned with the goal outlined by JonEs et al. With appropriate modeling of observation noise, the objective function posterior exactly represents our belief about the "signal component": the latent objective $f$. Simple reward evaluates progress directly with respect to this belief, only ascribing merit to observations that improve the maximum of the posterior mean and thus the expected outcome of an optimal terminal recommendation. Notably, an excessively noisy observation has weak correlation with the underlying objective function value and thus yields little change in the posterior mean (2.12). Therefore even an extremely high outcome would produce only a minor improvement to the simple reward. As a result, the expected improvement of such a point would be relatively small, exactly the desired behavior.

Unfortunately, observation noise renders expected improvement somewhat more complicated than the exact case, due to the nature of the updated simple reward. After an exact observation, the simple reward can only be achieved at one of two locations: either the observed point or the incumbent. However, with noisy observations, inherent uncertainty in the objective function implies that the updated simple reward could be achieved anywhere, including a point that previously appeared suboptimal. We must account for this possibility in the computation, increasing its complexity. Fortunately, exact computation is still possible for Gaussian processes and additive Gaussian noise by adopting a procedure originally described by FRAZIER et al. for computing the knowledge gradient on a discrete domain; ${ }^{4}$ we will return to this method in our discussion on knowledge gradient in the next section. If we define $\mu^{*}=\max \mu_{\mathcal{D}}(\mathbf{x})$ to represent the simple reward of the current data, then we must compute:

$$
\alpha_{\mathrm{EI}}(x ; \mathcal{D})=\int\left[\max \mu_{\mathcal{D}^{\prime}}\left(\mathbf{x}^{\prime}\right)-\mu^{*}\right] \mathcal{N}\left(y ; \mu, s^{2}\right) \mathrm{d} y .
$$

We first reduce this computation to an expectation of the general form

$$
g(\mathbf{a}, \mathbf{b})=\int \max (\mathbf{a}+\mathbf{b} z) \phi(z) \mathrm{d} z,
$$

where $\mathbf{a}, \mathbf{b} \in \mathbb{R}^{n}$ are arbitrary vectors and $z$ is a standard normal random variable. ${ }^{5}$ Note that given the observation $y$, the updated posterior mean at $\mathbf{x}^{\prime}$ is a vector we may compute in closed form (2.19):

$$
\mu_{\mathcal{D}^{\prime}}\left(\mathbf{x}^{\prime}\right)=\mu_{\mathcal{D}}\left(\mathbf{x}^{\prime}\right)+\frac{K_{\mathcal{D}}\left(\mathbf{x}^{\prime}, x\right)}{s} \frac{y-\mu}{s} .
$$

This update is linear in $y$. Applying the transformation $y=\mu+s z$ yields

$$
\mu_{\mathcal{D}^{\prime}}\left(\mathbf{x}^{\prime}\right)=\mathbf{a}+\mathbf{b} z
$$

where

$$
\mathbf{a}=\mu_{\mathcal{D}}\left(\mathbf{x}^{\prime}\right) ; \quad \mathbf{b}=\frac{K_{\mathcal{D}}\left(\mathbf{x}^{\prime}, x\right)}{s},
$$

and we may express expected improvement in the desired form:

$$
\alpha_{\mathrm{EI}}(x ; \mathcal{D})=g(\mathbf{a}, \mathbf{b})-\mu^{*}
$$

As a function of $z, \mathbf{a}+\mathbf{b} z$ is a set of lines with intercepts and slopes given by the entries of the $\mathbf{a}$ and $\mathbf{b}$ vectors, respectively. In the context of expected improvement, these lines represent the updated posterior mean values for each point of interest $\mathbf{x}^{\prime}$ as a function of the $z$-score of the noisy observation $y$. See Figure 8.2 for an illustration. Note that the points with the highest correlation with the proposed point have the greatest slope (8.12), as our belief at these locations will be strongly affected by the outcome.

Now $\max (\mathbf{a}+\mathbf{b} z)$ is the upper envelope of these lines, a convex piecewise linear function shown in Figure 8.3. The interpretation of this envelope is the simple reward of the updated dataset given the $z$-score of $y$, which will be achieved at some point in $\mathbf{x}^{\prime}$. For this example, the updated posterior mean could be maximized at one of four locations: either at one of the points on the far right given a relatively high observation, or at a backup point farther left given a relatively low observation. Note that in the latter case the simple reward will decrease. ${ }^{6}$

With this geometric intuition in mind, we can deduce that $g$ is invariant to transformations that do not alter the upper envelope. In particular, $g$ is invariant both to reordering the lines by applying an identical permutation to $\mathbf{a}$ and $\mathbf{b}$ and also to the deletion of lines that never dominate. In the interest of notational simplicity, we will take advantage of these invariances and only consider evaluating $g$ when every given line achieves current simple reward, $\mu^{*}$

reduction to evaluation of $g(\mathbf{a}, \mathbf{b})$

5 The dimensions of $\mathbf{a}$ and $\mathbf{b}$ can be arbitrary as long as they are equal.

definition of $\mathbf{a}, \mathbf{b}$

geometric intuition of $g$

\footnotetext{
6 However, the expected marginal gain is always positive.

invariance to permutation

invariance to deletion of always dominated lines
} 

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-06.jpg?height=532&width=1634&top_left_y=445&top_left_x=154)

Figure 8.2: The geometric intuition of the $g(\mathbf{a}, \mathbf{b})$ function. Left: if we make a measurement at $x$, the $z$-score of the observed value completely determines the updated posterior mean at that point and all previously observed points. Right: as a function of $z$, the updated posterior mean at each of these points is linear; here the lightness of each line corresponds to the matching point on the left. The slope and intercept of each line can be determined from the posterior (8.12). Not all lines are visible.

7 Briefly, we sort the lines in ascending order of slope, then add each line in turn to a set of dominating lines, checking whether any previously added lines need to be removed and updating the intervals of dominance.
8 Reordering the lines in order of increasing slope guarantees this correspondence: the line with minimal slope is always the "leftmost" in the upper envelope, etc. maximal value on some interval and the lines appear in strictly increasing order of slope:

$$
b_{1}<b_{2}<\cdots<b_{n} .
$$

FRAZIER et al. give a simple and efficient algorithm to process a set of $n$ lines to eliminate any always-dominated lines, reorder the remainder in increasing slope, and identify their intervals of dominance in $\mathcal{O}(n \log n)$ time. $^{7}$ The output of this procedure is a permutation matrix $\mathbf{P}$, possibly with some rows deleted, such that

$$
g(\mathbf{a}, \mathbf{b})=g(\mathbf{P a}, \mathbf{P b})=g(\boldsymbol{\alpha}, \boldsymbol{\beta}),
$$

and the new inputs $(\boldsymbol{\alpha}, \boldsymbol{\beta})$ satisfy the desired properties. We will assume below that the inputs have been preprocessed in such a manner.

Given a set of lines in the desired form, we may partition the real line into a collection of $n$ intervals

$$
\left(-\infty=c_{1}, c_{2}\right) \cup\left(c_{2}, c_{3}\right) \cup \cdots \cup\left(c_{n}, c_{n+1}=+\infty\right),
$$

such that the $i$ th line $a_{i}+b_{i} z$ dominates on the corresponding interval $\left(c_{i}, c_{i+1}\right){ }^{8}$ This allows us to decompose the desired expectation (8.10) into a sum of contributions on each interval:

$$
g(\mathbf{a}, \mathbf{b})=\sum_{i} \int_{c_{i}}^{c_{i+1}}\left(a_{i}+b_{i} z\right) \phi(z) \mathrm{d} z .
$$

Finally, we may compute each integral in the sum in closed form:

$$
g(\mathbf{a}, \mathbf{b})=\sum_{i} a_{i}\left[\Phi\left(c_{i+1}\right)-\Phi\left(c_{i}\right)\right]+b_{i}\left[\phi\left(c_{i}\right)-\phi\left(c_{i+1}\right)\right],
$$


![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-07.jpg?height=388&width=1630&top_left_y=444&top_left_x=270)

Figure 8.3: After a measurement at $x$, the updated simple reward can be achieved at one of four points (left), whose corresponding lines comprise the upper envelope $\max (\mathbf{a}+\mathbf{b z})$ (right). The lightness of the line segments on the right correspond to the possible updated maximum locations on the left. The lightest point on the left serves as a "backup option" if the observed value is low.

allowing efficient and exact computation of expected improvement in the noisy case. The main bottleneck is the modest $\mathcal{O}(n \log n)$ preprocessing step required.

This expression reverts to that for exact measurements (8.9) in the absence of noise. In that case, we have $\mathbf{b}=[\mathbf{0}, s]^{\top}$, and the upper envelope only contains lines corresponding to the incumbent and newly observed point, which intersect at the incumbent value $c_{1}=\phi^{*}$. Geometrically, the situation collapses to that in Figure 8.1, and it is easy to confirm (8.9) and (8.16) coincide.

Although tedious, we may compute the gradient of $g$ and thus the gradient of expected improvement (8.13); the details are in an appendix.

\section*{Alternative formulations of noisy expected improvement}

Although thematically consistent and mathematically straightforward, our approach of computing expected improvement as the expected marginal gain in simple reward is not common and may be unfamiliar to some readers.

Over the years, numerous authors have grappled with the best definition of noisy expected improvement. A typical approach is to begin with the convenient formula (8.9) in the noiseless regime, then work "backwards" by identifying potential issues with its application to noisy data and suggesting a heuristic correction. This strategy of "fixing" exact expected improvement is in opposition to our ground-up approach of first defining a well-grounded utility function and only then working out the expected marginal gain. PICHENY et al. provided a survey of such approximations - representing a total of 11 different acquisition strategies - accompanied by a thorough empirical investigation. ${ }^{9}$ Notably, the authors were familiar with the exact computation we outlined in the previous section and recommended it for use in practice.

One popular idea in this direction is to take the formula from the exact case (8.9) and substitute a plug-in estimate for the now unknown compatibility with noiseless case

gradient of noisy expected improvement: § . 3, p. 308
9 V. PICHENY et al. (2013b). A Benchmark of Kriging-Based Infill Criteria for Noisy Optimization. Structural and Multidisciplinary Optimization 48(3):607-626.

expected improvement with plug-in estimator of $\phi^{*}$ 

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-08.jpg?height=224&width=1631&top_left_y=516&top_left_x=158)

— plug-in estimate, $\phi^{*} \approx \max y(8.17) \quad \boldsymbol{\nabla}$ next observation location

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-08.jpg?height=154&width=1631&top_left_y=820&top_left_x=157)

— plug-in estimate, $\phi^{*} \approx \max \mu_{\mathcal{D}}(\mathbf{x})(8.18)$

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-08.jpg?height=160&width=1628&top_left_y=1039&top_left_x=157)

- expected improvement (8.16)

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-08.jpg?height=172&width=1628&top_left_y=1256&top_left_x=160)

Figure 8.4: Expected improvement using different plug-in estimators (8.17-8.18) compared with the noisy expected improvement as the expected marginal gain in simple reward (8.7).

maximum noisy value "utility" function: $§ 6.1$, p. 113

inflating expected improvement threshold: $\S 7.10$, p. 154

10 v. PICHENy et al. (2013b). A Benchmark of Kriging-Based Infill Criteria for Noisy Optimization. Structural and Multidisciplinary Optimization 48(3):607-626.

11 v. NGUYEN et al. (2017). Regret for Expected Improvement over the Best-Observed Value and Stopping Condition. ACML 2017. incumbent value $\phi *$ Several possibilities for this estimate have been put forward. One option is to plug in the maximum noisy observation:

$$
\phi^{*} \approx \max \mathbf{y} .
$$

However, this may not always behave as expected for the same reason the maximum observed value does not serve as a sensible utility function (6.6). With very noisy data, the maximum observed value is most likely spurious rather than a meaningful goalpost. Further, as we are likely to overestimate our progress due to bias in this estimate, the resulting behavior may become excessively exploratory. The approximation will eventually devolve to expected improvement against an inflated threshold (7.21), which may overly encourage exploration; see Figure 7.23. An especially spurious observation can bias the estimate in (8.17) (and our behavior) for a considerable time.

Opinions on the proposed approximation using this simple plug-in estimate (8.17) vary dramatically. PICHENY et al. discarded the idea out of hand as "naïve" and lacking robustness. ${ }^{10}$ The authors also found it empirically inferior in their investigation and described the same over-exploratory effect and explanation given above. On the other hand, NGUYEN et al. described the estimator as "standard" and concluded it was empirically preferable! ${ }^{11}$ An alternative estimator is the simple reward of the data $(6.3):^{12,13}$

$$
\phi^{*} \approx \max \mu_{\mathcal{D}}(\mathbf{x}),
$$

which is less biased and may be preferable. A simple extension is to maximize other predictive quantiles, ${ }^{10,14}$ and HUANG et al. recommend using a relatively low quantile, specifically $\Phi(-1) \approx 0.16$, in the interest of risk aversion.

In Figure 8.4 we compare noisy expected improvement with two plug-in approximations. The plug-in estimators agree that sampling on the right-hand side of the domain is the most promising course of action, but our formulation of noisy expected improvement prefers a less explored region. This decision is motivated by the interesting behavior of the posterior, which shows considerable disagreement regarding the updated posterior mean; see Figure 8.6. This nuance is only revealed as our formulation reasons about the joint predictive distribution of $\mathbf{y}^{\prime}$, whereas the plug-in estimators only inspect the marginals.

Another proposed approximation scheme for noisy expected improvement is reinterpolation. We fit a noiseless Gaussian process to imputed values of the objective function at the observed locations $\phi=f(\mathbf{x})$, then compute the exact expected improvement for this surrogate. A natural choice considered by FORRESTER et al. is to impute using the posterior mean: ${ }^{15}$

$$
\phi \approx \mu_{\mathcal{D}}(\mathbf{x})
$$

resulting in the approximation (computed with respect to the surrogate):

$$
\alpha_{\mathrm{EI}}(x ; \mathcal{D}) \approx \alpha_{\mathrm{EI}}(x ; \mathbf{x}, \boldsymbol{\phi}) \text {. }
$$

This procedure is illustrated in Figure 8.5. The resulting decision is very similar to that made by noisy expected improvement.

LETHAM et al. also promoted this basic approach, but proposed marginalizing rather than imputing the latent objective function values: ${ }^{16}$

$$
\alpha_{\mathrm{EI}}(x ; \mathcal{D}) \approx \int \alpha_{\mathrm{EI}}(x ; \mathbf{x}, \boldsymbol{\phi}) p(\boldsymbol{\phi} \mid \mathbf{x}, \mathcal{D}) \mathrm{d} \boldsymbol{\phi}
$$

The approximate acquisition function is the expectation of the exact expected improvement if we had access to exact observations. Although this integral cannot be computed exactly, the authors described a straightforward and effective quasi-Monte Carlo approximation. LETHAM et al.'s approximation is illustrated in Figure 8.6. There is good agreement with the expected improvement acquisition function from Figure 8.4, except near the observation location chosen by that policy.

This stark disagreement is a consequence of reinterpolation: the acquisition function vanishes at previously observed locations - just as the exact expected improvement does - regardless of the observed values. Thus repeated measurements at the same location are barred, strongly encouraging exploration. HUANG et al. cite this property as undesirable, ${ }^{17}$ as repeated measurements can reinforce our understanding
12 E. vazQuez et al. (2008). Global Optimization Based on Noisy Evaluations: An Empirical Study of Two Statistical Approaches. ICIPE 2008.

13 Z. WANG and N. DE FREITAS (2014). Theoretical Analysis of Bayesian Optimization with Unknown Gaussian Process Hyper-Parameters. arXiv: 1406.7758 [stat.ML].

14 D. HUANG et al. (2006b). Global Optimization of Stochastic Black-Box Systems via Sequential Kriging Meta-Models. Journal of Global Optimization 34(3):441-466

approximating through reinterpolation

15 A. I. J. FORRESTER et al. (2006). Design and Analysis of "Noisy" Computer Experiments. AIAA fournal 44(10):2331-2339.

16 B. LETHAm et al. (2019). Constrained Bayesian Optimization with Noisy Experiments. Bayesian Analysis 14(2):495-519.

reduction to zero at observed locations

17 D. HUANG et al. (20o6b). Global Optimization of Stochastic Black-Box Systems via Sequential Kriging Meta-Models. Journal of Global Optimization 34(3):441-466. 
![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-10.jpg?height=446&width=1648&top_left_y=515&top_left_x=152)

FORRESTER et al.'s approximation (8.19) $\quad \boldsymbol{\nabla}$ next observation location

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-10.jpg?height=200&width=1645&top_left_y=1034&top_left_x=157)

Figure 8.5: FORRESTER et al.'s approximation to noisy expected improvement (8.19). Given a Gaussian process fit to noisy data, we compute the exact expected improvement (8.9) for a noiseless Gaussian process fit to the posterior mean.

adjusting for observation noise

18 D. HUANG et al. (2006b). Global Optimization of Stochastic Black-Box Systems via Sequential Kriging Meta-Models. fournal of Global Optimization 34(3):441-466.

19 This is only sensible when the signal-to-noise ratio is at least one. of the optimum by reducing uncertainty. LETHAM et al. on the other hand suggest the exploration boost is beneficial for optimization and point out we may reduce uncertainty through measurements in neighboring locations if desired.

A weakness shared by all these approximations is that the underlying noiseless expected improvement incorrectly assumes that our observation will reveal the exact objective value. In fact, observation noise is ignored entirely, as all these acquisition functions are expectations with respect to the unobservable quantity $\phi$ rather than the observed quantity $y$. This represents a disconnect between the reasoning of the optimization policy and the true nature of the observation process. HUANG et al. acknowledged this issue and proposed an augmented expected improvement measure accounting for observation noise ${ }^{18}$ by multiplying by the factor $1-\sigma_{n} / s,{ }^{19}$ penalizing locations with low signal-to-noise ratios.

Ignoring the distribution of the observed value can be especially problematic with heteroskedastic noise, as shown in Figure 8.7. Both the augmented ${ }^{18}$ and noisy expected improvement acquisition functions are biased toward the left-hand side of the domain, where observations reveal more information. Plug-in approximations, on the other hand, are oblivious to this distinction and elect to explore the noisy region on the right; see Figure 8.4.

Our opinion is that all approximation schemes based on reducing to exact expected improvement should be avoided with significant noise levels, but can be a reasonable choice otherwise. With low noise, observed 
![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-11.jpg?height=502&width=1632&top_left_y=519&top_left_x=268)

- sampled acquisition functions
![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-11.jpg?height=384&width=1630&top_left_y=1062&top_left_x=268)

Figure 8.6: LETHAM et al.'s approximation to noisy expected improvement (8.20). We take the expectation of the exact expected improvement (8.9) for a noiseless Gaussian process fit to exact observations at the observed locations. The middle panels show realizations of the reinterpolated process and the resulting expected improvement.

values cannot stray too far from the true underlying objective function value. Thus we have $y \approx \phi ; s \approx \sigma$, and any inaccuracy in (8.17), (8.18), or (8.20) will be minor. However, this heuristic argument breaks down in high-noise regimes.

\subsection*{PROBABILITY OF IMPROVEMENT}

Probability of improvement (7.5) represents the probability that the simprobability of improvement: $§ 7.5$, p. 131 ple reward of our data (6.3) will exceed a threshold $\tau$ after obtaining an observation at $x$ :

$$
\alpha_{\mathrm{PI}}(x ; \mathcal{D})=\operatorname{Pr}\left(u\left(\mathcal{D}^{\prime}\right)>\tau \mid x, \mathcal{D}\right) .
$$

Like expected improvement, this quantity can be computed exactly for our chosen model class.

\section*{Probability of improvement without noise}

The probability of improvement after an exact observation at $x$ is simply the probability that the observed value will exceed $\tau$ (7.6). For a Gaussian 

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-12.jpg?height=343&width=1630&top_left_y=448&top_left_x=156)

_ augmented expected improvement (HUANG et al., 2006b)

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-12.jpg?height=206&width=1628&top_left_y=865&top_left_x=157)

noisy expected improvement (8.16)

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-12.jpg?height=206&width=1625&top_left_y=1136&top_left_x=158)

Figure 8.7: A comparison of noiseless expected improvement with a plug-in estimate for $\phi^{*}(8.17)$ and the expected one-step marginal gain in simple reward (7.2). Here the noise variance increases linearly from zero on the left-hand side of the domain to a signal-to-noise ratio of 1 on the right-hand side. The larger enveloping area shows $95 \%$ credible intervals for the noisy observation $y$, whereas the smaller area provides the same credible intervals for the objective function.

equivalent acquisition function, $\alpha_{\mathrm{PI}}^{\prime}$

partial derivatives with respect to predictive distribution parameters process, this probability is given by the complementary Gaussian CDF:

$$
\alpha_{\mathrm{PI}}(x ; \mathcal{D}, \tau)=\Phi\left(\frac{\mu-\tau}{\sigma}\right) .
$$

As our policy will ultimately be determined by maximizing the probability of improvement, it is prudent to transform this expression into the simpler and better-behaved acquisition function

$$
\alpha_{\mathrm{PI}}^{\prime}(x ; \mathcal{D}, \tau)=\frac{\mu-\tau}{\sigma}
$$

which shares the same maxima but is slightly cheaper to compute and does not suffer from a vanishing gradient for extreme values of $\tau$.

We may gain some insight into the behavior of probability of improvement by computing the gradient of this alternative expression (8.22) with respect to the parameters of the predictive distribution:

$$
\frac{\partial \alpha_{\mathrm{PI}}^{\prime}}{\partial \mu}=\frac{1}{\sigma} ; \quad \frac{\partial \alpha_{\mathrm{PI}}^{\prime}}{\partial \sigma}=\frac{\tau-\mu}{\sigma^{2}} .
$$

We observe that probability of improvement is monotonically increasing with $\mu$, universally encouraging exploitation. Probability of improvement 

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-13.jpg?height=289&width=802&top_left_y=461&top_left_x=273)

is also increasing with $\sigma$ when the target value is greater than the predictive mean, or equivalently when the probability of improvement is less than $1 / 2$. Therefore probability of improvement also tends to encourage exploration in this typical case. However, when the predictive mean is greater than the improvement threshold, probability of improvement is, perhaps surprisingly, decreasing with $\sigma$, discouraging exploration in this favorable regime. Probability of improvement favors a relatively safe option returning a certain but modest improvement over a more-risky alternative offering a potentially larger but less-certain improvement, as demonstrated in Figure 7.8.

With the partial derivatives computed above, we may readily compute the gradient of (8.22) with respect to the proposed evaluation location $x$ in the noiseless case (8.6):

$$
\frac{\partial \alpha_{\mathrm{PI}}^{\prime}}{\partial x}=\frac{1}{\sigma}\left[\frac{\partial \mu}{\partial x}-\alpha_{\mathrm{PI}}^{\prime} \frac{\partial \sigma}{\partial x}\right] .
$$

\section*{Probability of improvement with noise}

As with expected improvement, computing probability of improvement with noisy observations is slightly more complicated than with exact observations, and for the same reason. A noisy observation can affect the posterior mean at every previously observed location, and thus improvement to the simple reward can occur at any point. However, we can compute probability of improvement exactly by adapting the techniques we used to compute noisy expected improvement (8.16).

As before, the key observation is that the updated posterior mean is linear in the observed value $y$ (8.11), allowing us to express the updated simple reward as

$$
u\left(\mathcal{D}^{\prime}\right)=\max (\mathbf{a}+\mathbf{b} z),
$$

where $\mathbf{a}$ and $\mathbf{b}$ are defined in (8.12) and $z$ is the $z$-score of the observation. Now we can write the probability of improvement as the expectation of an indicator against a standard normal random variable; see Figure 8.8:

$$
\alpha_{\mathrm{PI}}(x ; \mathcal{D})=\int[\max (\mathbf{a}+\mathbf{b} z)>\tau] \phi(z) \mathrm{d} z .
$$

Due to the convexity of the updated simple reward, improvement occurs on at-most two intervals: ${ }^{20}$

$$
(-\infty, \ell) \cup(u, \infty) .
$$

risk aversion of probability of improvement: $\S 7.5$, p. 132

gradient of probability of improvement without noise The probability of improvement is the probability this value exceeds a threshold $\tau$, the area of the shaded region.

20 In the case that improvement is impossible or occurs at a single point, the probability of improvement is zero. gradient of noisy probability of improvement: $\S$ c.3, p. 309

upper confidence bound: § 7.8, p. 145
21 A word of warning: the exploration parameter is sometimes denoted $\sqrt{\beta}$, so that $\beta$ is a weight on the variance $\sigma^{2}$ instead.

gradient of upper confidence bound
The endpoints of these intervals may be computed directly by inspecting the intersections of the lines $\mathbf{a}+\mathbf{b} z$ with the threshold:

$$
\ell=\max _{i}\left\{\left(\tau-a_{i}\right) / b_{i} \mid b_{i}<0\right\} ; \quad u=\min _{i}\left\{\left(\tau-a_{i}\right) / b_{i} \mid b_{i}>0\right\} .
$$

Note that one of these endpoints may not exist, for example if every slope in the $\mathbf{b}$ vector were positive; see Figure 8.3 for an example. In this case we may take $\ell=-\infty$ or $u=\infty$ as appropriate. Now given the endpoints $(\ell, u)$, the probability of improvement may be computed in terms of the standard normal $\mathrm{CDF}$ :

$$
\alpha_{\mathrm{PI}}(x ; \mathcal{D})=\Phi(\ell)+\Phi(-u) .
$$

We may compute the gradient of this noisy formulation of probability of improvement; details are given in the appendix.

\subsection*{UPPER CONFIDENCE BOUND}

Computing an upper confidence bound (7.18) for a Gaussian process is trivial. Given a confidence parameter $\pi \in(0,1)$, we must compute a pointwise upper bound for the objective function with that confidence:

$$
\alpha_{\mathrm{UCB}}(x ; \mathcal{D}, \pi)=q(\pi ; x, \mathcal{D}),
$$

where $q$ is the quantile function of the predictive distribution. For a Gaussian process, this quantile takes the simple form

$$
\alpha_{\mathrm{UCB}}(x ; \mathcal{D}, \pi)=\mu+\beta \sigma,
$$

where $\beta=\Phi^{-1}(\pi)$ depends on the confidence level and can be computed from the inverse Gaussian CDF. This acquisition function is normally parameterized directly in terms of $\beta$ rather than the confidence level $\pi$. In this case $\beta$ can be interpreted as an "exploration parameter," as higher values clearly reward uncertainty more than smaller values. ${ }^{21}$ No special care is required in computing (8.25), and its gradient with respect to the proposed observation location $x$ can also be computed easily:

$$
\frac{\partial \alpha_{\mathrm{UCB}}}{\partial x}=\frac{\partial \mu}{\partial x}+\beta \frac{\partial \sigma}{\partial x}
$$

\section*{Correspondence with probability of improvement}

For Gaussian processes, we can derive an intimate correspondence between the (noiseless) probability of improvement and upper confidence bound acquisition functions. Namely, a point maximizing probability of improvement for a given target also maximizes some statistical upper bound of the objective and vice versa, a fact that has been noted by several authors, including JONES ${ }^{22}$ and, independently, wANG et al. ${ }^{23}$

To establish this result, let an arbitrary exploration parameter $\beta$ be given. Consider a point optimizing the upper confidence bound:

$$
x \in \underset{x^{\prime} \in \mathcal{X}}{\arg \max } \mu+\beta \sigma,
$$

and define

$$
\tau(\beta)=\max _{x^{\prime} \in \mathcal{X}} \mu+\beta \sigma
$$

to be its optimal value. Then the following equalities are satisfied at $x$ :

$$
\tau(\beta)=\mu+\beta \sigma ; \quad \beta=\frac{\tau(\beta)-\mu}{\sigma} .
$$

Now it is easy to show that $x$ also minimizes the score

$$
x \in \underset{x^{\prime} \in \mathcal{X}}{\arg \min } \frac{\tau(\beta)-\mu}{\sigma},
$$

because if there were some other point $x^{\prime}$ with

$$
\frac{\tau(\beta)-\mu^{\prime}}{\sigma^{\prime}}<\beta,
$$

then we would have

$$
\mu^{\prime}+\beta \sigma^{\prime}>\tau(\beta),
$$

contradicting the optimality of $x$. Therefore $x$ also maximizes probability of improvement with target $\tau(\beta)$ (8.22).

It is important to note that the value of this target $\tau(\beta)$ is data- and model-dependent and may change from iteration to iteration. Therefore sequentially maximizing an upper confidence bound with a fixed exploration parameter is not equivalent to maximizing probability of improvement with a fixed improvement target.

\subsection*{APPROXIMATE COMPUTATION FOR ONE-STEP LOOKAHEAD}

Unfortunately we have exhausted the acquisition functions for which exact computation is possible with Gaussian process models. However, we can still proceed effectively with appropriate approximations. We will begin by discussing the implementation of arbitrary one-step lookahead policies when exact computation is not possible.

Recall that one-step lookahead entails maximizing the expected marginal gain to a utility function $u(\mathcal{D})$ after making an observation at a proposed location $x$ :

$$
\alpha(x ; \mathcal{D})=\int\left[u\left(\mathcal{D}^{\prime}\right)-u(\mathcal{D})\right] \mathcal{N}\left(y ; \mu, s^{2}\right) \mathrm{d} y
$$

If this integral is intractable, we must resort to analytic approximation or numerical integration to evaluate and optimize the acquisition function. Fortunately in the case of sequential optimization, this is a one-dimensional integral that can be approximated using standard tools.

It will be convenient below to introduce simple notation for the marginal gain in utility resulting from a putative observation $(x, y)$. We will write

$$
\Delta(x, y)=u\left(\mathcal{D}^{\prime}\right)-u(\mathcal{D})
$$

data-dependence of relationship

one-step lookahead: § 5.3, p. 101

marginal gain from observation $(x, y)$, $\Delta(x, y)$ Gauss-Hermite quadrature

24 P. J. DAVIS and P. RABINOwITZ (1984). Methods of Numerical Integration. Academic Press.

integration nodes, $\left\{z_{i}\right\}$

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-16.jpg?height=183&width=523&top_left_y=1062&top_left_x=161)

Gauss-Hermite nodes/weights for $n \in\{5,8\}$.

25 T. S. SHAO et al. (1964). Tables of Zeros and Gaussian Weights of Certain Associated Laguerre Polynomials and the Related Generalized Hermite Polynomials. Mathematics of Computation 18(88):598-616.

26 We take

$$
y=\mu+\sqrt{2} s z \quad \Leftrightarrow \quad z=\frac{y-\mu}{\sqrt{2} s} .
$$

Note we must account for the normalization factor $(\sqrt{2 \pi} s)^{-1}$ of the Gaussian distribution, hence the constant that appears.

approximating gradient via Gauss-Hermite quadrature: § C.3, p. 309

knowledge gradient: $§ 7.4$, p. 129 for this quantity, leaving the dependence on $\mathcal{D}$ implicit. Now we seek to approximate

$$
\alpha(x ; \mathcal{D})=\int \Delta(x, y) \mathcal{N}\left(y ; \mu, s^{2}\right) \mathrm{d} y .
$$

We recommend using off-the-shelf quadrature methods to estimate the expected marginal gain (8.27). ${ }^{24}$ The most natural approach is GaussHermite quadrature, a classical approach for approximating integrals of the form

$$
I=\int h(z) \exp \left(-z^{2}\right) \mathrm{d} z .
$$

Like all numerical integration methods, Gauss-Hermite quadrature entails measuring the integrand $h$ at a set of $n$ points, called nodes, $\left\{z_{i}\right\}$, then approximating the integral by a weighted sum of the measured values:

$$
I \approx \sum_{i=1}^{n} w_{i} h\left(z_{i}\right)
$$

Remarkably, this estimator is exact when the integrand is a polynomial of degree less than $2 n-1$. This guarantee provides some guidance for selecting the order of the quadrature rule depending on how well the marginal gain $\Delta$ may be approximated by a polynomial over the range of plausible observations. Tables of integration nodes and quadrature weights for $n$ up to 64 are readily available, although such a high order should rarely be needed. ${ }^{25}$

Through a simple transformation ${ }^{26}$ we can rewrite the arbitrary Gaussian expectation (8.27) in the Gauss-Hermite form (8.28):

$$
\int \Delta(x, y) \mathcal{N}\left(y ; \mu, s^{2}\right) \mathrm{d} y=\frac{1}{\sqrt{\pi}} \int \Delta(x, \mu+\sqrt{2} s z) \exp \left(-z^{2}\right) \mathrm{d} z .
$$

If we define appropriately renormalized weights

$$
\bar{w}_{i}=w_{i} / \sqrt{\pi},
$$

then we arrive at the following approximation to the acquisition function:

$$
\alpha(x ; \mathcal{D}) \approx \sum_{i=1}^{n} \bar{w}_{i} \Delta\left(x, y_{i}\right) ; \quad y_{i}=\mu+\sqrt{2} z_{i} s .
$$

We may also extend this scheme to approximate the gradient of the acquisition function using the same nodes and weights; details are provided in the accompanying appendix.

\subsection*{KNOWLEDGE GRADIENT}

The knowledge gradient is the expected one-step gain in the global reward (6.5). If we define

$$
\mu^{*}=\max _{x \in \mathcal{X}} \mu_{\mathcal{D}}(x)
$$

to be the utility of the current dataset, then we must compute:

$$
\alpha_{\mathrm{KG}}(x ; \mathcal{D})=\int\left[\max _{x^{\prime} \in \mathcal{X}} \mu_{\mathcal{D}^{\prime}}\left(x^{\prime}\right)-\mu^{*}\right] p(y \mid x, \mathcal{D}) \mathrm{d} y .
$$

The global optimization in the expectation renders the knowledge gradient nontrivial to compute in most cases, so we must resort to quadrature or other approximation.

\section*{Exact computation in discrete domains}

FRAZIER et al. first proposed the knowledge gradient for optimization on a discrete domain $\mathcal{X}=\{1,2, \ldots, n\} .{ }^{27}$ In this case, the objective is simply a vector $\mathbf{f} \in \mathbb{R}^{n}$, and a Gaussian process belief about the objective is simply a multivariate normal distribution:

$$
p(\mathbf{f} \mid \mathcal{D})=\mathcal{N}(\mathbf{f} ; \boldsymbol{\mu}, \Sigma) .
$$

The knowledge gradient now reduces to the expected marginal gain in the maximum of the posterior mean vector:

$$
\alpha_{\mathrm{KG}}(x ; \mathcal{D})=\int\left[\max \boldsymbol{\mu}^{\prime}-\mu^{*}\right] p(y \mid x, \mathcal{D}) \mathrm{d} y .
$$

We may compute this expectation in closed form following our analysis of noisy expected improvement, which was merely a slight adaptation of FRAZIER et al.'s approach.

The updated posterior mean vector $\boldsymbol{\mu}^{\prime}$ after observing an observation $(x, y)$ is linear in the observed value $y$ :

$$
\boldsymbol{\mu}^{\prime}=\boldsymbol{\mu}+\frac{\Sigma_{x}}{s} \frac{y-\mu_{x}}{s},
$$

where $\mu_{x}$ is the entry of $\boldsymbol{\mu}$ corresponding to the index $x$, and $\Sigma_{x}$ is similarly the corresponding column of $\Sigma$. If we define

$$
\mathbf{a}=\boldsymbol{\mu} ; \quad \mathbf{b}=\frac{\Sigma_{x}}{s},
$$

then we may rewrite the knowledge gradient in terms of the $g$ function introduced in the context of noisy expected improvement (8.10):

$$
\alpha_{\mathrm{KG}}(x ; \mathcal{D})=g(\mathbf{a}, \mathbf{b})-\mu^{*}
$$

We may evaluate this expression exactly in $\mathcal{O}(n \log n)$ time following our previous discussion. Thus we may compute the knowledge gradient policy with a discrete domain in $\mathcal{O}\left(n^{2} \log n\right)$ time per iteration by exhaustive computation of the acquisition function.

\section*{Approximation via numerical quadrature}

Although the knowledge gradient acquisition function can be computed exactly over discrete domains, the situation becomes significantly more
27 P. Frazier et al. (2009). The KnowledgeGradient Policy for Correlated Normal Beliefs. INFORMS fournal on Computing 21(4):599-613.

computation of noisy expected improvement: $\S 8.2$, p. 160

computation in terms of $g(\mathbf{a}, \mathbf{b})$

computation of $g$ : $\S 8.2$, p. 161 
![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-18.jpg?height=592&width=1604&top_left_y=470&top_left_x=182)

Figure 8.9: The complex behavior of the updated global reward. Above: the location of the posterior mean maximum given the $z$-score of an observation at two points, $x$ and $x$. An observation at $x^{\prime}$ always results in shoring up the existing maximum, whereas an observation at $x$ reveals a new maximum given a sufficiently high observation. Below: the marginal gain in the global reward as a function of the $z$-score of an observation at these locations.

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-18.jpg?height=263&width=808&top_left_y=1168&top_left_x=972)

exact computation for special cases

28 H. J. KUSHNER (1964). A New Method of Locating the Maximum Point of an Arbitrary Multipeak Curve in the Presence of Noise. Fournal of Basic Engineering 86(1):97-106.

29 J. MоскUS (1972). Bayesian Methods of Search for an Extremum. Avtomatika $i$ Vychislitel'naya Tekhnika (Automatic Control and Computer Sciences) 6(3):53-62. complicated in continuous domains. The culprit is the nonconvex global optimization in (8.30), which makes the knowledge gradient intractable except in a few special cases.

Some Gaussian processes give rise to convenient structure in the posterior distribution facilitating computation of the knowledge gradient. For example, in one dimension the Wiener or Ohrstein-Uhlenbeck processes satisfy a Markov property guaranteeing the posterior mean is always maximized at an observed location. As a result the simple reward and global reward are always equal, and the knowledge gradient reduces to expected improvement. This structure was often exploited in the early literature on Bayesian optimization. ${ }^{28,29}$

Figure 8.9 give some insight into the complexity of the knowledge gradient integral (8.30). We illustrate the possible results from adding an observation at two locations as a function of the $z$-score of the observed value. For the point on the left, the behavior is similar to expected improvement: a sufficiently high value moves the maximum of the posterior mean to that point; otherwise, we retain the incumbent. The marginal gain in utility is a piecewise linear function corresponding to these two outcomes, as in expected improvement. The point on the right displays entirely different behavior - the marginal gain in utility is smooth, and nearly any outcome would be beneficial. 

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-19.jpg?height=634&width=1630&top_left_y=454&top_left_x=270)

Figure 8.10: A comparison of the knowledge gradient acquisition function and the KGCP approximation for an example scenario. In this case the KGCP approximation reverts to expected improvement.

When exact computation is not possible, we must resort to numerical integration or an analytic approximation when adopting the knowledge gradient policy. The former path was explored in depth by wu et al. ${ }^{30} \mathrm{To}$ compute the acquisition function at a given point, we can compute a highaccuracy approximation following the numerical techniques outlined in the previous section. Order- $n$ Gauss-Hermite quadrature for example, would approximate with (8.29):

$$
\alpha_{\mathrm{KG}}(x ; \mathcal{D}) \approx \frac{1}{\sqrt{\pi}} \sum_{i=1}^{n} w_{i} \Delta\left(x, y_{i}\right) ; \quad \Delta\left(x, y_{i}\right)=\max _{x^{\prime} \in \mathcal{X}} \mu_{\mathcal{D}^{\prime}}\left(x^{\prime}\right)-\mu^{*}
$$

With some care, we can also approximate the gradient of the knowledge gradient in this scheme; details are given in an appendix.

\section*{Knowledge gradient for continuous parameters approximation}

scotT et al. suggested an alternative and lightweight approximation scheme for the knowledge gradient the authors called the knowledge gradient for continuous parameters (KGCP). ${ }^{31}$ The KGCP approximation entails replacing the domain of the maximization in the definition of the global reward utility, normally the entire domain $\mathcal{X}$, with a conveniently chosen discrete set: the already observed locations $\mathbf{x}$ and the proposed new observation location $x$. Let $\mathbf{x}^{\prime}$ represent this set. We approximate the current and future utility with

$$
u(\mathcal{D}) \approx \max \mu_{\mathcal{D}}\left(\mathbf{x}^{\prime}\right) ; \quad u\left(\mathcal{D}^{\prime}\right) \approx \max \mu_{\mathcal{D}^{\prime}}\left(\mathbf{x}^{\prime}\right),
$$

yielding the approximation

$$
\alpha_{\mathrm{KG}}(x ; \mathcal{D}) \approx \mathbb{E}\left[\max \mu_{\mathcal{D}^{\prime}}\left(\mathbf{x}^{\prime}\right) \mid x, \mathcal{D}\right]-\max \mu_{\mathcal{D}}\left(\mathbf{x}^{\prime}\right) .
$$

30 J. wu et al. (2017). Bayesian Optimization with Gradients. NeurIPS 2017.

approximating gradient of knowledge gradient: § C.3, p. 310

31 w. scotT et al. (2011). The Correlated Knowledge Gradient for Simulation Optimization of Continuous Parameters Using Gaussian Process Regression. SIAM fournal on Optimization 21(3):996-1026. comparison with expected improvement

gradient of KGCP approximation

example and discussion

Thompson sampling: § 7.9, p. 148

32 One notable example is the Wiener process, where $x^{*}$ famously has an arcsine distribution:

P. LÉvy (1948). Processus stochastiques et mouvement brownien. Gauthier-Villars.
This expression is almost identical to expected improvement (8.7)! In fact, if we define

$$
\mu^{*}=\max \mu_{\mathcal{D}}(\mathbf{x}),
$$

then a simple manipulation gives:

$$
\alpha_{\mathrm{KG}}(x ; \mathcal{D}) \approx \alpha_{\mathrm{EI}}(x ; \mathcal{D})-\max \left(\mu-\mu^{*} 0\right) \text {. }
$$

Effectively, the KGCP approximation is a simple adjustment to expected improvement where we punish points for already having large expected value. From the point of view of the global reward, these points already represent success and their large expected values are already reflected in the current utility. Therefore, we should not necessarily waste precious evaluations confirming what we already believe.

The gradient of the KGCP approximation may also be computed in terms of the gradient of expected improvement and the posterior mean:

$$
\frac{\partial \alpha_{\mathrm{KG}}}{\partial x} \approx \frac{\partial \alpha_{\mathrm{EI}}}{\partial x}-\left[\mu>\mu^{*}\right] \frac{\partial \mu}{\partial x} .
$$

In the case of our example scenario, the posterior mean never exceeds the highest observed point. Therefore, the KGCP approximation to the knowledge gradient globally reduces to the expected improvement; see Figure 8.10 and compare with the expected improvement in Figure 7.3. Comparing with true knowledge gradient, we can see that this approximation cannot necessarily be trusted to be unconditionally faithful. However, the KGCP approximation has the advantage of efficient computation and may be used as a drop-in replacement for expected improvement that may offer a slight boost in performance when the global reward utility is preferred to simple reward.

\subsection*{THOMPSON SAMPLING}

Thompson sampling designs each observation by sampling a point proportional to its probability of maximizing the objective (7.19):

$$
x \sim p\left(x^{*} \mid \mathcal{D}\right)
$$

A major barrier to Thompson sampling with Gaussian processes is the complex nature of this distribution. Except in a small number of special cases, ${ }^{32}$ this distribution cannot be computed analytically. Figure 8.11 illustrates the complicated nature of this distribution for our running example, which was only revealed via brute-force sampling. However, a straightforward implementation strategy is to maximize a draw from the objective function posterior, which assumes the role of an acquisition function:

$$
\alpha_{\mathrm{TS}}(x ; \mathcal{D}) \sim p(f \mid \mathcal{D}) .
$$

The global optimum of $\alpha_{\mathrm{TS}}$ is then a sample from the desired distribution.

This procedure in fact yields a sample from the joint distribution of the location and value of the optimum, $p\left(x^{*}, f^{*} \mid \mathcal{D}\right)$, as the value of 

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-21.jpg?height=431&width=1630&top_left_y=453&top_left_x=270)

$\triangle$ Thompson samples

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-21.jpg?height=175&width=1628&top_left_y=969&top_left_x=271)

Figure 8.11: The distribution of the location of the global maximum, $p\left(x^{*} \mid \mathcal{D}\right)$, for an example scenario, and 1oo samples drawn from this distribution.

the sampled objective function $\alpha_{\mathrm{TS}}$ at its maximum provides a sample from $p\left(f^{*} \mid x^{*} \mathcal{D}\right)$; see the margin. We discuss Thompson sampling now because the ability to sample from these distributions will be critical for computing mutual information, our focus in the following sections.

\section*{Exhaustive sampling}

In "small" domains, we can realize Thompson sampling via brute force. If the domain can be exhaustively covered by a sufficiently small set of points $\xi$ (for example, with a dense grid or a low-discrepancy sequence) then we can simply sample the associated objective function values $\phi=f(\xi)$ and maximize: ${ }^{33}$

$$
x=\arg \max \phi ; \quad \phi \sim p(\phi \mid \xi, \mathcal{D}) .
$$

The distribution of $\boldsymbol{\phi}$ is multivariate normal, making sampling easy:

$$
p(\phi \mid \xi, \mathcal{D})=\mathcal{N}(\phi ; \mu, \Sigma) ; \quad \boldsymbol{\mu}=\mu_{\mathcal{D}}(\xi) ; \quad \Sigma=K_{\mathcal{D}}(\xi, \xi) .
$$

The running time of this procedure grows quickly with the size of $\xi$, although sophisticated numerical methods enable scaling to roughly 50 ooo points. ${ }^{34}$

Figure 8.11 shows 100 Thompson samples for our example scenario generated via exhaustive sampling, taking $\xi$ to be a grid of 1000 points.

\section*{On-demand sampling}

An alternative to exhaustive sampling is to use off-the-shelf optimization routines to maximize a draw from the objective function posterior we

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-21.jpg?height=276&width=508&top_left_y=1324&top_left_x=1388)

Maximizing a draw from a Gaussian process naturally samples from $p\left(x^{*} f^{*} \mid \mathcal{D}\right)$

33 We are taking a slight liberty with notation here as we have previously used $\boldsymbol{\phi}$ for $f(\mathbf{x})$, the latent objective function values at the observed locations. However, for the remainder of this discussion we will be assuming the general, potentially noisy case where our data will be written $\mathcal{D}=(\mathbf{x}, \mathbf{y})$ and we will have no need to refer to $f(\mathbf{x})$.

34 G. PLEISS et al. (2020). Fast Matrix Square Roots with Applications to Gaussian Processes and Bayesian Optimization. NeurIPS 2020. Algorithm 8.1: On-demand sampling.

low-rank updates: § 9.1, p. 202 running time using gradient methods

35 M. LÁZARO-GREDILLA et al. (2010). Sparse Spectrum Gaussian Process Regression. fournal of Machine Learning Research 11(Jun):1865-1881.

36 J. M. HERnÁNDEZ-LOBATO et al. (2014). Predictive Entropy Search for Efficient Global Optimization of Black-Box Functions. NeurIPS 2014.

stationarity: § 3.2, p. 50

37 Recall the convention of writing a stationary covariance with respect to a single input, $K\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=K\left(\mathbf{x}-\mathbf{x}^{\prime}\right)$.

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-22.jpg?height=335&width=1013&top_left_y=455&top_left_x=790)

build progressively on demand. Namely, when an optimization routine requests an evaluation of $\alpha_{\mathrm{TS}}$ at a point $x$, we sample an objective function value at that location:

$$
\phi \sim p(\phi \mid x, \mathcal{D}),
$$

then augment our dataset with the simulated observation $(x, \phi)$. We proceed in this manner until the optimizer terminates. This procedure avoids simulating the entire objective function while guaranteeing the joint distribution of the provided evaluations is correct via the chain rule of probability. Pseudocode is provided in Algorithm 8.1 in the form of a generator function; the "yield" statement returns a value while maintaining state. Using rank-one updates to update the posterior, the computational cost of generating each evaluation scales quadratically with the size of the fictitious dataset $\mathcal{D}_{\text {TS }}$.

If desired, we may use gradient methods to optimize the generated sample by sampling from the joint posterior of the function value and its gradient at each requested location:

$$
p\left(\phi, \nabla \phi \mid x, \mathcal{D}_{\mathrm{TS}}\right)
$$

However, in high dimensions, the additional cost required to condition on these gradient observations may become excessive. In $d$ dimensions, returning gradients effectively reduces the number of function evaluations we can allow for the optimizer by a factor of $(d+1)$ if we wish to maintain the same total computational effort.

\section*{Sparse spectrum approximation for stationary covariance functions}

If the prior covariance function $K$ of the Gaussian process is stationary, we may use a sparse spectrum approximation ${ }^{35}$ to the posterior Gaussian process to dramatically accelerate Thompson sampling. ${ }^{36}$

Consider a stationary covariance function $K$ on $\mathbb{R}^{d}$ with spectral density $\kappa$. The key idea in sparse spectrum approximation is to interpret the characterization in (3.10) as an expectation with respect to the spectral density: ${ }^{37}$

$$
K\left(\mathbf{x}-\mathbf{x}^{\prime}\right)=K(\mathbf{0}) \mathbb{E}_{\xi}\left[\exp \left(2 \pi i\left(\mathbf{x}-\mathbf{x}^{\prime}\right)^{\top} \boldsymbol{\xi}\right)\right],
$$

and approximate via Monte Carlo integration. We first sample a set of $m$ frequencies, called spectral points, from the spectral density: $\left\{\xi_{i}\right\} \sim \kappa(\xi)$. posterior mean, exact

posterior $95 \%$ credible interval, exact

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-23.jpg?height=200&width=1015&top_left_y=591&top_left_x=272)

- posterior mean, sparse spectrum approximation posterior $95 \%$ credible interval, sparse spectrum approximation

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-23.jpg?height=199&width=1011&top_left_y=937&top_left_x=271)

To enforce the symmetry around the origin inherent to the spectral density (theorem 3.2), we augment each sample $\xi_{i}$ with its negation, $-\xi_{i}$.

Using these samples for a Monte Carlo approximation to (8.32) has the effect of approximating a Gaussian process $\mathcal{G P}(f ; \mu, K)$ with a finitedimensional Gaussian process:

$$
f(\mathbf{x}) \approx \mu(\mathbf{x})+\boldsymbol{\beta}^{\top} \boldsymbol{\psi}(\mathbf{x}) .
$$

Here $\boldsymbol{\beta}$ is a vector of normally distributed weights and $\psi: \mathbb{R}^{d} \rightarrow \mathbb{R}^{2 m}$ is a feature representation determined by the spectral points:

$$
\psi_{2 i-1}(\mathbf{x})=\cos \left(2 \pi \xi_{i}^{\top} \mathbf{x}\right) ; \quad \psi_{2 i}(\mathbf{x})=\sin \left(2 \pi \xi_{i}^{\top} \mathbf{x}\right) .
$$

For the additive Gaussian noise model, if $\mathrm{N}$ is the covariance of the noise contributions to the observed values $\mathbf{y}$, the posterior moments of the weight vector in this approximation are:

$$
\begin{aligned}
\mathbb{E}[\boldsymbol{\beta} \mid \mathcal{D}] & =\Psi^{\top}\left(\Psi \Psi^{\top}+a^{-2} \mathbf{N}\right)^{-1}(\mathbf{y}-\boldsymbol{\mu}) ; \\
\operatorname{cov}[\boldsymbol{\beta} \mid \mathcal{D}] & =a^{2}\left[\mathbf{I}-\Psi^{\top}\left(\Psi \Psi^{\top}+a^{-2} \mathbf{N}\right)^{-1} \Psi\right],
\end{aligned}
$$

where $a^{2}=K(\mathbf{0}) / m, \boldsymbol{\mu}$ is the prior mean of $\mathbf{y}$, and $\Psi$ is a matrix whose rows comprise the feature representations of the observed locations.

The sparse spectrum approximation allows us to generate a posterior sample of the objective function in time $\mathcal{O}\left(\mathrm{nm}^{2}\right)$ by drawing a sample from the weight posterior and appealing to the representation in (8.33). The resulting sample is nontrivial to maximize due to the nonlinear nature of the representation, but with the sampled weights in hand, we can generate requested function and gradient evaluations in constant time, enabling exhaustive optimization via gradient methods.

A sparse spectrum approximation is illustrated in Figure 8.12, using a total of 100 spectral points. The approximation is quite reasonable near data and acceptable in extrapolatory regions as well.
Figure 8.12: Sparse spectrum approximation. Top: the exact posterior belief (about noisy observations rather than the latent function) for a Gaussian process conditioned on $200 \mathrm{ob}-$ servations. Bottom: a sparse spectrum approximation using 100 spectral points sampled from the spectral density.

weight vector, $\boldsymbol{\beta}$

feature representation, $\psi$

noise covariance matrix, $\mathrm{N}$

training features, $\Psi$

sampling from a multivariate normal distribution: § A.2, p. 299 

\subsection*{MUTUAL INFORMATION WITH $x^{*}$}

mutual information: § 7.6, p. 135

$38 \mathrm{~J}$. villemonteix et al. (2009). An Informational Approach to the Global Optimization of Expensive-to-evaluate Functions. fournal of Global Optimization 44(4):509-534.

39 P. HENNIG and C. J. SCHULER (2012). Entropy Search for Information-Efficient Global Optimization. Fournal of Machine Learning Research 13 (Jun):1809-1837.
4O J. M. HERNÁNDEZ-LOBATO et al. (2014). Predictive Entropy Search for Efficient Global Optimization of Black-Box Functions. NeurIPS 2014.

41 With careful implementation, the two schemes are comparable in terms of computational cost. We focus on the predictive formulation because it is now more commonly encountered and serves as the foundation for several extensions described in Chapter 11.

predictive entropy search
Mutual information measures the expected information gain (6.3) provided by an observation at $x$ about some random variable of interest $\omega$, which can be expressed in two equivalent forms (7.14-7.15):

$$
\begin{aligned}
\alpha_{\mathrm{MI}}(x ; \mathcal{D}) & =H[\omega \mid \mathcal{D}]-\mathbb{E}_{y}\left[H\left[\omega \mid \mathcal{D}^{\prime}\right] \mid x, \mathcal{D}\right] \\
& =H[y \mid x, \mathcal{D}]-\mathbb{E}_{\omega}[H[y \mid \omega, x, \mathcal{D}] \mid x, \mathcal{D}]
\end{aligned}
$$

In the context of Bayesian optimization, the most natural choices for $\omega$ are the location $x^{*}$ and value $f^{*}$ of the global optimum, both of which were discussed in the previous chapter. Unfortunately, a Gaussian process belief on the objective function in general induces complex distributions for these quantities, which makes even approximating mutual information somewhat involved. However, several effective approximation schemes are available for both options. We will consider each in turn, beginning with $x^{*}$

\section*{Direct form of mutual information}

Of the two equivalent formulations of mutual information above, the former (8.34) is perhaps the most natural, as it reasons directly about changes in the entropy of the variable of interest. Initial work on mutual information with $x^{*}$ considered approximations to this direct expression $^{38}$ - including the work coining the now-common moniker entropy search for information-theoretic optimization policies ${ }^{39}$ - but this approach has fallen out of favor in preference for the latter formulation (8.35), discussed below.

The main computational difficulty in approximating (8.34) is the unwieldy (and potentially high-dimensional) distribution $p\left(x^{*} \mid \mathcal{D}\right)$; see even the simple example in Figure 8.11. There is no general method for computing the entropy of this distribution in closed form, and overcoming this barrier requires approximation. The usual approach is to make a discrete approximation to this distribution via a set of carefully maintained so-called representer points, then reason about changes in the entropy of this surrogate distribution via further layers of approximation.

\section*{Predictive form of mutual information}

HERNÁNDEZ-LOBATO et al. proposed a faithful approximation to the mutual information with $x^{*}$ based on the formulation in (8.35): ${ }^{40}$

$$
\alpha_{x^{*}}(x ; \mathcal{D})=H[y \mid x, \mathcal{D}]-\mathbb{E}\left[H\left[y \mid x, x^{*} \mathcal{D}\right] \mid x, \mathcal{D}\right] .
$$

Compared to the direct form (8.34), this formulation is attractive as all entropy computations are restricted to the one-dimensional predictive distribution $p(y \mid x, \mathcal{D})$; however, considerable approximation is still required to realize an effective policy, as we will see. ${ }^{41}$ The authors named their approach predictive entropy search to highlight this focus. Let us consider the evaluation of (8.36) for a Gaussian process model with additive Gaussian noise. To begin, the first term is simply the differential entropy of a one-dimensional Gaussian distribution (8.5) and may be computed in closed form (A.17):

$$
H[y \mid x, \mathcal{D}]=\frac{1}{2} \log \left(2 \pi e s^{2}\right) .
$$

Unfortunately, the second term is significantly more complicated to work with, and we can identify two primary challenges.

\section*{Approximating expectation with respect to $x^{*}$}

First, we must compute an expectation with respect to the location of the

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-25.jpg?height=43&width=1008&top_left_y=1001&top_left_x=273)
but Monte Carlo integration remains a viable option. We use Thompson sampling to generate samples of the optimal location $\left\{x_{i}^{*}\right\}_{i=1}^{n} \sim p\left(x^{*} \mid \mathcal{D}\right)$ and estimate the expected updated entropy with:

$$
\mathbb{E}\left[H\left[y \mid x, x^{*} \mathcal{D}\right] \mid x, \mathcal{D}\right] \approx \frac{1}{n} \sum_{i=1}^{n} H\left[y \mid x, x_{i}^{*}, \mathcal{D}\right] .
$$

When the prior covariance function is stationary, HERNÁNDEZ-LOBATO et al. further propose to exploit the efficient approximate Thompson sampling scheme via sparse spectrum approximation described in the last section. When feasible, this reduces the cost of drawing the samples, but it is not necessary for correctness.

\section*{Gaussian approximation to conditional predictive distribution}

Now we must address the predictive distribution conditioned on the location of the global optimum, $p\left(y \mid x, x^{*} \mathcal{D}\right)$, which we may express as the result of marginalizing the latent objective value $\phi$ :

$$
p\left(y \mid x, x^{*} \mathcal{D}\right)=\int p(y \mid x, \phi) p(\phi \mid x, x, \mathcal{D}) \mathrm{d} \phi
$$

It is unclear how we can condition our belief on the objective function given knowledge of the optimum, and - as the resulting posterior on $\phi$ will be non-Gaussian - how we can resolve the resulting integral (8.37)

A key insight is that if the predictive distribution $p\left(\phi \mid x, x^{*} \mathcal{D}\right)$ were Gaussian, we could compute (8.37) in closed form, ${ }^{42}$ suggesting a promising path forward. In particular, consider an arbitrary Gaussian approximation:

$$
p(\phi \mid x, x, \mathcal{D}) \approx \mathcal{N}\left(\phi ; \mu_{*}, \sigma_{*}^{2}\right),
$$

whose parameters depend $x^{*}$ as their subscripts indicate. Plugging into (8.37), we may estimate the predictive variance of $y$ given $x^{*}$ with (A.15):

$$
\operatorname{var}\left[y \mid x, x^{*} \mathcal{D}\right] \approx \sigma_{*}^{2}+\sigma_{n}^{2}=s_{*}^{2},
$$

computing first term

Thompson sampling for GPs: § 8.7, p. 176

sparse spectrum approximation for Thompson sampling: $§ 8.7$, p. 178
42 In this case, the integral in (8.37) becomes the convolution of two Gaussians, which may be interpreted as the distribution of the sum of independent Gaussian random variables - see $\S$ A.2, p. 300 .

approximate variance of $y$ given $x, s_{*}^{2}$ 

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-26.jpg?height=265&width=1633&top_left_y=467&top_left_x=157)

Figure 8.13: The example scenario we will consider for illustrating the predictive entropy search approximation to $p\left(f \mid x, x^{*} \mathcal{D}\right)$, using the marked location for $x^{*}$

gradient, Hessian at $x^{*}: \nabla^{*} \mathbf{H}^{*}$

43 We discussed a simpler, one-dimensional ana$\log$ of this task in $\S 2.8$, p. 39 .

gradient constraint: $\nabla^{*}=0$

first Hessian constraint: diagonal entries are negative and thus its differential entropy with (A.17):

$$
H\left[y \mid x, x^{*} \mathcal{D}\right] \approx \frac{1}{2} \log \left(2 \pi e s_{*}^{2}\right) .
$$

After simplification, the resulting approximation to (8.36) becomes:

$$
\alpha_{x^{*}}(x ; \mathcal{D}) \approx \alpha_{\mathrm{PES}}(x ; \mathcal{D})=\log s-\frac{1}{n} \sum_{i=1}^{n} \log s_{*_{i}} .
$$

\section*{Approximation via Gaussian expectation propagation}

To realize a complete algorithm, we need some way of finding a suitable Gaussian approximation to $p\left(\phi \mid x, x^{*} \mathcal{D}\right)$, and HERNÁNDEZ-LOBATO et al. describe one effective approach. The high-level idea is to begin with the objective function posterior $p(f \mid \mathcal{D})$, impose a series of constraints implied by knowledge of $x$, then approximate the desired posterior via approximate inference. We will describe the procedure for an arbitrary putative optimum $x^{*}$, illustrating each step of the approximation for the example scenario and assumed optimum location shown in Figure 8.13. For the moment, we will proceed with complete disregard for computational efficiency, and return to the question of implementation shortly.

\section*{Ensuring $x^{*}$ is a local optimum}

We first condition our belief on $x^{*}$ being a local optimum by insisting that the point satisfies the second partial derivative test. Let $\nabla^{*}$ and $\mathbf{H}^{*}$ respectively represent the gradient and Hessian of the objective function at $x$. Local optimality implies that the gradient is zero and the Hessian is negative definite: ${ }^{43}$

$$
\begin{aligned}
\nabla^{*} & =\mathbf{0} ; \\
\mathbf{H}^{*} & <0 .
\end{aligned}
$$

Enforcing the gradient constraint (8.42) is straightforward, as we can directly condition on a gradient observation. We show the result for our example scenario in the top panel of Figure 8.14.

The Hessian constraint (8.43) however is nonlinear and we must resort to approximate inference. HERNÁNDEZ-LOBATO et al. approximate 
![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-27.jpg?height=550&width=1634&top_left_y=468&top_left_x=266)

Figure 8.14: Top: the posterior for our example after conditioning on the derivative being zero at $x^{*}(8.42)$. Bottom: the approximate posterior after conditioning on the second derivative being negative at $x^{*}(8.44)$.

this condition by breaking it into two complementary components. First, we compel every diagonal entry of the Hessian to be negative. Letting $\mathbf{h}^{*}=\operatorname{diag} \mathbf{H}^{*}$; we assume:

$$
\forall i: h_{i}^{*}<0
$$

We then fix the off-diagonal entries of the Hessian to values of our choosing. One simple option is to set all off-diagonal entries to zero:

$$
\text { upper } \mathbf{H}^{*}=\mathbf{0} \text {, }
$$

which combined with the diagonal constraint guarantees negative definiteness. ${ }^{44}$ The combination of these conditions $\left(8.44^{-8.45)}\right.$ is stricter than mere negative definiteness, as we eliminate all degrees of freedom for the off-diagonal entries. However, an advantage of this approach is that we can enforce the off-diagonal constraint via exact conditioning.

To proceed we dispense with the constraints we can condition on exactly. Let $\mathcal{D}^{\prime}$ represent our dataset augmented with the gradient (8.42) and off-diagonal Hessian (8.45) observations. The joint distribution of the latent objective value $\phi$, the purportedly optimal value $\phi^{*}=f\left(x^{*}\right)$, and the diagonal of the Hessian $\mathbf{h}^{*}$ given this additional information is multivariate normal:

$$
p\left(\phi, \phi^{*} \mathbf{h}^{*} \mid x^{*}, \mathcal{D}^{\prime}\right)=\mathcal{N}\left(\phi, \phi, \mathbf{h}^{*} ; \boldsymbol{\mu}^{*} \Sigma^{*}\right) .
$$

We will now subject this initial belief to a series of factors corresponding to desired nonlinear constraints. These factors will be compatible with Gaussian expectation propagation, which we will use to finally derive the desired Gaussian approximation to the posterior (8.38). To begin, the Hessian diagonal constraint (8.44) contributes one factor for each entry:

$$
\prod_{i}\left[h_{i}^{*}<0\right] .
$$

diagonal of Hessian at $x^{*} \mathbf{h}^{*}$

second Hessian constraint: off-diagonal entries are fixed

44 HERNÁNDEZ-LOBATO et al. point out this may not be faithful to the model and suggest the alternative of matching the off-diagonal entries of the Hessian of the objective function sample that generated $x$. However, this does not guarantee negative definiteness without tweaking (8.44).

Gaussian EP: § B.2, p. 302

truncating a variable with EP: § B.2, p. 305 
![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-28.jpg?height=588&width=1634&top_left_y=468&top_left_x=156)

Figure 8.15: Top: the approximate posterior after conditioning on $\phi^{*}$ exceeding the function values at previously measured locations (8.49). Bottom: the approximate posterior after conditioning on $\phi^{*}$ dominating elsewhere (8.50).

45 For this and the following demonstrations, we show the expectation propagation approximation to the entire objective function posterior; this is not required to approximate the marginal predictive distribution and is only for illustration. truncating with respect to an unknown threshold with EP: § B.2, p. 306

46 C. E. CLARK (1961). The Greatest of a Finite Set of Random Variables. Operations Research 9(2): 145-162.
Our approximate posterior after incorporating (8.47) and performing expectation propagation is shown in the bottom panel of Figure $8.14 \cdot{ }^{45}$

\section*{Ensuring $x^{*}$ is a global optimum}

Our belief now reflects our desire that $x^{*}$ be a local maximum; however, we wish for $x^{*}$ to be the global maximum. Global optimality is not easy to enforce, as it entails infinitely many constraints bounding the objective at every point in the domain. HERNÁNDEZ-LOBATO et al. instead approximate this condition with optimality at the most relevant locations: the already-observed points $\mathrm{x}$ and the proposed point $x$.

To enforce that $\phi^{*}$ exceed the objective function values at the observed points, we could theoretically add $\phi=f(\mathbf{x})$ to our prior (8.46), then add one factor for each observation: $\prod_{j}\left[\phi_{j}<\phi^{*}\right]$. However, this approach requires an increasing number of factors as we gather more data, rendering expectation propagation (and thus the acquisition function) increasingly expensive. Further, factors corresponding to obviously suboptimal observations are uninformative and simply represent extra work for no benefit.

Instead, we enforce this constraint through a single factor truncating with respect to the maximal value of $\phi$ : $\left[\phi^{*}<\max \phi\right]$. In general, this threshold will be a random variable unless our observations are noiseless. Fortunately, expectation propagation enables tractable approximate truncation at an unknown, Gaussian-distributed threshold. Define

$$
\mu_{\max }=\mathbb{E}[\max \boldsymbol{\phi} \mid \mathcal{D}] ; \quad \sigma_{\max }^{2}=\operatorname{var}[\max \boldsymbol{\phi} \mid \mathcal{D}] ;
$$

these moments can be approximated via either sampling or an assumed density filtering approach described by CLARK. ${ }^{46}$ Taking a momentmatched Gaussian approximation to max $\phi$ and integrating yields the 

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-29.jpg?height=642&width=1650&top_left_y=450&top_left_x=266)

Figure 8.16: The predictive entropy search approximation (8.41) to the mutual information with $x^{*}$ acquisition function (8.36) using the 100 Thompson samples from Figure 8.11.

factor (B.13): ${ }^{47}$

$$
\Phi\left(\frac{\phi^{*}-\mu_{\max }}{\sigma_{\max }}\right) .
$$

We show the approximate posterior for our running example after incorporating this factor in the top panel of Figure 8.15 . The probability mass at $x^{*}$ has shifted up dramatically.

Finally, to constrain $\phi^{*}$ to dominate the objective function value at a point of interest $x$, we add one additional factor:

$$
\left[\phi<\phi^{*}\right] \text {. }
$$

We obtain the final approximation to $p\left(\phi \mid x, x^{*} \mathcal{D}\right)$ by combining the prior in (8.46) with the factors $\left(8.47,8.49^{-8.50}\right)$ :

$$
\mathcal{N}\left(\phi, \phi^{*} \mathbf{h}^{*} ; \boldsymbol{\mu}^{*} \Sigma^{*}\right)\left[\phi<\phi^{*}\right] \Phi\left(\frac{\phi^{*}-\mu_{\max }}{\sigma_{\max }}\right) \prod_{i}\left[h_{i}^{*}<0\right],
$$

approximating with Gaussian expectation propagation, and deriving the marginal belief about $\phi$. The resulting final approximate posterior for our example scenario is shown in the bottom panel of Figure 8.15. Our predictive uncertainty has been moderated in response to the final constraint (8.50).

With the ability to approximate the latent predictive posterior, we can now compute the predictive entropy search acquisition function (8.41) via Thompson sampling and following the above procedure for each sample. Figure 8.16 shows the approximation computed with 1000 Thompson samples. The approximation is excellent and induces a nearoptimal decision. Any deviation from the truth mostly reflects bias in the Thompson sample distribution.
47 With high noise, this approximation could be improved slightly by computing moments of $\max \phi$ given $\mathcal{D}^{\prime}$, at additional expense.

ensuring $\phi^{*}>\phi$

enforcing order with EP: § B.2, p. 306 48 In the interest of numerical stability, the inverse of $\mathbf{V}_{*}$ should not be stored directly; for relevant practical advice see:

C. E. RASMUSSEN and C. K. I. WILLIAMS (2006). Gaussian Processes for Machine Learning. MIT Press.

J. P. cunningham et al. (2011). Gaussian Probabilities and Expectation Propagation. arXiv: 1111.6832 [stat.ML].
49 This can be derived by marginalizing $\mathrm{z}^{*}$ according to its approximate posterior from step 3 above:

$$
\int p\left(\phi, \phi^{*} \mid \mathrm{z}^{*}\right) p\left(\mathrm{z}^{*}\right) \mathrm{dz} .
$$

\section*{Efficient implementation}

A practical realization of predictive entropy search can benefit from careful precomputation and reuse of partial results. We will outline an efficient implementation strategy, beginning with three steps of one-time initial work.

1. Estimate the moments of $\max \phi(8.48)$.

2. Generate a set of Thompson samples $\left\{x_{i}^{*}\right\}$.

3. For each sample $x^{*}$, derive the joint distribution of the function value, gradient, and Hessian at $x^{*}$. Let $\mathbf{z}^{*}=\left(\phi^{*}, \mathbf{h}\right.$, upper $\left.\mathbf{H}^{*}, \nabla^{*}\right)$ represent a vector comprising these random variables, with those that will be subjected to expectation propagation first. We compute:

$$
p\left(\mathbf{z}^{*} \mid x^{*} \mathcal{D}\right)=\mathcal{N}\left(\mathbf{z}^{*} ; \boldsymbol{\mu}^{*} \Sigma^{*}\right)
$$

Find the marginal belief over $\left(\phi_{,}^{*} \mathbf{h}^{*}\right)$ and use Gaussian expectation propagation to approximate the posterior after incorporating the factors (8.47, 8.49). Let vectors $\tilde{\boldsymbol{\mu}}$ and $\tilde{\boldsymbol{\sigma}}^{2}$ denote the site parameters at termination. Finally, precompute ${ }^{48}$

$$
\mathbf{V}_{*}^{-1}=\left[\Sigma^{*}+\left[\begin{array}{cc}
\tilde{\Sigma} & 0 \\
0 & 0
\end{array}\right]\right]^{-1} ; \quad \boldsymbol{\alpha}^{*}=\mathrm{V}_{*}^{-1}\left[\left[\begin{array}{c}
\tilde{\mu} \\
0
\end{array}\right]-\boldsymbol{\mu}^{*}\right],
$$

where $\tilde{\Sigma}=\operatorname{diag} \tilde{\sigma}^{2}$. These quantities do not depend on $x$ and will be repeatedly reused during prediction.

After completing the preparations above, suppose a proposed observation location $x$ is given. For each sample $x^{*}$, we compute the joint distribution of $\phi$ and $\phi^{*}$ :

$$
p\left(\phi, \phi^{*} \mid x, x^{*} \mathcal{D}\right)=\mathcal{N}\left(\phi, \phi^{*} ; \boldsymbol{\mu}, \Sigma\right),
$$

and derive the approximate posterior given the exact gradient (8.42) and off-diagonal Hessian (8.45) observations and the factors $(8.47,8.49)$. Defining

$$
\mathbf{K}=\left[\begin{array}{l}
\mathbf{k}^{\top} \\
\mathbf{k}_{*}^{\top}
\end{array}\right]=\operatorname{cov}\left(\left[\phi, \phi^{*}\right]^{\top}, \mathbf{z}^{*}\right),
$$

the desired distribution is $\mathcal{N}\left(\phi, \phi^{*} ; \mathbf{m}, \mathbf{S}\right)$, where: ${ }^{49}$

$$
\mathbf{m}=\left[\begin{array}{l}
m \\
m^{*}
\end{array}\right]=\boldsymbol{\mu}+\mathbf{K} \boldsymbol{\alpha}^{*} \quad \mathbf{S}=\left[\begin{array}{cc}
\varsigma^{2} & \rho \\
\rho & \varsigma_{*}^{2}
\end{array}\right]=\Sigma-\mathbf{K V}_{*}^{-1} \mathbf{K}^{\top}
$$

We now apply the prediction constraint (8.50) with one final step of expectation propagation. Define (в.7, в.11-в.12):

$$
\begin{gathered}
\bar{\mu}=m-m^{*} ; \quad \bar{\sigma}^{2}=\varsigma^{2}-2 \rho+\varsigma_{*}^{2} ; \\
z=-\frac{\bar{\mu}}{\bar{\sigma}} ; \quad \alpha=-\frac{\phi(z)}{\Phi(z) \bar{\sigma}} ; \quad \gamma=-\frac{\bar{\sigma}}{\alpha}\left(\frac{\phi(z)}{\Phi(z)}+z\right)^{-1} .
\end{gathered}
$$

The final approximation to the predictive variance of $\phi$ given $x^{*}$ is

$$
\sigma_{*}^{2}=\varsigma^{2}-\left(\varsigma^{2}-\rho\right)^{2} / \gamma
$$

from which we can compute the contribution to the acquisition function from this sample with (8.39-8.40).

Although it may seem unimaginable at this point, in Euclidean domains we may compute the gradient of the predictive entropy search acquisition function; see the accompanying appendix for details.

\subsection*{MUTUAL INFORMATION With $f^{*}$}

Finally, we consider the computation of the mutual information between the observed value $y$ and the value of the global maximum $f^{*}(8.53)$. Several authors have considered this acquisition function in its predictive form (8.35), which is the most convenient choice for Gaussian process models:

$$
\alpha_{f^{*}}(x ; \mathcal{D})=H[y \mid x, \mathcal{D}]-\mathbb{E}\left[H\left[y \mid x, f^{*} \mathcal{D}\right] \mid x, \mathcal{D}\right]
$$

Unfortunately, this expression cannot be computed exactly due to the complexity of the distribution $p\left(f^{*} \mid \mathcal{D}\right)$; see Figure 8.17 for an example. However, several effective approximations have been proposed, including max-value entropy search (MES) ${ }^{50}$ and output-space predictive entropy search (OPES). ${ }^{51}$

The issues we face in estimating (8.53), and the strategies we use to overcome them, largely mirror those in predictive entropy search. To begin, the first term is the differential entropy of a Gaussian and may be computed exactly:

$$
H[y \mid x, \mathcal{D}]=\frac{1}{2} \log \left(2 \pi e s^{2}\right)
$$

The second term, however, presents some challenges, and the available approximations to (8.53) diverge in their estimation approach. We will discuss the MES and OPES approximations in parallel, as they share the same basic strategy and only differ in some details along the way.

\section*{Approximating expectation with respect to $f^{*}$}

The first complication in evaluating the second term of (8.53) is that we must compute an expectation with respect to $f *$ Although Thompson sampling and simple Monte Carlo approximation is one way forward, we can exploit the fact that $f^{*}$ is one dimensional to pursue more sophisticated approximations. One convenient and rapidly converging strategy is to design $n$ samples $\left\{f_{i}^{*}\right\}_{i=1}^{n}$ to be equally spaced quantiles of $f^{*}$, then use the familiar estimator

$$
\mathbb{E}\left[H\left[y \mid x, f^{*} \mathcal{D}\right] \mid x, \mathcal{D}\right] \approx \frac{1}{n} \sum_{i=1}^{n} H\left[y \mid x, f_{i}^{*}, \mathcal{D}\right]
$$

gradient of predictive entropy search acquisition function: $\S \mathrm{C} .3, \mathrm{p} .311$

mutual information with $f^{*}$ : § 7.6, p. 140

50 z. WANG and S. JEGELKA (2017). Max-value Entropy Search for Efficient Bayesian Optimization. ICML 2017.

51 M. W. HOFFMAN and z. GHAHRAMANI (2015). Output-Space Predictive Entropy Search for Flexible Global Optimization. Bayesian Optimization Workshop, NeurIPS 2015. 52 R. E. CAfLISCH (1998). Monte Carlo and QuasiMonte Carlo Methods. Acta Numerica 7:1-49.

53 The chosen samples are the first $n$ points from a base- $n$ van de Corput sequence:

J. G. VAN DE CORPUT (1935). Verteilungsfunktionen: Erste Mitteilung. Proceedings of the Koninklijke Nederlandse Akademie van Wetenschappen 38:813-821

mapped to the desired distribution via the inverse CDF. These points have the minimum possible discrepancy for a set of size $n$.

54 C. E. CLARK (1961). The Greatest of a Finite Set of Random Variables. Operations Research 9(2): $145^{-162 .}$

55 A. M. Ross (2010). Computing Bounds on the Expected Maximum of Correlated Normal Variables. Methodology and Computing in Applied Probability 12(1):111-138.

56 R. P. BRENT (1973). Algorithms for Minimization without Derivatives. Prentice-Hall. [chapter 4]

57 D. SLEPIAN (1962). The One-Sided Barrier Problem for Gaussian Noise. The Bell System Technical fournal 41(2):463-501.

58 This result requires the posterior covariance function to be positive everywhere, which is not guaranteed even if true for the prior covariance function. However, it is "usually true" for typical models in high-dimensional spaces.
This represents a quasi-Monte Carlo ${ }^{52}$ approximation that converges considerably faster than simple Monte Carlo integration. ${ }^{53}$ Further, this scheme only requires the ability to estimate quantiles of $f^{*}$.

\section*{Approximating quantiles of $f^{*}$}

WANG and JEGELKA proposed estimating the quantiles of $f^{*}$ via a convenient analytic approximation. To build this approximation, they proposed selecting a set of representer points $\xi$ and approximating the distribution of $f^{*}$ with that of the maximum function value restricted to these locations. Let $\boldsymbol{\phi}=f(\xi)$ and define the random variable $\phi^{*}=\max \boldsymbol{\phi}$. We estimate:

$$
p\left(f^{*} \mid \mathcal{D}\right) \approx p\left(\phi^{*} \mid \xi, \mathcal{D}\right) .
$$

Unfortunately, the distribution of the maximum of dependent Gaussian random variables is intractable in general, even if the dimension is finite,$^{54,55}$ so we must resort to further approximation.

We could proceed in several ways, but WANG and JEGELKA propose using an exhaustive, dense set of representer points (for example, covering the domain with a low-discrepancy sequence) and then making the simplifying assumption that the associated function values are independent. If the marginal belief at each representer point is

$$
p\left(\phi_{i} \mid \xi_{i}, \mathcal{D}\right)=\mathcal{N}\left(\phi_{i} ; \mu_{i}, \sigma_{i}^{2}\right),
$$

then we may approximate the cumulative distribution function of $\phi^{*}$ with a product of normal CDFs:

$$
\operatorname{Pr}\left(\phi^{*}<z \mid \xi, \mathcal{D}\right) \approx \prod_{i} \Phi\left(\frac{z-\mu_{i}}{\sigma_{i}}\right) .
$$

We can now estimate the quantiles of $f^{*}$ via numerical inversion of (8.56); Brent's ${ }^{56}$ method applied to the log CDF would offer rapid convergence. The resulting approximation for our example scenario is shown in Figure 8.17 using 100 equally spaced representer points covering the domain. In general, the independence assumption tends to overestimate the maximal value, a finding WANG and JEGELKA explained heuristically by appealing to SLEPIAN's lemma. ${ }^{57,58}$

A diametrically opposing alternative to using dense representer points with a crude approximation (8.56) would be using a few, carefully chosen representer points with a better approximation to the distribution of $\phi^{*}$. For example, we could generate a set of Thompson samples, $\xi_{i} \sim p\left(x^{*} \mid \mathcal{D}\right)$, then approximate the quantiles of $\phi^{*}$ by repeatedly sampling their corresponding function values $\phi$. Figure 8.17 shows such an approximation for our example scenario using 100 Thompson samples. The agreement with the true distribution is excellent, but the computational cost was significant compared to the independent approximation. However, in high-dimensional spaces where dense coverage of the domain is not possible, such a direct approach may become appealing. 
![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-33.jpg?height=594&width=1626&top_left_y=457&top_left_x=266)

Figure 8.17: Left: the true distribution $p\left(f^{*} \mid \mathcal{D}\right)$ for our running scenario, estimated using exhaustive sampling. Middle: WANG and JEGELKA's approximation to $p\left(f^{*} \mid \mathcal{D}\right)(8.56)$ using a grid of 100 equally spaced representer points. Right: an approximation to $p\left(f^{*} \mid \mathcal{D}\right)$ from sampling the function values at the 100 Thompson samples from Figure 8.11.

\section*{Predictive entropy with exact observations}

Regardless of the exact estimator we use to approximate the second term of the acquisition function, we will need to compute the entropy of the predictive distribution for $y$ given the optimal value $f^{*}$ :

$$
p\left(y \mid x, f^{*} \mathcal{D}\right)=\int p(y \mid x, \phi) p\left(\phi \mid x, f^{*} \mathcal{D}\right) \mathrm{d} \phi .
$$

The latent predictive distribution of $\phi$ given $f^{*}$ can be reasonably approximated as a normal distribution with upper tail truncated at $f^{*}:^{59}$

$$
p\left(\phi \mid x, f^{*} \mathcal{D}\right) \approx \mathcal{T N}\left(\phi ; \mu, \sigma^{2}\left(-\infty, f^{*}\right)\right) .
$$

Remarkably, this approximation is sufficient to provide a closed-form upper bound on the differential entropy of the true marginal: ${ }^{60}$

$$
H\left[\phi \mid x, f^{*} \mathcal{D}\right] \leq \frac{1}{2}\left[\log \left(2 \pi e \sigma^{2} \Phi(z)^{2}\right)-z \frac{\phi(z)}{\Phi(z)}\right] ; \quad z=\frac{f^{*}-\mu}{\sigma} .
$$

In the absence of observation noise, this result is sufficient to realize a complete algorithm by combining (8.59) with (8.53-8.55). After simplification, this yields the final MEs approximation, which ignores the effect of observation noise in the predictive distribution:

$$
\alpha_{\mathrm{MES}}(x ; \mathcal{D}) \approx \frac{1}{2 n} \sum_{i=1}^{n}\left[z_{i} \frac{\phi\left(z_{i}\right)}{\Phi\left(z_{i}\right)}-\log \Phi\left(z_{i}\right)^{2}\right] ; \quad z_{i}=\frac{f_{i}^{*}-\mu}{\sigma} .
$$

Figure 8.18 illustrates this approximation for our running example using the independent approximation to $p\left(f^{*} \mid \mathcal{D}\right)(8.56)$. The approximation is faithful and induces a near-optimal decision.

59 It is possible, but expensive, to compute better approximations by "correcting" this truncated normal to account for correlations between $\phi$ and the rest of $f$, which must satisfy the constraint globally:

J. CARTinhour (1989). One-Dimensional Marginal Density Functions of a Truncated Multivariate Normal Density Function. Communications in Statistics - Theory and Methods 19(1):197-203.

6o This is a consequence of the fact that "information never hurts." We are conditioning $\phi$ to be less than $f^{*}$, but every other function value is similarly bounded - a fact we ignore. 

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-34.jpg?height=637&width=1630&top_left_y=450&top_left_x=156)

Figure 8.18: An approximation to the mutual information between the observed value $y$ and the value of the global optimum $\alpha_{f^{*}}=I\left(y ; f^{*} \mid x, \mathcal{D}\right)$ for our running example using the independent approximation to $p\left(f^{*} \mid \mathcal{D}\right)$ (8.56) and numerical integration.

direct computation of predictive entropy

61 s. TURBAn (2010). Convolution of a Truncated Normal and a Centered Normal Variable. Technical report. Columbia University.

62 Thus equality is only approximate due to the approximation in (8.58); the convolution of the truncated normal and normal distribution is exact.

analytic approximation to predictive entropy

63 However, the closed form for the predictive distribution above was not discussed by either and perhaps deserves further consideration.

\section*{Approximating predictive entropy with noisy observations}

In the case of additive Gaussian noise, however, the predictive distribution of $y(8.57)$ is better approximated as the convolution of a centered normal distribution and a truncated normal distribution; see Figure 8.19 for an illustration of a particularly noisy scenario. TURBAN provides a closed form for the resulting probability density function: ${ }^{61,62}$

$$
p\left(y \mid x, f^{*} \mathcal{D}\right) \approx \gamma \Phi\left(\frac{\alpha(y)-y+f^{*}}{\beta}\right) \exp \left(-\frac{(y-\mu)^{2}}{2 s^{2}}\right),
$$

where

$$
\alpha(y)=\frac{\sigma_{n}^{2}(y-\mu)}{s^{2}} ; \quad \beta=\frac{\sigma_{n} \sigma}{s} ; \quad \gamma=\frac{1}{\sqrt{2 \pi} s \Phi(z)},
$$

and $z$ is defined in (8.59). Unfortunately this distribution does not admit a closed-form expression for its entropy, although we can approximate the entropy effectively via numerical quadrature.

Alternatively, we can appeal to analytic approximation to estimate the predictive entropy, and both WANG and JEGELKA and HOFFMAN and GHAHRAMANI take this approach. ${ }^{63}$ The MES approximation simply ignores the observation noise in the acquisition function (8.60) and instead computes the mutual information between the latent function value $\phi$ and $f *$ This approach overestimates the true mutual information by assuming exact observations, but serves as a reasonable approximation when the signal-to-noise ratio is high. This approximation is illustrated in Figure 8.20 for a scenario with relatively high noise; although the approximation is not perfect, the chosen point is close to optimal. 
![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-35.jpg?height=352&width=1648&top_left_y=452&top_left_x=252)

Figure 8.19: Left: an example of the latent predictive distribution $p\left(\phi \mid x, f^{*} \mathcal{D}\right)$, which takes the form of a truncated normal distribution (8.58), and the resulting predictive distribution $p(y \mid x, f, \mathcal{D})$, which is a convolution with a centered normal distribution accounting for Gaussian observation noise. Right: a Gaussian expectation propagation approximation to the predictive distribution (8.62).

HOFFMAN and GHAHRAMANI instead approximate the latent predictive truncating a variable with EP: § B.2, p. 305 distribution (8.58) using Gaussian expectation propagation:

$$
p\left(\phi \mid x, f^{*} \mathcal{D}\right) \approx \mathcal{N}\left(\phi ; \mu_{*}, \sigma_{*}^{2}\right),
$$

where

$$
\sigma_{*}^{2}=\sigma^{2}\left[1-z \frac{\phi(z)}{\Phi(z)}-\frac{\phi(z)^{2}}{\Phi(z)^{2}}\right]
$$

is the variance of the truncated normal latent predictive distribution (8.58) and $z$ is defined in (8.59). We may now approximate the differential entropy of $y$ with

$$
\begin{aligned}
\operatorname{var}\left[y \mid x, f^{*} \mathcal{D}\right] & \approx \sigma_{*}^{2}+\sigma_{n}^{2}=s_{*}^{2} \\
H\left[y \mid x, f^{*} \mathcal{D}\right] & \approx \frac{1}{2} \log \left(2 \pi e s_{*}^{2}\right) .
\end{aligned}
$$

This is similar to the strategy used in predictive entropy search, although computing the approximate latent posterior is considerably easier in this case, only requiring the single factor $\left[\phi<f^{*}\right]$. Figure 8.19 shows the resulting Gaussian approximation to the predictive distribution for an example point; the approximation is excellent. Estimating the expectation over $f^{*}$ with (8.55) and simplifying, the final opEs approximation to $(8.53)$ is

$$
\alpha_{\mathrm{OPES}}(x ; \mathcal{D})=\log s-\frac{1}{n} \sum_{i=1}^{n} \log s_{*_{i}},
$$

where $s_{*_{i}}$ is the approximate predictive standard deviation corresponding to the sample $f_{i}^{*}$. This takes the same form as the predictive entropy search approximation to the mutual information with $x^{*}(8.41)$, although the approximations to the predictive distribution are of course different. The OPES approximation (8.63) is shown for a high-noise scenario in Figure 8.20; the approximation is almost perfect and in this case yields the optimal decision.

Both the MES and OPES approximations to the mutual information can be differentiated with respect to the proposed observation location. 

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-36.jpg?height=215&width=1630&top_left_y=515&top_left_x=156)

- mutual information with $f^{*} \alpha_{f^{*}} \quad \boldsymbol{\nabla}$ next observation location

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-36.jpg?height=137&width=1628&top_left_y=820&top_left_x=160)

- MES approximation (8.6o)

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-36.jpg?height=143&width=1628&top_left_y=1025&top_left_x=160)

OPES approximation (8.62)

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-36.jpg?height=143&width=1625&top_left_y=1236&top_left_x=160)

Figure 8.20: The MES and OPES approximations to the mutual information with $f^{*}$ for an example high-noise scenario with unit signal-to-noise ratio. Both use the independent representer point approximation to $p\left(f^{*} \mid \mathcal{D}\right)(8.56)$.

gradient of OPES acquisition function: § c.3, p. 311
For the former, we simply differentiate (8.6o):

$$
\frac{\partial \alpha_{\mathrm{MES}}}{\partial x}=\frac{1}{2 n \sigma} \sum_{i=1}^{n} \frac{\phi\left(z_{i}\right)}{\Phi\left(z_{i}\right)}\left[\frac{\partial \mu}{\partial x}+z_{i} \frac{\partial \sigma}{\partial x}\right]\left[1+z_{i} \frac{\phi\left(z_{i}\right)}{\Phi\left(z_{i}\right)}+z_{i}^{2}\right]
$$

note the $\left\{f_{i}^{*}\right\}$ samples will in general not depend on $x$, so we do not need to worry about any dependence of their distribution on the observation location. The details for the opes approximation are provided in the appendix.

\section*{AVERAGING OVER A SPACE OF GAUSSIAN PROCESSES}

Throughout this chapter we have assumed our belief regarding the objective function is given by a single Gaussian process. However, especially when our dataset is relatively small, we may seek robustness to model misspecification by averaging over multiple plausible models. Here we will provide some guidance for policy computation when performing Bayesian model averaging over a space of Gaussian processes. Although this may seem straightforward, there is some nuance involved.

Below we will consider a space of Gaussian processes indexed by a vector of hyperparameters $\boldsymbol{\theta}$, which determines both the moments of the objective function prior and any relevant parameters of the observation noise process. Bayesian model averaging over this space yields marginal posterior and predictive distributions:

$$
\begin{aligned}
p(f \mid \mathcal{D}) & =\int p(f \mid \mathcal{D}, \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathcal{D}) \mathrm{d} \boldsymbol{\theta} ; \\
p(y \mid x, \mathcal{D}) & =\int p(y \mid x, \mathcal{D}, \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathcal{D}) \mathrm{d} \boldsymbol{\theta},
\end{aligned}
$$

which are integrated against the model posterior $p(\theta \mid \mathcal{D})(4.7)$. Both of these distributions are in general intractable, but we developed several viable approximations in Chapter 3, all of which approximate the objective function posterior (8.64) with a mixture of Gaussian processes and the posterior predictive distribution (8.65) with a mixture of Gaussians.

\section*{Noiseless expected improvement and probability of improvement}

When observations are exact, the marginal gain in utility underlying both expected improvement and probability of improvement depends only on the objective function value $\phi$ and the value of the incumbent $\phi^{*}:$

$$
\Delta_{\mathrm{EI}}(x, \phi)=\max (\phi-\phi, 0) ; \quad \Delta_{\mathrm{PI}}(x, \phi)=\left[\phi>\phi^{*}\right] .
$$

As a result, the formulas we derived for these acquisition functions given a GP belief on the objective $(8.9,8.21)$ depend only on the moments of the (Gaussian) predictive distribution $p(\phi \mid x, \mathcal{D}) .{ }^{64} \mathrm{With}$ a Gaussian mixture approximation to the predictive distribution, the expected marginal gain,

$$
\alpha(x ; \mathcal{D})=\int \Delta(x, \phi) p(\phi \mid x, \mathcal{D}) \mathrm{d} \phi,
$$

is simply a weighted combination of these results by linearity of expectation.

\section*{Model-dependent utility functions}

For the remaining decision-theoretic acquisition functions, noisy expected improvement and probability of improvement, knowledge gradient, and mutual information with any relevant random variable $\omega$, the situation is somewhat more complicated, because the underlying utility functions depend on the model of the objective function. The first three depend on the posterior mean function, and the last depends on the posterior belief $p(\omega \mid \mathcal{D})$, both of which are induced by our belief regarding the objective function. To make this dependence explicit, for a model $\boldsymbol{\theta}$ in our space of interest, let us respectively notate the utility, marginal gain in utility, and expected marginal gain in utility with:

$$
u(\mathcal{D} ; \boldsymbol{\theta}) ; \quad \Delta(x, y ; \boldsymbol{\theta}) ; \quad \alpha(x ; \mathcal{D}, \boldsymbol{\theta}) .
$$

There are two natural ways we might address this dependence when averaging over models. One is to seek to maximize the expected utility model posterior, $p(\theta \mid \mathcal{D})$

64 In fact, a Gaussian process belief is not required at all, only Gaussian predictive distributions. This will be important in the next section.

dependence on objective function model average model-conditional utility, $\mathbb{E} u$

marginal gain in $\mathbb{E} u, \mathbb{E} \Delta$ expected marginal gain in $\mathbb{E} u, \mathbb{E} \alpha$

marginal posterior mean

marginal belief about $\omega$

utility of marginal model, $u \mathbb{E}$ (expected) marginal gain in utility of marginal model: $\Delta \mathbb{E}, \alpha \mathbb{E}$
65 These models may be interpreted as degenerate Gaussian processes with covariance $K \equiv 0$. of the data, averaged over the choice of model:

$$
\mathbb{E} u(\mathcal{D})=\int u(\mathcal{D} ; \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathcal{D}) \mathrm{d} \boldsymbol{\theta} .
$$

Writing the marginal gain in expected utility as $\mathbb{E} \Delta$, we may derive an acquisition function via one-step lookahead:

$$
\begin{aligned}
\mathbb{E} \alpha(x ; \mathcal{D}) & =\int \mathbb{E} \Delta(x, y) p(y \mid x, \mathcal{D}) \mathrm{d} y \\
& =\int\left[\int \Delta(x, y ; \boldsymbol{\theta}) p(y \mid x, \mathcal{D}, \boldsymbol{\theta}) \mathrm{d} y\right] p(\boldsymbol{\theta} \mid \mathcal{D}) \mathrm{d} \boldsymbol{\theta} \\
& =\int \alpha(x ; \mathcal{D}, \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathcal{D}) \mathrm{d} \boldsymbol{\theta} .
\end{aligned}
$$

As hinted by its notation, this is simply the expectation of the conditional acquisition functions, which we can approximate via standard methods.

Although this approach is certainly convenient, it may overestimate optimization progress as utility is only measured under the assumption of a perfectly identified model - the utility function is "blind" to model uncertainty. An arguably more appealing alternative is to evaluate utility with respect to the marginal objective function model (8.64) from the start, defining simple and global reward with respect to the marginal posterior mean:

$$
\int \mu_{\mathcal{D}}(x ; \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathcal{D}) \mathrm{d} \boldsymbol{\theta},
$$

and information gain about $\omega$ with respect to its marginal belief:

$$
\int p(\omega \mid \mathcal{D}, \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathcal{D}) \mathrm{d} \boldsymbol{\theta}
$$

Let us notate a utility function defined in this manner with $u \mathbb{E}(\mathcal{D})$, contrasting with its post hoc averaging equivalent, $\mathbb{E} u(\mathcal{D})(8.66)$. Similarly, let us notate its marginal gain with $\Delta \mathbb{E}$ and its expected marginal gain with:

$$
\alpha \mathbb{E}(x ; \mathcal{D})=\int \Delta \mathbb{E}(x, y) p(y \mid x, \mathcal{D}) \mathrm{d} y .
$$

\section*{Example and discussion}

We may shed some light on the differences between these approaches with a barebones example. Let us work with the knowledge gradient, and consider a pair of simple models for an objective function on the interval $\mathcal{X}=[-1,1]$ : either $f=x$ or $f=-x$, with equal probability. ${ }^{65}$

The model-conditional global reward in either case is 1 , and thus the expected utility (8.66) does not depend on the data: $\mathbb{E} u(\mathcal{D}) \equiv 1$. What a strange set of affairs - although we know the maximal value of the objective a priori, we need data to tell us where it is! For this particular model space, an optimal recommendation for either model is maximally suboptimal if that model is incorrect. In contrast, the global reward of the marginal model does depend on the data via the model posterior. The global reward is monotonic in the probability of the model most favored by the data, $\pi$ :

$$
u \mathbb{E}(\mathcal{D})=2 \pi-1 .
$$

A priori, the utility is $u \mathbb{E}(\varnothing)=0$ - the marginal mean function is $\mu \equiv 0$, and no compelling recommendation is possible until we determine the correct model.

As the expected utility $\mathbb{E} u$ (8.66) is independent of the data, it does not lead to an especially insightful policy. In contrast, the marginal gain in $u \mathbb{E}(8.70)$ can differentiate potential observation locations via their expected impact on the model posterior. We illustrate this acquisition function in the margin given an empty dataset and assuming moderate additive Gaussian noise. As one might hope, it prefers evaluating on the boundary of the domain, where observations are expected to provide more information regarding the model and thus greater improvement in model-marginal utility.

\section*{Computing expected gain in marginal utility}

Unfortunately, the expected gain in utility for the marginal model (8.70) does not simplify as before (8.66), as we must account for the effect of the observed data on the model posterior in the utility function. However, we may sketch a Monte Carlo approximation using samples from the model posterior, $\left\{\boldsymbol{\theta}_{i}\right\} \sim p(\boldsymbol{\theta} \mid \mathcal{D})$.

For utility functions based on the marginal posterior mean (8.68), the resulting approximation to the expected marginal gain (8.70) is a weighted sum of Gaussian expectations of $\Delta \mathbb{E}$, each of which we may approximate via Gauss-Hermite quadrature. To approximate $\Delta \mathbb{E}$ for a putative observation $(x, y)$, a simple sequential Monte Carlo approximation to the updated model posterior would reweight each sample $\boldsymbol{\theta}_{i}$ by $p\left(y \mid x, \mathcal{D}, \boldsymbol{\theta}_{i}\right)$, then approximate the updated marginal posterior mean by a weighted sum.

To approximate the predictive form of mutual information with $x^{*}$ or $f^{*}(8.36,8.53)$ :

$$
H[y \mid x, \mathcal{D}]-\mathbb{E}_{\omega}[H[y \mid x, \omega, \mathcal{D}] \mid x, \mathcal{D}],
$$

we first note that the first term is the entropy of a Gaussian mixture, which we can approximate via quadrature. We may approximate the expectation in the second term via Thompson sampling (see below); for each of these samples, we may approximate the updated predictive distribution as before, reweighting the mixture by $p(\omega \mid \mathcal{D}, \theta){ }^{66}$

\section*{Upper confidence bound and Thompson sampling}

We may compute any desired upper confidence bound of a Gaussian process mixture at $x$ by bisecting the cumulative distribution function of $\phi$, which is a weighted sum of Gaussian CDFs.

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-39.jpg?height=363&width=529&top_left_y=612&top_left_x=1366)

Above: the two possible objectives for our comparison of the $\mathbb{E} u$ and $u \mathbb{E}$. Below: the expected marginal gain in $u \mathbb{E}$, which prefers sampling on the boundary to reveal more information regarding the model. The expected marginal gain in $\mathbb{E} u$ is constant.

Gauss-Hermite quadrature: § 8.5, p. 171

66 This requires one final layer of approximation, which is produced as a byproduct of expectation propagation in the case of $x^{*}$ (в.6), and may be dealt with without much trouble in the case of the univariate random variable $f^{*}$. Thompson sampling for GPs: § 8.7, p. 176

\section*{1}

67 L. BREIMAN (2001). Random Forests. Machine Learning 45(1):5-32.

68 M. FERNÁNDEZ-DELGAdo et al. (2014). Do We Need Hundreds of Classifiers to Solve Real World Classification Problems? Journal of $\mathrm{Ma}$ chine Learning Research 15(90):3133-3181.

69 F. HUtTER et al. (2014). Algorithm Runtime Prediction: Methods \& Evaluation. Artificial Intelligence 206:79-111.

70 F. HUTTER et al. (2011). Sequential Model-Based Optimization for General Algorithm Configuration. LION 5

71 HUTTER et al. then fit a single Gaussian distribution to this mixture via moment matching, although this is not strictly necessary.

72 A similar approach can also be used to estimate arbitrary predictive quantiles:

N. MEINSHAUSEN (2006). Quantile Regression Forests. Journal of Machine Learning Research 7(35):983-999.
Finally, to perform Thompson sampling from the marginal posterior, we first sample from the model posterior, $\theta \sim p(\theta \mid \mathcal{D})$; the conditional posterior $p(f \mid \mathcal{D}, \theta)$ is then a GP, and we may proceed via the previous discussion.

11 ALTERNATIVE MODELS: BAYESIAN NEURAL NETWORKS, ETC.

Although Gaussian processes are without question the most prominent objective function model used in Bayesian optimization, we are of course free to use any other model when prudent. Below we briefly outline some notable alternative model classes that have received some attention in the context of Bayesian optimization and comment on any issues arising in computing common policies with these surrogates.

\section*{Random forests}

Random forest ${ }^{67}$ are a popular model class renown for their excellent off-the-shelf performance ${ }^{68}$ offering good generalization, strong resistance to overfitting, and efficient training and prediction. Of particular relevance for optimization, random forests are adept at handling highdimensional data and categorical and conditional features, and may be a better choice than Gaussian processes for objectives featuring any of these characteristics.

Algorithm configuration is one setting where these capabilities are critical: complex algorithms such as compilers or SAT solvers often have complex configuration schemata with many mutually dependent parameters, and it can be difficult to build nontrivial covariance functions for such inputs. Random forests require no special treatment in this setting and have delivered impressive performance in predicting algorithmic performance measures such as runtime. ${ }^{69}$ They are thus a natural choice for Bayesian optimization of these same measures. ${ }^{70}$

Classical random forests are not particularly adept at quantifying uncertainty in predictions off-the-shelf. Seeking more nuanced uncertainty quantification, HUTTER et al. proposed a modification of the vanilla model wherein leaves store both the mean (as usual) and the standard deviation of the training data terminating there. ${ }^{69} \mathrm{We}$ then estimate the predictive distribution with a mixture of Gaussians with moments corresponding to the predictions of the member trees. ${ }^{71,72}$ Figure 8.21 compares the predictions of a Gaussian process and a random forest model on a toy dataset. Although they differ in their extrapolatory behavior, the models make very similar predictions otherwise.

To realize an optimization policy with a random forest, HUTTER et al. suggested approximating acquisition functions depending only on marginal predictions - such as (noiseless) expected improvement or probability of improvement - by simply plugging this Gaussian approximation into the expressions derived in this chapter (8.9, 8.22). Either can be computed easily from a Gaussian mixture predictive distribution as well due to linearity of expectation. 

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-41.jpg?height=517&width=1031&top_left_y=458&top_left_x=270)

Density ratio estimation

BERGSTRA et al. described a lightweight Bayesian optimization algorithm that operates by maximizing probability of improvement (7.3) via a reduction to density ratio estimation. ${ }^{73,74}$

We begin each iteration of this algorithm by choosing some reference value $y^{*}$ that we wish to exceed with our next observation; we will then select the next observation location to maximize the probability of improvement over this threshold: $:^{75}$

$$
\alpha_{\mathrm{PI}}\left(x ; \mathcal{D}, y^{*}\right)=\operatorname{Pr}\left(y>y^{*} \mid x, \mathcal{D}\right) .
$$

We will discuss the selection of the improvement target $y^{*}$ shortly.

We now consider two conditional probability density functions depending on this threshold:

$$
\begin{aligned}
& g(x)=p\left(x \mid y>y^{*} \mathcal{D}\right) \\
& \ell(x)=p\left(x \mid y \leq y^{*} \mathcal{D}\right)
\end{aligned}
$$

that is, $g$ is the probability density of observation locations exceeding the threshold, and $\ell$ of locations failing to. Note that both densities are proportional to quantities related to the probability of improvement:

$$
\begin{aligned}
& g(x) \propto \operatorname{Pr}\left(y>y^{*} \mid x, \mathcal{D}\right) p(x)=\quad \alpha_{\mathrm{PI}} p(x) ; \\
& \ell(x) \propto \operatorname{Pr}\left(y \leq y^{*} \mid x, \mathcal{D}\right) p(x)=\left(1-\alpha_{\mathrm{PI}}\right) p(x),
\end{aligned}
$$

where $p(x)$ is an arbitrary prior density that we will dispose of shortly.

BERGSTRA et al. now define an acquisition function as the ratio of these densities:

$$
\alpha(x ; \mathcal{D})=\frac{g(x)}{\ell(x)} \propto \frac{\alpha_{\mathrm{PI}}}{1-\alpha_{\mathrm{PI}}},
$$

where the prior density $p(x)$ appearing in (8.73) cancels. ${ }^{76}$ As this ratio is monotonically increasing with the probability of improvement (8.71), maximizing the density ratio yields the same policy.

To proceed, we require some mechanism for estimating this density ratio (8.74). BERGSTRA et al. appealed to density estimation, constructing
Figure 8.21: The predictions of a Gaussian process model (above) and a random forest model comprising 100 regression trees (below) for an example dataset; the credible intervals of the latter are not symmetric as the predictive distribution is estimated with a Gaussian mixture.
73 J. BERGSTRA et al. (2011). Algorithms for HyperParameter Optimization. NeurIPS 2011.

74 Although the source material claims that expected improvement is maximized, this was the result of a minor mathematical error.

75 Although we have spent considerable effort arguing against maximizing "improvement" over a noisy observation - at least in the presence of high noise - we will see that this choice provides considerable computational benefit.
76 There is, however, a tacit assumption that the prior density has support over the whole domain to enable this cancellation. Figure 8.22: BERGSTRA et al.'s Parzen estimation optimization policy. The top panel shows an example dataset along with a threshold $y^{*}$ set to be the 85 th percentile of the observed values. The next two panels illustrate the central kernel density estimates (8.72) for the density of observation locations above and below this threshold. The "wiggliness" of these estimates stems from the kernel bandwidth scheme proposed by BERGSTRA et al. The bottom panel shows the ratio of these densities, which is monotonic in the probability of improvement over $y^{*}$ (8.74).

77 B. W. Silverman (1986). Density Estimation for Statistics and Data Analysis. Chapman \& Hall.

78 The details of BERGSTRA et al.'s scheme are somewhat complicated due to the particular optimization problems they were considering. Their focus was on hyperparameter tuning, where conditional dependence among variables can lead to complex structure in the domain. To address this dependence, BERGSTRA et al. considered domains with tree structure, where each node represents a variable that can be assigned provided the assignments of its parents. They then built separate conditional density estimates for each variable appearing in the tree.

79 L. C. TIAO et al. (2021). BORE: Bayesian Optimization by Density-Ratio Estimation. ICML 2021.

role of improvement threshold in probability of improvement: $\S 7.5$, p. 133
![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-42.jpg?height=658&width=1068&top_left_y=456&top_left_x=720)

$g(x) / \ell(x)$

![](https://cdn.mathpix.com/cropped/2023_09_22_3ee31cf1ba660015b9dag-42.jpg?height=157&width=562&top_left_y=1144&top_left_x=1221)

kernel (Parzen) density estimates ${ }^{77}$ of $g$ and $\ell$ and directly maximizing their ratio (8.74) to realize a policy. ${ }^{78}$ One advantage of this scheme is that the cost of computing the policy only scales linearly with respect to the number of observations. Figure 8.22 illustrates the key components of this algorithm for a toy problem in one dimension.

An alternative to density estimation is to approximate the density ratio in (8.74) directly, an option promoted by TIAO et al. ${ }^{79}$ Here we effectively reduce the problem to binary classification, building a probabilistic classifier to predict the binary label $\left[y>y^{*}\right]$, then interpreting the predictions of this classifier in terms of the probability of improvement. Any probabilistic classifier could be used to this end, so the scheme offers a great deal of latitude in modeling beyond Gaussian processes or related models; in fact we do not need to model the objective function at all!

For either of these approaches to be feasible, we must have enough observations to either accurately estimate the required densities (8.72) or to accurately train a classifier predicting $\left[y>y^{*}\right]$. BERGSTRA et al. addressed this issue by taking $y^{*}$ to be a relatively (but not exceedingly) high quantile of the observed data, using the 85th percentile in their experiments. TIAO et al. provided some discussion on the role of $y^{*}$ in balancing exploration and exploitation, but ultimately used the 66th percentile of the observed data in their experiments for simplicity.

\section*{Bayesian neural networks}

Neural networks represent the state-of-the-art in numerous learning tasks due to their flexibility in modeling and impressive predictive perfor- mance. For this reason, it is tempting to use (Bayesian) neural networks for modeling in Bayesian optimization. However, there are two obstacles that must be addressed in order to build a successful system. First, the typically considerable cost of obtaining new data in Bayesian optimization limits the amount of data that might be used to train a neural network, imposing a ceiling on model complexity and perhaps ruling out deep architectures. Second, in order to guide an optimization policy, a neural network must yield useful estimates of uncertainty.

SNOEK et al. took an early step in this direction and reported success with a relatively simple construction. ${ }^{80}$ The authors first trained a typical (non-Bayesian) neural network using empirical loss minimization, then replaced the final layer with Bayesian linear regression. The weights in all but the final layer were fixed for the remainder of the procedure; the resulting model can thus be interpreted as Bayesian linear regression using neural basis functions. As this is in fact a Gaussian process, ${ }^{81}$ we may appeal to the computational details outlined earlier in this chapter to compute any policy we desire.

A somewhat more involved (and, arguably, "more Bayesian") approach was proposed by SPRINGENBERG et al., ${ }^{82}$ who combined a parametric objective function model $f(x ; \mathbf{w})$ - the output of a neural network with input $x$ and weights $\mathbf{w}$ - with an additive Gaussian noise observation model:

$$
p(y \mid x, \mathbf{w}, \sigma)=\mathcal{N}\left(y ; f(x ; \mathbf{w}), \sigma^{2}\right)
$$

Bayesian inference proceeds by selecting a prior $p(\mathbf{w}, \sigma)$ and computing the posterior from the observed data. The posterior is intractable, but SPRINGENBERG et al. described a Hamiltonian Monte Carlo scheme for drawing samples $\left\{\mathbf{w}_{i}, \sigma_{i}\right\}_{i=1}^{s}$ from the posterior, from which we may form a Gaussian mixture approximation to the predictive distribution. ${ }^{83}$ This is sufficient to compute policies such as (noiseless) expected improvement and probability of improvement $(8.9,8.22)$.

Several other neural and/or deep models have also been explored in the context of Bayesian optimization, including deep Gaussian processes $^{84}$ and (probabilistic) transformers. ${ }^{85}$

The impressive performance of modern deep neural networks is largely due to their ability to learn sophisticated feature representations of complex data, a process that requires enormous amounts of data and may be out of reach in most Bayesian optimization settings. However, when the domain consists of structured objects with sufficiently many unlabeled examples available, one path forward is to train a generative latent variable model - such as a variational autoencoder or generative adversarial network - using unsupervised (or semi-supervised) methods. We can then perform optimization in the resulting latent space, for example by simply constructing a Gaussian process over the learned neural representation. ${ }^{86}$ This approach has notably proven useful in the design of molecules ${ }^{87}$ and biological sequences ${ }^{88}$ In both of these settings, enormous databases (on the order of hundreds of millions of examples) are available for learning a representation prior to optimization. ${ }^{89,90}$
80 J. SNOEK et al. (2015). Scalable Bayesian Optimization Using Deep Neural Networks. ICML 2015.

81 See the result in (3.6), setting $K \equiv 0$ as there is no nonlinear component in the model. One beneficial side effect of this model is that the cost of inference for Bayesian linear regression is only linear with the number of observations.

82 J. T. SPRINGENBERG et al. (2016). Bayesian Optimization with Robust Bayesian Neural Networks. NeurIPS 2016.

83 Namely, we approximate with:

$$
p(y \mid x, \mathcal{D}) \approx \frac{1}{s} \sum_{i=1}^{s} \mathcal{N}\left(y ; f\left(x ; \mathbf{w}_{i}\right), \sigma_{i}^{2}\right) .
$$

Following HUTTER et al.'s approach to optimization with random forests, SPRINGENBERG et al. fit a single Gaussian distribution to this mixture via moment matching, but this is optional.

84 Z. DAI et al. (2016). Variational Auto-encoded Deep Gaussian Processes. ICLR 2016. arXiv: 1511.06455 [cs.LG].

85 A. MARAVAL et al. (2022). Sample-Efficient Optimisation with Probabilistic Transformer Surrogates. arXiv: 2205.13902 [cs.LG].

86 This construction represents a realization of a manifold Gaussian process/deep kernel; see § 3.4, p. 59 .

87 R. GÓMEZ-BOMBARELLI et al. (2018). Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules. ACS Central Science 4(2):268-276.

88 Numerous example systems are outlined in:

B. L. HIE and K. K. YANG (2021). Adaptive Machine Learning for Protein Engineering. arXiv: 2106.05466 [q-bio.QM]. [table 1]

89 J. J. IRWIN et al. (2020). ZINC2O - A Free Ultralarge-Scale Chemical Database for Ligand Discovery. Fournal of Chemical Information and Modeling 6o(12):6065-6073.

90 THE UNIPROT CONSORTIUM (2021). UniProt: The Universal Protein Knowledgebase in 2021 Nucleic Acids Research 49(D1):D480-D489. 91 D. zHOU et al. (2020). Neural Contextual Bandits with UCB-Based Exploration. ICML 2020.

92 w. ZHANG et al. (2O21). Neural Thompson Sampling. ICLR 2021. arXiv: 2010.00827 [CS.LG].

93 C. RiQuelme et al. (2018). Deep Bayesian Bandits Showdown. ICLR 2018. arXiv: 1802.09127 [CS.LG].

computation of expected improvement and probability of improvement: $\S \S 8.2-8.3$, p. 167

approximate computation for one-step lookahead: $§ 8.5$, p. 171 computation of knowledge gradient: $§ 8.6$, p. 172

Thompson sampling: § 8.7, p. 176 approximate computation of mutual information: $\S \S 8.8-8.9$, p. 180

averaging over a space of GPS: § 8.10, p. 192

alternatives to GPs: $§ 8.11$, p. 196
Finally, neural networks have also received some attention in the bandit literature as models for contextual bandits, which has led to theoretical regret bounds. ${ }^{91,92}$ RIQUELME et al. conducted an empirical "showdown" of various neural models in this setting. ${ }^{93}$ Although there was no clear "winner," the authors concluded that "decoupled" models - where uncertainty quantification is handled post hoc by feeding the output layer of a neural network into a simple model such as Bayesian linear regression - tended to perform better than end-to-end training.

\subsection*{SUMMARY OF MAJOR IDEAS}

In this chapter we considered the computation of the popular optimization policies described in the last chapter for Gaussian process models of an objective function with an exact or additive Gaussian noise observation model. The acquisition functions for most of these policies represent the one-step expected marginal gain to some underlying utility function:

$$
\alpha(x ; \mathcal{D})=\int \Delta(x, y) \mathcal{N}\left(y ; \mu, s^{2}\right) \mathrm{d} y,
$$

where $\Delta(x, y)$ is the gain in utility resulting from the observation $(x, y)$ (8.27). When $\Delta$ is a piecewise linear function of $y$, this integral can be resolved analytically in terms of the standard normal CDF. This is the case for the expected improvement and probability of improvement acquisition functions, both with and without observation noise.

However, when $\Delta$ is a more complicated function of the putative observation, we must in general rely on approximate computation to resolve this integral. When the predictive distribution is normal - as in the model class considered in this chapter - Gauss-Hermite quadrature provides a useful and sample-efficient approximation via a weighted average of carefully chosen integration nodes. This allows us to address some more complex acquisition functions such as the knowledge gradient.

The computation of mutual information with $x^{*}$ or $f^{*}$ entails an expectation with respect to these random variables, which cannot be approximated using simple quadrature schemes. Instead, we must rely on schemes such as Thompson sampling - a notable policy in its own right - to generate samples and proceed via (simple or quasi-) Monte Carlo integration, and in some cases, further approximations to the conditional predictive distributions resulting from these samples.

Finally, Bayesian optimization is of course not limited to a single Gaussian process belief on the objective function, which may be objectionable even when Gaussian processes are the preferred model class due to uncertainty in hyperparameters or model structure. Averaging over a space of Gaussian processes is possible - with some care - by adopting a Gaussian process mixture approximation to the marginal objective function posterior and relying on results from the single GP case. If desired, we may also abandon the model class entirely and compute policies with respect to an alternative such as random forests or Bayesian neural networks.