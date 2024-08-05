\section*{UTILITY FUNCTIONS FOR OPTIMIZATION}

In the last chapter we introduced Bayesian decision theory, a framework for decision making under uncertainty through which we can derive theoretically optimal optimization policies. Central to this approach is the notion of a utility function evaluating the quality of a dataset returned from an optimization routine. Given a model of the objective function - conveying our beliefs in the face of uncertainty - and a utility function - expressing our preferences over outcomes - computing the optimal policy is purely mechanical: we design every observation to maximize the expected utility of the returned dataset (5.15-5.17). Setting aside computational issues, adopting this approach entails only two major hurdles: building an objective function model consistent with our beliefs and designing a utility function consistent with our preferences.

Neither of these tasks is trivial! Beliefs and preferences are so innate to the human experience that distilling them down to mathematical symbols can be challenging. Fortunately, expressive and mathematically convenient options for both are readily available. We devoted significant attention to model building in the first part of this book, and we will address the construction of utility functions in this chapter. We will introduce a number of common utility functions designed for optimization, each carrying a different perspective on how optimization performance should be quantified. We hope that the underlying motivation for these utility functions may inspire the design of novel alternatives when called for. In the next chapter, we will demonstrate how approximating the optimal optimization policy corresponding to the utility functions described here yields many widespread Bayesian optimization algorithms.

Although we will be using Gaussian process models in our illustrations throughout the chapter, we will not assume the objective function model is a Gaussian process in our discussion. As in the previous chapters, we will use the notation $\mu_{\mathcal{D}}(x)=\mathbb{E}[\phi \mid x, \mathcal{D}]$ for the posterior mean of the objective function; this should not be interpreted as implying any particular model structure beyond admitting a posterior mean.

\subsection*{EXPECTED UTILITY OF TERMINAL RECOMMENDATION}

The purpose of optimization is often to explore a space of possibilities in search of the single best alternative, and after investing in optimization, we commit to using some chosen point in a subsequent procedure. In this context, the only purpose of the data collected during optimization is to help select this final point. For example, in hyperparameter tuning, we may evaluate numerous hyperparameters during model development, only to use the apparently best settings found in a production system.

Selecting a point for permanent use represents a decision, which we may analyze using Bayesian decision theory. If the sole purpose of optimization is to inform a final decision, it is natural to design the policy to maximize the expected utility of the terminal decision directly, and several popular policies are defined in this manner.
Chapter 7: Common Bayesian Optimization Policies, p. 123

posterior mean function, $\mu_{\mathcal{D}}$
Chapter 5: Decision Theory for Optimization, p. 87 1 Dependence on $\phi$ alone is not strictly necessary. For example, in the interest of robustness we might wish to ensure that function values are high in the neighborhood of our recommendation as well. This would be possible in the same framework by redefining the utility function as desired.

![](https://cdn.mathpix.com/cropped/2023_09_22_d7bf10e8271fa8c98208g-02.jpg?height=257&width=366&top_left_y=1059&top_left_x=251)

We may also interpret this class of utility functions as augmenting the decision tree in Figure 5.4 with a final layer corresponding to the terminal decision. The utility of the data is then the expected utility of this subtree, assuming optimal behavior.

\section*{Formalization of terminal recommendation decision}

Suppose we have run an optimization routine, which returned a dataset $\mathcal{D}=(\mathbf{x}, \mathbf{y})$, and suppose we now wish to recommend a point $x \in \mathcal{X}$ for use in some task, with performance determined by the underlying objective function value $\phi=f(x)$. ${ }^{1}$ This represents a decision under uncertainty about $\phi$, informed by the predictive distribution, $p(\phi \mid x, \mathcal{D})$.

To completely specify the decision problem, we must identify an action space $\mathcal{A} \subset \mathcal{X}$ for our recommendation and a utility function $v(\phi)$ evaluating a recommendation in hindsight according to its objective value $\phi$. Given these, a rational recommendation maximizes the expected utility:

$$
x \in \underset{x^{\prime} \in \mathcal{A}}{\arg \max } \mathbb{E}\left[v\left(\phi^{\prime}\right) \mid x^{\prime}, \mathcal{D}\right] .
$$

The expected utility of the optimal recommendation only depends on the data returned by the optimizer; it does not depend on the optimal recommendation $x$ (due to maximization) nor its objective value $\phi$ (due to expectation). This suggests a natural utility for use in optimization: the expected quality of an optimal terminal recommendation given the data,

$$
u(\mathcal{D})=\max _{x^{\prime} \in \mathcal{A}} \mathbb{E}\left[v\left(\phi^{\prime}\right) \mid x^{\prime}, \mathcal{D}\right] .
$$

In the context of the sequential decision tree from Figure 5.4, this utility function effectively "collapses" the expected utility of a final decision into a utility for the returned data; see the illustration in the margin. We are free to select the action space and utility function for the final recommendation as we see fit; we provide some advice below.

\section*{Choosing an action space}

We begin with the action space $\mathcal{A} \subset \mathcal{X}$. One extreme option is to restrict our choice to only the visited points $\mathbf{x}$. This ensures at least some knowledge of the objective function at the recommended point, which may be prudent when the objective function model may be misspecified. The other extreme is the maximally permissive alternative: the entire domain $\mathcal{X}$, allowing us to recommend any point, including those arbitrarily far from our observations. The wisdom of recommending an unvisited point for perpetual use is ultimately a question of faith in the model's beliefs.

Compromises between these extremes have also been occasionally suggested in the literature. OSBORNE et al. for example proposed restricting the choice of final recommendation to only those points where the objective function is known with acceptable tolerance. ${ }^{2}$ Such a scheme can limit unwanted surprise from recommending points where the objective function value is not known with sufficient certainty. One might accomplish this in several ways; OsBORNE et al. adopted a parametric, data-dependent action space of the form

$$
\mathcal{A}(\varepsilon ; \mathcal{D})=\{x \mid \operatorname{std}[\phi \mid x, \mathcal{D}] \leq \varepsilon\},
$$

where $\varepsilon$ is a threshold specifying the largest acceptable uncertainty.
M. A. OsBorne et al. (2009). Gaussian Processes for Global Optimization. LION 3. Choosing a utility function and risk tolerance

In addition to selecting an action space, we must also select a utility function $v(\phi)$ evaluating a recommendation at $x$ in light of the corresponding function value $\phi$. As our focus is on maximization (1.1), it is clear that the utility should be monotonically increasing in $\phi$, but it is not necessarily clear what shape this function should assume. The answer depends on our risk tolerance, a concept demonstrated in the margin. When making our final recommendation, we may wish to consider not only the expected function value of a given point but also our uncertainty in this value, as points with greater uncertainty may result in more surprising and potentially disappointing results.

By controlling the shape of the utility function $v(\phi)$, we may induce different behavior with respect to risk. The simplest and most common option encountered in Bayesian optimization is a linear utility:

$$
v(\phi)=\phi \text {. }
$$

In this case, the expected utility from recommending $x$ is simply the posterior mean of $\phi$, as we have already seen (5.4):

$$
\mathbb{E}[v(\phi) \mid x, \mathcal{D}]=\mu_{\mathcal{D}}(x),
$$

and an optimal recommendation maximizes the posterior mean over the action space:

$$
x=\underset{x^{\prime} \in \mathcal{A}}{\arg \max } \mu_{\mathcal{D}}\left(x^{\prime}\right) .
$$

Uncertainty in the objective function is not considered in this decision at all! Rather, we are indifferent between points with equal expected value, regardless of their uncertainty - that is, we are risk neutral.

Risk neutrality is computationally convenient due to the simple form of the expected utility, but may not always reflect our true preferences. In the margin we show beliefs over the objective values for two potential recommendations with equal expected value but significantly different risk. In many scenarios we would have a clear preference between the two alternatives, but a risk-neutral utility induces complete indifference.

A useful concept when reasoning about risk preferences is the socalled certainty equivalent. Consider a risky potential recommendation $x$, that is, a point for which we do not know the objective value exactly. The certainty equivalent for $x$ is the value of a hypothetical risk-free alternative for which our preferences would be indifferent. That is, the certainty equivalent for $x$ corresponds to an objective function value $\phi^{\prime}$ such that

$$
v\left(\phi^{\prime}\right)=\mathbb{E}[v(\phi) \mid x, \mathcal{D}] .
$$

Under a risk-neutral utility function, the certainty equivalent of a point $x$ is simply its expected value: $\phi^{\prime}=\mu_{\mathcal{D}}(x)$. Thus we would abandon a potential recommendation for another only if it had greater expected value, independent of risk. However, we may encode risk-aware preferences with appropriately designed nonlinear utility functions.

![](https://cdn.mathpix.com/cropped/2023_09_22_d7bf10e8271fa8c98208g-03.jpg?height=223&width=523&top_left_y=548&top_left_x=1372)

Consider the illustrated beliefs about the objective function value corresponding to two possible recommendations. One option has a higher expected value, but also greater uncertainty, and proposing it entails some risk. The alternative has a lower expected value but is perhaps a safer option. A risk-averse agent might prefer the latter option, whereas a risk-tolerant agent might prefer the former.

![](https://cdn.mathpix.com/cropped/2023_09_22_d7bf10e8271fa8c98208g-03.jpg?height=357&width=351&top_left_y=1204&top_left_x=1432)

A risk-neutral (linear) utility function.

![](https://cdn.mathpix.com/cropped/2023_09_22_d7bf10e8271fa8c98208g-03.jpg?height=225&width=523&top_left_y=1692&top_left_x=1372)

Beliefs over two recommendations with equal expected value. A risk-neutral agent would be indifferent between these alternatives, a risk-averse agent would prefer the more certain option, and a risk-seeking agent would prefer the more uncertain option. 

![](https://cdn.mathpix.com/cropped/2023_09_22_d7bf10e8271fa8c98208g-04.jpg?height=359&width=351&top_left_y=503&top_left_x=224)

A risk-averse (concave) utility function.

![](https://cdn.mathpix.com/cropped/2023_09_22_d7bf10e8271fa8c98208g-04.jpg?height=97&width=523&top_left_y=1017&top_left_x=161)

A risk-averse agent may be indifferent between a risky recommendation (the wide distribution) and its risk-free certainty equivalent with lower expected value (the Dirac delta).

3 One flexible family is the hyperbolic absolute risk aversion (HARA) class, which includes many popular choices as special cases:

J. E. INGERSOLL JR. (1987). Theory of Financial Decision Making. Rowman \& Littlefield. [chapter 1]

4 Under Gaussian beliefs on function values, one can find a family of concave (or convex) utility functions inducing equivalent recommendations. See $\S 8.26$, p. 171 for related discussion.

5 Note that expected reward $(\mu)$ and risk $(\sigma)$ have compatible units in this formulation, so this weighted combination is sensible.

6 This name contrasts with the cumulative reward: § 6.2, p. 114 .

7 One technical caveat is in order: when the dataset is empty, the maximum degererates and we have $u(\varnothing)=-\infty$.
If our preferences indicate risk aversion, we might be willing to recommend a point with lower expected value if it also entailed less risk. We may induce risk-averse preferences by adopting a utility function that is a concave function of the objective value. In this case, by Jensen's inequality we have

$$
v\left(\phi^{\prime}\right)=\mathbb{E}[v(\phi) \mid x, \mathcal{D}] \leq v(\mathbb{E}[\phi \mid x, \mathcal{D}])=v\left(\mu_{\mathcal{D}}(x)\right),
$$

and thus the certainty equivalent of a risky recommendation is less than its expected value; see the example in the margin. Similarly, we may induce risk-seeking preferences with a convex utility function, in which case the certainty equivalent of a risky recommendation is greater than its expected value - our preferences encode an inclination toward gambling. Risk-averse and risk-seeking utilities are rarely encountered in the Bayesian optimization literature; however, they may be preferable in some practical settings, as risk neutrality is often questionable.

Numerous risk-averse utility functions have been proposed in the economics and decision theory literature, ${ }^{3}$ and a full discussion is beyond the scope of this book. However, one natural approach is to quantify the risk associated with recommending an uncertain value $\phi$ by its standard deviation:

$$
\sigma=\operatorname{std}[\phi \mid x, \mathcal{D}]
$$

Now we may establish preferences over potential recommendations consistent with ${ }^{4}$ a weighted combination of a point $x$ 's expected reward, $\mu=\mu_{\mathcal{D}}(x)$, and its risk, $\sigma:^{5}$

$$
\mu+\beta \sigma \text {. }
$$

Here $\beta$ serves as a tunable risk-tolerance parameter: values $\beta<0$ penalize risk and induce risk-averse behavior, values $\beta>0$ reward risk and induce risk-seeking behavior, and $\beta=0$ induces risk neutrality (6.2).

Two particular utility functions from this general framework are widely encountered in Bayesian optimization, both representing the expected utility of a risk-neutral optimal terminal recommendation.

\section*{Simple reward}

Suppose an optimization routine returned data $\mathcal{D}=(\mathbf{x}, \mathbf{y})$ to inform a terminal recommendation, and that we will make this decision using the risk-neutral utility function $v(\phi)=\phi(6.2)$. If we limit the action space of this recommendation to only the locations evaluated during optimization $\mathbf{x}$, the expected utility of the optimal recommendation is the so-called simple reward: $:^{6,7}$

$$
u(\mathcal{D})=\max \mu_{\mathcal{D}}(\mathbf{x})
$$

In the special case of exact observations, where $\mathbf{y}=f(\mathbf{x})=\boldsymbol{\phi}$, the simple reward reduces to the maximal function value encountered during optimization:

$$
u(\mathcal{D})=\max \phi
$$



![](https://cdn.mathpix.com/cropped/2023_09_22_d7bf10e8271fa8c98208g-05.jpg?height=565&width=1031&top_left_y=460&top_left_x=267)

One-step lookahead with the simple reward utility function produces a widely used acquisition function known as expected improvement, which we will discuss in detail in the next two chapters.

\section*{Global reward}

Another prominent utility is the global reward. ${ }^{8}$ Here we again consider a risk-neutral terminal recommendation, but now expand the action space for this recommendation to the entire domain $\mathcal{X}$. The expected utility of this recommendation is the global maximum of the posterior mean:

$$
u(\mathcal{D})=\max _{x \in \mathcal{X}} \mu_{\mathcal{D}}(x) .
$$

An example dataset exhibiting a large discrepancy between the simple reward (6.3) and global reward (6.5) utilities is illustrated in Figure 6.1. The larger action space underlying global reward leads to a markedly different and somewhat riskier recommendation.

One-step lookahead with global reward (6.5) yields the knowledge gradient acquisition function, which we will also consider at length in the following chapters.

\section*{A tempting, but nonsensical alternative}

There is an alternative utility deceptively similar to the simple reward that is sometimes encountered in the Bayesian optimization literature, namely, the maximum noisy observed value contained in the dataset: ${ }^{9}$

$$
u(\mathcal{D}) \stackrel{?}{=} \max \mathbf{y} .
$$

In the case of exact observations of the objective function, this value coincides with the simple reward (6.4), which has a natural interpretation as the expected utility of a particular optimal terminal recommendation. However, this correspondence does not hold in the case of inexact or noisy observations, and the proposed utility is rendered absurd.
Figure 6.1: The terminal recommendations corresponding to the simple reward and global reward for an example dataset comprising five observations. The prior distribution for the objective for this demonstration is illustrated in Figure 6.3.

expected improvement: $§ 7.3$ p. 127

8 "Global simple reward" would be a more accurate (but annoyingly bulky) name.

knowledge gradient: $§ 7.4$, p. 129

\footnotetext{
9 The "questionable equality" symbol $\stackrel{?}{=}$ is reserved for this single dubious equation.
} 

![](https://cdn.mathpix.com/cropped/2023_09_22_d7bf10e8271fa8c98208g-06.jpg?height=331&width=1650&top_left_y=443&top_left_x=160)

Figure 6.2: The utility $u(\mathcal{D})=\max y$ would prefer the excessively noisy dataset on the left to the less-noisy dataset on the right with smaller maximum value. The data on the left reveal little information about the objective function, and the maximum observed value is very likely to be an outlier, whereas the data on the right indicate reasonable progress.

large-but-noisy observations are not necessarily preferable

example and discussion

approximation to the simple reward

expected improvement: § 7.3, p. 127
This is simple to demonstrate by contemplating the preferences over outcomes encoded in the utility, which may not align with intuition. This disparity is especially notable in situations with excessively noisy observations, where the maximum value observed will likely reflect spurious noise rather than actual optimization progress.

Figure 6.2 shows an extreme but illustrative example. We consider two optimization outcomes over the same domain, one with excessively noisy observations and the other with exact measurements. The noisy dataset contains a large observation on the right-hand side of the domain, but this is almost certainly the result of noise, as indicated by the objective function posterior. Although the other dataset has a lower maximal value, the observations are more trustworthy and represent a plainly better outcome. But the proposed utility (6.6) prefers the noisier dataset! On the other hand, both the simple and global reward utilities prefer the noiseless dataset, as the data produce a larger effect on the posterior mean - and thus yield more promising recommendations.

Of course, errors in noisy measurements are not always as extreme as in this example. When the signal-to-noise ratio is relatively high, the utility (6.6) can serve as a reasonable approximation to the simple reward. We will discuss this approximation scheme further in the context of expected improvement.

\subsection*{CUMULATIVE REWARD}

Simple and global reward are motivated by supposing that the goal of optimization is to discover the best single point from a space of alternatives. To this end, we evaluate data according to the highest function value revealed and assume that the values of any suboptimal points encountered are irrelevant.

In other settings, the value of every individual observation might be significant, for example, if the optimization procedure is controlling a critical external system. If the consequences of these decisions are nontrivial, we might wish to discourage observing where we might encounter unexpectedly low objective function values. Cumulative reward encourages obtaining observations with large average value. For a dataset $\mathcal{D}=(\mathbf{x}, \mathbf{y})$, its cumulative reward is simply the sum of the observed values:

$$
u(\mathcal{D})=\sum_{i} y_{i}
$$

One notable use of cumulative reward is in active search, a simple mathematical model of scientific discovery. Here, we successively select points for investigation seeking novel members of a rare, valuable class $\mathcal{V} \subset \mathcal{X}$. Observing at a point $x \in \mathcal{X}$ yields a binary observation indicating membership in the desired class: $y=[x \in \mathcal{V}]$. Most studies of active search seek to maximize the cumulative reward (6.7) of the gathered data, hoping to discover as many valuable items as possible.

\subsection*{INFORMATION GAIN}

Simple, global, and cumulative reward judge optimization performance based solely on having found high objective function values, a natural and pragmatic concern. Information theor $y^{10}$ provides an alternative approach to measuring utility that is often used in Bayesian optimization. An information-theoretic approach to sequential experimental design (including optimization) identifies some random variable that we wish to learn about through our observations. We then evaluate performance by quantifying the amount of information about this random variable revealed by data, favoring datasets containing more information. This line of reasoning gives rise to the notion of information gain.

Let $\omega$ be a random variable of interest that we wish to determine through the observation of data. The choice of $\omega$ is open-ended and should be guided by the application at hand. Natural choices aligned with optimization include the location of the global optimum, $x^{*}$, and the maximal value of the objective, $f^{*}(1.1)$, each of which has been considered in depth in this context.

We may quantify our initial uncertainty about $\omega$ via the (differential) entropy of its prior distribution, $p(\omega)$ :

$$
H[\omega]=-\int p(\omega) \log p(\omega) \mathrm{d} \omega
$$

The information gain offered by a dataset $\mathcal{D}$ is then the reduction in entropy when moving from the prior to the posterior distribution:

$$
u(\mathcal{D})=H[\omega]-H[\omega \mid \mathcal{D}],
$$

where $H[\omega \mid \mathcal{D}]$ is the differential entropy of the posterior: ${ }^{11}$

$$
H[\omega \mid \mathcal{D}]=-\int p(\omega \mid \mathcal{D}) \log p(\omega \mid \mathcal{D}) \mathrm{d} \omega .
$$

Somewhat confusingly, some authors use an alternative definition of information gain - the Kullback-Leibler $(K L)$ divergence between the active search: $\S 11.11$, p. 282

10 A broad introduction to information theory is provided by the classical text:

T. M. COVER and J. A. THOMAs (2006). Elements of Information Theory. John Wiley \& Sons,

and a treatment focusing on the connections to Bayesian inference can be found in:

D. J. C. MACKAY (2003). Information Theory, Inference, and Learning Algorithms. Cambridge University Press.

entropy

11 A caveat is in order regarding this notation, which is not standard. In information theory $H[\omega \mid \mathcal{D}]$ denotes the conditional entropy of $\omega$ given $\mathcal{D}$, which is the expectation of the given quantity over the observed values y. For our purposes it will be more useful for this to signify the differential entropy of the notationally parallel posterior $p(\omega \mid \mathcal{D})$. When needed, we will write conditional entropy with an explicit expectation: $\mathbb{E}[H[\omega \mid \mathcal{D}] \mid \mathbf{x}]$.

Kullback-Leibler (KL) divergence 12 A simple example: suppose $\omega \in(0,1)$ is the unknown bias of a coin, with prior

$$
p(\omega)=\operatorname{Beta}(\omega ; 2,1) ; \quad H \approx-0.193 .
$$

After flipping and observing "tails," the posterior becomes

$$
p(\omega \mid \mathcal{D})=\operatorname{Beta}(\omega ; 2,2) ; \quad H \approx-0.125 .
$$

The information "gained" was

$$
H[\omega]-H[\omega \mid \mathcal{D}] \approx-0.068<0 .
$$

Of course, the most likely outcome of the flip a priori was "heads," so the outcome was surprising. Indeed the expected information gain before the experiment was

$$
H[\omega]-\mathbb{E}[H[\omega \mid \mathcal{D}]] \approx 0.137>0 .
$$

13 See p. 138 for a proof.

mutual information, entropy search: § 7.6,

information-theoretic policies as the scientific method: $\S 7.6$, p. 136

model averaging: $\S \S 4 \cdot 4-4 \cdot 5$, p. 74 model-agnostic alternatives

14 The effect on simple and global reward is to maximize a model-marginal posterior mean, and the effect on information gain is to evaluate changes in model-marginal beliefs about $\omega$. posterior distribution and the prior distribution:

$$
u(\mathcal{D})=D_{\text {КL }}[p(\omega \mid \mathcal{D}) \| p(\omega)]=\int p(\omega \mid \mathcal{D}) \log \frac{p(\omega \mid \mathcal{D})}{p(\omega)} \mathrm{d} \omega .
$$

That is, we quantify the information contained in data by how much our belief in the $\omega$ changes as a result of collecting it. This definition has some convenient properties compared to the previous one (6.8); namely, the expression in (6.9) is invariant to reparameterization of $\omega$ and always nonnegative, whereas "surprising" observations may cause the information gain in (6.8) to become negative. ${ }^{12}$ However, the previous definition as the direct reduction in entropy may be more intuitive.

Fortunately (and perhaps surprisingly!), there is a strong connection between these two "information gains" (6.8-6.9) in the context of sequential decision making. Namely, their expected values with respect to observed values are equal, and thus maximizing expected utility with either leads to identical decisions. ${ }^{13}$ For this reason, the reader may simply choose whichever definition they find more intuitive.

One-step lookahead with (either) information gain yields an acquisition function known as mutual information. This is the basis for a family of related Bayesian optimization procedures sharing the moniker entropy search, which we will discuss further in the following chapters.

Unlike the other utility functions discussed thus far, information gain is not intimately linked to optimization, and may be adapted to a wide variety of tasks by selecting the random variable $\omega$ appropriately. Rather, this scheme of refining knowledge through experiment is effectively a mathematical formulation of scientific inquiry.

\subsection*{DEPENDENCE ON MODEL OF OBJECTIVE FUNCTION}

One striking feature of most of the utility functions defined in this chapter is implicit dependence on an underlying model of the objective function. Both the simple and global reward are defined in terms of the posterior mean function $\mu_{\mathcal{D}}$, and information gain about the location or value of the optimum is defined in terms of the posterior belief about these values, $p\left(x^{*} f^{*} \mid \mathcal{D}\right)$; both of these quantities are byproducts of the objective function posterior.

One way to mitigate model dependence in the computation of utility is via model averaging $(4.11,4.23) .{ }^{14}$ We may also attempt to define purely model-agnostic utility functions in terms of the data alone, without reference to a model; however, the possibilities are somewhat limited if we wish the resulting utility to be sensible. Cumulative reward (6.7) is one example, as it depends only on the observed values $y$. The maximum function value observed is another possibility (6.6), but, as we have shown, it is dubious when observations are corrupted by noise. Other similarly defined alternatives may suffer the same fate - for additive noise with zero mean, the expected contribution from noise to the cumulative reward is zero; however, noise will bias many other natural measures such as order statistics (including the maximum) of the observations. 

![](https://cdn.mathpix.com/cropped/2023_09_22_d7bf10e8271fa8c98208g-09.jpg?height=529&width=1585&top_left_y=455&top_left_x=270)

Figure 6.3: The objective function prior used throughout our utility function comparison. Marginal beliefs of function values are shown, as well as the induced beliefs over the location of the global optimum, $p\left(x^{*}\right)$, and the value of the global optimum, $p\left(f^{*}\right)$. Note that there is a significant probability that the global optimum is achieved on the boundary of the domain, reflected by large point masses.

\subsection*{COMPARISON OF UTILITY FUNCTIONS}

We have now presented several utility functions for evaluating a dataset returned by an optimization routine. Each utility quantifies progress on our model optimization problem (1.1) in some way, but it may be difficult at this point to appreciate their, sometimes subtle, differences in approach. Here we will present and discuss example datasets for which different utility functions diverge in their opinion of quality.

We particularly wish to contrast the behavior of the simple reward (6.3) with other utility functions. Simple reward is probably the most prevalent utility in the Bayesian optimization literature (especially in applications), as it corresponds to the widespread expected improvement acquisition function. A distinguishing feature of simple reward is that it evaluates data based only on local properties of the objective function posterior. This locality is both computationally convenient and pragmatic. Simple reward is derived from the premise that we will be recommending one of the points observed during the course of optimization for permanent use, and thus it is sensible to judge performance based on the objective function values at the observed locations alone.

Several alternatives instead measure global properties of the objective function posterior. The global reward (6.5), for example, considers the entire posterior mean function, reflecting a willingless to recommend an unevaluated point after termination. Information gain (6.8) about the location or value of the optimum considers the posterior entropy of these quantities, again a global property. The consequences of reasoning about local or global properties of the posterior can sometimes lead to significant disagreement between the simple reward and other utilities.

In the following examples, we consider optimization on an interval with exact measurements. We model the objective function with

local vs. global properties of posterior

expected improvement: § 7.3, p. 127 a Gaussian process with constant mean (3.1) and squared exponential 

![](https://cdn.mathpix.com/cropped/2023_09_22_d7bf10e8271fa8c98208g-10.jpg?height=765&width=1579&top_left_y=457&top_left_x=221)

Figure 6.4: An example dataset of five observations and the resulting posterior belief of the objective function. This dataset exhibits relatively low simple reward (6.3) but relatively high global reward (6.5) and information gain (6.8) about the location $x^{*}$ and value $f^{*}$ of the optimum.

15 For this model, a unique optimum will exist with probability one; see $\S 2.7$, p. 34 for more details. covariance (3.12). This prior is illustrated in Figure 6.3, along with the induced beliefs about the location $x^{*}$ and value $f^{*}$ of the global optimum. Both distributions reflect considerable uncertainty. ${ }^{15}$ We will examine two datasets that might be returned by an optimizer using this model and discuss how different utility functions would evaluate these outcomes.

\section*{Good global outcome but poor local outcome}

Consider the dataset in Figure 6.4 and the resulting posterior belief about the objective and its optimum. In this example, the simple reward is

low simple reward

high global reward

final recommendations

high information gain relatively low as the posterior mean at our observations is unremarkable. unlucky outcome. However, the global reward is relatively high: the data imply a steep derivative in one location, inducing high values of the posterior mean away from our data. This is a significant accomplishment from the point of view of the global reward, as the model expects a terminal recommendation in that region to be especially valuable.

Figure 6.1 shows the optimal final recommendations associated with these two utility functions. The simple reward recommendation prioritizes safety over reward, whereas the global reward recommendation reflects more risk tolerance. Neither is inherently better: although the global reward recommendation has a larger expected value, this expectation is computed using a model that might be mistaken. Further, comparing the posterior distribution in Figure 6.4 with the prior in Figure In fact, every observation was lower than the prior mean, a seemingly - observations _ posterior mean posterior $95 \%$ credible interval

![](https://cdn.mathpix.com/cropped/2023_09_22_d7bf10e8271fa8c98208g-11.jpg?height=403&width=1582&top_left_y=615&top_left_x=268)

Figure 6.5: An example dataset containing a single observation and the resulting posterior belief of the objective function. This dataset exhibits relatively high simple reward (6.3) but relatively low global reward (6.5) and information gain (6.8) about the location $x^{*}$ and value $f^{*}$ of the optimum.

6.3, we see this example dataset also induces a significant reduction in our uncertainty about both the location and value of the global optimum, despite not containing any particularly notable values itself. Therefore, despite a somewhat low simple reward, observing this dataset results in relatively high information gain about these quantities.

\section*{Good local outcome but poor global outcome}

We illustrate a different example dataset in Figure 6.5. The dataset contains a single observation with value somewhat higher than the prior mean. Although this dataset may not appear particularly impressive, its simple reward is higher than the previous dataset, as this observation exceeds every value seen in that scenario.

However, this dataset has lower value than the previous dataset when evaluated by other utility functions. Its global reward is lower than in the first scenario, as the global maximum of the posterior mean is lower. This can be verified by visual inspection of Figures 6.4 and 6.5, whose vertical axes are compatible. Further, the single observation in this scenario provides nearly no information regarding the location nor the value of the global maximum. The observation of a moderately high value provides only weak evidence that the global optimum may be located nearby, barely influencing our posterior belief about $x$. The observation does truncate our belief of the value of the global optimum $f^{*}$ but only rules out a relatively small portion of its lower tail.

\subsection*{SUMMARY OF MAJOR IDEAS}

In Bayesian decision theory, preferences over outcomes are encoded by a utility function, which in the context of optimization policy design,

high simple reward

low global reward

low information gain 16 Just like human taste, there is no right or wrong when it comes to preferences, at least not over certain outcomes. The von NeumannMorgenstern theorem mentioned on p. 90 entails rationality axioms, but these only apply to preferences over uncertain outcomes.

expected utility of terminal recommendation: $\S 6.1$, p. 109

risk tolerance: $§ 6.1$, p. 111

simple reward: $§ 6.1$, p. 112 global reward: §6.1, p. 113

cumulative reward: $§ 6.2$, p. 114

information gain: $§ 6.3$, p. 115

comparison of utility functions: $§ 6.5$, p. 117 assesses the quality of data returned by an optimization routine, $u(\mathcal{D})$. The optimization policy then seeks to design observations to maximize the expected utility of the returned data. The general theory presented in the last chapter makes no assumptions regarding the utility function. ${ }^{16}$ However, in the context of optimization, some utility functions are particularly easy to motivate.

- In many cases there is a decision following optimization in which we must recommend a single point in the domain for perpetual use. In this case, it is sensible to define an optimization utility function in terms of the expected utility of the optimal terminal recommendation informed by the returned data. This requires fully specifying that terminal recommendation, including its action space and utility function, after which we may "pass through" the optimal expected utility (6.1).

- When designing a terminal recommendation - especially when we may recommend points with residual uncertainty in their underlying objective value - it may be prudent to consider our risk tolerance. Careful design of the terminal utility allows for us to tune our appetite for risk, in terms of trading off a point's expected value against its uncertainty. Most utilities encountered in Bayesian optimization are risk neutral, but this need not necessarily be the case.

- Two notable realizations of this scheme are simple reward (6.3) and global reward (6.5), both of which represent the expected utility of an optimal terminal recommendation with a risk-neutral utility. The action space for simple reward is the points visited during optimization, and the action space for global reward is the entire domain.

- The simple reward simplifies when observations are exact (6.4).

- An alternative to the simple reward is the cumulative reward (6.7), which evaluates a dataset based on the average, rather than maximum, value observed. This does not see too much direct use in policy design, but is an important concept for the analysis of algorithms.

- Information gain provides an information-theoretic approach to quantifying the value of data in terms of the information provided by the data regarding some quantity of interest. This can be quantified by either measuring the reduction in differential entropy moving from the prior to the posterior (6.8) or by the KL divergence between the posterior and prior (6.9) - either induces the same one-step lookahead policy.

- In the context of optimization, information gain regarding either the location $x^{*}$ or value $f^{*}$ of the global optimum (1.1) are judicious realizations of this general approach to utility design.

- An important feature distinguishing simple reward from most other utility functions is its dependence on the posterior belief at the observed locations alone, rather than the posterior belief over the entire objective function. Even in relatively simple examples, this may lead to disagreement between simple reward and other utility functions in judging the quality of a given dataset. The utility functions presented in this chapter form the backbone of the most popular Bayesian optimization algorithms. In particular, many common policies are realized by maximizing the one-step expected marginal gain to one of these utilities, as we will show in the next chapter.

one-step lookahead: §5·3, p. 101 