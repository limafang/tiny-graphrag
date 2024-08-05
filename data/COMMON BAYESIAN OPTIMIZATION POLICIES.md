\section*{COMMON BAYESIAN OPTIMIZATION POLICIES}

The heart of an optimization routine is its policy, which sequentially designs each observation in light of available data. ${ }^{1}$ In the Bayesian approach to optimization, policies are designed with reference to a probabilistic belief about the objective function, with this belief guiding the policy in making decisions likely to yield beneficial outcomes. Numerous Bayesian optimization policies have been proposed in the literature, many of which enjoy widespread use. In this chapter we will present an overview of popular Bayesian optimization policies and emphasize common themes in their construction. In the next chapter we will provide explicit computational details for implementing these policies with Gaussian process models of the objective function.

Nearly all Bayesian optimization algorithms result from one of two primary approaches to policy design. The most popular is Bayesian decision theory, the focus of the previous two chapters. In Chapter 5 we introduced Bayesian decision theory as a general framework for deriving optimal, but computationally prohibitive, optimization policies. In this chapter, we will apply the ideas underlying these optimal procedures to realize computationally tractable and practically useful policies. We will see that a majority of popular Bayesian optimization algorithms can be interpreted in a uniform manner as performing one-step lookahead for some underlying utility function.

Another avenue for policy design is to adopt algorithms for multiarmed bandits to the optimization setting. A multi-armed bandit is a finitedimensional model of sequential optimization with noisy observations. We consider an agent faced with a finite set of alternatives ("arms"), who is compelled to select a sequence of items from this set. Choosing a given item yields a stochastic reward drawn from an unknown distribution associated with that arm. We seek a sequential policy for selecting arms maximizing the expected cumulative reward (6.7). ${ }^{2}$

Multi-armed bandits have seen decades of sustained study, and some policies have strong theoretical guarantees on their performance, suggesting these policies may also be useful for optimization. To this end, we may model optimization as an infinite-armed bandit, where each point in the domain $x \in \mathcal{X}$ represents an arm with uncertain reward depending on the objective function value $\phi=f(x)$. Our belief about the objective function then provides a mechanism to reason about these rewards and derive a policy. This analogy has inspired several Bayesian optimization policies, many of which enjoy strong performance guarantees.

A central concern in bandit problems is the exploration-exploitation dilemma: we must repeatedly decide whether to allocate resources to an arm already known to yield high reward ("exploitation") or to an arm with uncertain reward to learn about its reward distribution ("exploration"). Exploitation may yield a high instantaneous reward, but exploration may provide valuable information for improving future rewards. This tradeoff between instant payoff and learning for the future has been called "a conflict evident in all human action.", A similar choice is faced
1 The reader may wish to recall our model optimization procedure: Algorithm 1.1, p. 3 .

Chapter 8: Computing Policies with Gaussian Processes, p. 157

Chapter 5: Decision Theory for Optimization, p. 87

one-step lookahead: $§ 5.3$, p. 101

Chapter 6: Utility Functions for Optimization: p. 109

multi-armed bandits

2 The name references a gambler contemplating how to allocate their bankroll among a wall of slot machines. Slot machines are known as "one-armed bandits" in American vernacular, as they eventually steal all your money.

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-01.jpg?height=226&width=528&top_left_y=1880&top_left_x=1369)

Exploration vs. exploitation. We show reward distributions for two possible options. The more certain option returns higher expected reward, but the alternative reflects more uncertainty and may actually be superior. Which should we prefer?

3 P. Whittle (1982). Optimization over Time: Dynamic Programming and Stochastic Control. Vol. 1. John Wiley \& Sons. 

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-02.jpg?height=297&width=1630&top_left_y=460&top_left_x=156)

Figure 7.1: The scenario we will consider for illustrating optimization policies. The objective function prior is a Gaussian process with constant mean and Mátern covariance with $v=5 / 2$ (3.14). We show the marginal predictive distributions and three samples from the posterior conditioned on the indicated observations.

4 Take note of the legend; it will not be repeated.

Chapter 2: Gaussian Processes, p. 15

objective function for simulation

Chapter 6: Utility Functions for Optimization: throughout optimization, as we must continually decide whether to focus on a suspected local maximum (exploitation) or to explore unknown regions of the domain seeking new maxima (exploration). We will see that typical Bayesian optimization policies reflect consideration of this dilemma in some way, whether by explicit design on or as a consequence of decision-theoretic reasoning.

Before diving into policy design, we pause to introduce a running example we will carry through the chapter and notation to facilitate our discussion. We will then derive a series of policies stemming from Bayesian decision theory, and finally consider bandit-inspired algorithms.

\subsection*{EXAMPLE OPTIMIZATION SCENARIO}

Throughout this chapter we will demonstrate the behavior of optimization policies on an example scenario illustrated in Figure 7.1. ${ }^{4} \mathrm{We}$ consider a one-dimensional objective function observed without noise and adopt a Gaussian process prior belief about this function. The prior mean function is constant (3.1), and the prior covariance function is a Mátern covariance with $v=5 / 2(3.14)$. The parameters are fixed so that the domain spans exactly 30 length scales. We condition this prior on three observations, inducing two local maxima in the posterior mean and a range of marginal predictive uncertainty.

We will illustrate the behavior of policies by simulating optimization to design a sequence of additional observations for this running example. The ground truth objective function we will use for these simulations is shown in Figure 7.2 and was drawn from the corresponding model. The objective features numerous undiscovered local maxima and exhibits an unusually high global maximum on the left-hand side of the domain.

\subsection*{DECISION-THEORETIC POLICIES}

Central to decision-theoretic optimization is a utility function $u(\mathcal{D})$ measuring the quality of a dataset returned by an optimizer. After selecting a utility function and a model of the objective function and our observations, we may design each observation to maximize the expected utility 

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-03.jpg?height=277&width=1630&top_left_y=461&top_left_x=270)

Figure 7.2: The true objective function used for simulating optimization policies.

of the returned data (5.16). This policy is optimal in the average case: it maximizes the expected utility of the returned dataset over the space of all possible policies. ${ }^{5}$ Unfortunately, optimality comes at a great cost. Computing the optimal policy requires recursive simulation of the entire remainder of optimization, a random process due to uncertainty in the outcomes of our observations. In general, the cost of computing the optimal policy grows exponentially with the horizon, the number of observations remaining before termination.

However, the structure of the optimal policy suggests a natural family of lookahead approximations based on fixing a computationally tractable maximum horizon throughout optimization. This line of reasoning has led to many of the practical policies available for Bayesian optimization. In fact, most popular algorithms represent one-step lookahead, where in each iteration we greedily maximize the expected utility after obtaining only a single additional observation. Although these policies are maximally myopic, they are also maximally efficient among lookahead approximations and have delivered impressive empirical performance in a wide range of settings.

It may seem surprising that such dramatically myopic policies have any use at all. There is a huge difference between the scale of reasoning in one-step lookahead compared with the optimal procedure, which may consider hundreds of future decisions or more when designing an observation. However, the situation is somewhat more nuanced than it might appear. In a seminal paper, KUSHNER argued that myopic policies may in fact show better empirical performance than a theoretically optimal policy, and his argument remains convincing: ${ }^{6}$

Since a mathematical model of $[f]$ is available, it is theoretically possible, once a criterion of optimality is given, to determine the mathematically optimal sampling policy. However... determination of the optimum sampling policies is extremely difficult. Because of this, the development of our sampling laws has been guided primarily by heuristic considerations. ${ }^{7}$ There are some advantages to the approximate approach... [and] its use may yield better results than would a procedure that is optimum for the model. Although the model selected for $[f]$ is the best we have found for our purposes, it is sometimes too general...
5 To be precise, optimality is defined with respect to a model for the objective function $p(f)$, an observation model $p(y \mid x, \phi)$, a utility function $u(\mathcal{D})$, and an upper bound on the number of observations allowed $\tau$. Bayesian decision theory provides a policy achieving the maximal expected utility at termination with respect to these choices.

running time of optimal policy and efficient approximations: $\S 5 \cdot 3$, p. 99

limited lookahead: $§ 5 \cdot 3$, p. 101

6 H. J. Kushner (1964). A New Method of Locating the Maximum Point of an Arbitrary Multipeak Curve in the Presence of Noise. Fournal of Basic Engineering 86(1):97-106.

7 Specifically, maximizing probability of improvement: § 7.5, p. 131. notation for one-step lookahead policies

proposed next point $x$ with putative value $y$ updated dataset $\mathcal{D}^{\prime}=\mathcal{D} \cup(x, y)$

expected marginal gain

acquisition functions: $§ 5$, p. 88

value of sample information
What could possibly cause such a seemingly contradictory finding? As KUSHNER suggests, one possible reason could be model misspecification. The optimal policy is only defined with respect to a chosen model of the objective function and our observations, which is bound to be imperfect. By relying less on the model's belief, we may gain some robustness alongside considerable computational savings.

The intimate relationship between many Bayesian optimization methods and one-step lookahead is often glossed over, with a policy often introduced ex nihilo and the implied choice of utility function left unstated. This disconnect can sometimes lead to policies that are nonsensical from a decision-theoretic perspective or that incorporate implicit approximations that may not always be appropriate. We intend to clarify these connections here. We hope that our presentation can help guide practitioners in navigating the increasingly crowded space of available policies when presented with a novel scenario.

\section*{One-step lookahead}

Let us review the generic procedure for developing a one-step lookahead policy and adopt standard notation to facilitate their description. Suppose we have selected an arbitrary utility function $u(\mathcal{D})$ to evaluate a returned dataset. Suppose further that we have already gathered an arbitrary dataset $\mathcal{D}=(\mathbf{x}, \mathbf{y})$ and wish to select the next evaluation location. This is the fundamental role of an optimization policy.

If we were to choose some point $x$, we would observe a corresponding value $y$ and update our dataset, forming $\mathcal{D}^{\prime}=\left(\mathbf{x}^{\prime}, \mathbf{y}^{\prime}\right)=\mathcal{D} \cup\{(x, y)\}$. Note that in our discussion on decision theory in Chapter 5 , we notated this updated dataset with the symbol $\mathcal{D}_{1}$, as we needed to be able to distinguish between datasets after the incorporation of a variable number of additional observations. As our focus in this chapter will be on one-step lookahead, we can simplify notation by dropping subscripts indicating time. Instead, we will systematically use the prime symbol to indicate future quantities after the acquisition of the next observation.

In one-step lookahead, we evaluate a proposed point $x$ via the expected marginal gain in utility after incorporating an observation there (5.8):

$$
\alpha(x ; \mathcal{D})=\mathbb{E}\left[u\left(\mathcal{D}^{\prime}\right) \mid x, \mathcal{D}\right]-u(\mathcal{D}),
$$

which serves as an acquisition function inducing preferences over possible observation locations. We design each observation by maximizing this score:

$$
x \in \underset{x^{\prime} \in \mathcal{X}}{\arg \max } \alpha\left(x^{\prime} ; \mathcal{D}\right) .
$$

When the utility function $u(\mathcal{D})$ represents the expected utility of a decision informed by the data, such as a terminal recommendation following optimization, the expected marginal gain is also known as the value of sample information from observing at $x$. This term originates from the study of decision making in an economic context. Consider 

\begin{tabular}{ll}
\hline utility function, $u(\mathcal{D})$ & expected one-step marginal gain \\
\hline simple reward, (6.3) & expected improvement, § 7.3 \\
global reward, (6.5) & knowledge gradient, § 7.4 \\
improvement to simple reward & probability of improvement, § 7.5 \\
information gain, (6.8) or (6.9) & mutual information, § 7.6 \\
cumulative reward, (6.7) & posterior mean, § 7.10 \\
\hline
\end{tabular}

an agent who must make a decision under uncertainty, and suppose they have access to a third party who is willing to provide potentially insightful advice in exchange for a fee. By reasoning about the potential impact of this advice on the ultimate decision, we may quantify the expected value of the information, ${ }^{8,9}$ and determine whether the offered advice is worth the investment.

Due to its simplicity and inherent computational efficiency, one-step lookahead is a pervasive approximation scheme in Bayesian optimization. Table 7.1 provides a list of common acquisition functions, each representing the expected one-step marginal gain to a corresponding utility function. We will discuss each in detail below.

\subsection*{EXPECTED IMPROVEMENT}

Adopting the simple reward utility function (6.3) and performing onestep lookahead defines the expected improvement acquisition function. Sequential maximization of expected improvement is perhaps the most widespread policy in all of Bayesian optimization.

Suppose that we wish to locate a single location in the domain with the highest possible objective value and ultimately wish to recommend one of the points investigated during optimization for permanent use. The simple reward utility function evaluates a dataset $\mathcal{D}$ precisely by the expected value of an optimal final recommendation informed by the data, assuming risk neutrality:

$$
u(\mathcal{D})=\max \mu_{\mathcal{D}}(\mathbf{x}) .
$$

Suppose we have already gathered observations $\mathcal{D}=(\mathbf{x}, \mathbf{y})$ and wish to choose the next evaluation location. Expected improvement is derived by measuring the expected marginal gain in utility, or the instantaneous improvement, $u\left(\mathcal{D}^{\prime}\right)-u(\mathcal{D}),{ }^{10}$ offered by making the next observation at a proposed location $x:^{11}$

$$
\alpha_{\mathrm{EI}}(x ; \mathcal{D})=\int\left[\max \mu_{\mathcal{D}^{\prime}}\left(\mathbf{x}^{\prime}\right)\right] p(y \mid x, \mathcal{D}) \mathrm{d} y-\max \mu_{\mathcal{D}}(\mathbf{x}) .
$$

Expected improvement reduces to a particularly nice expression in the case of exact observations of the objective, where the utility takes a simpler form (6.4). Suppose that, when we elect to make an observation at a location $x$, we observe the exact objective value $\phi=f(x)$. Consider
Table 7.1: Summary of one-step lookahead optimization policies.

8 J. MARSCHAK and R. RADNER (1972). Economic Theory of Teams. Yale University Press. [§ 2.12]

9 H. RAIFFA and R. SCHLAIFER (1961). Applied Statistical Decision Theory. Division of Research, Graduate School of Business Administration, Harvard University. [§ 4.5]

simple reward: §6.1, p. 109

risk neutrality: §6.1, p. 109

10 This reasoning is the same for all one-step lookahead policies, which could all be described as maximizing "expected improvement." But this name has been claimed for the simple reward utility alone.

11 As mentioned in the last chapter, simple reward degenerates with an empty dataset; expected improvement does as well. In that case we can simply ignore the second term and compute the first, which for zero-mean additive noise becomes the mean function of the prior process.

expected improvement without noise 

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-06.jpg?height=419&width=1630&top_left_y=453&top_left_x=156)

Figure 7.3: The expected improvement acquisition function (7.2) corresponding to our running example.

maximal value observed, incumbent $\phi^{*}$

12 The value $\phi^{*}$ is incumbent as it is currently "holding office" as our standing recommendation until it is deposed by a better candidate.

simulated optimization and interpretation

exploitative behavior resulting from myopia a dataset $\mathcal{D}=(\mathbf{x}, \boldsymbol{\phi})$, and define $\phi^{*}=\max \phi$ to be the so-called incumbent: the maximal objective value yet seen. ${ }^{12}$ As a consequence of exact observation, we have

and thus

$$
u(\mathcal{D})=\phi^{*} ; \quad u\left(\mathcal{D}^{\prime}\right)=\max (\phi, \phi)
$$

$$
u\left(\mathcal{D}^{\prime}\right)-u(\mathcal{D})=\max (\phi-\phi, 0)
$$

Substituting into (7.2), in the noiseless case we have

$$
\alpha_{\mathrm{EI}}(x ; \mathcal{D})=\int \max (\phi-\phi, 0) p(\phi \mid x, \mathcal{D}) \mathrm{d} \phi .
$$

Expected improvement is illustrated for our running example in Figure 7.3. In this case, maximizing expected improvement will select a point near the previous best point found, an example of exploitation. Notice that the expected improvement vanishes near regions where we have existing observations. Although these locations may be likely to yield values higher than $\phi^{*}$ due to relatively high expected value, the relatively narrow credible intervals suggest that the magnitude of any improvement is likely to be small. Expected improvement is thus considering the exploration-exploitation dilemma in the selection of the next observation location, and the tradeoff between these two concerns is considered automatically.

Figure 7.4 shows the posterior belief of the objective after sequentially maximizing expected improvement to gather 20 additional observations of our example objective function. The global optimum was efficiently located. The distribution of the sample locations, with more evaluations in the most promising regions, reflects consideration of the explorationexploitation dilemma. However, there seems to have been a focus on exploitation throughout the entire process; the first ten observations for example never strayed from the initially known local optimum. This behavior is a reflection of the simple reward utility function underlying the policy, which only rewards the discovery of high objective function values at observed locations. As a result, one-step lookahead may 

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-07.jpg?height=765&width=1648&top_left_y=463&top_left_x=267)

Figure 7.4: The posterior after 10 (top) and 20 (bottom) steps of the optimization policy induced by the expected improvement acquisition function (7.2) on our running example. The tick marks show the points chosen by the policy, progressing from top to bottom, during iterations 1-10 (top) and 11-20 (bottom). Observations within 0.2 length scales of the optimum have thicker marks; the optimum was located on iteration 19.

rationally choose to make marginal improvements to the value of the best-seen point, even if the underlying function value is known with a fair amount of confidence.

\subsection*{KNOWLEDGE GRADIENT}

Adopting the global reward utility (6.5) and performing one-step-lookahead yields an acquisition function known as the knowledge gradient.

Assume that, just as in the situation leading to the derivation of expected improvement, we again wish to identify a single point in the domain maximizing the objective function. However, imagine that at termination we are willing to commit to a location possibly never evaluated during optimization. To this end, we adopt the global reward utility function to measure our progress:

$$
u(\mathcal{D})=\max _{x \in \mathcal{X}} \mu_{\mathcal{D}}(x),
$$

which rewards data for increasing the posterior mean, irrespective of location. Computing the one-step marginal gain to this utility results in the knowledge gradient acquisition function:

$$
\alpha_{\mathrm{KG}}(x ; \mathcal{D})=\int\left[\max _{x^{\prime} \in \mathcal{X}} \mu_{\mathcal{D}^{\prime}}\left(x^{\prime}\right)\right] p(y \mid x, \mathcal{D}) \mathrm{d} y-\max _{x^{\prime} \in \mathcal{X}} \mu_{\mathcal{D}}\left(x^{\prime}\right) .
$$

The knowledge gradient moniker was coined by FRAZIER and POwELL, ${ }^{13}$ who interpreted the global reward as the amount of "knowledge" global reward: §6.1, p. 109

13 P. FRAZiER and w. POWELL (2007). The Knowledge Gradient Policy for Offline Learning with Independent Normal Rewards. ADPRL 2007. 

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-08.jpg?height=423&width=1630&top_left_y=454&top_left_x=156)

Figure 7.5: The knowledge gradient acquisition function (7.4) corresponding to our running example.

Figure 7.6: Samples of the updated posterior mean when evaluating at the location chosen by the knowledge gradient, illustrated in Figure 7.5. Only the right-hand section of the domain is shown.

example and interpretation

reason for selected observation about the global maximum offered by a dataset $\mathcal{D}$. The knowledge gradient $\alpha_{\mathrm{KG}}(x ; \mathcal{D})$ can then be interpreted as the expected change in knowledge (that is, a discrete-time gradient) offered by a measurement at $x$.

The knowledge gradient is illustrated for our running example in Figure 7.5. Perhaps surprisingly, the chosen observation location is remarkably close to the previously best-seen point. At first glance, this may seem wasteful, as we are already fairly confident about the value we might observe.

However, the knowledge gradient seeks to maximize the global maximum of the posterior mean, regardless of its location. With this in mind, we may reason as follows. There must be a local maximum of the objective function in the neighborhood of the best-seen point, but our current knowledge is insufficient to pinpoint its location. Further, as the relevant local maximum is probably not located precisely at this point, the objective function is either increasing or decreasing as it passes through. If we were to learn the derivative of the objective at this point, we would adjust our posterior belief to reflect that knowledge. Regardless of the sign or exact value of the derivative, our updated belief would reflect the discovery of a new, higher local maximum of the posterior mean in the indicated direction. By evaluating at the location selected by the knowledge gradient, we can effectively estimate the derivative of the objective; this is the principle behind finite differencing.

In Figure 7.6, we show samples of the updated posterior mean function $\mu_{\mathcal{D}^{\prime}}(x)$ derived from sampling from the predictive distribution at the chosen evaluation location and conditioning. Indeed, these samples exhibit newly located global maxima on either side of the selected point, depending on the sign of the implied derivative. Note that the locations 
![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-09.jpg?height=778&width=1632&top_left_y=463&top_left_x=269)

Figure 7.7: The posterior after 10 (top) and 20 (bottom) steps of the optimization policy induced by the knowledge gradient acquisition function (7.4) on our running example. The tick marks show the points chosen by the policy, progressing from top to bottom, during iterations 1-10 (top) and 11-20 (bottom). Observations within 0.2 length scales of the optimum have thicker marks; the optimum was located on iteration 15 .

of these new maxima coincide with local maxima of the expected improvement acquisition function; see Figure 7.3 for comparison. This is not a coincidence! One way to interpret this relation is that, due to rewarding large values of the posterior mean at observed locations only, expected improvement must essentially guess on which side the hidden local optimum of the objective lies and hope to be correct. The knowledge gradient, on the other hand, considers identifying this maximum on either side a success, and guessing is not necessary.

Figure 7.7 illustrates the behavior of the knowledge gradient policy on our example optimization scenario. The global optimum was located efficiently. Comparing the decisions made by the knowledge gradient to those made by expected improvement (see Figure 7.4), we can observe a somewhat more even exploration of the domain, including in local maxima. The knowledge gradient policy does not necessarily need to expend observations to verify a suspected maximum, instead putting more trust into the model to have correct beliefs in these regions.

\section*{PROBABILITY OF IMPROVEMENT}

As its name suggests, the probability of improvement acquisition function computes the probability of an observed value to improve upon some chosen threshold, regardless of the magnitude of this improvement. simulated optimization and interpretation

more exploration than expected improvement from more-relaxed utility 

- - improvement target
$-p(\phi \mid x, \mathcal{D})$
$-p\left(\phi^{\prime} \mid x^{\prime}, \mathcal{D}\right)$

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-10.jpg?height=314&width=1062&top_left_y=460&top_left_x=720)

Figure 7.8: An illustrative example comparing the behavior of probability of improvement with expected improvement computed with respect to the dashed target. The predictive distributions for two points $x$ and $x^{\prime}$ are shown. The distributions have equal mean but the distribution at $x^{\prime}$ has larger predictive standard deviation. The shaded regions represent the region of improvement. The relatively safe $x$ is preferred by probability of improvement, whereas the more-risky $x^{\prime}$ is preferred by expected improvement.

simple reward: $§ 6.1$, p. 109

desired margin of improvement, $\varepsilon$ desired improvement threshold, $\tau$

utility formulation

noiseless case

comparison with expected improvement
Consider the simple reward of an already gathered dataset $\mathcal{D}=(\mathbf{x}, \mathbf{y})$ :

$$
u(\mathcal{D})=\max \mu_{\mathcal{D}}(\mathbf{x}) .
$$

The probability of improvement acquisition function scores a proposed observation location $x$ according to the probability that an observation there will improve this utility by at least some margin $\varepsilon$. Let us denote the desired utility threshold with $\tau=u(\mathcal{D})+\varepsilon$; we will use both the absolute threshold $\tau$ and the marginal threshold $\varepsilon$ in the following discussion as convenient. The probability of improvement is then the probability that the updated utility $u\left(\mathcal{D}^{\prime}\right)$ exceeds the chosen threshold:

$$
\alpha_{\mathrm{PI}}(x ; \mathcal{D}, \tau)=\operatorname{Pr}\left(u\left(\mathcal{D}^{\prime}\right)>\tau \mid x, \mathcal{D}\right) .
$$

We may interpret probability of improvement in the Bayesian decisiontheoretic framework as computing the expected one-step marginal gain in a peculiar choice of utility function: a utility offering unit reward for each observation increasing the simple reward by the desired amount.

In the case of exact observation, we have

$$
u(\mathcal{D})=\max f(\mathbf{x})=\phi^{*} ; \quad u\left(\mathcal{D}^{\prime}\right)=\max \left(\phi^{*} \phi\right),
$$

and we may write the probability of improvement in the somewhat simpler form

$$
\alpha_{\mathrm{PI}}(x ; \mathcal{D}, \tau)=\operatorname{Pr}(\phi>\tau \mid x, \mathcal{D}) .
$$

In this case, the probability of improvement is simply the complementary cumulative distribution function of the predictive distribution evaluated at the improvement threshold $\tau$. This form of probability of improvement is sometimes encountered in the literature, but our modification in terms of the simple reward allows for inexact observations as well.

It can be illustrative to compare the preferences over observation locations implied by the probability of improvement and expected improvement acquisition functions. In general, probability of improvement is somewhat more risk-averse than expected improvement, because probability of improvement would prefer a certain improvement of modest 

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-11.jpg?height=851&width=1630&top_left_y=451&top_left_x=270)

Figure 7.9: The probability of improvement acquisition function (7.5) corresponding to our running example for different values of the target improvement $\varepsilon$. The target is expressed as a fraction of the range of the posterior mean over the space. Increasing the target improvement leads to increasingly exploratory behavior.

magnitude to an uncertain improvement of potentially large magnitude. Figure 7.8 illustrates this phenomenon. Shown are the predictive distributions for the objective function values at two points $x$ and $x$. Both points have equal predictive means; however, $x^{\prime}$ has a significantly larger predictive standard deviation. We consider improvement with respect to the illustrated target. The shaded regions represent the regions of improvement; the probability mass of these regions equal the probabilities of improvement. Improvement is near certain at $x\left(\alpha_{\mathrm{PI}}=99.9 \%\right)$, whereas it is somewhat smaller at $x^{\prime}\left(\alpha_{\mathrm{PI}}=72.6 \%\right)$, and thus probability of improvement would prefer to observe at $x$. The expected improvement at $x$, however, is small compared to $x^{\prime}$ with its longer tail:

$$
\frac{\alpha_{\mathrm{EI}}\left(x^{\prime} ; \mathcal{D}\right)}{\alpha_{\mathrm{EI}}(x ; \mathcal{D})}=1.28
$$

The expected improvement at $x^{\prime}$ is $28 \%$ larger than at $x$, indicating a preference for a less-certain but potentially larger payout.

\section*{The role of the improvement target}

The magnitude of the required improvement plays a crucial role in shaping the behavior of probability of improvement policies. By adjusting this parameter, we may encourage exploration (with large $\varepsilon$ ) or exploitation (with small $\varepsilon$ ). Figure 7.9 shows the probability of improvement for 
![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-12.jpg?height=782&width=1650&top_left_y=461&top_left_x=154)

Figure 7.10: The posterior after 10 (top) and 20 (bottom) steps of the optimization policy induced by probability of improvement with $\varepsilon=0.1\left[\max \mu_{\mathcal{D}}(x)-\min \mu_{\mathcal{D}}(x)\right]$ (7.5) on our running example. The tick marks show the points chosen by the policy, progressing from top to bottom, during iterations 1-10 (top) and 11-20 (bottom). Observations within 0.2 length scales of the optimum have thicker marks; the optimum was located on iteration 15 . provement, a modest improvement, and a significant improvement. The shift toward exploratory behavior for larger improvement thresholds can be clearly seen.

In Figure 7.10, we see 20 evaluations chosen by maximizing probability of improvement with the target dynamically set to $10 \%$ of the range of the posterior mean function. The global optimum was located, and the domain appears sufficiently explored. Although performance was quite reasonable here, the improvement threshold was set somewhat arbitrarily, and it is not always clear how one should set this parameter.

On one extreme, some authors define a parameter-free (and probably too literal) version of probability of improvement by fixing the improvement target to $\varepsilon=0$, rewarding even infinitesimal improvement to the current data. Intuitively, this low bar can induce overly exploitative behavior. Examining the probability of improvement with $\varepsilon=0$ for our running example in Figure 7.9, we see that the acquisition function is maximized directly next to the previously best-found point. This decision represents extreme exploitation and potentially undesirable behavior. The situation after applying probability of improvement with $\varepsilon=0$ to select 20 additional observation locations, shown in Figure 7.11, clearly demonstrates a drastic focus on exploitation. Notably, the global optimum was not identified. 

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-13.jpg?height=475&width=1632&top_left_y=462&top_left_x=266)

Figure 7.11: The posterior after 20 steps of the optimization policy induced by probability of improvement with $\varepsilon=0$ (7.5) on our running example. The tick marks show the points chosen by the policy, progressing from top to bottom.

Evidently we must carefully select the desired improvement threshold to achieve ideal behavior. JONEs provided some simple, data-driven advice for choosing improvement thresholds that remains sound. ${ }^{14}$ Define

$$
\mu^{*}=\max _{x \in \mathcal{X}} \mu_{\mathcal{D}}(x) ; \quad r=\max \mu_{\mathcal{D}}(\mathbf{x})-\min \mu_{\mathcal{D}}(\mathbf{x})
$$

to represent the global maximum of the posterior mean and the range of the posterior mean at the observed locations. JONEs suggests considering targets of the form

$$
\mu^{*}+\alpha r
$$

where $\alpha \geq 0$ controls the amount of desired improvement in terms of the range of observed data. He provides a table of 27 suggested values for $\alpha$ in the range $[0,3]$ and remarks that the points optimizing the set of induced acquisition functions typically cluster together in a small number of locations, each representing a different tradeoff between exploration and exploitation. ${ }^{15}$ JONES continued to recommend selecting one point from each of these clusters to evaluate in parallel, defining a batch optimization policy. Although this may not always be possible, the recommended parameterization of the desired improvement is natural and would be appropriate for general use.

This proposal is illustrated for our running example in Figure 7.12. We begin with the posterior after selecting 10 points in our previous demo (see Figure 7.10), and indicate the points maximizing the probability of improvement for Jones's proposed improvement targets. The points cluster together in four regions reflecting varying exploration-exploitation tradeoffs.

\section*{MUTUAL INFORMATION AND ENTROPY SEARCH}

A family of information-theoretic optimization policies have been proposed in recent years, most with variations on the name entropy search.
14 D. R. JONEs (2001). A Taxonomy of Global Optimization Methods Based on Response Surfaces. fournal of Global Optimization 21(4):345383 .

15 The proposed values for $\alpha$ given by JONES are compiled below.

\begin{tabular}{lll}
\multicolumn{3}{c}{$\alpha$} \\
\hline 0 & 0.07 & 0.25 \\
0.0001 & 0.08 & 0.3 \\
0.001 & 0.09 & 0.4 \\
0.01 & 0.1 & 0.5 \\
0.02 & 0.11 & 0.75 \\
0.03 & 0.12 & 1 \\
0.04 & 0.13 & 1.5 \\
0.05 & 0.15 & 2 \\
0.06 & 0.2 & 3
\end{tabular}



![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-14.jpg?height=267&width=1628&top_left_y=495&top_left_x=160)

Figure 7.12: The points maximizing probability of improvement using the 27 improvement thresholds proposed by JONES, beginning with the posterior from Figure 7.10 after 10 total observations have been obtained. The tick marks show the chosen points and cluster together in four regions representing different tradeoffs between exploration and exploitation.

16 T. M. COVER and J. A. THOMAS (2006). Elements of Information Theory. John Wiley \& Sons.

17 D. J. C. MACKAY (2003). Information Theory, Inference, and Learning Algorithms. Cambridge University Press.

information-theoretic decision making as a model of the scientific method

information gain: $§ 6.3$, p. 115

18 D. V. LINDLEY (1956). On a Measure of the Information Provided by an Experiment. The Annals of Mathematical Statistics 27(4):986-1005.

19 B. Settles (2012). Active Learning. Morgan \& Claypool.
The acquisition function in these methods is mutual information, a measure of dependence between random variables that is a central concept in information theory. ${ }^{16,17}$

The reasoning underlying entropy search policies is somewhat different from and more general than the other acquisition functions we have considered thus far, all of which ultimately focus on maximizing the posterior mean function. Although this is a pragmatic concern, it is intimately linked to optimization. Information-theoretic experimental design is instead motivated by an abstract pursuit of knowledge, and may be interpreted as a mathematical formulation of the scientific method.

We begin by identifying some unknown feature of the world that we wish to learn about; in the context of Bayesian inference, this will be some random variable $\omega$. We then view each observation we make as an opportunity to learn about this random variable, and seek to gather data that will, in aggregate, provide considerable information about $\omega$. This process is analogous to a scientist designing a sequence of experiments to understand some natural phenomenon, where each experiment may be chosen to challenge or confirm constantly evolving beliefs.

The framework of information theory allows us to formalize this process. We may quantify the amount of information provided about a random variable $\omega$ by a dataset $\mathcal{D}$ via the information gain, a concept for which we provided two definitions in the last chapter. Adopting either definition as a utility function and performing one-step lookahead yields mutual information as an acquisition function.

Information-theoretic optimization policies select $\omega$ such that its determination gives insight into our optimization problem (1.1). However, by selecting different choices for $\omega$, we can generate radically different policies, each attempting to learn about a different aspect of the system of interest. Maximizing mutual information has long been promoted as a general framework for optimal experimental design,${ }^{18}$ and this framework has been applied in numerous active learning settings. ${ }^{19}$

Before showing how mutual information arises in a decision-theoretic context, we pause to define the concept and derive some important properties. 

\section*{Mutual information}

Let $\omega$ and $\psi$ be random variables with probability density functions $p(\omega)$ and $p(\psi)$. The mutual information between $\omega$ and $\psi$ is

$$
I(\omega ; \psi)=\iint p(\omega, \psi) \log \frac{p(\omega, \psi)}{p(\omega) p(\psi)} \mathrm{d} \omega \mathrm{d} \psi .
$$

This expression may be recognized as the Kullback-Leibler divergence between the joint distribution of the random variables and the product of their marginal distributions:

$$
I(\omega ; \psi)=D_{\mathrm{KL}}[p(\omega, \psi) \| p(\omega) p(\psi)] .
$$

We may extend this definition to conditional probability distributions as well. Given an arbitrary set of observed data $\mathcal{D}$, we define the conditional mutual information between $\omega$ and $\psi$ by: $:^{20}$

$$
I(\omega ; \psi \mid \mathcal{D})=\iint p(\omega, \psi \mid \mathcal{D}) \log \frac{p(\omega, \psi \mid \mathcal{D})}{p(\omega \mid \mathcal{D}) p(\psi \mid \mathcal{D})} \mathrm{d} \omega \mathrm{d} \psi .
$$

Here we have simply conditioned all distributions on the data and applied the definition in (7.7) to the posterior beliefs.

Several properties of mutual information are immediately evident from its definition. First, mutual information is symmetric in its arguments:

$$
I(\omega ; \psi)=I(\psi ; \omega) .
$$

We also have that if $\omega$ and $\psi$ are independent, then $p(\omega, \psi)=p(\omega) p(\psi)$ and the mutual information is zero:

$$
I(\omega ; \psi)=\iint p(\omega, \psi) \log \frac{p(\omega) p(\psi)}{p(\omega) p(\psi)} \mathrm{d} \omega \mathrm{d} \psi=0 .
$$

Further, recognition of mutual information as a Kullback-Leibler divergence implies several additional inherited properties, including nonnegativity. Thus mutual information attains its minimal value when $\omega$ and $\psi$ are independent.

We may also manipulate (7.7) by twice applying the identity

$$
p(\omega, \psi)=p(\psi) p(\omega \mid \psi)
$$

to derive an equivalent expression for the mutual information:

$$
\begin{aligned}
I(\omega ; \psi) & =\iint p(\omega, \psi) \log \frac{p(\omega, \psi)}{p(\omega) p(\psi)} \mathrm{d} \omega \mathrm{d} \psi \\
& =\iint p(\omega, \psi) \log p(\omega \mid \psi) \mathrm{d} \omega \mathrm{d} \psi-\int p(\omega) \log p(\omega) \mathrm{d} \omega \\
& =\int p(\psi)\left[\int p(\omega \mid \psi) \log p(\omega \mid \psi) \mathrm{d} \omega\right] \mathrm{d} \psi+H[\omega] \\
& =H[\omega]-\mathbb{E}[H[\omega \mid \psi]] .
\end{aligned}
$$

definition

conditional mutual information, $I(\omega ; \psi \mid \mathcal{D})$

20 Some authors use the notation $I(\omega ; \psi \mid \mathcal{D})$ to represent the expectation of the given quantity with respect to the dataset $\mathcal{D}$. In optimization, we will always have an explicit dataset in hand, in which case the provided definition is more useful.

symmetry

nonnegativity

expected reduction in entropy 21 It is important to note that this is true only in expectation. Consider two random variables $x$ and $y$ with the following joint distribution. $x$ takes value 0 or 1 with probability $1 / 2$ each. If $x$ is $0, y$ takes value 0 or 1 with probability $1 / 2$ each. If $x$ is $1, y$ takes value 0 or -1 with probability $1 / 2$ each. The entropy of $x$ is 1 bit and the entropy of $y$ is 1.5 bits. Observing $x$ always yields 0.5 bits about $y$. However, observing $y$ produces either no information about $x(0$ bits), with probability $1 / 2$, or complete information about $x$ (1 bit), with probability $1 / 2$. So the information gain about $x$ from $y$ and about $y$ from $x$ is actually never equal. How ever, the expected information gain is equal, $I(x ; y)=0.5$ bits
22 Setting $u(\mathcal{D})=D_{\text {кL }}[p(\omega \mid \mathcal{D}) \| p(\omega)]$, we have:

$$
\begin{aligned}
& \mathbb{E}\left[u\left(\mathcal{D}^{\prime}\right) \mid x, \mathcal{D}\right] \\
& =\mathbb{E}\left[\int p\left(\omega \mid \mathcal{D}^{\prime}\right) \log p\left(\omega \mid \mathcal{D}^{\prime}\right) \mathrm{d} \omega \mid x, \mathcal{D}\right] \\
& \quad-\int p(\omega \mid \mathcal{D}) \log p(\omega) \mathrm{d} \omega \\
& =-\mathbb{E}\left[H\left[\omega \mid x, \mathcal{D}^{\prime}\right] \mid \mathcal{D}\right] \\
& \quad-\int p(\omega \mid \mathcal{D}) \log p(\omega) \mathrm{d} \omega .
\end{aligned}
$$

Here the second term is known as the cross entropy between $p(\omega)$ and $p(\omega \mid \mathcal{D})$. We can also rewrite the utility in similar terms:

$$
\begin{aligned}
u(\mathcal{D})= & \int p(\omega \mid \mathcal{D}) \frac{\log p(\omega \mid \mathcal{D})}{\log p(\omega)} \mathrm{d} \omega \\
= & \int p(\omega \mid \mathcal{D}) \log p(\omega \mid \mathcal{D}) \mathrm{d} \omega \\
& \quad-\int p(\omega \mid \mathcal{D}) \log p(\omega) \mathrm{d} \omega \\
= & -H[\omega \mid \mathcal{D}] \\
& \quad-\int p(\omega \mid \mathcal{D}) \log p(\omega) \mathrm{d} \omega .
\end{aligned}
$$

If we subtract, the cross-entropy terms cancel and we obtain mutual information:

$$
\begin{aligned}
& \mathbb{E}\left[u\left(\mathcal{D}^{\prime}\right) \mid x, \mathcal{D}\right]-u(\mathcal{D})= \\
& \quad H[\omega \mid \mathcal{D}]-\mathbb{E}\left[H\left[\omega \mid \mathcal{D}^{\prime}\right] \mid \mathcal{D}\right] .
\end{aligned}
$$

Thus the mutual information between $\omega$ and $\psi$ is the expected decrease in the differential entropy of $\omega$ if we were to observe $\psi$. Due to symmetry (7.8), we may swap the roles of $\omega$ and $\psi$ to derive an equivalent expression in the other direction:

$$
I(\omega ; \psi)=H[\omega]-\mathbb{E}_{\psi}[H[\omega \mid \psi]]=H[\psi]-\mathbb{E}_{\omega}[H[\psi \mid \omega]] .
$$

Observing either $\omega$ or $\psi$ will, in expectation, provide the same amount of information about the other: the mutual information $I(\omega ; \psi) .^{21}$

\section*{Maximizing mutual information as an optimization policy}

Mutual information arises naturally in Bayesian sequential experimental design as the one-step expected information gain resulting from an observation. In the previous chapter, we introduced two different methods for quantifying this information gain. The first was the reduction in the differential entropy of $\omega$ from the prior to the posterior:

$$
\begin{aligned}
u(\mathcal{D}) & =H[\omega]-H[\omega \mid \mathcal{D}] \\
& =\int p(\omega \mid \mathcal{D}) \log p(\omega \mid \mathcal{D}) \mathrm{d} \omega-\int p(\omega) \log p(\omega) \mathrm{d} \omega .
\end{aligned}
$$

The second was the Kullback-Leibler divergence between the posterior and the prior:

$$
u(\mathcal{D})=D_{\text {КL }}[p(\omega \mid \mathcal{D}) \| p(\omega)]=\int p(\omega \mid \mathcal{D}) \log \frac{p(\omega \mid \mathcal{D})}{p(\omega)} \mathrm{d} \omega .
$$

Remarkably, performing one-step lookahead with either choice yields mutual information as an acquisition function.

Let us first compute the expected marginal gain in (7.11). In this case the marginal information gain is:

$$
H[\omega \mid \mathcal{D}]-H\left[\omega \mid \mathcal{D}^{\prime}\right],
$$

and the expected marginal information gain is then:

$$
\begin{aligned}
\alpha_{\mathrm{MI}}(x ; \mathcal{D}) & =H[\omega \mid \mathcal{D}]-\mathbb{E}\left[H\left[\omega \mid \mathcal{D}^{\prime}\right] \mid x, \mathcal{D}\right] \\
& =I(y ; \omega \mid x, \mathcal{D}),
\end{aligned}
$$

where we have recognized the expected reduction in entropy in (7.13) as the mutual information between $y$ and $\omega$ given the putative location $x$ and the available data $\mathcal{D}(7.9)$. It is simple to verify that the expected marginal improvement to the alternative information gain definition (7.12) gives the same expression; several terms cancel when computing the expectation, and those that remain are identical to those in $(7.13){ }^{22}$

Due to the symmetry of mutual information, we have several equivalent forms for this acquisition function (7.10):

$$
\begin{aligned}
\alpha_{\mathrm{MI}}(x ; \mathcal{D}) & =I(y ; \omega \mid x, \mathcal{D}) \\
& =H[\omega \mid \mathcal{D}]-\mathbb{E}_{y}\left[H\left[\omega \mid \mathcal{D}^{\prime}\right] \mid x, \mathcal{D}\right] \\
& =H[y \mid x, \mathcal{D}]-\mathbb{E}_{\omega}[H[y \mid \omega, x, \mathcal{D}] \mid x, \mathcal{D}]
\end{aligned}
$$



![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-17.jpg?height=518&width=1586&top_left_y=452&top_left_x=266)

Figure 7.13: The posterior belief about the location of the global optimum, $p\left(x^{*} \mid \mathcal{D}\right)$, and about the value of the global optimum, $p\left(f^{*} \mid \mathcal{D}\right)$, for our running example. Note the significant probability mass associated with the optimum lying on the boundary.

Depending on the application, one of these two forms may be preferable, and maximizing either results in the same policy.

Adopting mutual information as an acquisition function for optimization requires that $\omega$ be selected to support the optimization task. Two natural options present themselves: the location of the global optimum, $x^{*}$, and the maximum value attained, $f^{*}=f\left(x^{*}\right)(1.1)$. Both have received extensive consideration, and we will discuss each in turn.

\section*{Mutual information with $x^{*}$}

Several authors have proposed mutual information with the location of the global optimum $x^{*}$ as an acquisition function: ${ }^{23}$

$$
\alpha_{x^{*}}(x ; \mathcal{D})=I\left(y ; x^{*} \mid x, \mathcal{D}\right) .
$$

The distribution of $x^{*}$ is illustrated for our running example in Figure 7.13. Even for this simple example, the distribution of the global optimum is nontrivial and multimodal. In fact, in this case, there is a significant probability that the global maximum occurs on the boundary of the domain, which has Lebesgue measure zero, so $x^{*}$ does not even have a proper probability density function. ${ }^{24}$ We will nonetheless use the notation $p\left(x^{*} \mid \mathcal{D}\right)$ in our discussion below.

The middle panel of Figure 7.14 shows the mutual information with $x^{*}$ (7.16) for our running example. The next evaluation location will be chosen to investigate the neighborhood of the best-seen point. It is interesting to compare the behavior of the mutual information with other acquisition functions near the boundary of the domain. Although there is a significant probability that the maximum is achieved on the boundary, mutual information indicates that we cannot expect to reveal much information regarding $x^{*}$ by measuring there. More information tends to be revealed by evaluating away from the boundary, as we can reduce our

23 We tacitly assume the location of the global optimum is unique to simplify discussion. Technically $x^{*}$ is a set-valued random variable. For Gaussian process models, the uniqueness of $x^{*}$ can be guaranteed under mild assumptions (§ 2.7, p. 34).

24 A proper treatment would separate the probability density on the interior of $\mathcal{X}$ from the distribution restricted to the boundary, but in practice we will only ever be sampling from this distribution, as it is simply too complicated to work with directly. 

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-18.jpg?height=642&width=1647&top_left_y=450&top_left_x=153)

Figure 7.14: The mutual information between the observed value and the location of the global optimum, $\alpha_{x^{*}}$ (middle panel), and between the observed value and the value of the global optimum, $\alpha_{f^{*}}$ (bottom panel), for our running example.

25 Again we assume in this discussion that a global optimum $f^{*}$ exists almost surely. This assumption can be guaranteed for Gaussian process models under mild assumptions $(\S 2.7$, p. 34). uncertainty about the objective function over a larger volume. Expected improvement (Figure 7.3) and probability of improvement (Figure 7.9), on the other hand, are computed only from inspection of the marginal predictive distribution $p(y \mid x, \mathcal{D})$. As a result, they cannot differentiate observation locations based on their global impact on our belief.

In Figure 7.15, we demonstrate optimization by maximizing the mutual information with $x^{*}$ (7.16) for our scenario. We also show the posterior belief about the maximum location at termination, $p\left(x^{*} \mid \mathcal{D}\right)$. The global optimum was discovered efficiently and with remarkable confidence. Further, the posterior mode matches the true optimal location.

\section*{Mutual information with $f^{*}$}

Mutual information with the value of the global optimum $f^{*}$ has also been investigated as an acquisition function: ${ }^{25}$

$$
\alpha_{f^{*}}(x ; \mathcal{D})=I\left(y ; f^{*} \mid x, \mathcal{D}\right) .
$$

The distribution of this quantity is illustrated for our running example in Figure 7.13. There is a sharp mode in the distribution corresponding to the best-seen value in fact being near-optimal, and there is no mass below the best-seen value, as it serves as a lower bound on the maximum due to the assumption of exact observation.

The bottom panel of Figure 7.14 shows the mutual information with $f^{*}(7.17)$ for our running example. The next evaluation location will be chosen to investigate the neighborhood of the best-seen point. It is interesting to contrast this surface with that of the mutual information with $x^{*}$ in Figure 7.16. Mutual information with $f^{*}$ is heavily penalized near 
![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-19.jpg?height=976&width=1650&top_left_y=464&top_left_x=266)

Figure 7.15: The posterior after 10 (top) and 20 (bottom) steps of maximizing the mutual information between the observed value $y$ and the location of the global maximum $x^{*}(7.16)$ on our running example. The tick marks show the points chosen by the policy, progressing from top to bottom. Observations within 0.2 length scales of the optimum have thicker marks; the optimum was located on iteration 16.

existing observations, even those with relatively high values. Observing at these points would contribute relatively little information about the value of the optimum, as their predictive distributions already reflect a narrow range of possibilities and contribute little to the distribution of $f^{*}$ Further, points on the boundary were less favored when seeking to learn about $x$. However, these points are expected to provide just as much information about $f^{*}$ as neighboring points.

Figure 7.16 illustrates 25 evaluations chosen by sequentially maximizing the mutual information with $f^{*}$ (7.17) for our scenario, along with the posterior belief about the value of the maximum given these observations, $p\left(f^{*} \mid \mathcal{D}\right)$. The global optimum was discovered after 23 iterations, somewhat slower than the alternatives described above. The value of the optimum is known with almost complete confidence at termination.

\subsection*{MULTI-ARMED BANDITS AND OPTIMIZATION}

Several Bayesian optimization algorithms have been inspired by policies for multi-armed bandits, a model system for sequential decision making 

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-20.jpg?height=814&width=1585&top_left_y=461&top_left_x=224)

Figure 7.16: The posterior after 10 (top) and 25 (bottom) steps of maximizing the mutual information between the observed value $y$ and the location of the global maximum $f^{*}(7.17)$ on our running example. The tick marks show the points chosen by the policy, progressing from top to bottom. Observations within 0.2 length scales of the optimum have thicker marks; the optimum was located on iteration 23.

set of arms, $\mathcal{X}$

cumulative reward: §6.2, p. 114

26 W. R. тномpson (1933). On the Likelihood That One Unknown Probability Exceeds Another in View of the Evidence of Two Samples. Biometrika 25(3-4):285-294 under uncertainty. A multi-armed bandit problem can be interpreted as a particular finite-dimensional analog of sequential optimization, and effective algorithm design in both settings requires addressing many shared concerns.

The classical multi-armed bandit problem considers a finite set of "arms" $\mathcal{X}$ and an agent who must select a sequence of items from this set. Selecting an arm $x$ results in a stochastic reward $y$ drawn from an unknown distribution $p(y \mid x)$ associated with that arm; these rewards are assumed to be independent of time and conditionally independent given the chosen arm. The goal of the agent is to select a sequence of arms $\left\{x_{i}\right\}$ to maximize the cumulative reward (6.7) received, $\sum y_{i}$.

Multi-armed bandits have been studied as a model of many sequential decision processes arising in practice. For example, consider a doctor caring for a series of patients with the same condition, who must determine which of two possible treatments is the best course of action. We could model the sequence of treatment decisions as a two-armed bandit, with patient outcomes determining the rewards. The Hippocratic oath compels the doctor to discover the optimal arm (the best treatment) as efficiently and confidently as possible to minimize patient harm, creating a dilemma of effective assignment. In fact, this scenario of sequential clinical trials was precisely the original motivation for studying multi-armed bandits. $^{26}$ To facilitate the following discussion, for each arm $x \in \mathcal{X}$, we define $\phi=\mathbb{E}[y \mid x]$ to be its expected reward and will aggregate these into a vector $\mathbf{f}$ when convenient. We also define

$$
x^{*} \in \underset{x \in \mathcal{X}}{\arg \max } \mathbb{E}[y \mid x]=\arg \max \mathbf{f} ; \quad f^{*}=\max _{x \in \mathcal{X}} \mathbb{E}[y \mid x]=\max \mathbf{f}
$$

to be the index of an arm with maximal expected reward and the value of that optimal reward, respectively. ${ }^{27}$ If the reward distributions associated with each arm were known a priori, the optimal policy would be trivial: we would always select the arm with the highest expected reward. This policy generates expected reward $f^{*}$ in each iteration, and it is clear from linearity of expectation that this is optimal. Unfortunately, the reward distributions are unknown to the agent and must be learned from observations instead. This complicates policy design considerably.

The only way we can learn about the reward distributions is to allocate resources to each arm and observe the outcomes. If the reward distributions have considerable spread and/or overlap with each other, a large number of observations may be necessary before the agent can confidently conclude which arm is optimal. The agent thus faces an exploration-exploitation dilemma, constantly forced to decide whether to select an arm believed to have high expected reward (exploitation) or whether to sample an uncertain arm to better understand its reward distribution (exploration). Ideally, the agent would have a policy that efficiently explores the arms, so that in the limit of many decisions the agent would eventually allocate an overwhelming majority of resources to the best possible alternative.

Dozens of policies for the multi-armed bandit problem have been proposed and studied from both the Bayesian and frequentist perspectives, and many strong convergence results are known. ${ }^{28}$ Numerous variations on the basic formulation outlined above have also received consideration in the literature, and the interested reader may refer to one of several available exhaustive surveys for more information. ${ }^{29,30,31}$

\section*{The Bayesian optimal policy}

A multi-armed bandit is fundamentally a sequential decision problem under uncertainty, and we may derive an optimal expected-case policy following our discussion in Chapter 5 . The selection of each arm is a decision with action space $\mathcal{X}$, and we must act under uncertainty about the expected reward vector $\mathbf{f}$. Over the course of $\tau$ decisions, we will gather a dataset $\mathcal{D}_{\tau}=\left(\mathbf{x}_{\tau}, \mathbf{y}_{\tau}\right)$, seeking to maximize the cumulative reward (6.7): $u\left(\mathcal{D}_{\tau}\right)=\sum y_{i}$.

The key to the Bayesian approach is maintaining a belief about the expected reward of each arm. We begin by choosing a prior over the expected rewards, $p(\mathbf{f})$, and an observation model for the observed rewards given the index of an arm and its expected reward, $p(y \mid x, \phi){ }^{32}$ Now given an arbitrary set of previous observations $\mathcal{D}$, we may derive a posterior belief about the expected rewards, $p(\mathbf{f} \mid \mathcal{D})$. optimal policy with known rewards

27 All of the notation throughout this section is chosen to align with that for optimization. In a multi-armed bandit, an $\operatorname{arm} x$ is associated with expected reward $\phi=\mathbb{E}[y \mid x]$. In optimization with zero-mean additive noise, a point $x$ is associated with expected observed value $\phi=f(x)=\mathbb{E}[y \mid x]$.

challenges in policy design

28 We will explore these connections in our discussion on theoretical convergence results for Bayesian optimization algorithms in Chapter 10 , p. 213.

29 D. A. BERRY and B. FRISTEDT (1985). Bandit Problems: Sequential Allocation of Experiments. Chapman \& Hall.

30 S. BUBECK and N. CESA-BIANCHI (2012). Regret Analysis of Stochastic and Nonstochastic Multi-Armed Bandit Problems. Foundations and Trends in Machine Learning 5(1):1-122.

31 T. LATTIMORE and C. SZEPESVÁRI (2020). Bandit Algorithms. Cambridge University Press.

belief about expected rewards

32 This model is often conditionally independent of the arm given the expected reward, allowing the definition of a single observation model $p(y \mid \phi)$. optimal policy: § 5.2, p. 91

33 See $\S 7.10$ for an analogous result in optimization.

running time of optimal policy: §5.3, p. 99

34 R. Agrawal (1995). The Continuum-Armed Bandit Problem. SIAM fournal on Control and Optimization 33(6):1926-1951.

correlation of rewards

35 In the multi-armed bandit literature, bandits with correlated rewards are known as restless bandits, as our belief about arms may change even when they are not selected (left alone):

P. Whittle (1988). Restless Bandits: Activity Allocation in a Changing World. Fournal of Applied Probability 25(A):287-298.

cumulative vs. simple reward
The optimal policy may now be derived following our previous analysis. We make each decision by maximizing the expected reward by termination, recursively assuming optimal future behavior (5.15-5.17) Notably, the optimal decision for the last round is the arm maximizing the posterior mean reward, reflecting pure exploitation: ${ }^{33}$

$$
x_{\tau} \in \arg \max \mathbb{E}\left[\mathbf{f} \mid \mathcal{D}_{\tau-1}\right] .
$$

More exploratory behavior begins with the penultimate decision and increases with the decision horizon.

Unfortunately, the cost of computing the optimal policy increases exponentially with the horizon. We must therefore find some mechanism to design computationally efficient but empirically effective policies for use in practice. This is precisely the same situation we face in optimization!

\section*{Optimization as an infinite-armed bandit}

We may model continuous optimization as an infinite-armed bandit problem, ${ }^{34}$ and this analogy has proven fruitful. Suppose we seek to optimize an objective function $f: \mathcal{X} \rightarrow \mathbb{R}$, where the domain $\mathcal{X}$ is now infinite. We assume as usual that we can observe this function at any point $x$ of our choosing, revealing an observation $y$ with distribution $p(y \mid x, \phi) ; \phi=f(x)$. For the bandit analogy to be maximally appropriate, we will further assume that the expected value of this observation is $\phi$ : $\mathbb{E}[y \mid \phi]=\phi$; this is not unduly restrictive and is satisfied for example by zero-mean additive noise.

Assuming an evaluation budget of $\tau$ observations, we may formulate this scenario to a multi-armed bandit. We interpret each point $x \in \mathcal{X}$ as one of infinitely many arms, with each arm returning an expected reward determined by the underlying objective function value $\phi$. With some care, we may now adapt a multi-armed bandit policy to this setting.

In the traditional multi-armed bandit problem, we assume that rewards are conditionally independent given the chosen arm index. As a consequence, the reward from any selected arm provides no information about the rewards of other arms. However, this independence would render an infinite-armed bandit hopeless, as we would never be able to determine the best arm with a finite budget. Instead, we must assume that the rewards are correlated over the domain, so that each observation can potentially inform us about the rewards of every other arm. ${ }^{35}$

This assumption is natural in optimization; the objective function must reflect some nontrivial structure, or optimization would also be hopeless. In the Bayesian framework, we formalize our assumptions regarding the structure of correlations between function values by choosing an appropriate prior distribution, which we may condition on available data to form the posterior belief, $p(f \mid \mathcal{D})$. In our bandit analogy, this distribution encapsulates beliefs about the expected rewards of each arm that can be used to derive effective policies.

Why should we reduce optimization to the multi-armed bandit at all? Notably, in optimization we are usually concerned with identifying a single point in the domain maximizing the function, and variations on the simple reward directly measure progress toward this end. It may seem odd to focus on maximizing the cumulative reward, which judges a dataset based on the average value observed rather than the maximum.

These aims are not necessarily incompatible. In the limit of many observations, we hope to guarantee the best arm will be eventually identified so that we can guarantee convergence to optimal behavior. Bandit algorithms are typically analyzed in terms of their cumulative regret, the difference between the cumulative reward received and that expected from the optimal policy. If this quantity decreases sufficiently quickly, we may conclude that the optimal arm is eventually identified and selected. In our infinite-armed case, this implies the global optimum of the objective will eventually be located ${ }^{36}$ suggesting the multi-armed bandit reduction is indeed reasonable.

\subsection*{MAXIMIZING A STATISTICAL UPPER BOUND}

Effective strategies for both bandits and optimization require careful consideration of the exploration-exploitation dilemma for success. $\mathrm{Nu}$ merous bandit algorithms have been built on the unifying principle of optimism in the face of uncertaint $y,{ }^{37}$ which has proven to be an effective heuristic for balancing these concerns. The key idea is to use any available data to both estimate the expected reward of each arm and to quantify the uncertainty in these estimates. When faced with a decision, we then always select the arm that would be optimal when allowing "the benefit of the doubt": the arm with the highest plausible expected reward given the currently available information. Arms with highly uncertain reward will have a correspondingly wide range of plausible values, and this mechanism thus provides underexplored arms a so-called exploration bonus $^{38}$ commensurate with their uncertainty, encouraging exploration of plausible optimal locations.

To be more precise, assume we have gathered an arbitrary dataset $\mathcal{D}$, and consider an arbitrary point $x$. Consider the quantile function associated with the predictive distribution $p(\phi \mid x, \mathcal{D})::^{39}$

$$
q(\pi ; x, \mathcal{D})=\inf \left\{\phi^{\prime} \mid \operatorname{Pr}\left(\phi \leq \phi^{\prime} \mid x, \mathcal{D}\right) \geq \pi\right\} .
$$

We can interpret this function as a statistical upper confidence bound on $\phi$ : the value will exceed the bound only with tunable probability $1-\pi$.

As a function of $x$, we can interpret $q(\pi ; x, \mathcal{D})$ as an optimistic estimate of the entire objective function. The principle of optimism in the face of uncertainty then suggests observing where this upper confidence bound is maximized, yielding the acquisition function

$$
\alpha_{\mathrm{UCB}}(x ; \mathcal{D}, \pi)=q(\pi ; x, \mathcal{D}) .
$$

Figure 7.17 shows upper confidence bounds for our example scenario corresponding to three values of the confidence parameter $\pi$. Unlike the acquisition functions considered previously in this chapter, an upper cumulative regret and convergence: $§ 10.1$, p. 214

36 In the infinite-armed case, establishing the noregret property ( $\S 10.1$, p. 214$)$ is not sufficient to guarantee the global optimum will ever be evaluated exactly; however, we can conclude that we evaluate points achieving objective values within any desired tolerance of the maximum.

optimism in the face of uncertainty

37 A.w. MOORE and C.G. ATKeson (1993) Memory-Based Reinforcement Learning: Efficient Computation with Prioritized Sweeping. NeurIPS 1992.

38 R. S. SUTton (1990). Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming. ICML 1990.

39 The quantile function satisfies the relation that $\phi \leq q(\pi ; x, \mathcal{D})$ with probability $\pi$.

upper confidence bound

example and discussion 

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-24.jpg?height=428&width=1630&top_left_y=454&top_left_x=156)

$-\alpha_{\mathrm{UCB}}(\pi=0.95)$
![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-24.jpg?height=384&width=1628&top_left_y=915&top_left_x=157)

Figure 7.17: The upper confidence bound acquisition function (7.18) corresponding to our running example for different values of the confidence parameter $\pi$. The vertical axis for each acquisition function is shifted to the largest observed function value. Increasing the confidence parameter leads to increasingly exploratory behavior.

correspondence between certainty parameter and exploration proclivity

adjusting the exploration parameter

4 O D. R. JONES (2001). A Taxonomy of Global Optimization Methods Based on Response Surfaces. fournal of Global Optimization 21(4):345383 confidence bound need not be nonnegative, so we shift the acquisition function in these plots so that the best-seen function value intersects the horizontal axis. We can see that relatively low confidence values $(\pi=0.8)$ give little credit to locations with high uncertainty, and exploitation is heavily favored. By increasing this parameter, our actions reflect more exploratory behavior.

In Figure 7.18, we sequentially maximize the upper confidence bound on our example function to select 20 observation locations using confidence parameter $\pi=0.999$, corresponding to the bottom and most exploratory example in Figure 7.17. The global maximum was located efficiently. Notably, the observation locations chosen in the early stages of the search reflect more exploration than all the other methods we have discussed thus far.

Using an upper confidence bound policy in practice requires specifying the exploration parameter $\pi$, and it may not always be clear how best to do so. We face a similar challenge when choosing the improvement target parameter for probability of improvement, and in fact this analogy is sometimes remarkably intimate. For some models of the objective function, including Gaussian processes, a point maximizing the probability of improvement over a given threshold $\tau$ also maximizes an upper confidence bound for some confidence parameter $\pi$, and vice versa. ${ }^{40}$ Therefore the sets of points obtained by maximizing these acquisition 

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-25.jpg?height=771&width=1648&top_left_y=465&top_left_x=267)

Figure 7.18: The posterior after 10 (top) and 20 (bottom) steps of the optimization policy induced by the upper confidence bound acquisition function (7.2) on our running example. The confidence parameter was set to $\pi=0.999$. The tick marks show the points chosen by the policy, progressing from top to bottom, during iterations 1-10 (top) and 11-20 (bottom). Observations within 0.2 length scales of the optimum have thicker marks; the optimum was located on iteration 15 .

functions over the range of their respective parameters are identical. We will establish this relationship for Gaussian process models in the next chapter.

Little concrete advice is available for selecting the confidence parameter, as concerns such as model selection and calibration may have effects on the upper confidence bound that are hard to foresee and account for Most authors select relatively large values in the approximate range $\pi \in(0.98,1)$, with values of $\pi \approx 0.999$ being perhaps the most common. In line with his advice regarding probability of improvement parameter selection, JONES suggests considering a wide range of confidence values, ${ }^{41}$ reflecting different exploration-exploitation tradeoffs. Figure 7.12 illustrates this concept by maximizing the probability of improvement for a range of improvement thresholds; by the correspondence between these policies for Gaussian process models, this is also illustrative for maximizing upper confidence bounds.

The policy realized by sequentially maximizing upper confidence bounds enjoys strong theoretical guarantees. For Gaussian process models, SRINIVAs et al. proved this policy is guaranteed to effectively maximize the objective at a nontrivial rate under reasonable assumptions. ${ }^{42}$ One of these assumptions is that the confidence parameter must increase asymptotically to 1 at a particular rate. Intuitively, the reason for this growth is that our uncertainty in the objective function will typically equivalence of $\alpha_{\mathrm{PI}}$ and $\alpha_{\mathrm{UCB}}$ for GPs: $\S 8.4$, p. 170

selecting improvement threshold: § 7.5, p. 134

41 D. R. JONEs (2001). A Taxonomy of Global Optimization Methods Based on Response Surfaces. Journal of Global Optimization 21(4):345383.

Chapter 10: Theoretical Analysis, p. 213

42 N. SRINIVAS et al. (2010). Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design. ICML 2010.

These results will be discussed at length in Chapter 10, p. 213. 43 W. R. THOMPSON (1933). On the Likelihood That One Unknown Probability Exceeds Another in View of the Evidence of Two Samples. Biometrika 25(3-4):285-294. definition

mutual information with $x^{*}:$ § 7.6, p. 139

alternative interpretation: optimizing a random acquisition function

44 We give the sample a suggestive name! decrease as we continue to gather more data. As a result, we must simultaneously increase the confidence parameter to maintain a sufficient rate of exploration. This idea of slowly increasing the confidence parameter throughout optimization may be useful as a practical heuristic as well as a theoretical device.

SSON SAMPLING

In the early 2oth century, THOMPSON proposed a simple and effective stochastic policy for the multi-armed bandit problem that has come to be known as Thompson sampling. ${ }^{43}$ Faced with a set of alternatives, the basic idea is to maintain a belief about which of these options is optimal in light of available information. We then design each evaluation by sampling from this distribution, yielding an adaptive stochastic policy. This procedure elegantly addresses the exploration-exploitation dilemma: sampling observations proportional to their probability of optimality automatically encourages exploitation, while the inherent randomness of the policy guarantees constant exploration. Thompson sampling can be adopted from finite-armed bandits to continuous optimization, and has enjoyed some interest in the Bayesian optimization literature.

Suppose we are at an arbitrary stage of optimization with data $\mathcal{D}$. The key object in Thompson sampling is the posterior distribution of the location of the global maximum $x^{*} p\left(x^{*} \mid \mathcal{D}\right)$, a distribution we have already encountered in our discussion on mutual information. Whereas maximizing mutual information carefully maximizes the information we expect to receive, Thompson sampling employs a considerably simpler mechanism. We choose the next observation location by sampling from this belief, yielding a nondeterministic optimization policy:

$$
x \sim p\left(x^{*} \mid \mathcal{D}\right)
$$

At first glance, Thompson sampling appears fundamentally different from the previous policies we have discussed, which were all defined in terms of maximizing an acquisition function. However, we may resolve this discrepancy while gaining some insight: Thompson sampling in fact designs each observation by maximizing a random acquisition function.

The location of the global maximum $x^{*}$ is completely determined by the objective function $f$, and we may exploit this relationship to yield a simple two-stage implementation of Thompson sampling. We first sample a random realization of the objective function from its posterior: ${ }^{44}$

$$
\alpha_{\mathrm{TS}}(x ; \mathcal{D}) \sim p(f \mid \mathcal{D}) .
$$

We then optimize this function to yield the desired sample from $p\left(x^{*} \mid \mathcal{D}\right)$, our next observation location:

$$
x \in \underset{x^{\prime} \in \mathcal{X}}{\arg \max } \alpha_{\mathrm{TS}}\left(x^{\prime} ; \mathcal{D}\right) .
$$

From this point of view, we can interpret the sampled objective function $\alpha_{\text {Ts }}$ as an ordinary acquisition function that is maximized as usual. 

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-27.jpg?height=934&width=1630&top_left_y=464&top_left_x=270)

Figure 7.19: An illustration of Thompson sampling for our example optimization scenario. At the top, we show the objective function posterior $p(f \mid \mathcal{D})$ and three samples from this belief. Thompson sampling selects the next observation location by maximizing one of these samples. In the bottom three panels we show three possible outcomes of this process, corresponding to each of the sampled objective functions illustrated in the top panel.

Rather than representing an expected utility or a statistical upper bound, the acquisition function used in each round of Thompson sampling is a hallucinated objective function that is plausible under our belief. Whereas a Bayesian decision-theoretic policy chooses the optimal action in expectation while averaging over the uncertain objective function, Thompson sampling chooses the optimal action for a randomly sampled objective function. This interpretation of Thompson sampling is illustrated for our example scenario in Figure 7.19, showing three possible outcomes for Thompson sampling. In this case, two samples would exploit the region surrounding the best-seen point, and one would explore the region around the left-most observation.

Figure 7.20 shows the posterior belief of the objective after 15 rounds of Thompson sampling for our example scenario. The global maximum was located remarkably quickly. Of course, as a stochastic policy, it is not guaranteed that the behavior of Thompson sampling will resemble this outcome. In fact, this was a remarkably lucky run! The most likely locations of the optimum were ignored in the first rounds, quickly leading to the discovery of the optimum on the left. Figure 7.21 shows another optimal decision for random sample

example and discussion

example of slow convergence 

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-28.jpg?height=457&width=1647&top_left_y=454&top_left_x=153)

Figure 7.20: The posterior after 15 steps of Thompson sampling (7.19) on our running example. The tick marks show the points chosen by the policy, progressing from top to bottom. Observations within 0.2 length scales of the optimum have thicker marks; the optimum was located on iteration 8 .

approximate dynamic programming: $§ 5.3$, p. 101 two-step lookahead run of Thompson sampling on the same scenario. The global optimum was not found nearly as quickly; however, it was eventually located after approximately 80 iterations. In 100 repetitions of the policy varying the random seed, the median iteration for discovering the optimum on this example was 38 , with only 12 seeds resulting in discovery in the first 20 iterations. Despite this sometimes slow convergence, the distribution of the chosen evaluation locations nonetheless demonstrates continual management of the exploration-exploitation tradeoff.

\subsection*{OTHER IDEAS IN POLICY CONSTRUCTION}

We have now discussed the two most pervasive approaches to policy construction in Bayesian optimization: one-step lookahead and adapting policies for multi-armed bandits. We have also introduced the most popular Bayesian optimization policies encountered in the literature, all of which stem from one of these two methods. However, there are some additional ideas worthy of discussion.

\section*{Approximate dynamic programming beyond one-step lookahead}

One-step lookahead offers tremendous computational benefits, but the cost of these savings is extreme myopia in decision making. As we are oblivious to anything that might happen beyond the present observation, one-step lookahead can focus too much on exploitation. However, some less myopic alternatives have been proposed based on more complex (and more costly!) approximations to the optimal policy.

The simplest idea in this direction is to extend the lookahead horizon, and the most tractable option is of course two-step lookahead. Given an arbitrary one-step lookahead acquisition function $\alpha(x ; \mathcal{D})$, the two-step analog is $(5.12)$ :

$$
\alpha_{2}(x ; \mathcal{D})=\alpha(x ; \mathcal{D})+\mathbb{E}\left[\max _{x^{\prime} \in \mathcal{X}} \alpha\left(x^{\prime} ; \mathcal{D}^{\prime}\right) \mid x, \mathcal{D}\right]
$$



![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-29.jpg?height=614&width=1634&top_left_y=453&top_left_x=268)

Figure 7.21: The posterior after 8o steps of Thompson sampling (7.19) on our running example, using a random seed different from Figure 7.20. The global optimum was located on iteration 78 .

Although this door is open for any of the decision-theoretic policies considered in this chapter, two-step expected improvement has received the most attention. OSBORNE et al. derived two-step expected improvement and demonstrated good empirical performance on some test functions compared with the one-step alternative. ${ }^{45}$ However, it is telling that they restricted their investigation to a limited number of functions due to the inherent computational expense. GINSBOURGER and LE RICHE completed a contemporaneous exploration of two-step expected improvement and provided an explicit example showing superior behavior from the less myopic policy ${ }^{46}$ Recently, several authors have revisited (2+)-step lookahead and developed sophisticated implementation schemes rendering longer horizons more feasible. ${ }^{47,48}$

We provided an in-depth illustration and deconstruction of two-step expected improvement for our example scenario in Figures 5.2-5.3. Note that the two-step expected improvement is appreciable even for the (useless!) options of evaluating at the previously observed locations, as we can still make conscientious use of the following observation.

Figure 7.22 illustrates the progress of 20 evaluations designed by maximizing two-step expected improvement for our example scenario. Comparing with the one-step alternative in Figure 7.4, the less myopic policy exhibits somewhat more exploratory behavior and discovered the optimum more efficiently - after 15 rather than 19 evaluations.

Rollout has also been considered as an approach to building nonmyopic optimization policies. Again the focus of these investigations has been on expected improvement (or the related knowledge gradient), but the underlying principles could be extended to other policies.

LAM et al. combined expected improvement with several steps of rollout, again maximizing expected improvement as the base policy. ${ }^{49} \mathrm{The}$ authors also proposed optionally adjusting the utility function through
45 M. A. OsBorne et al. (2009). Gaussian Processes for Global Optimization. LION 3 .

46 D. GINSBOURGER and R. LE RICHE (2010). Towards Gaussian Process-Based Optimization with Finite Time Horizon. MODA 9.

47 J. WU and P. I. Frazier (2019). Practical Two-Step Look-Ahead Bayesian Optimization. NeurIPS 2019

48 s. JIANG et al. (202ob). Efficient Nonmyopic Bayesian Optimization via One-Shot MultiStep Trees. NeurIPS 2020.

rollout: $§ 5 \cdot 3$, p. 102

49 R. R. LAM et al. (2016). Bayesian Optimization with a Finite Budget: An Approximate Dynamic Programming Approach. NeurIPS 2016. 
![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-30.jpg?height=796&width=1648&top_left_y=462&top_left_x=152)

Figure 7.22: The posterior after 10 (top) and 20 (bottom) steps of the optimization policy induced by maximizing two-step expected improvement on our running example. The tick marks show the points chosen by the policy, progressing from top to bottom, during iterations 1-10 (top) and 11-20 (bottom). Observations within 0.2 length scales of the optimum have thicker marks; the optimum was located on iteration 15.

50 Such a discount factor is common in infinitehorizon decision problems:

D. P. BERTSEKAs (2017). Dynamic Programming and Optimal Control. Vol. 1. Athena Scientific.

51 X. YUE and R. AL KONTAR (2020). Why NonMyopic Bayesian Optimization Is Promising and How Far Should We Look-ahead? A Study via Rollout. AISTATS 2020.

52 See p. 125

batch rollout: $§ 5 \cdot 3$, p. 103

53 J. GONZÁLEZ et al. (2016b). GLASSES: Relieving the Myopia of Bayesian Optimisation. AISTATS 2016.

54 The policy used in GLASSES is described in

J. GONZÁlEz et al. (2016a). Batch Bayesian Optimization via Local Penalization. AISTATS 2016,

as well as in $§ 11.3$, p. 257; however, any desired alternative could also be used. multiplicative discounting to encourage earlier rather than later progress during the rollout steps. ${ }^{50}$ For some combinations of the rollout horizon and the discount factor, the resulting policies outperformed common one-step lookahead policies on a suite of synthetic test functions. YUE and AL KONTAR described a mechanism for dynamically choosing the rollout horizon based on the potential impact of a misspecified model, ${ }^{51}$ exactly the issue that gave KUSHNER pause. ${ }^{52}$

The additional computational burden of rollout limited LAM et al. to a relatively short rollout horizon on the order of $4-5$. Although considerably less myopic than one-step lookahead, the true decision horizon can be much greater, especially during the early stages of optimization. GONZÁLEZ et al. proposed an alternative approach based on batch rollout that can effectively look farther ahead at the expense of ignoring dependence among future decisions. ${ }^{53}$ The algorithm - dubbed GLAsses by the authors as it counteracts myopia - augments expected improvement with a single rollout step that designs a batch of additional observation locations of size equal to the remaining evaluation budget. ${ }^{54}$ These points serve as a rough simulation of the decisions that might follow after making a proposed observation. A potential observation location is then evaluated by the expected improvement gained by simultaneously evaluating at that point as well as the constructed batch, realizing an efficient and budget-aware nonmyopic policy. GLASSES outperformed myopic and nonmyopic baselines on synthetic test functions, and the 

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-31.jpg?height=845&width=1630&top_left_y=454&top_left_x=270)

Figure 7.23: The modified expected improvement acquisition function (7.21) for our running example for different values of the target improvement $\varepsilon$. The target is expressed as a fraction of the range of the posterior mean over the space. Increasing the target improvement leads to increasingly exploratory behavior.

authors also demonstrated that dynamically setting the batch size to the remaining budget outperformed an arbitrary fixed size, suggesting that budget adaptation was important for success.

JIANG et al. continued this thread with an even more dramatic approximation dubbed BINOCULARs, potentially initiating an arms race toward increasingly nonmyopic acronyms. ${ }^{55}$ The idea is to construct a single batch observation in each iteration, then select a point from this batch for evaluation. This represents an extreme computational savings over GLASSES, which must construct a batch anew for every proposed observation location. However, the method retains the same fundamental motivation: well-designed batch policies automatically induce diversity among batch members, encouraging exploration in the resulting sequential policy (see Figure 11.3). The strong connection between the optimal batch and sequential policies (11.9-11.10) provides further motivation for this approach. JIANG et al. also conducted a study of optimization performance versus the cost of computing the policy: one-step lookahead, BINOCULARS, GLASSES, and rollout comprised the Pareto frontier, with each method increasing computational effort by an order of magnitude.

\section*{Artificially encouraging exploration}

Another more indirect approach to nonmyopic policy design is to modify a one-step lookahead acquisition function to artificially encourage more
55 s. JIANG et al. (2020a). BINOCULARs for Efficient, Nonmyopic Sequential Experimental Design. ICML $202 O$.

batch observations: $§ 11.3$, p. 252

Pareto frontier: $§ 11.7$, p. 269 

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-32.jpg?height=457&width=1651&top_left_y=454&top_left_x=154)

Figure 7.24: The posterior after 15 steps of the STRELTSOV and VAKILI optimization policy on our running example. The tick marks show the points chosen by the policy, progressing from top to bottom. Observations within 0.2 length scales of the optimum have thicker marks; the optimum was located on iteration 14.

56 H. J. KUSHNER (1964). A New Method of Locating the Maximum Point of an Arbitrary Multipeak Curve in the Presence of Noise. Fournal of Basic Engineering 86(1):97-106. exploratory behavior when the remaining budget is significant. This can be motivated by the nature of the optimal policy, which maximizes a combination of immediate reward (exploitation) with expected future reward (exploration) (5.18). As the decision horizon increases, the exploration term may become increasingly dominant in this score, suggesting that optimal behavior entails early exploration of the domain followed by later exploitation and refinement.

Both the probability of improvement (7.5) and upper confidence bound acquisition functions already feature a parameter controlling the exploration-exploitation tradeoff, which can be dynamically maintained throughout optimization. This idea is quite old, reaching back to the earliest papers on Bayesian optimization. KUSHNER for example provided detailed advice on adjusting the threshold in probability of improvement to transition from early exploration to increasing levels of exploitation when appropriate. ${ }^{56}$

In the case of exact observation, it is also possible to modify expected improvement (7.3) to incorporate a similar parameter. Rather than measuring improvement with respect to the utility of the current data $u(\mathcal{D})=\phi^{*}$, we measure with respect to an inflated value $\phi^{*}+\varepsilon$, with no credit given for improvements less than this amount. The result is a modified expected improvement acquisition function:

$$
\alpha_{\mathrm{EI}}^{\prime}(x ; \mathcal{D}, \varepsilon)=\mathbb{E}\left[\max \left\{\phi-\left[\phi^{*}+\varepsilon\right], 0\right\} \mid x, \mathcal{D}\right] .
$$

As with probability of improvement, larger improvement thresholds encourage increasing exploration, as illustrated in Figure 7.23. For extreme values, the modified expected improvement drops to effectively zero except in regions with significant uncertainty. It is not obvious how this idea can be extended to handle noisy observations, as the simple reward of the updated dataset may in fact decrease, raising the question of how to correctly define "sufficient improvement."

MOcKus proposed a scheme to set the threshold for Gaussian process models with constant mean and stationary covariance where the threshold was set dynamically based on the remaining budget, using the asymptotic behavior of the maximum of iid Gaussian random variables. ${ }^{57}$ This can be interpreted as an approximate batch rollout policy where remaining decisions are simulated by fictitious uncorrelated observations; for some models this serves as an efficient simulation of random rollout

The modified expected improvement (7.21) was also the basis for an unusual policy proposed by STRELTSOv and VAKILI. ${ }^{58}$ Let $c: \mathcal{X} \rightarrow \mathbb{R}^{>0}$ quantify the cost of making an observation at any proposed location; in the simplest case we could take the cost to be constant. To evaluate the promise of making an observation at $x$, we solve the equation

$$
\alpha_{\mathrm{EI}}^{\prime}\left(x ; \mathcal{D}, \alpha_{\mathrm{SV}}\right)=c(x)
$$

for $\alpha_{\mathrm{sv}}$, which will serve as the acquisition function value at $x .^{59}$ That is, we solve for the improvement threshold that would render an observation at $x$ cost-prohibitive in expectation, and design each observation to coincide with the last point to be ruled out when considering increasingly demanding thresholds. The resulting policy shows interesting behavior, at least on our running example; see Figure 7.24. After effective initial exploration, the global optimum was located on iteration 14. The behavior is similar to the upper confidence bound approach in Figure 7.18, and indeed STRELTSOV and VAKILI showed that the proposed method can be understood as a variation on this method with a locationdependent upper confidence quantile depending on observation cost and uncertainty.

An even simpler mechanism for injecting exploration into an existing policy is to occasionally make decisions randomly or via some other policy encouraging pure exploration. DE ATH et al. for example considered a family of $\varepsilon$-greedy policies where a one-step lookahead policy is interrupted, with probability $\varepsilon$ in each iteration, by evaluating at a location chosen uniformly at random from the domain. ${ }^{60}$ These policies delivered impressive empirical performance, even when the one-step lookahead policy was as simple as maximizing the posterior mean (see below).

\section*{Lookahead for cumulative reward?}

Notably missing from our discussion on one-step lookahead policies was the cumulative reward utility function. Unfortunately, in this case one-step lookahead does not produce a particularly useful optimization policy. Suppose we adopt the cumulative reward utility function (6.7), $u(\mathcal{D})=\sum y_{i}$. Then marginal gain in utility from a measurement at $x$ is simply the observed value $y$. Therefore the expected one-step marginal gain is the posterior predictive mean:

$$
\alpha(x ; \mathcal{D})=\mathbb{E}[y \mid x, \mathcal{D}],
$$

which for zero-mean additive noise reduces to the posterior mean of $f$ :

$$
\alpha(x ; \mathcal{D})=\mu_{\mathcal{D}}(x) .
$$

57 J. MOcKUs (1989). Bayesian Approach to Global Optimization: Theory and Applications. Kluwer Academic Publishers. [§ 2.5]

58 s. STRELTSOV and P. VAKILI (1999). A NonMyopic Utility Function for Statistical Global Optimization Algorithms. Fournal of Global Optimization 14(3):283-298.

59 As $\alpha_{\mathrm{EI}}^{\prime}$ is monotonically decreasing with re spect to $\alpha_{\mathrm{Sv}}$ and approaches zero as $\alpha_{\mathrm{SV}} \rightarrow \infty$, the unique solution can be found efficiently via bisection.

6o G. DE ATH et al. (2021). Greed Is Good: Exploration and Exploitation Trade-offs in Bayesian Optimisation. ACM Transactions on Evolutionary Learning and Optimization 1(1):1-22.

cumulative reward: $§ 6.2$, p. 114

posterior mean acquisition function 

![](https://cdn.mathpix.com/cropped/2023_09_22_14c6db3425b15adc7320g-34.jpg?height=414&width=1630&top_left_y=464&top_left_x=156)

Figure 7.25: The posterior after 15 steps of two-step lookahead for cumulative reward (7.22) on our running example. The tick marks show the points chosen by the policy, progressing from top to bottom. The policy becomes stuck on iteration 7 .

becoming "stuck"

two-step lookahead for cumulative reward
From cursory inspection of our example scenario in Figure 7.1, we can see that maximizing this acquisition function can cause the policy to become "stuck" with no chance of recovery. In our example, the posterior mean is maximized at the previously best-seen point, so the policy will select this point forevermore.

It is also interesting to consider two-step lookahead, where the acquisition function becomes (5.12):

$$
\alpha_{2}(x ; \mathcal{D})=\mu_{\mathcal{D}}(x)+\mathbb{E}\left[\max _{x^{\prime} \in \mathcal{X}} \mu_{\mathcal{D}^{\prime}}\left(x^{\prime}\right) \mid x, \mathcal{D}\right] .
$$

After subtracting a constant (the global reward of the current data), we have

$$
\alpha_{2}(x ; \mathcal{D})=\mu_{\mathcal{D}}(x)+\alpha_{\mathrm{KG}}(x ; \mathcal{D}),
$$

the sum of the posterior mean and the knowledge gradient (7.4), reflecting both exploitation and exploration. Unfortunately even this less myopic option can become stuck due to overexploitation, as illustrated in Figure 7.25 .

\subsection*{SUMMARY OF MAJOR IDEAS}

Although we have covered a lot of ground in this chapter, there were really only two big ideas in policy construction: one-step lookahead and adopting successful policies from multi-armed bandits. However, there remains significant opportunity for novelty in this space. one-step lookahead: § 7.2, p. 124 multi-armed bandits: $§ 7.7$, p. 141