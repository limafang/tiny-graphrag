\section*{EXTENSIONS AND RELATED SETTINGS}

Thus far we have focused exclusively on a simple model problem (Algorithm 1.1): sequential optimization of a single objective with either a fixed evaluation budget or known observation costs. These assumptions are convenient for study and often reasonable in practice, but not all optimization scenarios fit neatly into this mold. Numerous extensions of this setup have received serious attention in the literature, and we will provide an overview of the most important of these in this chapter.

A running theme throughout this discussion will be adapting the decision-theoretic framework developed in Chapter 5 to derive policies for each of these new settings. In that chapter, we derived optimal optimization policies for our model problem, but we want to stress that the overarching approach to decision making can be extended to effectively any scenario.

Namely, we may derive an optimal policy for any sequential experimental design problem by following a simple recipe, an abstraction of our previous presentation:

1. Identify the action space $\mathcal{A}$ of each decision.

2. Define preferences over outcomes with a utility function $u(\mathcal{D})$.

3. Identify the uncertain elements $\psi$ relevant for each decision and determine how to compute the posterior belief given data, $p(\psi \mid \mathcal{D})$.

4. Compute the one-step marginal gain $\alpha_{1}$ (5.8) for every action. ${ }^{1}$

5. Derive the optimal policy by induction on the horizon (5.15-5.16).

We will sketch how this scheme can be realized for notable optimization settings not yet addressed. Each will involve additional complexity in building effective policies due to additional complexity in the problem formulation, but we will demonstrate how we can adapt our existing tools by addressing the appropriate steps of the above general procedure. We will also provide a survey of proposed approaches to each of the optimization extensions we will consider, regardless of whether they are grounded in decision theory.

\section*{UNKNOWN OBSERVATION COSTS}

In our discussion on cost-aware optimization, we assumed that observation costs were prescribed by a known cost function $c$, which enabled a simple mechanism for dynamic termination through careful accounting. However, in some scenarios the cost of an observation may not be known a priori, but rather only determined through the course of acquisition. These unknown costs reflect additional uncertainty that must be accounted for in policy design, as the utility of the returned data depends critically on their values. The natural solution is to perform inference about observation costs throughout optimization and use the resulting beliefs to guide our decisions, just as we do with an unknown

\section*{1}

Chapter 5: Decision Theory for Optimization, p. 87

general procedure for optimal policies

1 Conveniently, this step also yields the one-step lookahead approximate policy as a side effect.

cost-aware optimization: $\S 5.4$, p. 103 cost function, $c$ notation for observation costs

cost function, $c$

observed cost at $x, z$ value of cost function at $x, \kappa=c(x)$

cost observation model, $p(z \mid x, \kappa)$

cost function prior, $p(c)$ cost function posterior, $p(c \mid \mathcal{D})$

predictive distribution for cost, $p(z \mid x, \mathcal{D})$

2 As an example use case, suppose evaluating the objective requires train a machine learning model on cloud hardware with variable (e.g., spot) pricing.

3 The related literature is substantial, but a Bayesian optimization perspective can be found in:

F. HUTTER et al. (2011). Sequential Model-Based Optimization for General Algorithm Configuration. LION 5 . objective function. To adapt the cost-aware policies from Chapter 5 to this setting, we must revisit steps $3-4$ of the above procedure.

\section*{Inference for unknown cost function}

At a high level, reasoning about an unknown cost function given observations is no different from reasoning about an unknown objective function, and we can apply any suitable regression model for this task. The details of this inference will obviously depend on the situation, but we can outline one rather general approach.

First, let us define some notation for cost observations mirroring our notation for the objective function. Suppose that evaluation costs are determined, perhaps stochastically, by an underlying cost function $c: \mathcal{X} \rightarrow \mathbb{R}$ we wish to infer. Suppose further that an evaluation at a point $x$ now returns both a measured value $y$, whose distribution depends on the corresponding objective function value $\phi=f(x)$, and an observation cost $z$, whose distribution depends on the corresponding cost function value $\kappa=c(x)$. We will accumulate these values throughout optimization in an augmented dataset of observations and their costs, $\mathcal{D}=(\mathbf{x}, \mathbf{y}, \mathbf{z})$.

If we can safely model cost observations as conditionally independent of objective observations given the chosen location, then we may follow our modeling strategy for the objective function and assume that each observed cost is generated by an observation model $p(z \mid x, \kappa)$. This allows for modeling a wide range of different scenarios, including nondeterministic costs. ${ }^{2}$ Now we can proceed with inference about the cost function as usual. We choose a suitable (possibly parametric) prior process $p(c)$, which we condition on observed costs to form the posterior $p(c \mid \mathcal{D})$. Finally, we can form the predictive distribution for the cost of making an observation at an arbitrary point $x$ by marginalizing the latent cost function:

$$
p(z \mid x, \mathcal{D})=\int p(z \mid x, \kappa) p(\kappa \mid x, \mathcal{D}) \mathrm{d} \kappa .
$$

In some applications, observation costs may be nontrivially correlated with the objective function. As an extreme example, consider a common problem in algorithm configuration, ${ }^{3}$ where the goal is to design the parameters of an algorithm so as to minimize its expected running time. Here the cost of evaluating a proposed configuration might be reasonably defined to be proportional to its running time. Up to scaling, the observation cost is precisely equal to the objective! To model such correlations, we could define a joint prior $p(f, c)$ over the cost and objective functions, as well as a joint observation model $p(y, z \mid x, \phi, \kappa)$. We could then continue as normal, computing expected utilities with respect to the joint predictive distribution

$$
p(y, z \mid x, \mathcal{D})=\iint p(y, z \mid x, \phi, \kappa) p(\phi, \kappa \mid x, \mathcal{D}) \mathrm{d} \phi \mathrm{d} \kappa,
$$

a setup offering considerable flexibility in modeling. - observations — posterior mean posterior $95 \%$ credible interval

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-03.jpg?height=212&width=1628&top_left_y=545&top_left_x=271)

cost-agnostic expected improvement, $\alpha_{\mathrm{EI}}$

posterior mean, cost

posterior $95 \%$ credible interval, cost

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-03.jpg?height=194&width=1622&top_left_y=877&top_left_x=274)

— cost-adjusted expected improvement $\quad \boldsymbol{\nabla}$ next observation location

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-03.jpg?height=246&width=1668&top_left_y=1159&top_left_x=271)

Figure 11.1: Decision making with uncertain costs. The middle panel shows the cost-agnostic expected improvement acquisition function along with a belief about an uncertain cost function, here assumed to be independent of the objective. The bottom panel shows the cost-adjusted expected improvement, marginalizing uncertainty in the objective and cost function (11.1).

Decision making with unknown costs

The approach outlined above suffices to maintain a belief about the potential cost of observations proposed throughout optimization, but we still must account for this uncertainty in the optimization policy. Thankfully, this is relatively straightforward in our decision-theoretic framework: we simply compute expected utility accounting for all relevant uncertainty as usual, here to include cost. Given an arbitrary dataset $\mathcal{D}$, now augmented with observation costs, consider the one-step expected marginal gain in utility:

$$
\alpha_{1}(x ; \mathcal{D})=\mathbb{E}\left[u\left(\mathcal{D}_{1}\right) \mid x, \mathcal{D}\right]-u(\mathcal{D})
$$

Computing this expectation now requires integrating over both the unknown measurement $y$ and the unknown observation cost $z$ :

$$
\mathbb{E}\left[u\left(\mathcal{D}_{1}\right) \mid x, \mathcal{D}\right]=\iint u(\mathcal{D} \cup\{x, y, z\}) p(y, z \mid x, \mathcal{D}) \mathrm{d} y \mathrm{~d} z
$$

Although there is some slight added complexity in this computation, the optimal policy otherwise remains exactly as derived in (5.15-5.17). ${ }^{4}$
4 Of course, all nested expectations must also be taken with respect to unknown observation costs! - observations posterior mean posterior $95 \%$ credible interval

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-04.jpg?height=200&width=1625&top_left_y=551&top_left_x=158)

cost-agnostic expected improvement, $\alpha_{\mathrm{EI}}$ posterior mean, cost posterior $95 \%$ credible interval, cost

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-04.jpg?height=197&width=1625&top_left_y=877&top_left_x=160)

- expected improvement per unit cost

$\mathbf{v}$ next observation location

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-04.jpg?height=197&width=1647&top_left_y=1141&top_left_x=153)

Figure 11.2: Expected gain per unit cost. The middle panel shows the cost-agnostic expected improvement acquisition function along with a belief about an uncertain cost function, here assumed to be independent of the objective. The bottom panel shows the expected improvement per unit cost (11.1).

5 S. Zilberstein (1996). Using Anytime Algorithms in Intelligent Systems. AI Magazine $17(3): 73-83$.

6 J. SNOEK et al. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. NeUrIPS 2012.

7 Assuming independence between the objective and cost, a second-order expansion is just as easy to compute:

$$
\frac{\alpha_{1}(x ; \mathcal{D})}{\mathbb{E}[\kappa \mid x, \mathcal{D}]}+\frac{\alpha_{1}(x ; \mathcal{D}) \operatorname{var}[\kappa \mid x, \mathcal{D}]}{\mathbb{E}[\kappa \mid x, \mathcal{D}]^{3}}
$$

Figure 11.1 illustrates this policy, combining expected improvement with an independent uncertain cost function.

\section*{Expected gain per unit cost}

Other approaches to dealing with unknown costs are also of course possible. SNOEK et al. proposed one notable option that has gained some popularity based on a heuristic common in anytime algorithms ${ }^{5}$ - that is, algorithms that seek to maximize the instantaneous rate of improvement under the premise that the procedure may be terminated at any time. ${ }^{6}$ The idea is simple: we can approximate the expected immediate marginal gain in utility per unit cost from an observation at $x$ by

$$
\mathbb{E}\left[\frac{u\left(\mathcal{D}_{1}\right)-u(\mathcal{D})}{\kappa} \mid x, \mathcal{D}\right] \approx \frac{\alpha_{1}(x ; \mathcal{D})}{\mathbb{E}[\kappa \mid x, \mathcal{D}]},
$$

where $\alpha_{1}$ is the expected gain in data utility. This is a first-order approximation to the expected gain-per-cost (which is not the ratio of their respective expectations, even in the independent case) that could be further refined if desired, ${ }^{7}$ but works well in practice. The motivating example for SNOEK et al. was hyperparameter tuning of machine learning algorithms with unknown training costs, and the simple heuristic of maximizing "expected improvement per (expected) second" delivered promising results in their experiments. This heuristic has since appeared in other contexts. ${ }^{8}$

Figure 11.2 illustrates this policy with the sample example in Figure 11.2. The chosen decision closely matches the decision reached in Figure 11.1. It is interesting to compare the behavior of the two acquisition functions on both sides of the domain: whereas these regions are not especially exciting in the additive cost approach, they are appealing from the anytime view - although they are expected to give only modest improvement, they are also relatively inexpensive.

\section*{CONSTRAINED OPTIMIZATION AND UNKNOWN CONSTRAINTS}

Many optimization problems feature constraints restricting allowable solutions in a potentially complex, even uncertain manner. To this end we may extend our running optimization problem (1.1) to incorporate arbitrary constraints; a common formulation is:

$$
x^{*} \in \underset{x \in \mathcal{X}}{\arg \max } f(x) \quad \text { subject to } \forall i: g_{i}(x) \leq 0,
$$

where the functions $\left\{g_{i}\right\}: \mathcal{X} \rightarrow \mathbb{R}$ comprise a set of inequality constraints. The subset of the domain where all constraints are satisfied is known as the feasible region:

$$
\mathcal{F}=\left\{x \in \mathcal{X} \mid \forall i: g_{i}(x) \leq 0\right\} .
$$

In some situations, the value of some or all of the constraint functions may in fact be unknown a priori and only revealed through experimentation, complicating policy design considerably. As an example, consider a business optimizing the parameters of a service to maximize revenue, subject to constraints on customer response. If customer response is measured experimentally - for example via a focus group - we cannot know the feasibility of a proposed solution until after the objective has been measured, and even then only with limited confidence.

Further, even if the constraint functions can be computed exactly on demand, constrained optimization of an uncertain objective function is not entirely straightforward. In particular, an observation of the objective at an infeasible point may yield useful information regarding behavior on the feasible region, and could represent an optimal decision if that information were compelling enough. Thus simply restricting the action space to the feasible region may not be the best approach to policy design. ${ }^{9}$ Instead, to derive effective policies for constrained optimization, we must reconsider steps $2-4$ of our general approach.

\section*{Modeling constraint functions}

To allow for uncertain constraint functions, we begin by modeling the joint observation process of the objective and constraint functions. As with an uncertain cost function, we will assume that each observation of
8 G. MALKOMES et al. (2016). Bayesian Optimization for Automated Model Selection. NeurIPS 2016.

inequality constraints, $\left\{g_{i}\right\}$

feasible region, $\mathcal{F}$

uncertain constraints

9 R. B. GRAMACY and H. K. H. LEE (2011). Optimization under Unknown Constraints. In: Bayesian Statistics 9 . simple reward: § 6.1, p. 112

10 M. A. GELBART et al. (2014). Bayesian Optimization with Unknown Constraints. UAI 2014 the objective is accompanied by some information regarding constraint satisfaction at the chosen location. Modeling this process is trivial when the constraint functions are known a priori, but otherwise may require some care.

It is difficult to provide concrete advice as information about constraint satisfaction may assume different forms, ranging from exact observation of the constraint functions to mere binary indicators of constraint satisfaction. In some situations, we may even face stochastic constraint processes where the feasibility of a given location is only achieved with some unknown probability. Fortunately, our discussion in the previous section regarding modeling an uncertain cost function is general enough to handle any of these situations by choosing an appropriate joint prior processes and observation model for the objective and constraint functions.

\section*{Defining a utility function}

Next we must define a utility function appropriate for constrained optimization. Most of the utility functions in Chapter 6 can be suitably modified to this end.

We can realize a constrained version of simple reward by considering the expected utility of a risk-neutral terminal recommendation following optimization. As before, the resulting utility will depend on the action space used for the terminal decision. Perhaps the simplest option would be to limit the recommendation to the feasible observed locations (if known!):

$$
u(\mathcal{D})=\max _{x \in \mathbf{x} \cap \mathcal{F}} \mu_{\mathcal{D}}(x),
$$

resulting in a natural adaptation of the simple reward (6.3). If the entire feasible region is known, we might instead allow recommending any feasible point, giving rise to an adaptation of the global simple reward (6.5):

$$
u(\mathcal{D})=\max _{x \in \mathcal{F}} \mu_{\mathcal{D}}(x) .
$$

Finally, in the case of uncertain constraints, we might limit our recommendation to those points believed to be feasible with sufficient confidence: ${ }^{10}$

$$
u(\mathcal{D})=\max _{x \in \mathcal{F}(\delta)} \mu_{\mathcal{D}}(x) ; \quad \mathcal{F}(\delta)=\{x \mid \operatorname{Pr}(x \in \mathcal{F} \mid \mathcal{D}) \geq 1-\delta\} .
$$

With some care, we could also modify other utility functions to be aware of a (possibly uncertain) feasible region, although the variations on simple reward above have received the most attention.

\section*{Deriving a policy}

After selecting a model for the constraint functions and a utility function for our observations, we can derive a policy for constrained optimization following the standard procedure of induction on the decision horizon. A policy that has received particular attention is the result of one-step lookahead with (11.3). ${ }^{10,11,12}$ If we assume that the constraint functions are conditionally independent of the objective function given the observation location, then the one-step expected marginal gain in utility becomes

$$
\alpha(x ; \mathcal{D})=\alpha_{\mathrm{EI}}^{\prime}\left(x ; \mathcal{D}, \mu^{*}\right) \operatorname{Pr}(x \in \mathcal{F} \mid x, \mathcal{D}) .
$$

This is simply the expected improvement, measured with respect to the feasible incumbent value $\mu^{*}=u(\mathcal{D})(7.21,11.3)$, weighted by the probability of feasibility, a natural policy we might arrive at via purely heuristic arguments. GELBART et al. point out this acquisition function has a slight pathology (also present with unconstrained expected improvement): the utility degenerates when no feasible observations are available, and (11.6) becomes ill-defined. In this case, the authors propose simply maximizing the probability of feasibility:

$$
\alpha(x ; \mathcal{D})=\operatorname{Pr}(x \in \mathcal{F} \mid x, \mathcal{D}) .
$$

The expected feasible improvement (11.6) encodes a strong preference for evaluating on the feasible region only, and in the case where the constraint functions are all known, the resulting policy will never evaluate outside the feasible region. ${ }^{13}$ This is a natural consequence of the one-step nature of the acquisition function: an infeasible observation cannot yield any immediate improvement to the utility (11.3) and thus cannot be one-step optimal. However, this behavior might be seen as undesirable given our previous comment that infeasible observations may yield valuable information about the objective on the feasible region.

If we wish to realize a policy more open to observing outside the feasible region, there are several paths forward. A less-myopic policy built on the same utility (11.3) is one option; even two-step lookahead could elect to obtain an infeasible measurement. Another possibility is one-step lookahead with a more broadly defined utility such as (11.4-11.5), which can see the merit of infeasible observations through more global evaluation of success.

To encourage infeasible observations when prudent, GRAMACY and LEE proposed a score they called the integrated expected conditional improvement: ${ }^{14}$

$$
\iint\left[\alpha_{\mathrm{EI}}\left(x^{\prime} ; \mathcal{D}\right)-\alpha_{\mathrm{EI}}\left(x^{\prime} ; \mathcal{D}^{\prime}\right)\right] \operatorname{Pr}\left(x^{\prime} \in \mathcal{F} \mid x^{\prime}, \mathcal{D}\right) p(y \mid x, \mathcal{D}) \mathrm{d} x^{\prime} \mathrm{d} y,
$$

where $\mathcal{D}^{\prime}$ is the putative updated dataset and the location $x^{\prime}$ is integrated over the domain. This is a measure of the expected impact of a measurement on the entire acquisition surface over the feasible region, which can effectively capture the potential impact of an infeasible observation when it is useful. A similar approach was taken by PICHENY, who integrated the change in probability of improvement against the feasibility probability. ${ }^{15}$ Although these approaches can be heuristically motivated, the required integrals over the acquisition surfaces are intractable, and no obvious approximations are available beyond standard methods.
11 M. SCHONLAU et al. (1998). Global versus Local Search in Constrained Optimization of Computer Models. In: New Developments and Applications in Experimental Design.

12 J. R. GARDNER et al. (2014). Bayesian Optimization with Inequality Constraints. ICML 2014.

observations in the infeasible region

13 In this case the policy would be equivalent to redefining the domain to encompass only the feasible region $\mathcal{F}$ and maximizing the unmodified expected improvement (7.2).

14 R. B. GRAMACY and H. K. H. LEE (2011). Optimization under Unknown Constraints. In: Bayesian Statistics 9.

15 V. PICHENY (2014). A Stepwise Uncertainty Reduction Approach to Constrained Global Optimization. AISTATS 2014 . decoupled observations

16 M. A. GELBART et al. (2014). Bayesian Optimization with Unknown Constraints. UAI 2014.

17 J. M. HERNÁNDEZ-LOBATO et al. (2016b). A General Framework for Constrained Bayesian Optimization Using Information-Based Search fournal of Machine Learning Research 17:1-53.

predictive entropy search: $\S 8.8$, p. 180

synchronous vs. asynchronous batch construction

asynchronous batch observations: § 11.4, p. 262

batch of observation locations, $\mathbf{x}$ corresponding observed values, $\mathbf{y}$

action space for batch observations, $\mathcal{A}=\mathcal{X}^{b}$

expected one-step marginal gain from batch observation, $\beta_{1}$
GELBART et al. considered a variation on the constrained optimization problem discussed above wherein observations of the objective and constraints can be "decoupled" - that is, when we can elect to measure any of these functions independent of the others, expanding the action space of the decision problem. ${ }^{16}$ The authors noted the expected feasible improvement (11.6) displayed undesirable behavior in this scenario and proposed an alternative policy based on mutual information, which was later refined and expanded into a fully fledged information-theoretic policy for constrained optimization (with or without decoupled observations) based on predictive entropy search. ${ }^{17}$

\subsection*{SYNCHRONOUS BATCH OBSERVATIONS}

Many optimization settings allow for the possibility of making multiple observations in parallel. In fact, some settings such as high-throughput screening for scientific discovery practically demand parallel experiments due to the growing capacity of sophisticated automated instruments. Numerous batch policies have been proposed for Bayesian optimization to harness this capability, including variants of virtually every popular sequential policy.

Here we can distinguish two settings: synchronous and asynchronous batch construction. In both cases, multiple experiments must be designed to run in parallel. The distinguishing factor is that in the synchronous case, the results from each entire batch of experiments are obtained before designing the next, whereas in the asynchronous case, each time an experiment completes, we may immediately design a new one in light of those still pending. Bayesian decision theory naturally offers one possible approach to both of these scenarios. We will focus on the synchronous case in this section and will discuss the asynchronous case in the next section.

\section*{Decision-theoretic batch construction}

Consider an optimization scenario where in each iteration we may design a batch of $b$ points $\mathbf{x}=\left\{x_{1}, x_{2}, \ldots x_{b}\right\}$ for simultaneous evaluation, resulting in a corresponding vector of measured values y obtained before our next action. The design of each batch represents a decision with action space $\mathcal{A}=\mathcal{X}^{b}$, a modification to step 3 of our general procedure.

We proceed by computing the one-step expected gain in utility from a proposed batch measurement $\mathbf{x}$. We will call the corresponding batch acquisition function $\beta_{1}$ to distinguish it from its sequential analog $\alpha_{1}$ :

$$
\beta_{1}(\mathbf{x} ; \mathcal{D})=\mathbb{E}\left[u\left(\mathcal{D}_{1}\right) \mid \mathbf{x}, \mathcal{D}\right]-u(\mathcal{D}) .
$$

Here $\mathcal{D}_{1}$ represents the data available after the batch observation is resolved: $\mathcal{D}_{1}=\mathcal{D} \cup\{(\mathbf{x}, \mathbf{y})\}$. Computing this expected marginal gain is an expectation with respect to the unknown values $\mathbf{y}$ :

$$
\beta_{1}(\mathbf{x} ; \mathcal{D})+u(\mathcal{D})=\int u\left(\mathcal{D}_{1}\right) p(\mathbf{y} \mid \mathbf{x}, \mathcal{D}) \mathrm{d} \mathbf{y},
$$

- observations posterior mean posterior $95 \%$ credible interval

batch expected improvement optimal batch

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-09.jpg?height=643&width=1334&top_left_y=655&top_left_x=384)

Figure 11.3: Optimal batch selection. The heatmap shows the expected one-step marginal gain in simple reward (6.1) from adding a batch of two points - corresponding in location to the belief about the objective plotted along the margins - to the current dataset. Note that the expected marginal gain is symmetric. The optimal batch will observe on both sides of the previously best-seen point. In this example, incorporating either one of the selected points alone would also yield relatively high expected marginal gain.

and an optimal batch with decision horizon $\tau=1$ maximizes this score:

$$
\mathbf{x} \in \underset{\mathbf{x}^{\prime} \in \mathcal{X}^{b}}{\arg \max } \beta_{1}\left(\mathbf{x}^{\prime} ; \mathcal{D}\right) \text {. }
$$

Finally, we can derive the optimal batch policy for a fixed evaluation budget by induction on the horizon, accounting for the expanded action space for each future decision:

$$
\begin{aligned}
\mathbf{x} & \in \underset{\mathbf{x}^{\prime} \in \mathcal{X}^{b}}{\arg \max } \underbrace{\beta_{1}\left(\mathbf{x}^{\prime} ; \mathcal{D}\right)+\mathbb{E}\left[\beta_{\tau-1}^{*}\left(\mathcal{D}_{1}\right) \mid \mathbf{x}^{\prime}, \mathcal{D}\right]}_{=\beta_{\tau}\left(\mathbf{x}^{\prime} ; \mathcal{D}\right)} ; \\
\beta_{\tau}^{*}(\mathcal{D}) & =\max _{\mathbf{x}^{\prime} \in \mathcal{X}^{b}} \beta_{\tau}\left(\mathbf{x}^{\prime} ; \mathcal{D}\right) .
\end{aligned}
$$

If desired, we could also allow for variable-cost observations and the option of dynamic termination by accounting for costs and including a termination option in the action space. Another compelling possibility would be to consider dynamic batch sizes by expanding the action space further and assigning an appropriate size-dependent cost function for proposed batch observations.

Optimal batch selection is illustrated in Figure 11.3 for designing a batch of two points with horizon $\tau=1$. We compute the expected

optimal batch policy with fixed evaluation budget variable costs and termination option

dynamic batch sizes

example of optimal batch policy connection to $b$-step lookahead

"unrolling" the optimal policy: §5.3, p. 99

18 J. VONDRÁK (2005). Probabilistic Methods in Combinatorial and Stochastic Optimization. Ph.D. thesis. Massachusetts Institute of Technology. one-step gain in utility (11.7) - here the simple reward (6.1), analogous to expected improvement $\alpha_{\mathrm{EI}}$ (7.2) - for every possible batch and observe where the score is maximized. The optimal batch evaluates on either side of the previously best-seen point, achieving distributed exploitation. The expected marginal gain surface has notably complex structure, for example expressing a strong preference for batches containing at least one of the chosen locations over any purely exploratory alternative, as well as severely punishing batches containing an observation too close to an existing one.

We may gain some insight into the optimal batch policy by decomposing the expected batch marginal gain in terms of corresponding quantities from the optimal sequential policy. Let us first consider the expected marginal gain from selecting a batch of two points, $\mathbf{x}=\left\{x, x^{\prime}\right\}$, resulting in observed values $\mathrm{y}=\left\{y, y^{\prime}\right\}$. Let $\mathcal{D}^{\prime}$ represent the current data augmented with the single observation $(x, y)$. We may rewrite the marginal gain from the batch observation $(\mathbf{x}, \mathbf{y})$ as a telescoping sum with terms corresponding to the impact of each individual observation:

$$
u\left(\mathcal{D}_{1}\right)-u(\mathcal{D})=\left[u\left(\mathcal{D}_{1}\right)-u\left(\mathcal{D}^{\prime}\right)\right]+\left[u\left(\mathcal{D}^{\prime}\right)-u(\mathcal{D})\right]
$$

which allows us to rewrite the one-step expected batch marginal gain as:

$$
\beta_{1}(\mathbf{x} ; \mathcal{D})=\alpha_{1}(x ; \mathcal{D})+\mathbb{E}\left[\alpha_{1}\left(x^{\prime} ; \mathcal{D}^{\prime}\right) \mid \mathbf{x}, \mathcal{D}\right]
$$

This expression is remarkably similar to the optimal two-step expected sequential marginal gain (5.12):

$$
\alpha_{2}(x ; \mathcal{D})=\alpha_{1}(x ; \mathcal{D})+\mathbb{E}\left[\max _{x^{\prime} \in \mathcal{X}} \alpha_{1}\left(x^{\prime} ; \mathcal{D}^{\prime}\right) \mid x, \mathcal{D}\right] .
$$

The main difference is that in the batch setting, we must commit to both observation locations a priori, whereas in the sequential setting, we can design our second observation optimally given the outcome of the first.

We can extend this relationship to the general case. Temporarily adopting compact notation, a horizon- $b$ optimal sequential decision satisfies:

$$
x \in \arg \max \left\{\alpha_{1}+\mathbb{E}\left[\operatorname { m a x } \left\{\alpha_{1}+\mathbb{E}\left[\max \left\{\alpha_{1}+\cdots\right]\right\},\right.\right.\right.
$$

and the optimal one-step batch of size $b$ satisfies:

$$
\mathbf{x} \in \arg \max \left\{\alpha_{1}+\mathbb{E}\left[\quad \alpha_{1}+\mathbb{E}\left[\quad \alpha_{1}+\cdots\right]\right\} .\right.
$$

Clearly the expected utility gained from making $b$ optimal sequential decisions surpasses the expected utility from a single optimal batch the same size: the sequential policy benefits from designing each successive observation location adaptively, whereas the batch policy must make all decisions simultaneously and cannot benefit from replanning. This unavoidable difference in performance is called the adaptivity gap in the analysis of algorithms. ${ }^{18}$ 

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-11.jpg?height=360&width=1005&top_left_y=457&top_left_x=294)

Unfortunately, working with the larger action space inherent to batch optimization requires significant computational effort. First, computing the expected marginal gain (11.7) is more expensive than in the sequential setting (5.8), as we must now integrate with respect to the joint distribution over outcomes $p(\mathbf{y} \mid \mathbf{x}, \mathcal{D})$. Thus even evaluating the acquisition function at a single point entails more work. Additionally, finding the optimal decision (11.8) requires optimizing this score over a significantly larger domain than in the sequential analog, a nontrivial task due to its potentially complex and multimodal nature - see Figure 11.3.

Despite these computational difficulties, synchronous batch Bayesian optimization has enjoyed significant attention from the research community. We can identify two recurring research thrusts: deriving general strategies for extending arbitrary sequential policies to batch policies and deriving batch extensions of specific sequential policies. We provide a brief survey below.

\section*{Batch construction via sequential simulation}

Sequential simulation is an efficient strategy for creating batch policies by simulating multiple steps of an existing sequential policy. Pseudocode for this procedure is listed in Algorithm 11.1. Given a sequential acquisition function $\alpha$, we choose the first batch member by maximization:

$$
x_{1} \in \underset{x \in \mathcal{X}}{\arg \max } \alpha(x ; \mathcal{D}),
$$

and commit to this choice. We now augment our dataset with the chosen point and a fictitious observed value $\hat{y}_{1}$, forming $\mathcal{D}_{1}=\mathcal{D} \cup\left\{\left(x_{1}, \hat{y}_{1}\right)\right\}$, then maximize the acquisition function again to choose the second point:

$$
x_{2} \in \underset{x \in \mathcal{X}}{\arg \max } \alpha\left(x ; \mathcal{D}_{1}\right) \text {. }
$$

We proceed in this manner until the desired batch size has been reached. Sequential simulation entails $b$ optimization problems on $\mathcal{X}$ rather than a single problem on $\mathcal{X}^{b}$, which can be a significant speed improvement.

When the sequential policy represents one-step lookahead, sequential simulation can be regarded as a natural greedy approximation to the one-step batch policy via the decomposition in (11.10): whereas the optimal batch policy must maximize this score jointly, sequential simulation maximizes the score pointwise, fixing each point once chosen.
Algorithm 11.1: Sequential simulation.

first batch member, $x_{1}$

fictitious observation at $x_{1}, \hat{y}_{1}$

second batch member, $x_{2}$

greedy approximation of one-step lookahead 
![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-12.jpg?height=904&width=1648&top_left_y=454&top_left_x=152)

Figure 11.4: Sequential simulation using the expected improvement (7.2) policy and the kriging believer (11.11) imputation strategy. The first point is selected by maximizing expected improvement, and we condition the model on an observation equal to the posterior mean (top panel). We then maximize the updated expected improvement, condition, and repeat as desired. The bottom panel indicates the locations of several further points selected in this manner.

19 D. GINSBOURGER et al. (2010). Kriging is WellSuited to Parallelize Optimization. In: Computational Intelligence in Expensive Optimization Problems.
This procedure requires some mechanism for generating fictitious observations as we build the batch. GINSBOURGER et al. described two simple heuristics that have been widely adopted. ${ }^{19}$ Perhaps the most natural option is to impute the expected value of each observation, a heuristic GINSBOURGER et al. dubbed the kriging believer strategy:

$$
\hat{y}=\mathbb{E}[y \mid x, \mathcal{D}] .
$$

This has the effect of fixing the posterior mean of the objective function throughout simulation. An even simpler option is to impute a constant value independent of the chosen point, which the authors called the constant liar strategy:

$$
\hat{y}=c .
$$

Although this might seem silly, this has the advantage of being model independent and has demonstrated surprisingly good performance in practice. Three natural options for the constant, ranging from the most optimistic to most pessimistic, are to impute the maximum, mean, or minimum of the known observed values $\mathbf{y}$.

Seven steps of sequential simulation with the expected improvement acquisition function (7.2) and the kriging believer strategy (11.11) are 
![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-13.jpg?height=948&width=1674&top_left_y=450&top_left_x=226)

Figure 11.5: Batch construction via local penalization of the expected improvement acquisition function (7.2). We select the first point by maximizing the expected improvement, after which the acquisition function is multiplied by a penalty factor discouraging future batch members in that area. We then maximize the updated acquisition function, penalize, and repeat as desired. The bottom panel indicates the locations of several further points selected in this manner.

demonstrated in Figure 11.4. The selected points appear reasonable: the first two exploit the best-seen observation (and are near optimal for a batch size of two; see Figure 11.3), the next two exploit another local optimum, and the remainder explore the domain.

\section*{Batch construction via local penalization}

GONZÁLEZ et al. proposed another general mechanism for extending a given sequential policy (defined by the acquisition function $\alpha$ ) to the batch setting. ${ }^{20}$ Like sequential simulation, we select the first point by

20 J. GONZÁLEZ et al. (2016a). Batch Bayesian Optimization via Local Penalization. AISTATS 2016. maximizing the acquisition function:

$$
x_{1} \in \underset{x \in \mathcal{X}}{\arg \max } \alpha(x ; \mathcal{D}) .
$$

We then incorporate a multiplicative penalty $\varphi\left(x ; x_{1}\right)$ into the acquisition function discouraging future batch members from being in a neighborpenalty from selecting $x_{1}, \varphi\left(x ; x_{1}\right)$ hood of the initial point. This penalty is designed to avoid redundancy between batch members without disrupting the differentiability of the original acquisition function. GONZÁLEZ et al. describe one simple and 21 Any continually differentiable function on a compact set is Lipschitz continuous.

approximate computation for sequential one-step lookahead: § 8.5, p. 171

22 For more discussion on this method, see

J. T. WILSON et al. (2018). Maximizing Acquisition Functions for Bayesian Optimization. NeurIPS 2018.

23 Gauss-Hermite quadrature, as we recommended in the one-dimensional case, does not scale well with dimension.

Monte Carlo approximation of gradient effective penalty function from an estimate of the global maximum and Lipschitz constant of the objective function. ${ }^{21}$ We now select the second batch member by maximizing the penalized acquisition function

$$
x_{2} \in \underset{x \in \mathcal{X}}{\arg \max } \alpha(x ; \mathcal{D}) \varphi\left(x ; x_{1}\right),
$$

after which we apply another penalty and continue in this manner as desired. This process is illustrated in Figure 11.5. Comparing with sequential simulation, the first batch members are very similar; however, there is some divergence in the final stages, with local penalization preferring to revisit the local optimum on the right.

Like sequential simulation, local penalization also entails $b$ optimization problems on $\mathcal{X}$ and is in fact even faster than sequential simulation, as the objective function model does not need to be updated along the way.

\section*{Approximation via Monte Carlo integration}

If we wish to proceed via joint optimization of (11.7) rather than one of the above heuristics, we will often face the complication that the expectation with respect to the observed values $y$ is intractable. We can offer some advice for approximating this quantity and its gradient for a Gaussian process model coupled with an exact or additive Gaussian noise observation model. This abbreviated discussion will closely follow the approach for the sequential case. ${ }^{22}$

First, we write the one-step marginal gain (11.7) as an expectation of the marginal gain in utility $\Delta(\mathbf{x}, \mathbf{y})=u\left(\mathcal{D}^{\prime}\right)-u(\mathcal{D})$ with respect to a multivariate normal belief on the observations (2.20):

$$
\beta(\mathbf{x} ; \mathcal{D})=\int \Delta(\mathbf{x}, \mathbf{y}) \mathcal{N}(\mathbf{y} ; \boldsymbol{\mu}, \mathbf{S}) \mathrm{d} \mathbf{y} .
$$

We may approximate this expectation via Monte Carlo integration ${ }^{23}$ by sampling from this belief. It is convenient to do so by sampling $n$ vectors $\left\{\mathbf{z}_{i}\right\}$ from a standard normal distribution, then transforming these via

$$
\mathrm{z}_{i} \mapsto \boldsymbol{\mu}+\Lambda \mathrm{z}_{i}=\mathrm{y}_{i},
$$

where $\Lambda$ is the Cholesky factor of S: $\Lambda \Lambda^{\top}=\mathrm{S}$. Monte Carlo estimation now gives

$$
\beta(\mathbf{x} ; \mathcal{D}) \approx \frac{1}{n} \sum_{i=1}^{n} \Delta\left(\mathbf{x}, \mathbf{y}_{i}\right) .
$$

As in the sequential case, we may reuse these samples to approximate the gradient of the acquisition function with respect to the proposed observation locations, under mild assumptions (c.4-C.5):

$$
\frac{\partial \beta}{\partial \mathbf{x}} \approx \frac{1}{n} \sum_{i=1}^{n} \frac{\partial \Delta}{\partial \mathbf{x}}\left(\mathbf{x}, \mathbf{y}_{i}\right) ; \quad \frac{\partial \Delta}{\partial x_{j}}\left(\mathbf{x}, \mathbf{y}_{i}\right)=\left[\frac{\partial \Delta}{\partial x_{j}}\right]_{\mathbf{y}}+\frac{\partial \Delta}{\partial \mathbf{y}}\left[\frac{\partial \mu}{\partial x_{j}}+\frac{\partial \Lambda}{\partial x_{j}} \mathbf{z}_{i}\right],
$$

where, in the second expression, we account for the dependence of the $\mathbf{y}$ samples on $\mathbf{x}$ through the transformation (11.14). The gradient of the Cholesky factor $\Lambda$ can be computed efficiently via automatic differentiation. ${ }^{24}$

We will spend the remainder of this section discussing the details of explicit batch extensions of popular sequential policies.

\section*{Expected improvement and knowledge gradient}

Both the expected improvement and knowledge gradient policies have been extended to the batch case, closely following the decision-theoretic approach outlined above.

Unlike its sequential counterpart, computation of batch expected improvement for Gaussian process models is rather involved, even in the case of exact observation. The primary challenge is evaluating the expectation of a multivariate truncated normal distribution, a computation whose difficulty increases rapidly with the batch size. There are exact formulas to compute batch expected improvement ${ }^{25}$ and its gradient ${ }^{26}$ based on the moment-generating function for the truncated multivariate normal derived by TALLIS; ${ }^{27}$ however, these formulas require $b$ evaluations of the $b$-dimensional and $b^{2}$ evaluations of the $(b-1)$-dimensional multivariate normal CDF, itself a notoriously difficult computation that can only effectively be approximated via Monte Carlo methods in dimension greater than four. This limits the utility of the direct approach to relatively small batch sizes, perhaps $b \leq 10$.

For larger batch sizes, GINSBOURGER et al. proposed sequential simulation, and found the constant liar strategy (11.12) using the "optimistic" estimate $\hat{y}=\max y$ to deliver good empirical performance in simulation. ${ }^{28}$ WANG et al. proposed an efficient alternative: joint optimization of batch expected improvement via multistart stochastic gradient ascent using the Monte Carlo estimators in (11.15-11.16) ${ }^{29}$ The authors demonstrated this procedure scaling up to batch sizes of $b=128$ with impressive performance and runtime compared with exact computation and sequential simulation.

A similar approach for approximating the batch knowledge gradient policy was described by wU and FRAZIER. ${ }^{30}$ The overall approach is effectively the same: multistart stochastic gradient ascent relying on (11.15-11.16). An additional complication in estimating the batch knowledge gradient is the global optimization inherent to the global reward (6.5). WU and FRAZIER suggest a discretization approach where the global reward is estimated on a dynamically managed discrete set of points drawn via Thompson sampling from the objective function posterior.

The batch expected improvement acquisition function is illustrated for an example scenario in Figure 11.3, and batch knowledge gradient for the same scenario in Figure 11.6. In both cases, the optimal batch measures on either side of the previously best-seen point; however, knowledge gradient is more tolerant of batches containing only one observation in this region.
24 I. MURRAY (2016). Differentiation of the Cholesky decomposition. arXiv: 1602.07527 [stat.co].

expected improvement: $§ 7.3$ p. 127 knowledge gradient: $§ 7.4$ p. 129

25 C. Chevalier and D. Ginsbourger (2013). Fast Computation of the Multi-Points Expected Improvement with Applications in Batch Selection. LION 7.

26 s. MARMIn et al. (2015). Differentiating the Multipoint Expected Improvement for Optimal Batch Design. MOD 2015.

27 G. M. TALlis (1961). The Moment Generating Function of the Truncated Multi-Normal Distribution. Fournal of the Royal Statistical Society Series B (Methodological) 23(1):223-229.

28 D. GINSBOURger et al. (2010). Kriging is WellSuited to Parallelize Optimization. In: Computational Intelligence in Expensive Optimization Problems.

29 J. WANG et al. (2020a). Parallel Bayesian Global Optimization of Expensive Functions. Operations Research 68(6):1850-1865.

30 J. WU and P. I. FrAZIER (2016). The Parallel Knowledge Gradient Method for Batch Bayesian Optimization. NeurIPS 2016.

Thompson sampling: 7.9, p. 148 - observations

- posterior mean

posterior $95 \%$ credible interval

batch knowledge gradient

$\times \quad$ optimal batch

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-16.jpg?height=537&width=1446&top_left_y=648&top_left_x=311)

Figure 11.6: The batch knowledge gradient acquisition function for an example scenario. The optimal batch exploits the local optimum, but any batch containing at least one point in that neighborhood is near-optimal.

mutual information with $x^{*}:$ § 7.6, p. 139

predictive entropy search: $§ 8.8$, p. 180

31 A. ShaH and z. GHAhramani (2015). Parallel Predictive Entropy Search for Batch Global Optimization of Expensive Objective Functions. NeurIPS 2015.

Thompson sampling: 7.9, p. 148

probability of improvement: § 7.5, p. 131

32 See $§ 7.5$, p. 134 .

33 D. R. JONEs (2001). A Taxonomy of Global Optimization Methods Based on Response Surfaces. Journal of Global Optimization 21(4):345383

\section*{Mutual information with $x^{*}$}

Compared with the difficulty faced in computing (or even approximating) batch analogs of the expected improvement and knowledge gradient acquisition functions, extending the predictive entropy search acquisition function to the batch setting is relatively straightforward. ${ }^{31}$ The mutual information between the observed values $\mathbf{y}$ and the location of the global maximum $x^{*}$ is (compare with (8.36)):

$$
\beta_{x^{*}}(\mathbf{x} ; \mathcal{D})=H[\mathbf{y} \mid \mathbf{x}, \mathcal{D}]-\mathbb{E}\left[H\left[\mathbf{y} \mid \mathbf{x}, x^{*} \mathcal{D}\right] \mid \mathbf{x}, \mathcal{D}\right] .
$$

The first term is the differential entropy of a multivariate normal and may be computed in closed form (A.16). The second term is somewhat difficult to approximate, but no innovation is required in the batch setting beyond the machinery already developed for the sequential case. We may approximate the expectation with respect to $x^{*}$ via Thompson sampling and may approximate $p\left(\mathbf{y} \mid \mathbf{x}, x^{*} \mathcal{D}\right)$ as a multivariate normal following the expectation propagation approach described previously.

\section*{Probability of improvement}

Batch probability of improvement has received relatively little attention, but JONES proposed one simple option in the context of threshold selection. ${ }^{32,33}$ The idea is to find the optimal sequential decisions using a range of improvement thresholds, representing a spectrum of exploration-exploitation tradeoffs. JONEs then recommends a simple clustering procedure to remove redundant points, resulting in a batch (of variable size) reflecting diversity in location and behavior. A compelling aspect of this procedure is that it is naturally nonmyopic, as each batch is explicitly constructed to address both immediate and long-term gain.

This approach is illustrated in Figure $7.12 ;{ }^{34}$ depending on the aggressiveness of the pruning procedure, the constructed batch would contain 2-4 points chosen from the visible clusters.

\section*{Upper confidence bound}

Due to the equivalence between the probability of improvement and upper confidence bound policies for Gaussian processes $(8.22,8.26)$, the procedure proposed by JONEs described above may also be used to realize a simple batch upper confidence bound policy for that model class. In this case, we would design each batch by maximizing the upper confidence bound for a range of confidence parameters, clustering, and pruning.

Several more direct batch upper confidence bound policies have been developed, all variations on a theme. DESAUTELS et al. proposed a strategy - dubbed simply batch upper confidence bound (BUCB) - based on sequential simulation with the kriging believer strategy (11.11) ${ }^{35}$ Batch diversity is automatically encouraged: each point added to the batch globally reduces the upper confidence bound, most dramatically at the locations with the most strongly correlated function values.

The вUсв algorithm was later refined by several authors to encourage more exploration, which can improve both empirical and theoretical performance. Like вUсв, we seed each batch with the maximum of the upper confidence bound (8.25). We now identify the so-called "relevant region" of the domain, defined to be the set of locations whose upper confidence bound exceeds the global maximum of the lower confidence bound. ${ }^{36}$ The intuition behind this region is that the objective value at any point in its complement is - with high probability - lower than at the point maximizing the lower confidence bound and can thus be discarded with some confidence; see the illustration in the margin.

With the relevant region in hand, we design the remaining batch members to promote maximal information gain about the objective on this region. For a Gaussian process, this is intimately related to a diversity-encouraging distribution known as a $k$-determinantal point process ( $k$-DPP) built from the posterior covariance function. ${ }^{37}$ We may proceed by either a simple greedy procedure ${ }^{38}$ or via more nuanced maximization or sampling using methods developed for $k$-DPPs. ${ }^{39}$

These schemes are all backed by strong theoretical analysis, including sublinear cumulative regret (10.3) bounds under suitable conditions.

\section*{Thompson sampling}

The stochastic nature of Thompson sampling enables trivial batch construction by drawing $b$ independent samples of the location of the global maximum (7.19):

$$
\left\{x_{i}\right\} \sim p\left(x^{*} \mid \mathcal{D}\right) \text {. }
$$

34 See p. 136.

maximizing an upper confidence bound: $§ 7.8$, p. 145

35 T. DESAUtels et al. (2014). Parallelizing Exploration-Exploitation Tradeoffs in Gaussian Process Bandit Optimization. fournal of Machine Learning Research 15(119):4053-4103.

36 N. DE FREITAS et al. (2012b). Exponential Regret Bounds for Gaussian Process Bandits with Deterministic Observations. ICML 2012.

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-17.jpg?height=197&width=528&top_left_y=1346&top_left_x=1369)

The (disconnected) relevant region (the darker portions of the uncertainty envelope) for an example Gaussian process. Points outside the region (the lighter portion in between) are unlikely to maximize the objective function.

37 A. KUlesza and B. TASKAR (2012). Determinantal Point Processes for Machine Learning. Foundations and Trends in Machine Learning 5(2-3):123-286.

38 E. contal et al. (2013). Parallel Gaussian Process Optimization with Upper Confidence Bound and Pure Exploration. ECML PKDD 2013.

39 T. KATHURIA et al. (2016). Batched Gaussian Process Bandit Optimization via Determinantal Point Processes. NeurIPS 2016. 40 J. M. HERNÁNDEZ-LOBATO et al. (2017). Parallel and Distributed Thompson Sampling for Large-Scale Accelerated Exploration of Chemical Space. ICML 2017.

41 K. KANDASAmy et al. (2018). Parallelised Bayesian Optimisation via Thompson Sampling. AISTATS 2018.
A remarkable advantage of this policy is that the samples may be generated entirely in parallel, which allows linear scaling to arbitrarily large batch sizes. Batch Thompson sampling has delivered impressive performance in a real-world setting with batch sizes up to $b=500,{ }^{40}$ and is backed by theoretical guarantees on the asymptotic reduction of the simple regret (10.1). ${ }^{41}$

\subsection*{ASYNCHRONOUS OBSERVATION WITH PENDING EXPERIMENTS}

Some situations allow parallel observation with asynchronous execution. For example, when optimizing the result of a computational simulation, access to more than one CPU core (or even better, a cluster of machines) could enable many simulations to be run in parallel. To maximize throughput, we could immediately start a new simulation upon the termination of a previous job, without waiting for the other running processes to finish. An effective optimization policy for this setting must account for the pending experiments when designing each observation.

We may consider a general case where we wish to design a batch of experiments $\mathbf{x} \in \mathcal{X}^{b}$ when another batch of experiments $\mathbf{x}^{\prime} \in \mathcal{X}^{b^{\prime}}$ is under current evaluation, where the number of running and pending experiments may be arbitrary. ${ }^{42}$ Here the action space for the current decision is $\mathcal{A} \in \mathcal{X}$, and we must make the decision under uncertainty both in the observations resulting from the chosen batch, $y$, and the observations resulting from the pending experiments, $\mathbf{y}^{\prime}$.

The one-step expected gain in utility from a set of proposed experiments and the pending experiments is

$$
\beta_{1}\left(\mathbf{x} ; \mathbf{x}^{\prime} \mathcal{D}\right)=\mathbb{E}\left[u\left(\mathcal{D}_{1}\right) \mid \mathbf{x}, \mathbf{x}^{\prime} \mathcal{D}\right]-u(\mathcal{D}),
$$

where $\mathcal{D}_{1}=\mathcal{D} \cup\left\{\left(\mathbf{x}, \mathbf{y}, \mathbf{x}^{\prime}, \mathbf{y}^{\prime}\right)\right\}$. This entails an expectation with respect to the unknown values $\mathrm{y}$ and $\mathrm{y}^{\prime}$ :

$$
\beta_{1}(\mathbf{x} ; \mathcal{D})+u(\mathcal{D})=\iint u\left(\mathcal{D}_{1}\right) p\left(\mathbf{y}, \mathbf{y}^{\prime} \mid \mathbf{x}, \mathbf{x}^{\prime}, \mathcal{D}\right) \mathrm{d} \mathbf{y} \mathrm{d} \mathbf{y}^{\prime}
$$

reduction to synchronous case

This is simply the one-step marginal gain for the combined batch $\mathbf{x} \cup \mathbf{x}^{\prime}$ from the synchronous case (11.7)! The only difference with respect to one-step lookahead is that we can only maximize this score with respect to $\mathbf{x}$ as we are already committed to the pending experiments. Thus a one-step lookahead policy for the asynchronous case can be reduced to maximizing the corresponding score from the synchronous case with some batch members fixed. This reduction has been pointed out by numerous authors, and effectively every batch policy discussed above may be modified with little effort to work in the asynchronous case.

Moving beyond one-step lookahead may be extremely challenging, however, due to the implications of uncertainty in the order of termination for pending experiments. A full treatment would require a model for the time of termination and accounting for how the decision tree may branch after the present decision. Exact computation of even two-step lookahead is likely intractable in most practical situations, but rollout might offer one path forward.

In Bayesian optimization, we typically assume that observations of the objective function are expensive and should be made as sparingly as possible. However, some scenarios offer a potential shortcut: indirect inspection of the system of interest via a cheaper surrogate, such as the output of a computer simulation. In some cases, we may even have access to multiple surrogates of varying cost and fidelity. It is tempting to try to accelerate optimization using these surrogates to guide the search. This is the inspiration for multifidelity optimization, a cost-aware extension of optimization that has received significant attention.

As a motivating example, consider a problem from materials science where we wish to optimize the properties of a material as a function of its composition, process parameters, etc. Materials scientists have several mechanisms available ${ }^{43}$ to investigate a proposed material, ranging from relatively inexpensive computer simulations (molecular dynamics, density functional theory, etc.) to extremely expensive synthesis and characterization in a laboratory. State-of-the-art materials discovery pipelines rely on these computational surrogates to winnow the search space for experimental campaigns.

Automated machine learning provides another motivating example. Suppose we wish to tune the hyperparameters of a model by minimizing validation error after training on a large training set. Although training on the full dataset may be costly, we can estimate performance by training on only a subset of the data ${ }^{44}$ or by terminating the training procedure early. ${ }^{45}$ We may reasonably hope to accelerate hyperparameter tuning by exploiting these noisy, but cost-effective surrogates.

We will outline a decision-theoretic approach to multifidelity optimization below. The complexity of this setting will require readdressing every step of the procedure outlined at the top of this chapter. We will focus on the first three steps, as deriving the optimal policy is mechanical once the model and decision problem are completely specified.

\section*{Formalization of problem and action space}

Suppose that in addition to the objective function $f$, we have access to one-or-more surrogate functions $\left\{f_{i}\right\}: \mathcal{X} \rightarrow \mathbb{R}$, indexed by a parameter $i \in \mathcal{I}$. Most often we take the surrogate functions to form a discrete set, but in some cases we may wish to consider multidimensional and/or continuous surrogate spaces. ${ }^{46}$ We denote the objective function itself with the special index $* \in \mathcal{I}$, writing $f=f_{*}$. We consider an optimization scenario where we may design each observation to be either of the objective or a surrogate as we see fit, by selecting a location $x \in \mathcal{X}$ for our next observation and an index $i \in \mathcal{I}$ specifying the desired surrogate. The action space for each such decision is $\mathcal{A}=\mathcal{X} \times \mathcal{I}$. rollout: $\S 5 \cdot 3$, p. 102

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-19.jpg?height=300&width=534&top_left_y=678&top_left_x=1366)

In multifidelity optimization, we wish to optimize an expensive objective function (the darker curve) aided by access to cheaper but still informative - surrogates (the lighter curves).

materials science applications: Appendix D, p.313

43 "Relatively" should be stressed; these simulations can be quite expensive in absolute terms, but still much cheaper than synthesis.

44 A. KLEIN et al. (2015). Towards Efficient Bayesian Optimization for Big Data. Bayesian Optimization Workshop, NeurIPS 2015.

45 K. SWERSKy et al. (2014). Freeze-Thaw Bayesian Optimization. arXiv: 1406 . 3896 [stat.ML].

surrogate functions, $\left\{f_{i}\right\}$

surrogate index set, $\mathcal{I}$

objective function, $f_{*}$

46 K. KANDASAmy et al. (2017). Multi-Fidelity Bayesian Optimisation with Continuous Approximations. ICML 2017.

action space, $\mathcal{A}=\mathcal{X} \times \mathcal{I}$ joint prior process, $p\left(\left\{f_{i}\right\}\right)$

\author{
observation model, $p(y \mid x, i, \phi)$ \\ posterior distribution, $p\left(\left\{f_{i}\right\} \mid \mathcal{D}\right)$ \\ posterior predictive distribution, \\ $p(y \mid x, i, \mathcal{D})$
}

joint Gaussian processes: § 2.4, p. 26
47 M. A. Álvarez et al. (2012). Kernels for VectorValued Functions: A Review. Foundations and Trends in Machine Learning 4(3):195-266.

48 K. ULRICH et al. (2015). GP Kernels for CrossSpectrum Analysis. NeurIPS 2015.

shared domain covariance, $K_{\mathcal{X}}$ cross-function covariance, $K_{\mathcal{I}}$

49 See pp. $28-29$.

\section*{Modeling surrogate functions and observations}

If surrogate observations are to be useful, they must provide information about the objective function, and the relationship between the objective and its surrogates is captured by a joint model over their values. We first design a joint prior process $p\left(\left\{f_{i}\right\}\right)$ specifying the expected structure of each individual function and the nature of correlations between the functions. Next we must create an observation model linking the value $y$ observed at a point $[x, i]$ to the underlying function value $\phi=f_{i}(x)$ : $p(y \mid x, i, \phi)$. Now, given a set of observed data $\mathcal{D}$, we may derive the posterior belief over the functions, $p\left(\left\{f_{i}\right\} \mid \mathcal{D}\right)$, and the posterior predictive distribution, $p(y \mid x, i, \mathcal{D})$, with which we can reason about proposed observations.

In practice, the joint prior process is usually a joint Gaussian process over the objective and its surrogates. The primary challenge in crafting a joint Gaussian process is in defining cross-covariance functions

$$
K_{i j}=\operatorname{cov}\left[f_{i}, f_{j}\right]
$$

that adequately encode the correlations between the functions of interest, which can take some care to ensure the resulting joint covariance function over the collection $\left\{f_{i}\right\}$ is positive definite.

Fortunately, a great deal of effort has been invested in developing this model class into a flexible and expressive family. ${ }^{47,48}$ One simple construction offering some intuition is the separable covariance

$$
K\left([x, i],\left[x^{\prime}, i^{\prime}\right]\right)=K_{\mathcal{X}}\left(x, x^{\prime}\right) K_{\mathcal{I}}\left(i, i^{\prime}\right),
$$

which decomposes the joint covariance into a covariance function on the domain $K_{\mathcal{X}}$ shared by each individual function and a covariance function between the functions, $K_{\mathcal{I}}$ (which would be a covariance matrix if $\mathcal{I}$ is finite). In this construction, the marginal covariance and cross-covariance functions are all scaled versions of $K_{\mathcal{X}}$, with the $K_{\mathcal{I}}$ covariance scaling each marginal belief and encoding (constant) cross-correlations across functions as well. Figures 2.5 and $2.6^{49}$ illustrate the behavior of this model for two highly correlated functions on a shared one-dimensional domain.

\section*{Defining a utility function}

We have now addressed steps 2 and 3 of our general procedure for multifidelity optimization, by identifying the expanded action space implied by the problem and determining how to reason about the potential outcomes of these actions. Before we can proceed with deriving a policy, however, we must establish preferences over outcomes with a utility function $u(\mathcal{D})$. It is difficult to provide specific guidance for this choice, as these preferences are inherently bound to a given situation. One natural approach would be to choose a cost-aware utility function measuring optimization progress limited to the objective function alone, adjusted objective, $f_{*}$

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-21.jpg?height=206&width=802&top_left_y=508&top_left_x=270)

cost-adjusted expected improvement

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-21.jpg?height=94&width=785&top_left_y=850&top_left_x=270)

surrogate, $f_{1}$

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-21.jpg?height=203&width=797&top_left_y=504&top_left_x=1095)

v next observation location

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-21.jpg?height=169&width=800&top_left_y=775&top_left_x=1096)

Figure 11.7: Cost-adjusted multifidelity expected improvement for a toy scenario. The objective (left) and its surrogate (right) are modeled as a joint Gaussian process with marginals equivalent to the example from Figure 7.1 and constant cross-correlation 0.8 . The cost of observation was assumed to be ten times greater for the objective than its surrogate. Maximizing the cost-adjusted multifidelity expected improvement elects to continue evaluating the surrogate.

for the variable costs of each observation obtained. For example we might quantify the cost of observing the objective function or each of the surrogate functions with values $\left\{c_{i}\right\}$, and adjust a data utility $u^{\prime}(\mathcal{D})$ appropriately, by defining

$$
u(\mathcal{D})=u^{\prime}(\mathcal{D})-\sum_{(x, i) \in \mathcal{D}} c_{i} .
$$

With an appropriate utility function in hand, we can then proceed to derive the optimal policy as usual.

As an example, we can realize an analog of expected improvement (7.2) through a suitable redefinition of the simple reward (6.3). Suppose that at termination we wish to recommend a location visited during optimization, with any fidelity, using a risk-neutral utility. Given a multifidelity dataset $\mathcal{D}=([\mathbf{x}, \mathbf{i}], \mathbf{y})$, the expected utility of this recommendation would be

$$
u^{\prime}(\mathcal{D})=\max _{x^{\prime} \in \mathbf{x}} \mu_{\mathcal{D}}\left(\left[x^{\prime} * *\right]\right),
$$

the maximum of the posterior mean for the objective function at the observed locations.

Figure 11.7 illustrates one-step lookahead policy with this utility function for a one-dimensional objective function $f_{*}$ (left) and a surrogate $f_{1}$ (right). The marginal belief about each function is a Gaussian process identical to our running example from Chapter 7; see Figure 7.1. These are coupled together via the separable covariance (11.18) with $K_{\mathcal{I}}(*, *)=K_{\mathcal{I}}(1,1)=1$ and cross-correlation $K_{\mathcal{I}}(1, *)=0.8$. We begin with three surrogate observations and compute the cost-adjusted expected improvement as described above, where the cost of observing the objective was set to ten times that of the surrogate. In this case, the one-step optimal decision is to continue evaluating the surrogate around the best-seen surrogate observation.

multifidelity expected improvement

example and discussion - objective, $f_{*}$

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-22.jpg?height=817&width=1645&top_left_y=545&top_left_x=157)

Figure 11.8: A simulation of optimization with the cost-adjusted multifidelity expected improvement starting from the scenario in Figure 11.7. We simulate sequential observations of either the objective or the surrogate, illustrated using the running tick marks. The lighter marks correspond to surrogate observations and the heavier marks correspond to objective observations. The optimum was found after 10 objective observations and 32 surrogate observations. The prior (top) and posterior (bottom) of the objective function conditioned on all observations are also shown.

50 D. HUANG et al. (2006a). Sequential Kriging Optimization Using Multiple-Fidelity Evaluations. Structural and Multidisciplinary Optimization 32(5):369-382.

51 v. PICHENy et al. (2013a). Quantile-Based Optimization of Noisy Computer Experiments with Tunable Precision. Technometrics 55(1): $2-13$.

52 J. wu et al. (2019). Practical Multi-Fidelity Bayesian Optimization for Hyperparameter Tuning. UAI 2019.

53 K. KANDASAmy et al. (2016). Gaussian Process Bandit Optimisation with Multi-Fidelity Evaluations. NeurIPS 2016

54 K. KANDASAmy et al. (2017). Multi-Fidelity Bayesian Optimisation with Continuous Approximations. ICML 2017.

55 K. swersky et al. (2013). Multi-Task Bayesian Optimization. NeurIPS 2013.
Figure 11.8 simulates sequential multifidelity optimization using this policy; here the optimum was discovered after only 10 evaluations of the objective, guided by 32 observations of the cheaper surrogate. A remarkable feature we can see in the posterior is that all evaluations made of the objective function are above the prior mean, nearly all with $z$-scores of approximately $z=1$ or greater. This can be ascribed not to extreme luck, but rather to efficient use of the surrogate to rule out regions unlikely to contain the optimum.

Multifidelity Bayesian optimization has enjoyed sustained interest from the research community, and numerous policies have been available. These include adaptations of the expected improvement, ${ }^{50,51}$ knowledge gradient,,$^{52}$ upper confidence bound, ${ }^{53,54}$ and mutual information with $x^{* 55,56}$ and $f^{* 57}$ acquisition functions, as well as novel approaches. ${ }^{58}$

\subsection*{MULTITASK OPTIMIZATION}

Multitask optimization addresses the sequential or simultaneous optimization of multiple objectives $\left\{f_{i}\right\}: \mathcal{X} \rightarrow \mathbb{R}$ representing performance on related tasks. Like multifidelity optimization, the underlying idea in multitask optimization is that if performance on the various tasks is correlated as a function of the input, we may accelerate optimization by transferring information between tasks.

As a motivating example of sequential multitask optimization, consider a web service wishing to retune the parameters of an ad placement algorithm on a regular basis to maximize revenue in the current climate. Here the revenue at each epoch represents the different tasks to be optimized, which are optimized individually one after another. Although we could treat each optimization problem separately, they are clearly related, and with some care we may be able to use past performance to provide a "warm start" to each new optimization problem rather than start from scratch.

We may also consider the simultaneous optimization of performance across tasks. For example, a machine learning practitioner may wish to tune model hyperparameters to maximize the average predictive performance on several related datasets. ${ }^{59}$ A naïve approach would formulate the problem as maximizing a single objective defined to be the average performance across tasks, with each evaluation entailing retraining the model for each dataset. However, this would be potentially wasteful, as we may be able to eliminate poorly performing hyperparameters with high confidence after training on a fraction of the datasets. A multitask approach would model each objective function separately (perhaps jointly) and consider evaluations of single-task performance to efficiently maximize the combined objective.

SWERSKY et al. described a particularly clever realization of this idea: selecting model hyperparameters via cross validation. ${ }^{60}$ Here we recognize the predictive performance on each validation fold as being correlated due to shared training data across folds. If we can successfully share information across folds, we may potentially accelerate cross validation by iteratively selecting (hyperparameter, fold index) pairs rather than training proposed hyperparameters on every fold each time.

\section*{Formulation and approach}

Let $\left\{f_{i}\right\}: \mathcal{X} \rightarrow \mathbb{R}$ be the set of objective functions we wish to consider, representing performance on the relevant tasks. As with multifidelity optimization, the key enabler of multitask optimization is a joint model $p\left(\left\{f_{i}\right\}\right)$ over the tasks and a joint observation model $p(y \mid x, i, \phi)$ over evaluations thereof; this joint model allows us to share information between the tasks. This could, for example, take the form of a joint Gaussian process, as discussed previously.

Once this model has been chosen, we can turn to the problem of designing a multitask optimization policy. If each task is to be solved one at a time, a natural approach would be to design the utility function to ultimately evaluate performance on that task only. In this case, the problem devolves to single-objective optimization, and we may use any of the approaches discussed earlier in the book to derive a policy. The
56 y. ZHANG et al. (2017). Information-Based Multi-Fidelity Bayesian Optimization. Bayesian Optimization Workshop, NeurIPS 2017.

57 S. TAKENO et al. (2020). Multi-Fidelity Bayesian Optimization with Max-value Entropy Search and Its Parallelization. ICML 2020.

58 J. song et al. (2019). A General Framework for Multi-Fidelity Bayesian Optimization with Gaussian Processes. AISTATS 2019.

simultaneous tasks

59 This could also be formulated as a scalarized version of a multiobjective optimization problem, discussed in the next section.

6o K. swersky et al. (2013). Multi-Task Bayesian Optimization. NeurIPS 2013.

modeling task objectives and observations

joint GPs for modeling multiple functions: $\S 11.5$, p. 264

sequential tasks current task, $f$
![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-24.jpg?height=364&width=1630&top_left_y=543&top_left_x=158)

Figure 11.9: A demonstration of sequential multitask optimization. Left: a prior distribution over an objective function along with 13 observations selected by maximizing expected improvement, revealing the global maximum with the last evaluation. Right: the posterior distribution over the same objective conditioned on the observations of the two functions (now interpreted as related tasks) in Figure 11.8. The global maximum is now found after three observations due to the better informed prior.

only difference is that the objective function model is now informed from our past experience with other tasks; as a result, our initial optimization decisions can be more targeted.

This procedure is illustrated in Figure 11.9. Both panels illustrate the optimization of an objective function by sequentially maximizing expected improvement. The left panel begins with no information and locates the global optimum after 13 evaluations. The right panel begins the process instead with a posterior belief about the objective conditioned on the data obtained from the two functions in Figure 11.8, modeled as related tasks with cross-correlation 0.8 . Due to the better informed initial belief, we now find the global optimum after only three evaluations.

The case of simultaneous multitask optimization - where we may evaluate any task objective with each observation - requires somewhat more care. We must now design a utility function capturing our joint performance across the tasks and design each observation with respect to this utility. One simple option would be to select utility functions $\left\{u_{i}\right\}$ quantifying performance on each task separately and then take a weighted average:

$$
u(\mathcal{D})=\sum_{i} w_{i} u_{i}(\mathcal{D})
$$

This could be further adjusted for (perhaps task- and/or input-dependent) observation costs if needed. Now we may write the expected marginal gain in this combined utility as a weighted average of the expected marginal gain on each separate task. One-step lookahead would then maximize the weighted acquisition function

$$
\alpha([x, i] ; \mathcal{D})=\sum_{i} w_{i} \alpha_{i}([x, i] ; \mathcal{D})
$$

over all possible observation location-task pairs $[x, i]$, where $\alpha_{i}$ is the one-step expected marginal gain in $u_{i}$. Note that observing a single task 

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-25.jpg?height=297&width=1630&top_left_y=457&top_left_x=267)

Figure 11.10: A simple multiobjective optimization example with two objectives $\left\{f_{1}, f_{2}\right\}$ on a one-dimensional domain. We compare four identified points: $x_{1}$, the global optimum of $f_{1}, x_{2}$, the global optimum of $f_{2}, x_{3}$, a compromise with relatively high values of both objectives, and $x_{4}$, a point with relatively low values of both objectives.

could in fact yield improvement in all utilities due to information sharing through the joint belief on task objectives.

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-25.jpg?height=26&width=246&top_left_y=1176&top_left_x=337)

Like multitask optimization, multiobjective optimization addresses the simultaneous optimization of multiple objectives $\left\{f_{i}\right\}: \mathcal{X} \rightarrow \mathbb{R}$. However, whereas in multitask optimization we usually seek to identify the global optimum of each function separately, in multiobjective optimization we seek to identify points jointly optimizing all of the objectives. Of course, this is not possible unequivocally unless all of the maxima happen to coincide, as we may need to sacrifice the value of one objective in order to increase another. Instead, we may consider the optimization of various tradeoffs between the objectives and rely on subjective preferences to determine which option is preferred in a given scenario. Multiobjective optimization may then be posed as the identification of one-or-more optimal tradeoffs among the objectives to support this analysis.

A classic example of multiobjective optimization can be found in finance, where we seek investment portfolios optimizing tradeoffs between risk (often captured by the standard deviation of return) and reward (often captured by the expected return). Generally, investments with higher risk yield higher reward, but the optimal investment strategy depends on the investor's risk tolerance - for example, when capital preservation is paramount, low-risk, low-reward investments are prudent. The set of investment portfolios maximizing reward for any given risk is known as the Pareto frontier, ${ }^{61}$ which jointly span all rational solutions to the given problem. We generalize this concept below.

\section*{Pareto optimality}

To illustrate the tradeoffs we may need to consider during multiobjective optimization, consider the two objectives in Figure 11.10. The first objective $f_{1}$ has its global maximum at $x_{1}$, which nearly coincides with the global minimum of the second objective $f_{2}$. The reverse is true in the other direction: the global maximum of the second objective, $x_{2}$, risk: standard deviation of return

reward: expected value of return

61 In modern portfolio theory the term "efficient frontier" is more common for the equivalent concept.

Pareto frontier 

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-26.jpg?height=303&width=1651&top_left_y=454&top_left_x=154)

Figure 11.11: The objectives from Figure 11.10 with the Pareto optimal solutions highlighted. All points along the intervals marked on the horizontal axis are Pareto optimal, with the highlighted corresponding objective values forming the Pareto frontier (see margin). Points $x_{1}, x_{2}$, and $x_{3}$ are Pareto optimal; $x_{4}$ is not.

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-26.jpg?height=545&width=531&top_left_y=1024&top_left_x=157)

The regions dominated by points $x_{1}$ and $x_{3}$ in Figure 11.10 are shaded. $x_{4}$ is dominated by both, and $x_{2}$ by neither.

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-26.jpg?height=528&width=531&top_left_y=1729&top_left_x=157)

The Pareto frontier for the scenario in Figures 11.10-11.11. The four components correspond to the highlighted intervals in Figure 11.11.

62 K. M. MIETTINEN (1998). Nonlinear Multiobjective Optimization. Kluwer Academic Publishers. achieves a relatively low value on the first objective. Neither point can be preferred over the other on any objective grounds, but a rational agent may have subjective preferences over these two locations depending on the relative importance of the two objectives. Some agents might even prefer a compromise location such as $x_{3}$ to either of these points, as it achieves relatively high - but suboptimal - values for both objectives.

It is clearly impossible to identify an unambiguously optimal location, even in this relatively simple example. We can, however, eliminate some locations as plainly subpar. For example, consider the point $x_{4}$ in Figure 11.10. Assuming preferences are nondecreasing with each objective, no rational agent would prefer $x_{4}$ to $x_{3}$ as the latter point achieves higher value for both objectives. We may formalize this intuition by defining a partial order on potential solutions consistent with this reasoning.

We will say that a point $x$ dominates another point $x^{\prime}$, denoted $x^{\prime}<x$, if no objective value is lower at $x$ than at $x^{\prime}$ and if at least one objective value is higher at $x$ than at $x$. Assuming preferences are consistent with nondecreasing objective values, no agent could prefer a dominated point to any point dominating it. This concept is illustrated in the margin for the example from Figure 11.10: all points in the shaded regions are dominated, and in particular $x_{4}$ is dominated by both $x_{1}$ and $x_{3}$. On the other hand, none of $x_{1}, x_{2}$, or $x_{3}$ is dominated by any of the other points.

A point $x \in \mathcal{X}$ that is not dominated by any other point is called Pareto optimal, and the image of all Pareto optimal points is called the Pareto frontier. The Pareto frontier is a central concept in multiobjective optimization - it represents the set of all possible solutions to the problem consistent with weakly monotone preferences for the objectives. Figure 11.11 shows the Pareto optimal points for our example from Figure 11.10, which span four disconnected intervals. We may visualize the Pareto frontier by plotting the image of this set, as shown in the margin.

There are several approaches to multiobjective optimization that differ in terms of when exactly preferences are elicited. ${ }^{62}$ So-called $a$ posteriori methods seek to identify the entire Pareto frontier for a given problem, with preferences among possible solutions to be determined afterwards. In contrast, a priori methods assume that preferences are already predetermined, allowing us to seek a single Pareto optimal so- lution consistent with those preferences. Bayesian realizations of both types of approaches have been realized, as we discuss below.

\section*{Formalization of decision problem and modeling objectives}

Nearly all Bayesian multiobjective optimization procedures model each decision as choosing a location $x \in \mathcal{X}$, where we make an observation of every objective function. We could also consider a setting analogous to multitask or multifidelity optimization where we observe only one objective at a time, but this idea has not been sufficiently explored. For the following discussion, we will write $\mathbf{y}$ for the vector-valued observation resulting from an observation at a given point $x$, with $y_{i}$ being associated with objective $f_{i}$.

As in the previous two sections, we build a joint model for the objectives $\left\{f_{i}\right\}$ and our observations of them via a prior process $p\left(\left\{f_{i}\right\}\right)$ and an observation model $p\left(\mathbf{y} \mid x,\left\{\phi_{i}\right\}\right)$. The models are usually taken to be independent Gaussian processes on each objective combined with standard observation models. Foint Gaussian processes could also be used when appropriate, but a direct comparison of independent versus dependent models did not demonstrate improvement when modeling correlations between objectives, perhaps due to an increased burden in estimating model hyperparameters. ${ }^{63}$

\section*{Expected hypervolume improvement}

The majority of Bayesian multiobjective optimization approaches are a posteriori methods, seeking to identify the entire Pareto frontier or a representative portion of it. Many algorithms represent one-step lookahead for some utility function evaluating progress on this task.

One popular utility for multiobjective optimization is the volume under an estimate of the Pareto frontier, also known as the $\mathcal{S}$ metric. ${ }^{64}$ Namely, given observations of the objectives, we may build a natural statistical lower bound of the Pareto frontier by eliminating the outcomes dominated by the observations with high confidence. When observations are exact, we may simply enumerate the dominated regions and take their union; when observations are corrupted by noise, we may use a statistical lower bound of the underlying function values instead. ${ }^{65}$ This procedure is illustrated in the margin for our running example, where we have made exact observations of the objectives at three mutually nondominating locations; see Figure 11.12. The upper-right boundary of the dominated region is a lower bound of the true Pareto frontier.

To evaluate progress on mapping out the Pareto frontier, we consider the volume of space dominated by the available observations and bounded below by an identified, clearly suboptimal reference point $\mathbf{r}$; see the shaded area in the margin. The reference point is necessary to ensure the dominated volume does not diverge to infinity. Assuming the reference point is chosen such that it will definitely be dominated, this utility is always positive and is maximized when the true Pareto fron- vector of observations at $x, \mathrm{y}$

63 J. SVENSON and T. SANTNER (2016). Multiobjective Optimization of Expensive-to-evaluate Deterministic Computer Simulator Models. Computational Statistics and Data Analysis 94: 250-264.
64 E. ZITZLER (1999). Evolutionary Algorithms for Multiobjective Optimization: Methods and Applications. Ph.D. thesis. Eidgenössische Technische Hochschule Zürich. [§ 3.1]

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-27.jpg?height=537&width=534&top_left_y=1676&top_left_x=1366)

A lower bound of the Pareto frontier for our example given the data in Figure 11.12.

65 M. EMMERICH and B. NAUJOKS (2004). Metamodel Assisted Multiobjective Optimisation Strategies and Their Application in Airfoil Design. In: Adaptive Computing in Design and Manufacture VI. Figure 11.12: The posterior belief about our example objectives from Figure 11.10 given observations at the marked locations. The beliefs are separated vertically (by an arbitrary amount) for clarity, with the belief over the other function shown for reference.

Figure 11.13: The expected hypervolume improvement acquisition function for the above example.

66 M. FLEISCHER (2003). The Measure of Pareto Optima: Applications to Multi-Objective Metaheuristics. EMO 2003.

67 M. T. M. EMmerich et al. (2006). Single- and Multiobjective Evolutionary Optimization Assisted by Gaussian Random Field Metamodels. IEEE Transactions on Evolutionary Computation $10(4): 421-439$.

68 w. PONWEISER et al. (2008). Multiobjective Optimization on a Limited Budget of Evaluations Using Model-Assisted $\mathcal{S}$-Metric Selection. PPSN X.

69 к. YANG et al. (2019b). Multi-Objective Bayesian Global Optimization Using Expected Hypervolume Improvement Gradient. Swarm and Evolutionary Computation 44:945-956.

70 K. yANG et al. (2019a). Efficient Computation of Expected Hypervolume Improvement Using Box Decomposition Algorithms. Journal of Global Optimization 75(1):3-34.
![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-28.jpg?height=660&width=1014&top_left_y=460&top_left_x=772)

— expected hypervolume improvement

$\nabla$ next observation location

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-28.jpg?height=200&width=1017&top_left_y=1208&top_left_x=771)

tier is revealed by the observed data ${ }^{66}$ Therefore it provides a sensible measure of progress for a posteriori multiobjective optimization.

The one-step marginal gain in this utility is known as expected hypervolume improvement (EHVI) ${ }^{67,68}$ and serves as a popular acquisition function. This score is shown in Figure 11.13 for our example; the optimal decision attempts to refine the central portion of the Pareto frontier, but many alternatives are almost as favorable due to the roughness of the current estimate. Computation of EHvi is involved, and its cost grows considerably with the number of objectives. The primary difficulty is enumerating and integrating with respect to the lower bound to the Pareto front, which can become a complex region in higher dimensions. However, efficient algorithms are available for computing $\mathrm{EVHI}^{69}$ and its gradient. ${ }^{70}$

\section*{Information-theoretic a posteriori methods}

Several popular information-theoretic policies for single-objective optimization have been adapted to a posteriori multiobjective optimization. The key idea behind these methods is to approximate the mutual information between a joint observation of the objectives and either the set of Pareto optimal points (in the domain), $\mathcal{X}^{*}$, or the Pareto frontier (in the codomain), $\mathcal{F}^{*}$. These approaches operate by maximizing the predictive form of mutual information (8.35) and largely follow the parallel single-objective cases in their approximation. HERNÁNDEZ-LOBATO et al. proposed maximizing the mutual information between the observations $y$ realized at a proposed observation location $x$ and the set of Pareto optimal points $\mathcal{X}^{*}$ :

$$
\alpha_{\mathrm{PESMO}}(x ; \mathcal{D})=H[\mathbf{y} \mid x, \mathcal{D}]-\mathbb{E}_{\mathcal{X}^{*}}\left[H\left[\mathbf{y} \mid x, \mathcal{D}, \mathcal{X}^{*}\right] \mid x, \mathcal{D}\right],
$$

calling their policy predictive entropy search for multiobjective optimization (PESMO). ${ }^{71}$ As in the single-objective case, for Gaussian process models with additive Gaussian noise, the first term of this expression can be computed exactly as the differential entropy of a multivariate normal distribution (A.16). However, the second term entails two computational barriers: computing an expectation with respect to $\mathcal{X}^{*}$ and conditioning our objective function belief on this set. The authors provide approximations for each of these tasks for Gaussian process models based on Gaussian expectation propagation; these are reminiscent of the procedure used in predictive entropy search.

BELAKARIA et al. meanwhile proposed maximizing the mutual information with the Pareto frontier $\mathcal{F}^{*}$,

$$
\alpha_{\text {MESMO }}(x ; \mathcal{D})=H[\mathbf{y} \mid x, \mathcal{D}]-\mathbb{E}_{\mathcal{F}^{*}}\left[H\left[\mathbf{y} \mid x, \mathcal{D}, \mathcal{F}^{*}\right] \mid x, \mathcal{D}\right] .
$$

The authors dubbed the resulting policy max-value entropy search for multiobjective optimization (MESMO). ${ }^{72}$ This policy naturally shares many features with the PESMO policy. Again the chief difficultly is in addressing the expectation and conditioning with respect to $\mathcal{F}^{*}$ in the second term of the mutual information; however, both of these tasks are rendered somewhat easier by working in the codomain. To approximate the expectation with respect to $\mathcal{F}^{*}$, the authors propose a Monte Carlo approach where posterior samples of the objectives are generated efficiently using a sparse-spectrum approximation, which are fed into an off-the-shelf, exhaustive multiobjective optimization routine. Conditioning y on a realization of the Pareto frontier now entails appropriate truncation of its multivariate normal belief.

\section*{A priori methods and scalarization}

When preferences regarding tradeoffs among the objective functions can be sufficiently established prior to optimization, perhaps with input from a domain expert, we may reduce multiobjective optimization to a single objective problem by explicitly maximizing the desired criterion. This is called scalarization and is a prominent a priori method. Reframing in terms of a single objective offers the obvious benefit of allowing us to appeal to the expansive methodology for that purpose we have built up throughout the course of this book.

Scalarization is an important component of some a posteriori multiobjective optimization methods as well. The idea is to construct a family of exhaustive parametric scalarizations of the objectives such that the solution to any such problem is Pareto optimal, and that by spanning the parameter range we may reveal the entire Pareto frontier one point at a time.
71 D. HERNÁNDEZ-LOBATO et al. (2016a). Predictive Entropy Search for Multi-Objective Bayesian Optimization. ICML 2016.

predictive entropy search: $\S 8.8$, p. 180

72 S. BELAKARIA et al. (2019). Max-value Entropy
Search for Multi-Objective Bayesian Optimization. NeurIPS 2019.

sparse spectrum approximation: § 8.7, p. 178

scalarization

a posteriori optimization via scalarization 

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-30.jpg?height=184&width=1631&top_left_y=479&top_left_x=158)

Figure 11.14: A series of linear scalarizations of the example objectives from Figure 11.10. One end of the spectrum is $f_{1}$, and the other end is $f_{2}$.

vector of objective values at $x, \phi$ scalarization function, $g$

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-30.jpg?height=525&width=529&top_left_y=1388&top_left_x=158)

The marked points represent the maxima of the linear scalarizations in Figure 11.14.
73 J. KNOwles (2005). Parego: A Hybrid Algorithm with On-Line Landscape Approximation for Expensive Multiobjective Optimization Problems. IEEE Transactions on Evolutionary Computation 10(1):50-66.

74 D. GOLOvin and Q. ZHANG (2020). Random Hypervolume Scalarizations for Provable MultiObjective Black Box Optimization. ICML 2020.
Let $\phi$ denote the vector of objective function values at an arbitrary location $x$, defining $\phi_{i}=f_{i}(x)$. A scalarization function $g: x \mapsto g(\phi) \in \mathbb{R}$ maps locations in the domain to scalars determined by their objective values. We may interpret the output as defining preferences over locations in a natural manner: namely, if $g(\boldsymbol{\phi})>g\left(\boldsymbol{\phi}^{\prime}\right)$, then the outcomes at $x$ are preferred to those at $x^{\prime}$ in the scalarization. With this interpretation, a scalarization function allows us to recast multiobjective optimization as a single-objective problem by maximizing $g$ with respect to $x$.

A scalarization function can in principle be arbitrary, and a priori multiobjective optimization can be framed in terms of maximizing any such function. However, several tunable scalarization functions have been described in the literature that may be used in a general context.

A straightforward and intuitive example is linear scalarization:

$$
g_{\mathrm{LIN}}(x ; \mathbf{w})=\sum_{i} w_{i} \phi_{i},
$$

where each weight $w_{i}$ is nonnegative. A range of linear scalarizations for our running example is shown in Figure 11.14, here constructed to smoothly interpolate between the two objectives. The maximum of a linear scalarization is guaranteed to lie on the Pareto frontier; however, not every Pareto optimal point can be recovered in this manner unless the frontier is strictly concave. That is the case for our example, illustrated in the marginal figure. If we model each objective with a (perhaps joint) Gaussian process, then the induced belief about any linear scalarization is conveniently also a Gaussian process, so no further modeling would be required for the scalarization function itself.

Another choice that has seen some use in Bayesian optimization is augmented Chebyshev scalarization, which augments the linear scalarization (11.19) with an additional, nonlinear term:

$$
g_{\mathrm{AC}}(x ; \mathbf{w}, \rho)=\min _{i}\left[w_{i}\left(\phi_{i}-r_{i}\right)\right]+\rho g_{\mathrm{LIN}}(x ; \mathbf{w}) .
$$

Here $\mathbf{r}$ is a reference point as above and $\rho$ is a small nonnegative constant; KNOWLEs for example took $\rho=0.05 .^{73}$ The augmented Chebyshev scalarization function has the benefit that all points on the Pareto frontier can be realized by maximizing with respect to some corresponding setting of the weights, even if the frontier is nonconcave. GOLOVIN and zHANG proposed a similar hypervolume scalarization related to the $\mathcal{S}$-metric. ${ }^{74}$ - posterior mean, $f_{1}$

- posterior mean, $f_{2}$

- observations
![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-31.jpg?height=638&width=1056&top_left_y=636&top_left_x=228)

Several authors have derived Bayesian methods for a posteriori multiobjective optimization by solving a series of carefully constructed scalarized problems. KNOWLEs for example proposed sampling random weights for the augmented Chebyshev scalarization (11.20) and optimizing the resulting objective, repeating this process until satisfied. ${ }^{73}$ PARIA et al. proposed a similar approach incorporating a prior distribution over the parameters of a chosen scalarization function to allow the user to focus on an identified region of the Pareto frontier if desired.$^{75}$ The procedure then proceeds by repeatedly sampling from that distribution and maximizing the resulting objective. This is effectively the same as KNOwLEs's approach, but the authors were able to establish theoretical regret bounds for their procedure. These results were improved by GOLOVIN and ZHANG, who derived (stronger) theoretical guarantees for a procedure based on repeatedly optimizing their proposed hypervolume scalarization.

\section*{Other approaches}

ZULUAGA et al. outlined an intriguing approach to a posteriori multiobjective optimization ${ }^{76}$ wherein the problem was recast as an active learning problem. ${ }^{77}$ Namely, the authors considered the binary classification problem of predicting whether a given observation location was (approximately) Pareto optimal or not, then designed observations to maximize expected performance on this task. Their algorithm is supported by theoretical bounds on performance and performed admirably compared with KNOwLEs's algorithm discussed above. ${ }^{73}$

PICHENY proposed a spiritually similar approach also based on sequentially reducing a measure of uncertainty in the Pareto frontier. ${ }^{78}$
Figure 11.15: The probability of nondominance by the available data for our running example (above).
75 B. PARIA et al. (2019). A Flexible Framework for Multi-Objective Bayesian Optimization Using Random Scalarizations. UAI 2019.
76 M. zuluaga et al. (2016). $\varepsilon$-PAL: An Active Learning Approach to the Multi-Objective Optimization Problem. Fournal of Machine Learning Research 17(104):1-32.

77 B. SETtLes (2012). Active Learning. Morgan \& Claypool.

78 v. PICHENY (2015). Multiobjective Optimization Using Gaussian Process Emulators via Stepwise Uncertainty Reduction. Statistics and Computing 25(6):1265-1280. conditioning a GP on derivative observations: $\S 2.6, \mathrm{p} .32$

79 This could be, for example, exact observation or additive Gaussian noise. Recall that for a GP on $f$, the joint distribution of $(\phi, \nabla \phi)$ is multivariate normal (2.28), so for these choices the predictive distribution of $(y, g)$ is jointly Gaussian and exact inference is tractable.
Given a dataset $\mathcal{D}=(\mathbf{x}, \mathbf{y})$, we consider the probability that a given point $x \in \mathcal{X}$ is not dominated by any point in the current dataset (that is, the probability $x$ may lie on the Pareto frontier but not yet be discovered), $\operatorname{Pr}(x \nless \mathbf{x} \mid \mathcal{D})$. The integral of this score over the domain can be interpreted as a measure of uncertainty in the Pareto frontier as determined by the data, and negating this measure provides a plausible utility function for a posteriori multiobjective optimization:

$$
u(\mathcal{D})=-\int \operatorname{Pr}(x \nprec \mathbf{x}) \mathrm{d} x .
$$

This probability is plotted for our running example in Figure 11.15; here, there is a significant probability for many points to be nondominated by the rather sparse available data, indicating a significant degree of uncertainty in our understanding of the Pareto frontier. However, as the Pareto frontier is increasingly well determined by the data, this probability will vanish globally and the utility above will tend toward its maximal value of zero. After motivating this score, PICHENy proceeds to recommend designing observations via one-step lookahead.

\section*{GRADIENT OBSERVATIONS}

Bayesian optimization is often described as a "derivative-free" approach to optimization, but this characterization is misleading. Although it is true that Bayesian optimization methods do not require the ability to observe derivatives, it is certainly not the case that we cannot make use of such observations when available. In fact, it is straightforward to condition a Gaussian process on derivative observations, even if corrupted by noise, and so from a modeling perspective we are already done.

Of course, ideally, our policy should also consider the acquisition of derivative information due to its influence on our belief and the utility of collected data, of which they now form a part. To do so is by now relatively simple. A fairly general scheme would assume that an observation at $x$ yields a pair of measurements $(y, g)$ respectively related to $(\phi, \nabla \phi)$ via a joint observation model. ${ }^{79}$ We can then compute the one-step expected marginal gain in utility from observing these values:

$$
\alpha_{1}(x ; \mathcal{D})=\iint\left[u\left(\mathcal{D}_{1}\right)-u(\mathcal{D})\right] p(y, \mathbf{g} \mid x, \mathcal{D}) \mathrm{d} y \mathrm{dg},
$$

where the updated dataset $\mathcal{D}_{1}$ will reflect the entire observation $(x, y, g)$. Induction on the horizon gives the optimal policy as usual.

Figure 11.16 compares derivative-aware and derivative-unaware versions of the knowledge gradient (7.4) (assuming exact observation) for an example scenario. The derivative-aware version dominates the derivativeunaware one, as the acquisition of more information naturally leads to a greater expected marginal gain in utility. When derivative information is unavailable, the optimal decision is to evaluate nearby the previously best-seen point, effectively to estimate the derivative via finite differencing; when derivative information is available, an observation at this 

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-33.jpg?height=460&width=1646&top_left_y=450&top_left_x=268)

Figure 11.16: The knowledge gradient acquisition function for an example scenario reflecting the expected gain in global reward (6.5) provided exact observations of the objective function and its derivative. The vanilla knowledge gradient (7.4) based on an observation of the objective alone is shown for reference.

location yields effectively the same expected gain. However, we can fare even better in expectation by moving a bit farther astray, where the derivative information is less redundant.

When working with Gaussian process models in high dimension, it may not be wise to augment each observation of the objective with a full observation of the gradient due to the cubic scaling of inference. However, the scheme outlined above opens the door to consider the observation of any measurement related to the gradient. For example, we might condition on the value of a directional derivative, reducing the measurement to a scalar regardless of dimension and limiting computation. Such a scheme was promoted by wu et al., who considered the acquisition of a single coordinate of the gradient; this could be extended to non-axis-aligned directional derivatives without major complication. ${ }^{80}$ We could also consider a "multifidelity" extension where we weigh various possible gradient observations (including none at all) in light of their expected utility and the cost of acquisition/inference.

\section*{STOCHASTIC AND ROBUST OPTIMIZATION}

In some applications, the performance of a given system configuration depends on stochastic exogenous factors. For example, consider optimizing the parameters of a mobile robot's gait to maximize some tradeoff of stability, efficiency, and speed. These objectives depend not only on the gait parameters, but also on the nature of the environment - such as the composition, slope, etc. of the surface to be traversed - which cannot be controlled by the robot at the time of performance and may vary over repeated sessions. In this scenario, we may seek to optimize some measure of performance accounting for uncertainty in the environment.

To model this and related scenarios, we may consider an objective function $g(x, \omega)$, where $x$ represents a configuration we wish to optimize, and $\omega$ represents relevant parameters of the environment not under scaling of Gaussian process inference: § 9.1, p. 201

80 J. wU et al. (2017). Bayesian Optimization with Gradients. NeurIPS 2017.

partially controllable objective function, $g(x, \omega)$ environmental parameter, $\omega$ special case: perturbed parameters
81 This is a heavily overloaded phrase, as it is used as an umbrella term for optimization involving any random elements. It is also used, for example, to describe optimization from noisy observations, to include methods such as stochastic gradient descent. As noisy observations are commonplace in Bayesian optimization, we reserve the phrase for stochastic environmental parameters only.

82 B. J. Williams et al. (200o). Sequential Design of Computer Experiments to Minimize Integrated Response Functions. Statistica Sinica $10(4): 1133-1152$.

83 If $g$ has distribution $\mathcal{G P}(\mu, K)$, then $f$ has distribution $\mathcal{G P}(m, C)$ with:

$$
\begin{aligned}
& m=\int \mu(x, \omega) p(\omega) \mathrm{d} \omega \\
& C=\iint K\left([x, \omega],\left[x^{\prime}, \omega^{\prime}\right]\right) p\left(\omega, \omega^{\prime}\right) \mathrm{d} \omega \mathrm{d} \omega^{\prime} .
\end{aligned}
$$

Bayesian quadrature: $§ 2.6$, p. 33 our control. In this context, $\omega$ is called an environmental parameter or environmental variable. A notable special case of this setup is where the environmental parameters represent a perturbation to the optimization parameters at the time of performance:

$$
g(x, \omega)=h(x+\omega),
$$

where $h$ is an objective function to be optimized as usual. Such a model would allow us to consider scenarios where there may be instability or imprecision in the specification of control parameters.

The primary challenge when facing such an objective is in identifying a clear optimization goal in the face of parameters that cannot be controlled; once this has been properly formalized, the development of optimization policies is then usually relatively straightforward. Several possible approaches have been studied to this end, which vary somewhat in their motivation and details. One detail with strong influence on algorithm design is whether the environment and/or any perturbations to inputs can be controlled during optimization; if so, we can accelerate optimization through careful manipulation of the environment.

\section*{Stochastic optimization}

In stochastic optimization, ${ }^{81}$ we seek to optimize the expected performance of the system of interest (also known as the integrated response ${ }^{82}$ ) under an assumed distribution for the environmental parameters, $p(\omega)$ :

$$
f=\mathbb{E}_{\omega}[g]=\int g(x, \omega) p(\omega) \mathrm{d} \omega .
$$

Optimizing this objective presents a challenge: in general, the expectation with respect to $\omega$ cannot be evaluated directly but only estimated via repeated trials in different environments, and estimating the expected performance (11.23) with some degree of precision may require numerous evaluations of $g(x, \omega)$. However, when $g$ itself is expensive to evaluate - for example, if every trial requires manual manipulation of a robot and its environment before we can measure performance - this may not be the most efficient approach, as we may waste significant resources shoring up our belief about suboptimal values.

Instead, when we can control the environment during optimization at will, we can gain some traction by designing a sequence of free parameter-environment pairs, potentially changing both configuration and environment in each iteration. The most direct way to design such an algorithm in our framework would be to model the environmentalconditional objective function $g$ directly and define a utility function and policy with respect to this function, in light of the environmentalmarginal objective $f$ (11.23) and its induced Gaussian process distribution.

A Gaussian process model on $g$ is particularly practical in this regard, as we may then use Bayesian quadrature to seamlessly estimate and quantify our uncertainty in the objective $f(11.23)$ from observations. ${ }^{83}$ This approach was explored in depth by TOSCANO-PALMERIN and FRAZIER, who also provided an excellent review of the related literature. ${ }^{84}$

Several algorithms have also been proposed for the special case of perturbed control parameters (11.22), where we seek to optimize an objective function $h$ under a random perturbation of its input, whose distribution may depend on location:

$$
f=\mathbb{E}_{\omega}[h]=\int h(x+\omega) p(\omega \mid x) \mathrm{d} \omega .
$$

In effect, the goal is to identify relatively "stable" (or "wide") optima of the objective function; see the margin for an illustration. Once again, a Gaussian process belief on $h$ induces a Gaussian process belief on the averaged objective (11.24), ${ }^{85}$ which can aid in the construction of policies.

When the perturbations follow a multivariate Gaussian distribution, and when inputs are assumed to be perturbed during optimization (and not only at performance time), one simple approach is to estimate the expectation of an arbitrary acquisition function $\alpha(x ; \mathcal{D})$ (defined with respect to the unperturbed objective $h$ ) with respect to the random perturbations,

$$
\mathbb{E}_{\omega}[\alpha]=\int \alpha(x+\omega ; \mathcal{D}) p(\omega) \mathrm{d} \omega,
$$

via an unscented transform. ${ }^{86}$ We may then maximize the expected acquisition function to realize an effective policy.

Under the assumption of unperturbed parameters during optimization, FRÖHLICH et al. described an adaptation of max-value entropy search for maximizing the expected objective (11.24) ${ }^{87}$ OLIVEIRA et al studied what is effectively the reversal of this problem: optimizing the unperturbed objective $h$ (11.24) under the assumption of perturbed inputs during optimization. The authors proposed a variation of the Gaussian process upper confidence bound algorithm for this setting with theoretical guarantees on convergence. ${ }^{88}$

\section*{Robust optimization}

Maximizing expected performance $(11.23,11.24)$ may not always be appropriate in all settings, as it is risk neutral by design. In the interest of risk aversion, in some scenarios we may instead seek parameters that ensure reasonable performance even in unfavorable environments. This is the goal of robust optimization. We can identify two families of approaches to robust optimization, differing according to whether we seek guarantees that are probabilistic (good performance in most environments) or adversarial (good performance in even the worst possible environment).

Probabilistic approaches focus on optimizing risk measures of performance such as the value-at-risk or conditional value-at-risk. In the former case, we seek to optimize a (lower) quantile of the distribution of $g(x, \omega)$, rather than its mean (11.23); see the figure in the margin:

$$
\operatorname{VAR}(x ; \pi)=\inf \{\gamma \mid \operatorname{Pr}(g(x, \omega) \leq \gamma \mid x) \geq \pi\} .
$$

84 S. TOSCANO-PALMERIN and P. I. FRAZIER (2018). Bayesian Optimization with Expensive Integrands. arXiv: 1803.08661 [stat.ML].

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-35.jpg?height=237&width=531&top_left_y=652&top_left_x=1368)

When stability to perturbation is critical, the relatively stable, but suboptimal, peak at $x^{\prime}$ might be preferred to the relatively narrow global optimum at $x$.

85 If $h$ has distribution $\mathcal{G P}(\mu, K)$, then $f$ has distribution $\mathcal{G P}(m, C)$ with:

$$
\begin{aligned}
& m=\int \mu(x+\omega) p(\omega \mid x) \mathrm{d} \omega ; \\
& C=\iint K\left(x+\omega, x^{\prime}+\omega^{\prime}\right) \\
& \quad p\left(\omega, \omega^{\prime} \mid x, x^{\prime}\right) \mathrm{d} \omega \mathrm{d} \omega^{\prime} .
\end{aligned}
$$

86 J. GARCÍA-BARCOS and R. MARTINEZ-CANTIN (2021). Robust Policy Search for Robot Navigation. IEEE Robotics and Automation Letters 6(2): 2389-2396.

The authors focused on expected improvement and also defined an "unscented incumbent" to approximate the maximum of (11.24).

87 L. P. FRÖHLICH et al. (2020). Noisy-Input Entropy Search for Efficient Robust Bayesian Optimization. AISTATS 2020 .

88 R. OlIVEIRA et al. (2019). Bayesian Optimisation Under Uncertain Inputs. AISTATS 2019.

risk tolerance: $§ 6.1$, p. 111

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-35.jpg?height=257&width=525&top_left_y=2070&top_left_x=1371)

The value-at-risk (at the $\pi=5 \%$ level) for an example distribution of $g(x, \omega)$. The VAR is a pessimistic lower bound on performance holding with probability $1-\pi$. 

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-36.jpg?height=251&width=523&top_left_y=503&top_left_x=161)

These distributions have the same value-atrisk at $\pi=5 \%$ (marked VAR); however, the wider distribution has greater conditional value-at-risk (the right-most of the unlabeled ticks), as its upper tail has a higher expectation than the narrower distribution's.

89 Q. P. NGUYEN et al. (2021b). Value-at-Risk Optimization with Gaussian Processes. ICML 2021.

90 Q. P. NGUYen et al. (2021a). Optimizing Conditional Value-At-Risk of Black-Box Functions. NeurIPS 2021.

91 S. CAKMAK et al. (2O2O). Bayesian Optimization of Risk Measures. NeurIPS 2020.

92 BOGUNOVIC et al.'s main focus was actually the related problem of adversarial perturbations:

$$
f(x)=\min _{\omega \in \Delta(x)} h(x+\omega) ;
$$

however, they also extend their algorithm to robust objectives of the form given in (11.25).

reproducing kernel Hilbert spaces: § 10.2,

93 R. MARTINEZ-CANTIN et al. (2018). Practical Bayesian Optimization in the Presence of Outliers. AISTATS 2018.

94 See also $§ 11.11$, p. 282 for related discussion.

95 I. Bogunovic et al. (2020). CorruptionTolerant Gaussian Process Bandit Optimization. AISTATS 2020.
We can interpret the value-at-risk as a pessimistic lower bound on performance with tunable failure probability $\pi$. Although a natural measure of robustness, the value-at-risk ignores the shape of the upper tail of the performance distribution entirely - any two distributions with equal $\pi$-quantiles are judged equal. The conditional value-at-risk measure accounts for differences in upper tail distributions by computing the expected performance in the favorable regime:

$$
\operatorname{CVAR}(x ; \pi)=\mathbb{E}_{\omega}[g(x, \omega) \mid g(x ; \omega) \geq \operatorname{VAR}(x ; \pi)] .
$$

Two performance distributions that cannot be distinguished by var but can be distinguished by their CVAR are illustrated in the marginal figure.

Optimizing both value-at-risk and conditional value-at-risk has received some attention in the literature, and optimization policies for this setting - under the assumption of a controllable environment during optimization - have been proposed based on upper confidence bounds, ${ }^{89,90}$ Thompson sampling, ${ }^{90}$ and the knowledge gradient. ${ }^{91}$

An alternative approach to robust optimization is to maximize some notion of worst-case performance. For example, we might consider an adversarial objective function of the form:

$$
f(x)=\min _{\omega \in \Delta(x)} g(x, \omega)
$$

where $\Delta(x)$ is a compact, possibly location-dependent subset of environments to consider. Note that value-at-risk is actually a special case of this construction, where $\Delta(x)$ is taken to be the upper tail of our belief over $g(x, \omega)$; however, we could also define such an adversarial objective without any reference to probability at all. BOGUNOvic et al. for example considered robust optimization in this adversarial setting ${ }^{92}$ and described a simple algorithm based on upper/lower confidence bounds with strong convergence guarantees.

KIRSCHNER et al. considered a related adversarial setting: distributionally robust optimization, where we seek to optimize effectively without perfect knowledge of the environment distribution. They considered an objective of the following form:

$$
f(x)=\inf _{p \in \mathcal{P}} \int g(x, \omega) p(\omega) \mathrm{d} \omega
$$

where $\mathcal{P}$ is a space of possible probability measures over the environment to consider - thus we seek to maximize expected performance under the worst-case environment distribution. The authors proposed an algorithm based on upper confidence bounds and proved convergence guarantees under technical assumptions on $g$ (bounded RKHS norm) and $\mathcal{P}$ (bounded maximum mean discrepancy from a reference distribution).

Finally, another type of robustness that has received some attention is robustness to extreme measurement errors (beyond additive Gaussian noise), including heavy-tailed noise such as Student- $t$ errors (2.31 $)^{93,94}$ and adversarial corruption. ${ }^{95}$ In some applications, the objective function is determined by a sequence of dependent steps eventually producing its final value. If we have access to this sequential process and can model its progression, we may be able to accelerate optimization via shrewd "early stopping": terminating evaluations still in progress when their final value can be forecasted with sufficient confidence.

Hyperparameter tuning presents one compelling example. Consider for example the optimization of neural network hyperparameters $\boldsymbol{\theta} \cdot{ }^{96}$ The objective function in this setting is usually defined to be the value of some loss function $\ell$ (for example, validation error) after the network has been trained with the chosen hyperparameters. However, this training is an iterative procedure: if the network is parameterized by a vector of weights $\mathbf{w}$, then the objective function might be defined by

$$
f(\boldsymbol{\theta})=\lim _{t \rightarrow \infty} \ell\left(\mathbf{w}_{t} ; \boldsymbol{\theta}\right)
$$

the loss of the network after the weights have converged. ${ }^{97}$ If we don't treat this objective function as a black box but take a peek inside, we may interpret it as the limiting value of the learning curve defined by the learning procedure's loss at each stage of training.

The iterative nature of this objective function offers the opportunity for innovation in policy design. Learning curves typically exhibit fairly regular behavior, with the loss in each step of training generally falling over time until settling on its final value. This suggests we may be able to faithfully extrapolate the final value of a learning curve from the early stages of training; a toy example is presented in the margin. When possible, we may then be able to speed up optimization by not always training to convergence with every setting of the hyperparameters we explore. With this motivation in mind, several sophisticated methods for extrapolating learning curves have been developed, including carefully crafted parametric models ${ }^{98,99}$ and flexible Bayesian neural networks. ${ }^{100}$

Exploiting the ability to extrapolate sequential objectives requires that we expand our action space (step 1) to allow fine-grained control over evaluation. One compelling scheme was proposed by SwERSKY et al.,98 who suggested maintaining a set of partial evaluations of the objective throughout optimization. In the context of our hyperparameter tuning example (11.26), we would maintain a set of thus-far investigated hyperparameters $\left\{\boldsymbol{\theta}_{i}\right\}$, each accompanied by a sequence (of variable length) of thus-far evaluated weights $\left\{\mathbf{w}_{t, i}\right\}$ and the associated losses $\left\{\ell_{t, i}\right\}$. Now, each action we design can either investigate a novel hyperparameter $\boldsymbol{\theta}$ or extend an existing partial evaluation by one step. The authors dubbed this scheme freeze-thaw Bayesian optimization, as after each action we "freeze" the current evaluation, saving enough state such that we can "thaw" it later for further investigation if desired.

Regardless of modeling details, this scheme offers the potential for considerable savings when optimizing sequential objective functions.
96 We will use $\theta$ for the variable to be optimized here rather than $x$ to be consistent with our previous discussion of Gaussian process hyperparameters in Chapter 3 .

97 One could consider early stopping in the training procedure as well, but this simple example is useful for exposition.

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-37.jpg?height=363&width=531&top_left_y=1366&top_left_x=1368)

Learning curve extrapolation from initial training results. At this point, we may already wish to abandon the apparently inferior option.

98 K. SWERsky et al. (2014). Freeze-Thaw Bayesian Optimization. arXiv: 1406 . 3896 [stat.ML].

99 T. DOMHAN et al. (2015). Speeding up Automatic Hyperparameter Optimization of Deep Neural Networks by Extrapolation of Learning Curves. IJCAI 2015.

100 A. KLEIN et al. (2017). Learning Curve Prediction with Bayesian Neural Networks. ICLR 2017 .

freeze-thaw Bayesian optimization 101 L. LI et al. (2018b). Hyperband: A Novel BanditBased Approach to Hyperparameter Optimization. Journal of Machine Learning Research 18(185):1-52.

Chapter 5: Decision Theory for Optimization, p. 87

Chapter 6: Utility Functions for Optimization, p. 109

GP inference with non-Gaussian observation models: $§ 2.8$, p. 35

102 A. SHAH et al. (2014). Student- $t$ Processes as Alternatives to Gaussian Processes. AISTATS 2014 .

103 R. MARTINEZ-CANTIN et al. (2018). Practical Bayesian Optimization in the Presence of Outliers. AISTATS 2018.

104 We explored this possibility at length in $\S 2.8$.

105 M. TESCH et al. (2013). Expensive Function Optimization with Stochastic Binary Outcomes. ICML 2013.

106 Note that the simple reward utility function (6.3) we have been working with can be written in this exact form if we assume any additive observation noise has zero mean.

107 N. HOULSBY et al. (2012). Collaborative Gaussian Processes for Preference Learning. NeurIPS 2012.
This idea of abandoning stragglers based on early progress is also the basis of the bandit-based hyperband algorithm. ${ }^{101}$

\subsection*{NON-GAUSSIAN OBSERVATION MODELS AND ACTIVE SEARCH}

Throughout this book, we have focused almost exclusively on the additive Gaussian noise observation model. There are good reasons for this: it is a reasonably faithful model of many systems and offers exact inference with Gaussian process models of the objective function. However, the assumption of Gaussian noise is not always warranted and may be fundamentally incompatible with some scenarios.

Fortunately, the decision-theoretic core of most Bayesian optimization approaches does not make any assumptions regarding our model of the objective function or our observations of it, and with some care the utility functions we developed for optimization can be adapted for virtually any scenario. Further, there are readily available pathways for incorporating non-Gaussian observation models into Gaussian process objective function models, so we do not need to abandon that rich model class in order to use alternatives.

\section*{Sequential optimization with non-Gaussian observation models}

A decision-theoretic approach to optimization entails first selecting an objective function model $p(f)$ and observation model $p(y \mid x, \phi)$, which together are sufficient to derive the predictive distribution $p(y \mid x, \mathcal{D})$ relevant to every sequential decision made during optimization. After selecting a utility function $u(\mathcal{D})$, we may then follow the iterative procedure developed in Chapter 5 to derive a policy.

This abstract approach has been realized in several specific settings. For example, both Student- $t$ processes ${ }^{102}$ and the Student- $t$ observation model $^{103}$ have been explored to develop Bayesian optimization routines that are robust to the presence of outliers. ${ }^{104}$

TESCH et al. explored the use of expected improvement for optimization from binary success/failure indicators, motivated by optimizing the probability of success of a robotic platform operating in an uncertain environment. ${ }^{105}$ Here the utility function was taken to be this success probability maximized over the observed locations:

$$
u(\mathcal{D})=\max _{\mathbf{x}} \mathbb{E}[y \mid x, \mathcal{D}]=\max _{\mathbf{x}} \operatorname{Pr}(y=1 \mid x, \mathcal{D}),
$$

where for this expression we have assumed binary outcomes $y \in\{0,1\}$ with $y=1$ interpreted as indicating success. ${ }^{106}$ The authors then derived a policy for this setting via one-step lookahead.

Another setting involving binary (or categorical) feedback is in optimizing human preferences, such as in A/B testing or user modeling. Here we might seek to optimize user preferences by repeatedly presenting a panel of options and asking for the most preferred item. HOULSBY et al. described a convenient reduction from preference learning to classification for Gaussian processes that allows the immediate use of standard policies such as expected improvement, ${ }^{107,105}$ although more sophisticated policies have also been proposed specifically for this setting. ${ }^{108}$

\section*{Active search}

GARNETT et al. introduced active search as a simple model of scientific discovery in a discrete domain $\mathcal{X}=\left\{x_{i}\right\} .{ }^{109}$ In active search, we assume that among these points is hidden a rare, valuable subset exhibiting desirable properties for the task at hand. Given access to an oracle that can - at significant cost - determine whether an identified point belongs to the sought after class, the problem of active search is to design a sequence of experiments seeking to maximize the number of discoveries in a given budget. A motivating application is drug discovery, where the domain would represent a list of candidate molecules to search for those rare examples exhibiting significant binding activity with a chosen biological target. As the space of candidates is expansive and the cost of even virtual screening is nontrivial, intelligent experimental design has the potential to greatly improve the rate of discovery.

To derive an active search policy in our framework, we must first model the observation process and determine a suitable utility function. The former requires consideration of the nuances of a given situation, but we may provide a barebones construction that is already sufficient to be of practical and theoretical interest. Given a discrete domain $\mathcal{X}$, we assume there is some identifiable subset $\mathcal{V} \subset \mathcal{X}$ of valuable points we wish to recover. We associate with each point $x \in \mathcal{X}$ a binary label $y=[x \in \mathcal{V}]$ indicating whether $x$ is valuable $(y=1)$ or not $(y=0)$. A natural observation model is then to assume that selecting a point $x$ for investigation reveals this binary label $y$ in response. ${ }^{110}$ Finally, we may define a natural utility function for active search by assuming that, all other things being held equal, we prefer a dataset containing more valuable points to one with fewer:

$$
u(\mathcal{D})=\sum_{x \in \mathcal{D}} y
$$

This is simply the cumulative reward utility (6.7), which here can be interpreted as counting the number of valuable points discovered. ${ }^{111}$

To proceed with the Bayesian decision-theoretic approach, we must build a model for the uncertain elements inherent to each decision. Here the primary object of interest is the predictive posterior distribution $\operatorname{Pr}(y=1 \mid x, \mathcal{D})$, the posterior probability that a given point $x$ is valuable. We may build this model in any number of ways, for example by combining a Gaussian process prior on a latent function with an appropriate choice of observation model. ${ }^{112}$

Equipped with a predictive model, deriving the optimal policy is a simple exercise. To begin, the one-step marginal gain in utility (5.8) is

$$
\alpha_{1}(x ; \mathcal{D})=\operatorname{Pr}(y=1 \mid x, \mathcal{D})
$$

108 See Appendix D, p. 329 for related references.

109 R. GARNETT et al. (2012). Bayesian Optimal Active Search and Surveying. ICML 2012.

virtual screening: Appendix D, p. 314

modeling observations

110 Other situations may call for other approaches; for example, if value is determined by thresholding a continuous measurement, we may wish to model that continuous observation process explicitly.

utility function

111 The assumption of a discrete domain is to avoid repeatedly observing effectively the same point to trivially "max out" this score.

posterior predictive probability, $\operatorname{Pr}(y=1 \mid x, \mathcal{D})$

112 C. E. RASMUSSEN and C. K. I. WILLIAMS (2006). Gaussian Processes for Machine Learning. MIT Press. [chapter 3] 113 R. GARNETT et al. (2012). Bayesian Optimal Active Search and Surveying. ICML 2012.

114 S. JIANG et al. (2017). Efficient Nonmyopic Active Search. ICML 2017.

115 R. GARNETT et al. (2015). Introducing the 'Active Search' Method for Iterative Virtual Screening. Fournal of Computer-Aided Molecular Design 29(4):305-314.

cost of computing optimal policy: §5.3, p. 99

moving beyond one-step lookahead: § 7.10, p. 150

batch rollout: $§ 5 \cdot 3$, p. 103 that is, the optimal one-step decision is to greedily maximize the probability of success. Although this is a simple (and somewhat obvious) policy that can perform well in practice, theoretical and empirical study on active search has established that massive gains can be had by adopting less myopic policies.

On the theoretical side, GARNETT et al. demonstrated by construction that the expected performance of any lookahead approximation can be exceeded by any arbitrary amount by extending the lookahead horizon even a single step. ${ }^{113}$ This result was strengthened by JIANG et al., who showed - again by construction - that no policy that can be computed in time polynomial in $|\mathcal{X}|$ can approximate the performance of the optimal policy within any constant factor. ${ }^{114}$ Thus the optimal active search policy is not only hard to compute, but also hard to approximate. These theoretical results, which rely on somewhat unnatural adversarial constructions, have been supported by empirical investigations on realworld data as well. ${ }^{114,115}$ For example, GARNETT et al. demonstrated that simply using two-step instead of one-step lookahead can significantly accelerate virtual screening for drug discovery across a broad range of biological targets. ${ }^{115}$

This is perhaps a surprising state of affairs given the success of one-step lookahead - and the relative lack of less myopic alternatives for traditional optimization. We can resolve this discrepancy by noting that utility functions used in that setting, such as the simple reward (6.3), inherently exhibit decreasing marginal gains as they are effectively bounded by the global maximum. Further, such a utility tends to remain relatively constant throughout optimization, punctuated by brief but significant increases when a new local maximum is discovered. On the other hand, for the cumulative reward (11.27), every observation has the potential to increase the utility by exactly one unit. As a result, every observation is on equal footing in terms of potential impact, and there is increased pressure to consider the entire search trajectory when designing each observation.

One thread of research on active search has focused on developing efficient, yet nonmyopic policies grounded in approximate dynamic programming, such as lookahead beyond one step. ${ }^{113}$ JIANG et al. proposed one significantly less myopic alternative policy based on batch rollout. ${ }^{114}$ The key observation is that we may construct the one-step optimal batch observation of size $k$ by computing the posterior predictive probability $\operatorname{Pr}(y=1 \mid x, \mathcal{D})$ for the unlabeled points, sorting, and taking the top $k$; this is a consequence of linearity of expectation and utility (11.27). With this, we may realize an efficient batch rollout policy for horizon $\tau$ by maximizing the acquisition function

$$
\operatorname{Pr}(y=1 \mid x, \mathcal{D})+\mathbb{E}_{y}\left[\sum_{\tau-1}^{\prime} \operatorname{Pr}\left(y^{\prime}=1 \mid x^{\prime}, \mathcal{D}_{1}\right) \mid x, \mathcal{D}\right]
$$

Here the sum-with-prime notation $\sum_{\tau-1}^{\prime}$ indicates the sum of the top$(\tau-1)$ values over the unlabeled data - the expected utility of the optimal 
![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-41.jpg?height=444&width=1014&top_left_y=454&top_left_x=270)

batch observation consuming the remaining budget. ${ }^{116}$ In experiments, this policy showed interesting emergent behavior: it underperforms lookahead policies in the early stages of search due to significant investment in early exploration. However, this exploration pays off dramatically by revealing numerous promising regions that may be exploited later.

Figure 11.17 illustrates active search in a $2 d$ domain, which for the purposes of this demonstration was discretized into a $100 \times 100$ grid. The example is constructed so that a small number of discrete regions yield valuable items, which must be efficiently uncovered for a successful search. The right-hand panel shows a sequence of 500 observations designed by iteratively maximizing (11.28); their distribution clearly reflects both exploration of the domain and exploitation of the most fruitful regions. The rate of discovery for the active search policy was approximately 3.8 times greater than would be expected from random search.

VANCHINATHAN et al. considered an expanded setting they dubbed adaptive valuable item discovery (AVID), for which active search is a special case. Here items may have nonnegative, continuous values and the cumulative reward utility (11.27) was augmented with a term encouraging diversity among the selected items. ${ }^{117}$ The authors proposed an algorithm called GP-SELECT for this setting based on an acquisition function featuring two terms: a standard upper confidence bound score (8.25) and a term encouraging diversity related to determinantal point processes. ${ }^{118}$ The authors were further able to establish theoretical regret bounds for this algorithm under standard assumptions on the complexity of the value function.

\subsection*{LOCAL OPTIMIZATION}

Historically, the primary focus of Bayesian optimization research has been global optimization. However, when the domain is too enormous to search systematically, ${ }^{119}$ or if we have prior knowledge regarding the most promising areas of the domain to search, we may prefer to perform local optimization, where we seek to optimize the objective only in the neighborhood of an initial "guess" $x_{0} \in \mathcal{X}$.

Line search and trust region methods are two broad approaches to unconstrained local optimization in Euclidean domains $\mathcal{X} \subset \mathbb{R}^{d}$. The
Figure 11.17: A demonstration of active search. Left: a $2 d$ domain shaded according to the probability of a positive result. Right: 500 points chosen by a nonmyopic active search policy reflecting consideration of exploration versus exploitation. The darker observations were positive and the lighter ones were negative.

116 The cost of computing this acquisition function is $\mathcal{O}\left(n^{2} \log n\right)$, where $n=|\mathcal{X}|$; this is roughly the same order as the two-step expected marginal gain, $\mathcal{O}\left(n^{2}\right)$.

117 H. P. VANChinathan et al. (2015). Discovering Valuable Items from Massive Data. KDD 2015.

118 A. KULESZA and B. TASKAR (2012). Determinantal Point Processes for Machine Learning. Foundations and Trends in Machine Learning 5(2-3):123-286.

119 For example, when faced with the curse of dimensionality; see $\S 3.5$, p. 61, § 9.2, p. 208. 

![](https://cdn.mathpix.com/cropped/2023_09_22_6de6b1b5cca16263b2a3g-42.jpg?height=249&width=508&top_left_y=612&top_left_x=177)

120 For useful discussion in this context, see:

P. HENNIG et al. (2022). Probabilistic Numerics: Computation as Machine Learning. Cambridge University Press. [part IV]

121 When the search direction is the one of steepest ascent from $\mathbf{x}$ - that is, the gradient $\nabla f(\mathbf{x})-$ this procedure is known as gradient ascent.

122 For some early work in this direction, see:

J. MOckus (1989). Bayesian Approach to Global Optimization: Theory and Applications. Kluwer Academic Publishers. [chapter 7]

123 See $§ 2.6$, p. 30.

124 M. MAHSERECI and P. HENNIG (2015). Probabilistic Line Searches for Stochastic Optimization. NeurIPS 2015.

125 For example, when optimizing a function of the form $f(\mathbf{x})=\frac{1}{n} \sum_{i=1}^{n} f_{i}(\mathbf{x})$, we may estimate the gradient by only partially evaluating the sum (a "minibatch"). In some situations, the central limit theorem applies, and even evaluating the gradient for a random subset provides an unbiased estimate of the gradient corrupted by (approximately) Gaussian noise - perfect for conditioning a Gaussian process!

126 S. MÜLlER et al. (2021). Local Policy Search with Bayesian Optimization. NeurIPS 2021.

127 Q. NGUYen et al. (2022). Local Bayesian Optimization via Maximizing Probability of Descent. arXiv: 2210.11662 [cs.LG].

\begin{tabular}{l}
$\mathbf{x} \leftarrow \mathbf{x}_{0}$ \\
repeat \\
$\quad$ observe at $\mathbf{x}$ \\
$\quad$ update trust region $\mathcal{T}$ \\
$\quad$ select next observation location $\mathbf{x} \in \mathcal{T}$ \\
until termination condition reached \\
\hline
\end{tabular}

128 D. ERIKSSON et al. (2019). Scalable Global Optimization via Local Bayesian Optimization. NeurIPS 2019. basic idea in both cases is to navigate a path $\left\{\mathbf{x}_{0}, \mathbf{x}_{1}, \ldots\right\}$ through the domain - guided by the gradient of the objective function - hoping to generally progress "uphill" over time.

Starting from some arbitrary location $\mathrm{x} \in \mathcal{X}$, line search methods design the next step of the path $\mathbf{x} \rightarrow \mathbf{x}^{\prime}$ through a two-stage process (pseudocode in margin). ${ }^{120}$ First, we determine a search direction $\mathbf{d},{ }^{121}$ which restricts our choice to the one-dimensional subspace extending from $\mathbf{x}$ in the given direction: $\mathcal{L}=\{\mathbf{x}+\alpha \mathbf{d} \mid \alpha>0\}$. Next, we explore the objective function along this subspace to choose a step size $\alpha>0$ to take in this direction, which finally determines the next location: $\mathbf{x}^{\prime}=\mathbf{x}+\alpha \mathbf{d}$.

Once a search direction has been chosen, classical methods for determining the step size boil down to searching along $\mathcal{L}$ for a point providing "sufficient progress" toward a local optimum. In the absence of noise, this notion of sufficiency is well-understood and can be captured by simple conditions on the objective function, such as the well-known ArmijoGoldstein and Wolfe conditions. ${ }^{120}$ However, these conditions become difficult to assess in the presence of even a small amount of noise.

To overcome this barrier, several authors have developed Bayesian procedures for determining the step size informed by a probabilistic model of the objective function. ${ }^{122}$ Notably, MAHSERECI and HENNIG proposed a Gaussian process model for the objective function (and thus, its directional derivative ${ }^{123}$ ) along $\mathcal{L},{ }^{120,124}$ which allows us to assess the probability that the Wolfe conditions are satisfied at a given location in light of noisy observations. The authors then proposed a lightweight and efficient line search procedure that explores along $\mathcal{L}$ guided by this probability and the expected improvement.

When we can effectively estimate the gradient of the objective function, we can often use an established procedure for stochastic gradient ascent to design effective search directions. ${ }^{125}$ When this is not possible, the ability of Gaussian processes to infer the gradient from noisy observations of the objective function alone allows us to nonetheless build efficient Bayesian routines for "zeroth-order" local optimization. ${ }^{123}$ This idea was pursued by MÜLLER et al., who proposed to alternate between efficiently estimating the gradient at the current location via Bayesian experimental design, then taking a step in the direction of the expected gradient. ${ }^{126}$ NGUYEN et al. noted that the expected gradient is not necessarily the most likely direction of ascent and proposed maximizing the latter quantity directly to identify trustworthy search directions. ${ }^{127}$

Trust region methods operate by maintaining their namesake trust region throughout optimization, a subset of the domain on which a model of the objective function is trusted. In each iteration, we use the model to select the next observation location from this region, then adjust the trust region in light of the outcome (pseudocode in margin). Classical methods rely on very simple (for example, quadratic) models, but ERIKSSON et al. demonstrated success using Gaussian process models and Bayesian optimization policies instead. ${ }^{128}$ The authors also showed how to use their Bayesian trust region routine in an outer global optimization procedure, with impressive results especially in high dimension.