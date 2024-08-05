\section*{DECISION THEORY FOR OPTIMIZATION}

Optimization entails a series of decisions. Most obviously, we must repeatedly decide where to make each observation guided by the available data. Some settings also demand we decide when to terminate optimization, weighing the potential benefit from continuing optimization against any costs that may be incurred. It is not obvious how we should make these decisions, especially in the face of incomplete and constantly evolving knowledge about the objective function that is only refined via the outcomes of our own actions.

In the previous four chapters, we established Bayesian inference as a framework for reasoning about uncertainty that offers partial guidance. The primary obstacle to decision making during optimization is uncertainty about the objective function, and, by extension, the outcomes of proposed observations. Bayesian inference allows us to reason about an unknown objective function with a probability distribution over plausible functions that we may seamlessly update as we gather new information. This belief over the objective function in turn enables prediction of proposed observations via the posterior predictive distribution.

How can we use these beliefs to guide our decisions? Bayesian inference offers no direct answer, but in this chapter we will bridge this gap. We will develop Bayesian decision theory as a principled means of decision making under uncertainty and apply this approach in the context of optimization, demonstrating how to use a probabilistic belief about an objective function to inform intelligent optimization policies.

Recall our model of sequential optimization outlined in Algorithm 1.1, repeated for convenience on the following page. We begin with an arbitrary set of data, which we build upon through a sequence of observations of our own design. The core of the procedure is an optimization policy, which examines any already gathered data and makes the fundamental decision of where to make the next observation. With a policy in hand, optimization proceeds by repeating a straightforward pattern: the policy selects the next observation location, then we acquire the requested measurement and update our data accordingly. We repeat this process until satisfied, at which point we return the collected data.

Barring the question of termination, the behavior of this procedure is entirely determined by the policy, and constructing optimization policies will be our primary concern in this and the following chapters. We will begin with sheer audacity: we will derive the optimal policy - in terms of maximizing the expected quality of the returned data - in a generic setting. The reader may wonder why this book is so long if the optimal policy is apparently so simple. As it turns out, this theoretically optimal procedure is usually impossible to compute and rarely of practical value. However, our careful derivation will shed light on how we might derive effective approximations. This is a common theme in Bayesian optimization and will be our focus in Chapters 7 and 8.

The question of when to terminate optimization also represents a decision that can be of critical importance in some applications. A
Bayesian decision theory

formalization of optimization, § 1.1, p. 2

optimization policy

optimal optimization policies: $§ 5.2$, p. 91

running time and approximation: §5.3, p. 99

Chapter 7: Common Bayesian Optimization Policies, p. 123

Chapter 8: Computing Policies with Gaussian Processes, p. 157 Algorithm 1.1: Sequential optimization.

stopping rule

1 A predominant example is a preallocated budget on the number of allowed observations, in which case we are compelled to stop after exhausting the budget regardless of progress.

optimal stopping rules: §5.4, p. 103

practical stopping rules: $\S 9.3$, p. 210

Chapter 6: Utility Functions for Optimization, p. 109

acquisition function, infill function, figure of merit

acquisition function, $\alpha(x ; \mathcal{D})$

![](https://cdn.mathpix.com/cropped/2023_09_22_d3cf6bd410ce34714365g-02.jpg?height=323&width=997&top_left_y=455&top_left_x=792)

procedure for inspecting an observed dataset and deciding whether to stop or continue optimization is called a stopping rule. In optimization, the stopping rule is often fixed and known before we begin, in which case we do not need to worry over its design. ${ }^{1}$ However, in some scenarios, we may wish instead to consider our evolving understanding of the objective function and the expected cost of further observations to dynamically decide when to stop, requiring more subtle adaptive stopping rules. We will also address termination decisions in this chapter and will again begin by deriving the optimal - but intractable - stopping procedure, which will inspire efficient and effective approximations.

Practical optimization routines will return datasets that reflect significant progress on our global optimization problem (1.1) in some way. For example, we may wish to return datasets containing near-optimal values of the objective function. Alternatively, we may be satisfied returning datasets that indirectly reveal likely locations of the global optimum or achieve some other related goal. We will formalize this notion of a returned dataset's utility shortly and use it to guide optimization. First, we pause to introduce a useful and pervasive technique for implicitly defining an optimization policy by maximizing a score function over the domain.

\section*{Defining optimization policies via acquisition functions}

A convenient mechanism for defining an optimization policy is by first specifying an intermediate so-called acquisition function (also called an infill function or figure of merit) that provides a score to each potential observation location commensurate with its propensity for aiding the optimization task. We may then define a policy by observing at a point judged most promising by the acquisition function. Nearly all Bayesian optimization policies are defined in this manner, and this relationship is so intimate that the phrase "acquisition function" is often used interchangeably with "policy" in the literature and conversation, with maximization of the acquisition function understood.

Specifically, an acquisition function $\alpha: \mathcal{X} \rightarrow \mathbb{R}$ assigns a score to each point in the domain reflecting our preferences over locations for the next observation. Of course, these preferences will presumably depend on the data we have already observed. To make this dependence explicit, we adopt the notation $\alpha(x ; \mathcal{D})$ for a general acquisition function, where available data serve as parameters shaping our preferences. In the Bayes- ian approach, acquisition functions are invariably defined by deriving the posterior belief of the objective function given the data, $p(f \mid \mathcal{D})$, then defining preferences with respect to this belief.

An acquisition function $\alpha$ encodes preferences over potential observation locations by inducing a total order over the domain: given data $\mathcal{D}$, observing at a point $x$ is preferred over another point $x^{\prime}$ whenever $\alpha(x ; \mathcal{D})>\alpha\left(x^{\prime} ; \mathcal{D}\right)$. Thus a rational action in light of these preferences is (any) one maximizing the acquisition function: ${ }^{2}$

$$
x \in \underset{x^{\prime} \in \mathcal{X}}{\arg \max } \alpha\left(x^{\prime} ; \mathcal{D}\right) .
$$

Solving (5.1) maps a set of observed data $\mathcal{D}$ to a point $x \in \mathcal{X}$ to observe next, exactly the role of an optimization policy.

At first this idea may sound absurd: we have proposed solving a global optimization problem (1.1) by repeatedly solving global optimization problems (5.1)! To resolve this apparent paradox, we note that acquisition functions in common use have properties rendering their optimization considerably more tractable than the problem we ultimately wish to solve. Typical acquisition functions are both cheap to evaluate and analytically differentiable, allowing the use of off-the-shelf optimizers when computing the policy (5.1). The objective function, on the other hand, is assumed to be expensive to evaluate, and its gradient is often unavailable. Therefore we can reduce a difficult, expensive problem to a series of simpler, inexpensive problems - a reasonable pursuit!

Numerous acquisition functions have been proposed for Bayesian optimization, and we will describe many popular choices in detail in Chapter 7. The most prominent means to constructing acquisition functions is Bayesian decision theory, an approach to optimal decision making we will discuss over the remainder of the chapter.

\section*{INTRODUCTION TO BAYESIAN DECISION THEORY}

Bayesian decision theory is a framework for decision making under uncertainty that is flexible enough to handle effectively any scenario. Instead of presenting the entire theory in complete abstraction, we will introduce the essential concepts with an eye to the context of optimization. For a more in-depth and theoretical treatment, the interested reader may refer to numerous comprehensive reviews of the subject. ${ }^{3}$ A good familiarity with this material can demystify some key ideas that are often glossed over in the Bayesian optimization literature, as it serves as the "hidden origin" of many common acquisition functions.

In this section we will introduce the Bayesian approach to decision making and demonstrate how to make optimal decisions in the case of a single isolated decision. Ultimately, we will require a theory for making a sequence of decisions to reason over an entire optimization session. In the next section, we will extend the line of reasoning presented below to address sequential decision making and the construction of optimization policies. encoding preferences with an acquisition function

2 Ties may be broken arbitrarily.

the paradox of Bayesian optimization: global optimization via...global optimization?

Chapter 7: Common Bayesian Optimization Policies, p. 123

\footnotetext{
3 The following would be excellent companion texts:

M. H. DEGROOT (1970). Optimal Statistical Decisions. McGraw-Hill.

J. O. BERGER (1985). Statistical Decision Theory and Bayesian Analysis. Springer-Verlag.
} action space, $\mathcal{A}$

unknown variables affecting decision outcome, $\psi$

relevant observed data, $\mathcal{D}$ posterior belief about $\psi, p(\psi \mid \mathcal{D})$

utility function, $u(a, \psi, \mathcal{D})$

4 Typical presentations of Bayesian decision theory omit the data from the utility function, but including it offers more generality, and this allowance will be important when we turn our attention to optimization policies expected utility

5 One may question whether this framework is complete in some sense: is it possible to make rational decisions in some other manner? The von Neumann-Morgenstern theorem shows that the answer is, surprisingly, no. Assuming a certain set of rationality axioms, any rational preferences over uncertain outcomes can be captured by the expectation of some utility function. Thus every rational decision maximizes an expected utility:

J. VON NEUMANN and O. MORGENSTERN (1944). Theory of Games and Economic Behavior. Princeton University Press. [appendix A]

\section*{Isolated decisions}

A decision problem under uncertainty has two defining characteristics. The first is the action space $\mathcal{A}$, the set of all available decisions. Our task is to select an action from this space. For example, in sequential optimization, an optimization policy decision must select a point in the domain $\mathcal{X}$ for observation, and so we have $\mathcal{A}=\mathcal{X}$.

The second critical feature is the presence of uncertain elements of the world influencing the outcomes of our actions, complicating our decision. Let $\psi$ represent a random variable encompassing any relevant uncertain elements when making and evaluating a decision. Although we may lack perfect knowledge, Bayesian inference allows us to reason about $\psi$ in light of data via the posterior distribution $p(\psi \mid \mathcal{D})$, and we will use this belief to inform our decision.

Suppose now we must select a decision from an action space $\mathcal{A}$ under uncertainty in $\psi$, informed by a set of observed data $\mathcal{D}$. To guide our choice, we select a real-valued utility function $u(a, \psi, \mathcal{D})$. This function measures the quality of selecting the action $a$ if the true state of the world were revealed to be $\psi$, with higher utilities indicating more favorable outcomes. The arguments to a utility function comprise everything required to judge the quality of a decision in hindsight: the proposed action $a$, what we know (the data $\mathcal{D}$ ), and what we don't know (the uncertain elements $\psi){ }^{4}$

We cannot know the exact utility that would result from selecting any given action a priori, due to our incomplete knowledge of $\psi$. We can, however, compute the expected utility that would result from selecting an action $a$, according to our posterior belief:

$$
\mathbb{E}[u(a, \psi, \mathcal{D}) \mid a, \mathcal{D}]=\int u(a, \psi, \mathcal{D}) p(\psi \mid \mathcal{D}) \mathrm{d} \psi
$$

This expected utility maps each available action to a real value, inducing a total order and providing a straightforward mechanism for making our decision. We pick an action maximizing the expected utility:

$$
a \in \underset{a^{\prime} \in \mathcal{A}}{\arg \max } \mathbb{E}\left[u\left(a^{\prime}, \psi, \mathcal{D}\right) \mid a^{\prime}, \mathcal{D}\right]
$$

This decision is optimal in the sense that no other action results in greater expected utility. (By definition!) This procedure for acting optimally under uncertainty - computing expected utility with respect to relevant unknown variables and maximizing to select an action - is the central tenant of Bayesian decision making. ${ }^{5}$

\section*{Example: recommending a point for use after optimization}

With this abstract decision-making framework established, let us analyze an example decision that might be faced in the context of optimization. Consider a scenario where the purpose of optimization is to identify a single point $x \in \mathcal{X}$ for perpetual use in a production system, preferring locations achieving higher values of the objective function. If we run an optimizer and it returns some dataset $\mathcal{D}$, which point should we select for our final recommendation?

We may model this choice as a decision problem with action space $\mathcal{A}=\mathcal{X}$, where we must reason under uncertainty about the objective function $f$. We first select a utility function quantifying the quality of a given recommendation $x$ in hindsight. One natural choice would be

$$
u(x, f)=f(x)=\phi,
$$

which rewards points for achieving high values of the objective function. Now if our optimization procedure returned a dataset $\mathcal{D}$, the expected utility from recommending a point $x$ is simply the posterior mean of the corresponding function value:

$$
\mathbb{E}[u(x, f) \mid x, \mathcal{D}]=\mathbb{E}[\phi \mid x, \mathcal{D}]=\mu_{\mathcal{D}}(x) .
$$

Therefore, an optimal recommendation maximizes the posterior mean:

$$
x \in \underset{x^{\prime} \in \mathcal{X}}{\arg \max } \mu_{\mathcal{D}}\left(x^{\prime}\right) .
$$

Of course, other considerations in a given scenario such as risk aversion might suggest some other utility function or action space would be more appropriate, in which case we are free to select any alternative as we see fit. We will discuss terminal recommendations at length in the next chapter, including alternative utility functions and action spaces.

\subsection*{SEQUENTIAL DECISIONS With A FIXED BUdGeT}

We have now introduced Bayesian decision theory as a framework for computing optimal decisions informed by data. The key idea is to measure the post hoc quality of a decision with an appropriately designed utility function, then choose actions maximizing expected utility according to our beliefs. We will now apply this idea to the construction of optimization policies. This setting is considerably more complicated because each decision we make over the course of optimization will shape the context of all future decisions.

\section*{Modeling policy decisions}

To define an optimization routine, we must design a policy to adaptively design a sequence of observations seeking the optimum. Following our discussion in the previous section, we will model each of these choices as a decision problem under uncertainty. Some aspects of this modeling will be straightforward and others will take some care. To begin, the action space of each decision is the domain $\mathcal{X}$, and we must act under uncertainty about the objective function $f$, which induces uncertainty about the outcomes of proposed observations. Fortunately, we may make each decision guided by any data obtained from previous decisions.

![](https://cdn.mathpix.com/cropped/2023_09_22_d3cf6bd410ce34714365g-05.jpg?height=380&width=528&top_left_y=644&top_left_x=1369)

Optimal terminal recommendation. Above: posterior belief about an objective function given the data returned by an optimizer, $p(f \mid \mathcal{D})$. Below: the expected utility for our example, the posterior mean $\mu_{\mathcal{D}}(x)$. The optimal recommendation maximizes the expected utility.

terminal recommendations: § 6.1, p. 109 Bayesian inference of the objective function: $\S 1.2$, p. 8

optimization utility function, $u(\mathcal{D})$

Chapter 6: Utility Functions for Optimization, p. 109

\footnotetext{
6 In fact, we have already begun by analyzing a
} decision after optimization has completed!
To reason about uncertainty in the objective function, we follow the path laid out in the preceding chapters and maintain a probabilistic belief throughout optimization, $p(f \mid \mathcal{D})$. We make no assumptions regarding the nature of this distribution, and in particular it need not be a Gaussian process. Equipped with this belief, we may reason about the result of making an observation at some point $x$ via the posterior predictive distribution $p(y \mid x, \mathcal{D})(1.7)$, which will play a key role below.

The ultimate purpose of optimization is to collect and return a dataset $\mathcal{D}$. Before we can reason about what data we should acquire, we must first clarify what data we would like to acquire. Following the previous section, we will accomplish this by defining a utility function $u(\mathcal{D})$ to evaluate the quality of data returned by an optimizer. This utility function will serve to establish preferences over optimization outcomes: all other things being equal, we would prefer to return a dataset with higher utility than any dataset with lower utility. As before, we will use this utility to guide the design of policies, by making observations that, in expectation, promise the biggest improvement in utility. We will define and motivate several utility functions used for optimization in the next chapter, and some readers may wish to jump ahead to that discussion for explicit examples before continuing. In the following, we will develop the general theory in terms of an arbitrary utility function.

\section*{Uncertainty faced during optimization}

Suppose $\mathcal{D}$ is a dataset of previous observations and that we must select the next observation location $x$. This is the core decision defining an optimization policy, and we will make all such decisions in the same manner: by maximizing the expected utility of the data we will return.

Although this sounds straightforward, let us consider the uncertainty faced when contemplating this decision in more detail. When evaluating a potential action $x$, uncertainty in the objective function induces uncertainty in the corresponding value $y$ we will observe. Bayesian inference allows us to reason about this uncertain outcome via the posterior predictive distribution (1.7), and we may hope to be able to address this uncertainty without much trouble. However, we must also consider that evaluating at $x$ would add the unknown observation $(x, y)$ to our dataset, and that the contents of this updated dataset would be consulted for all future decisions. Thus we must reason not only about the outcome of the present observation but also its impact on the entire remainder of optimization. This requires special attention and distinguishes sequential decisions from the isolated decisions discussed in the last section.

Intuitively, we might suspect that decisions made closer to termination should be easier, as fewer future decisions depend on their outcomes. This is indeed the case, and it will be prudent to define optimization policies in reverse. ${ }^{6}$ We will first reason about the final decision - when we are freed from the burden of having to ponder any future observations and proceed backwards to the choice of the first observation location, working out optimal behavior every step along the way. In this section we will consider the construction of optimization policies assuming that we have a fixed and known budget on the number of observations we will make. This scenario is both common in practice and convenient for analysis, as we can for now ignore the question of when to terminate optimization. Note that this assumption effectively implies that every observation has a constant acquisition cost, which may not always be reasonable. We will address variable observation costs and the question of when to stop optimization later in this chapter.

Assuming a fixed observation budget allows us to reason about optimization policies in terms of the number of observations remaining to termination, which will always be known. The problem we will consider in this section then becomes the following: provided an arbitrary set of data, how should we design our next evaluation location when exactly $\tau$ observations remain before termination? In sequential decision making, this value is known as the decision horizon, as it indicates how far we must look ahead into the future when reasoning about the present.

To facilitate our discussion, we pause to define notation for future data that will be encountered during optimization relative to the present. When considering an observation at some point $x$, we will call the value resulting from an observation there $y$. We will then call the dataset available at the next stage of optimization $\mathcal{D}_{1}=\mathcal{D} \cup\{(x, y)\}$, where the subscript indicates the number of future observations incorporated into the current data. We will write $\left(x_{2}, y_{2}\right)$ for the following observation, which when acquired will form $\mathcal{D}_{2}$, etc. Our final observation $\tau$ steps in the future will then be $\left(x_{\tau}, y_{\tau}\right)$, and the dataset returned by our optimization procedure will be $\mathcal{D}_{\tau}$, with utility $u\left(\mathcal{D}_{\tau}\right)$.

This utility of the data we return is our ultimate concern and will serve as the utility function used to design every observation. Note we may write this utility in the same form we introduced in our general discussion:

$$
u\left(\mathcal{D}_{\tau}\right)=u(\underbrace{\mathcal{D},}_{\text {known }} \underbrace{x,}_{\text {action }} \underbrace{y, x_{2}, y_{2}, \ldots, x_{\tau}, y_{\tau}}_{\text {unknown }}),
$$

which expresses the terminal utility in terms of a proposed current action $x$, the known data $\mathcal{D}$, and the unknown future data to be obtained: the not-yet observed value $y$, and the locations $\left\{x_{2}, \ldots, x_{\tau}\right\}$ and values $\left\{y_{2}, \ldots, y_{\tau}\right\}$ of any following observations.

Following our treatment of isolated decisions, we evaluate a potential observation location $x$ via the expected utility at termination ultimately obtained if we observe at that point next:

$$
\mathbb{E}\left[u\left(\mathcal{D}_{\tau}\right) \mid x, \mathcal{D}\right],
$$

and define an optimization policy via maximization:

$$
x \in \underset{x^{\prime} \in \mathcal{X}}{\arg \max } \mathbb{E}\left[u\left(\mathcal{D}_{\tau}\right) \mid x^{\prime}, \mathcal{D}\right] .
$$

On its surface, this proposal is relatively simple. However, we must now consider how to actually compute the expected terminal utility (5.5). fixed, known budget

cost-aware optimization: $\S 5.4$, p. 103

number of remaining observations (horizon), $\tau$

putative next observation and dataset: $(x, y)$, $\mathcal{D}_{1}$

putative following observation and dataset: $\left(x_{2}, y_{2}\right), \mathcal{D}_{2}$

putative final observation and dataset: $\left(x_{\tau}, y_{\tau}\right), \mathcal{D}_{\tau}$

expected terminal utility, $\mathbb{E}\left[u\left(\mathcal{D}_{\tau}\right) \mid x, \mathcal{D}\right]$ 7 This is known as BELLMAN's principle of optimality, and will be discussed further later in this section.

8 This procedure is often called "backward induction," where we consider the last decision first and work backward in time. Our approach of a forward induction on the horizon is equivalent.

defining a policy by maximizing an acquisition function: $\S 5$, p. 88

isolated decisions: §5.1, p. 89
Explicitly writing out the expectation over the future data in (5.5) yields the following expression:

$$
\int \ldots \int u\left(\mathcal{D}_{\tau}\right) p(y \mid x, \mathcal{D}) \prod_{i=2}^{\tau} p\left(x_{i}, y_{i} \mid \mathcal{D}_{i-1}\right) \mathrm{d} y \mathrm{~d}\left\{\left(x_{i}, y_{i}\right)\right\} .
$$

This integral certainly appears unwieldy! In particular, it is unclear how to reason about uncertainty in our future actions, as we should hope that these actions are made to maximize our welfare rather than generated by a random process. We will show how to compute this expression under the bold but rational assumption that we make all future decisions optimally, ${ }^{7}$ and this analysis will reveal the optimal optimization policy.

We will proceed via induction on the number of evaluations remaining before termination, $\tau$. We will first determine optimal behavior when only one observation remains and then inductively consider increasingly long horizons. ${ }^{8}$ For this analysis it will be useful to introduce notation for the expected increase in utility achieved when beginning from an arbitrary dataset $\mathcal{D}$, making an observation at $x$, and then continuing optimally until termination $\tau$ steps in the future. We will write

$$
\alpha_{\tau}(x ; \mathcal{D})=\mathbb{E}\left[u\left(\mathcal{D}_{\tau}\right) \mid x, \mathcal{D}\right]-u(\mathcal{D})
$$

for this quantity, which is simply the expected terminal utility (5.5) shifted by the utility of our existing data, $u(\mathcal{D})$. It is no coincidence this notation echoes our notation for acquisition functions! We will characterize the optimal optimization policy by a family of acquisition functions defined in this manner.

\section*{Fixed budget: one observation remaining}

We first consider the case where only one observation remains before termination; that is, the horizon is $\tau=1$. In this case the terminal dataset will be the current dataset augmented with a single additional observation. As there are no following decisions to consider, we may analyze the decision using the framework we have already developed for isolated decisions. The marginal gain in utility from a final evaluation at $x$ is an expectation over the corresponding value $y$ with respect to the posterior predictive distribution:

$$
\alpha_{1}(x ; \mathcal{D})=\int u\left(\mathcal{D}_{1}\right) p(y \mid x, \mathcal{D}) \mathrm{d} y-u(\mathcal{D})
$$

The optimal observation maximizes the expected marginal gain:

$$
x \in \underset{x^{\prime} \in \mathcal{X}}{\arg \max } \alpha_{1}\left(x^{\prime} ; \mathcal{D}\right),
$$

and leads to our returning a dataset with expected utility

$$
u(\mathcal{D})+\alpha_{1}^{*}(\mathcal{D}) ; \quad \alpha_{1}^{*}(\mathcal{D})=\max _{x^{\prime} \in \mathcal{X}} \alpha_{1}\left(x^{\prime} ; \mathcal{D}\right) .
$$

- observations _ posterior mean posterior $95 \%$ credible interval

![](https://cdn.mathpix.com/cropped/2023_09_22_d3cf6bd410ce34714365g-09.jpg?height=457&width=1693&top_left_y=548&top_left_x=267)

Figure 5.1: Illustration of the optimal optimization policy with a horizon of one. Above: we compute the expected marginal gain $\alpha_{1}$ over the domain and design our next observation $x$ by maximizing this score. Left: the computation of the expected marginal gain for the optimal point $x$ and a suboptimal point $x^{\prime}$ indicated above. In this example the marginal gain is a simple piecewise linear function of the observed value (5.11), and the optimal point maximizes its expectation.

Here we have defined the symbol $\alpha_{\tau}^{*}(\mathcal{D})$ to represent the expected increase in utility when starting with $\mathcal{D}$ and continuing optimally for $\tau$ additional observations. This is called the value of the dataset with a value of $\mathcal{D}$ with horizon $\tau, \alpha_{\tau}^{*}(\mathcal{D})$ horizon of $\tau$ and will serve a central role below. We have now shown how to compute the value of any dataset with a horizon of $\tau=1$ (5.10) and how to identify a corresponding optimal action (5.9). This completes the base case of our argument.

We illustrate the optimal optimization policy with one observation remaining in Figure 5.1. In this scenario the belief over the objective function $p(f \mid \mathcal{D})$ is a Gaussian process, and for simplicity we assume our observations reveal exact values of the objective. We consider an intuitive utility function: the maximal objective value contained in the data, $u(\mathcal{D})=\max f(\mathbf{x}) .{ }^{9}$ The marginal gain in utility offered by a putative final observation $(x, y)$ is then a piecewise linear function of the observed value:

$$
u\left(\mathcal{D}_{1}\right)-u(\mathcal{D})=\max \{y-u(\mathcal{D}), 0\}
$$

that is, the utility increases linearly if we exceed the previously bestseen value and otherwise remains constant. To design the optimal final observation, we compute the expectation of this quantity over the domain and choose the point maximizing it, as shown in the top panels. We also illustrate the computation of this expectation for the optimal choice and

illustration of one-step optimal optimization policy

9 This is a special case of the simple reward utility function, which we discuss further in the next chapter (§ 6.1, p. 109). The corresponding expected marginal gain is the well-known $e x-$ pected improvement acquisition function $(\S 7 \cdot 3$, p. 127). - observations posterior mean posterior $95 \%$ credible interval

![](https://cdn.mathpix.com/cropped/2023_09_22_d3cf6bd410ce34714365g-10.jpg?height=202&width=1625&top_left_y=550&top_left_x=158)

expected marginal gain, $\alpha_{2}$

v next observation location, $x$

Figure 5.2: Illustration of the optimal optimization policy with a horizon of two. Above: the expected two-step marginal gain $\alpha_{2}$. Right: computation of $\alpha_{2}$ for the optimal point $x$. The marginal gain is decomposed into two components (5.13): the immediate gain $u\left(\mathcal{D}_{1}\right)-u(\mathcal{D})$ and the expected future gain $\mathbb{E}\left[u\left(\mathcal{D}_{2}\right)-u\left(\mathcal{D}_{1}\right)\right]$. The chosen point offers a high expected future reward even if the immediate reward is zero; see the facing page for the scenarios resulting from the marked values.

![](https://cdn.mathpix.com/cropped/2023_09_22_d3cf6bd410ce34714365g-10.jpg?height=371&width=831&top_left_y=1082&top_left_x=955)

a suboptimal alternative in the bottom panel. We expect an observation at the chosen location to improve utility by a greater amount than any alternative.

\section*{Fixed budget: two observations remaining}

Rather than proceeding immediately to the inductive case, let us consider the specific case of two observations remaining: $\tau=2$. Suppose we have obtained an arbitrary dataset $\mathcal{D}$ and must decide where to make the penultimate observation $x$. The reasoning for this special case presents the inductive argument most clearly.

We again consider the expected increase in utility by termination, now after two observations:

$$
\alpha_{2}(x ; \mathcal{D})=\mathbb{E}\left[u\left(\mathcal{D}_{2}\right) \mid x, \mathcal{D}\right]-u(\mathcal{D})
$$

Nominally this expectation requires marginalizing the observation $y$, as well as the final observation location $x_{2}$ and its value $y_{2}$ (5.7). However, if we assume optimal future behavior, we can simplify our treatment of the final decision $x_{2}$. First, we rewrite the two-step expected gain $\alpha_{2}$ in terms of the one-step expected gain $\alpha_{1}$, a function for which we have already established a good understanding. We write the two-step difference in 
![](https://cdn.mathpix.com/cropped/2023_09_22_d3cf6bd410ce34714365g-11.jpg?height=736&width=1212&top_left_y=458&top_left_x=201)

Figure 5.3: The posterior of the objective function given two possible observations resulting from the optimal two-step observation $x$ illustrated on the facing page. The relatively low value $y^{\prime}$ offers no immediate reward, but reveals a new local optimum and the expected future reward from the optimal final decision $x_{2}$ is high. The relatively high value $y^{\prime \prime}$ offers a large immediate reward and respectable prospects from the optimal final decision as well.

utility as a telescoping sum:

$$
u\left(\mathcal{D}_{2}\right)-u(\mathcal{D})=\left[u\left(\mathcal{D}_{1}\right)-u(\mathcal{D})\right]+\left[u\left(\mathcal{D}_{2}\right)-u\left(\mathcal{D}_{1}\right)\right]
$$

which yields

$$
\alpha_{2}(x ; \mathcal{D})=\alpha_{1}(x ; \mathcal{D})+\mathbb{E}\left[\alpha_{1}\left(x_{2} ; \mathcal{D}_{1}\right) \mid x, \mathcal{D}\right]
$$

That is, the expected increase in utility after two observations can be decomposition of expected marginal gain decomposed as the expected increase after our first observation $x$ - the expected immediate gain - plus the expected additional increase from the final observation $x_{2}$ - the expected future gain.

It is still not clear how to address the second term in this expression. However, from our analysis of the base case, we can reason as follows. Given $y$ (and thus knowledge of $\mathcal{D}_{1}$ ), the optimal final decision $x_{2}$ (5.9) results in an expected marginal gain of $\alpha_{1}^{*}\left(\mathcal{D}_{1}\right)$, a quantity we know how to compute (5.10). Therefore, assuming optimal future behavior, we have:

$$
\alpha_{2}(x ; \mathcal{D})=\alpha_{1}(x ; \mathcal{D})+\mathbb{E}\left[\alpha_{1}^{*}\left(\mathcal{D}_{1}\right) \mid x, \mathcal{D}\right]
$$

which expresses the desired quantity as an expectation with respect to the current observation $y$ only - the future value $\alpha_{1}^{*}(5.10)$ does not depend on either $x_{2}$ (due to maximization) or $y_{2}$ (due to expectation). The optimal penultimate observation location maximizes the expected gain as usual:

$$
x \in \underset{x^{\prime} \in \mathcal{X}}{\arg \max } \alpha_{2}\left(x^{\prime} ; \mathcal{D}\right),
$$

and provides an expected terminal utility of

$$
u(\mathcal{D})+\alpha_{2}^{*}(\mathcal{D}) ; \quad \alpha_{2}^{*}(\mathcal{D})=\max _{x^{\prime} \in \mathcal{X}} \alpha_{2}\left(x^{\prime} ; \mathcal{D}\right)
$$

illustration of two-step optimal optimization policy
10 Namely:

$$
\begin{aligned}
& u\left(\mathcal{D}_{\tau}\right)-u(\mathcal{D})= \\
& {\left[u\left(\mathcal{D}_{1}\right)-u(\mathcal{D})\right]+\left[u\left(\mathcal{D}_{\tau}\right)-u\left(\mathcal{D}_{1}\right)\right] .}
\end{aligned}
$$

This demonstrates we can achieve optimal behavior for a horizon of $\tau=2$ and compute the value of any dataset with this horizon.

The optimal policy with two observations remaining is illustrated in Figures 5.2 and 5.3. The former shows the expected two-step marginal gain $\alpha_{2}$ and the optimal action. This quantity depends both on the immediate gain from the next observation and the expected future gain from the optimal final action. The chosen observation appears quite promising: even if the result offers no immediate gain, it will likely provide information that can be exploited with the optimal final decision $x_{2}$. We show the situation that would be faced in the final stage of optimization for two potential values in Figure 5.3. The relatively low value $y^{\prime}$ offers no immediate gain but sets up an encouraging final decision, whereas the relatively high value $y^{\prime \prime}$ offers a significant immediate gain with some chance of further improvement.

\section*{Fixed budget: inductive case}

We now present the general inductive argument, which closely follows the $\tau=2$ analysis above. Let $\tau$ be an arbitrary decision horizon, and for the sake of induction assume we can compute the value of any dataset with a horizon of $\tau-1$. Suppose we have an arbitrary dataset $\mathcal{D}$ and must decide where to make the next observation. We will show how to do so optimally and how to compute its value with a horizon of $\tau$.

Consider the $\tau$-step expected gain in utility from observing at some point $x$ :

$$
\alpha_{\tau}(x ; \mathcal{D})=\mathbb{E}\left[u\left(\mathcal{D}_{\tau}\right) \mid x, \mathcal{D}\right]-u(\mathcal{D}),
$$

which we seek to maximize. We decompose this expression in terms of shorter-horizon quantities through a telescoping sum: ${ }^{10}$

$$
\alpha_{\tau}(x ; \mathcal{D})=\alpha_{1}(x ; \mathcal{D})+\mathbb{E}\left[\alpha_{\tau-1}\left(x_{2} ; \mathcal{D}_{1}\right) \mid x, \mathcal{D}\right] .
$$

Now if we knew $y$ (and thus $\mathcal{D}_{1}$ ), optimal continued behavior would provide an expected further gain of $\alpha_{\tau-1}^{*}\left(\mathcal{D}_{1}\right)$, a quantity we can compute via the inductive hypothesis. Therefore, assuming optimal behavior for all remaining decisions, we have:

$$
\alpha_{\tau}(x ; \mathcal{D})=\alpha_{1}(x ; \mathcal{D})+\mathbb{E}\left[\alpha_{\tau-1}^{*}\left(\mathcal{D}_{1}\right) \mid x, \mathcal{D}\right],
$$

which is an expectation with respect to $y$ of a function we can compute. To find the optimal decision and the $\tau$-step value of the data, we maximize:

$$
\begin{array}{r}
x \in \underset{x^{\prime} \in \mathcal{X}}{\arg \max } \alpha_{\tau}\left(x^{\prime} ; \mathcal{D}\right) \\
\alpha_{\tau}^{*}(\mathcal{D})=\max _{x^{\prime} \in \mathcal{X}} \alpha_{\tau}\left(x^{\prime} ; \mathcal{D}\right) .
\end{array}
$$

This demonstrates we can achieve optimal behavior for a horizon of $\tau$ given an arbitrary dataset and compute its corresponding value, establishing the inductive case and completing our analysis. We pause to note that the value of any dataset with null horizon is $\alpha_{0}^{*}(\mathcal{D})=0$, and thus the expressions in $\left(5.15^{-5.17)}\right.$ are valid for any horizon and compactly express the proposed policy. Further, we have actually shown that this policy is optimal in the sense of maximizing expected terminal utility over the space of all policies, at least with respect to our model of the objective function and observations. This follows from our induction: the base case is established in (5.9), and the inductive case by the sequential maximization in $(5 \cdot 16) .{ }^{11}$

\section*{Bellman optimality and the Bellman equation}

Substituting (5.15) into (5.17), we may derive the following recursive definition of the value in terms of the value of future data:

$$
\alpha_{\tau}^{*}(\mathcal{D})=\max _{x^{\prime} \in \mathcal{X}}\left\{\alpha_{1}\left(x^{\prime} ; \mathcal{D}\right)+\mathbb{E}\left[\alpha_{\tau-1}^{*}\left(\mathcal{D}_{1}\right) \mid x^{\prime}, \mathcal{D}\right]\right\} .
$$

This is known as the Bellman equation and is a central result in the theory of optimal sequential decisions. ${ }^{12}$ The treatment of future decisions in this equation - recursively assuming that we will always act to maximize expected terminal utility given the available data - reflects BELLMAN's principle of optimality, which characterizes optimal sequential decision policies in terms of the optimality of subpolicies: $:^{13}$

An optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision.

That is, to make a sequence of optimal decisions, we make the first decision optimally, then make all following decisions optimally given the outcome!

\section*{COST AND APPROXIMATION OF THE OPTIMAL POLICY}

Although the framework presented in the previous section is conceptually simple and theoretically attractive, the optimal policy is unfortunately prohibitive to compute except for very short decision horizons.

To demonstrate the key computational barrier, consider the selection of the penultimate observation location. The expected two-step marginal gain to be maximized is (5.13):

$$
\alpha_{2}(x ; \mathcal{D})=\alpha_{1}(x ; \mathcal{D})+\mathbb{E}\left[\alpha_{1}^{*}\left(\mathcal{D}_{1}\right) \mid x, \mathcal{D}\right]
$$

The second term appears to be a straightforward expectation over the one-dimensional random variable $y$. However, evaluating the integrand in this expectation requires solving a nontrivial global optimization problem (5.10)! Even with only two evaluations remaining, we must solve a doubly nested global optimization problem, an onerous task.

Close inspection of the recursively defined optimal policy $\left(5 \cdot 15^{-5.16}\right)$ reveals that when faced with a horizon of $\tau$, we must solve $\tau$ nested optimal policy: compact notation

optimality

11 Since ties in (5.16) may be broken arbitrarily, this argument does not rule out the possibility of there being multiple, equally good optimal policies.

12 R. Bellman (1952). On the Theory of Dynamic Programming. Proceedings of the $\mathrm{Na}$ tional Academy of Sciences 38(8):716-719.

Bellman equation

BELLMAN's principle of optimality

13 R. Bellman (1957). Dynamic Programming. Princeton University Press.

"unrolling" the optimal sequential policy 

![](https://cdn.mathpix.com/cropped/2023_09_22_d3cf6bd410ce34714365g-14.jpg?height=363&width=1442&top_left_y=455&top_left_x=267)

Figure 5.4: The optimal optimization policy as a decision tree. Squares indicate decisions (the choice of each observation), and circles represent expectations with respect to random variables (the outcomes of observations). Only one possible optimization path is shown; dangling edges lead to different futures, and all possibilities are always considered. We maximize the expected terminal utility $u\left(\mathcal{D}_{\tau}\right)$, recursively assuming optimal future behavior.

running time of optimal policy

evaluation budget for optimization, $n$ evaluation budget for quadrature, $q$

14 Detailed references are provided by:

W. B. POWELL (2011). Approximate Dynamic Programming: Solving the Curses of Dimensionality. John Wiley \& Sons.

D. P. BERTSEKAS (2017). Dynamic Programming and Optimal Control. Vol. 1. Athena Scientific. optimization problems to find the optimal decision. Temporarily adopting compact notation, we may "unroll" the optimal policy as follows:

$$
\begin{aligned}
& x \in \arg \max \alpha_{\tau} ; \\
& \alpha_{\tau}=\alpha_{1}+\mathbb{E}\left[\alpha_{\tau-1}^{*}\right] \\
& =\alpha_{1}+\mathbb{E}\left[\max \alpha_{\tau-1}\right] \\
& =\alpha_{1}+\mathbb{E}\left[\max \left\{\alpha_{1}+\mathbb{E}\left[\alpha_{\tau-2}^{*}\right]\right\}\right] \\
& =\alpha_{1}+\mathbb{E}\left[\operatorname { m a x } \left\{\alpha_{1}+\mathbb{E}\left[\operatorname { m a x } \left\{\alpha_{1}+\mathbb{E}\left[\max \left\{\alpha_{1}+\cdots\right]\right. \text {. }\right.\right.\right.\right.
\end{aligned}
$$

The design of each optimal decision requires repeated maximization over the domain and expectation over unknown observations until the horizon is reached. This computation is visualized as a decision tree in Figure 5.4, where it is clear that each unknown quantity contributes a significant branching factor. Computing the expected utility at $x$ exactly requires a complete traversal of this tree.

The cost of computing the optimal policy clearly grows with the horizon. Let us perform a careful running time analysis for a naïve implementation via exhaustive traversal of the decision tree in Figure 5.4 with off-the-shelf procedures. Suppose we use an optimization routine for each maximization and a numerical quadrature routine for each expectation encountered in this computation. If we allow $n$ evaluations of the objective for each call to the optimizer and $q$ observations of the integrand for each call to the quadrature routine, then each decision along the horizon will contribute a multiplicative factor of $\mathcal{O}(n q)$ to the total running time. Computing the optimal decision with a horizon of $\tau$ thus requires $\mathcal{O}\left(n^{\tau} q^{\tau}\right)$ work, an exponential growth in running time with respect to the horizon.

Evidently, the computational effort required for realizing the optimal policy quickly becomes intractable, and we must find some alternative mechanism for designing effective optimization policies. General approximation schemes for the optimal policy have been studied in depth under the name approximate dynamic programming, ${ }^{14}$ and usually operate as 

![](https://cdn.mathpix.com/cropped/2023_09_22_d3cf6bd410ce34714365g-15.jpg?height=223&width=1036&top_left_y=485&top_left_x=270)

follows. We begin with the intractable optimal expected marginal gain (5.15):

$$
\alpha_{\tau}(x ; \mathcal{D})=\alpha_{1}(x ; \mathcal{D})+\mathbb{E}\left[\alpha_{\tau-1}^{*}\left(\mathcal{D}_{1}\right) \mid x, \mathcal{D}\right],
$$

and substitute a tractable approximation for the "hard" part of the expression: the recursively defined future value $\alpha^{*}$ (5.18). The result is an acquisition function inducing a suboptimal - but rationally guided - approximate policy. Two particular approximations schemes have proven useful in Bayesian optimization: limited lookahead and rollout.

\section*{Limited lookahead}

One widespread and surprisingly effective approximation is to simply limit how many future observations we consider in each decision. This is practical as decisions closer to termination require substantially less computation than earlier decisions.

With this in mind, we can construct a natural family of approximations to the optimal policy defined by artificially limiting the horizon used throughout optimization to some computationally feasible maximum $\ell$. When faced with an infeasible decision horizon $\tau$, we make the crude approximation

$$
\alpha_{\tau}(x ; \mathcal{D}) \approx \alpha_{\ell}(x ; \mathcal{D})
$$

and by maximizing this score, we act optimally under the incorrect but convenient assumption that only $\ell$ observations remain. This effectively assumes $u\left(\mathcal{D}_{\tau}\right) \approx u\left(\mathcal{D}_{\ell}\right) .{ }^{15}$ This may be reasonable if we expect decreasing marginal gains, implying a significant fraction of potential gains can be attained within the truncated horizon. This scheme is often described (sometimes disparagingly) as myopic, as we limit our sight to only the next few observations rather than looking ahead to the full horizon.

A policy that designs each observation to maximize the limitedhorizon acquisition function $\alpha_{\min \{\ell, \tau\}}$ is called an $\ell$-step lookahead policy. ${ }^{16}$ This is also called a rolling horizon strategy, as the fixed horizon "rolls along" with us as we go. By limiting the horizon, we bound the computational effort required for each decision to at-most $\mathcal{O}\left(n^{\ell} q^{\ell}\right)$ time with the implementation described above. This can be a considerable speedup when the observation budget is much greater than the selected lookahead. A lookahead policy is illustrated as a decision tree in Figure 5.5. Comparing to the optimal policy in Figure 5.4, we simply "cut off" and ignore any portion of the tree lying deeper than $\ell$ steps in the future.
Figure 5.5: A lookahead approximation to the optimal optimization policy. We choose the optimal decision for a limited horizon $\ell \ll \tau$ decisions, ignoring any observations that would follow.
15 Equivalently, we approximate the true future value $\alpha_{\tau-1}^{*}$ with $\alpha_{\ell-1}^{*}$.

myopic approximations

$\ell$-step lookahead

rolling horizon

16 We take the minimum to ensure we don't look beyond the true horizon, which would be nonsense. 

![](https://cdn.mathpix.com/cropped/2023_09_22_d3cf6bd410ce34714365g-16.jpg?height=245&width=1422&top_left_y=457&top_left_x=274)

Figure 5.6: A decision tree representing a rollout policy. Comparing to the optimal policy in Figure 5.4, we simulate future decisions starting with $x_{2}$ using an efficient but suboptimal heuristic policy, rather than the intractable optimal policy. We maximize the expected terminal utility $u\left(\mathcal{D}_{\tau}\right)$, assuming potentially suboptimal future behavior.

one-step lookahead

Chapter 7: Common Bayesian Optimization Policies, p. 123

base policy, heuristic policy

choice of base policy
Particularly important in Bayesian optimization is the special case of one-step lookahead, which successively maximizes the expected marginal gain after acquiring a single additional observation, $\alpha_{1}$. One-step lookahead is the most efficient lookahead approximation (barring the absurdity that would be "zero-step" lookahead), and it is often possible to derive closed-form, analytically differentiable expressions for $\alpha_{1}$, enabling efficient implementation. Many well-known acquisition functions represent one-step lookahead approximations for some implicit choice of utility function, as we will see in Chapter 7 .

\section*{Rollout}

The optimal policy evaluates a potential observation location by simulating the entire remainder of optimization following that choice, recursively assuming we will use the optimal policy for every future decision. Although sensible, this is clearly intractable. Rollout is an approach to approximate policy design that emulates the structure of the optimal policy, but using a tractable suboptimal policy to simulate future decisions.

A rollout policy is illustrated as a decision tree in Figure 5.6. Given a putative next observation $(x, y)$, we use an inexpensive so-called base or heuristic policy to simulate a plausible - but perhaps suboptimal realization of the following decision $x_{2}$. Note there is no branching in the tree corresponding to this decision, as it does not depend on the exhaustively enumerated subtree required by the optimal policy. We then take an expectation with respect to the unknown value $y_{2}$ as usual. Given a putative value of $y_{2}$, we use the base policy to select $x_{3}$ and continue in this manner until reaching the decision horizon. We use the terminal utilities in the resulting pruned tree to estimate the expected marginal gain $\alpha_{\tau}$, which we maximize as a function of $x$.

There are no constraints on the design of the base policy used in rollout; however, for this approximation to be sensible, we must choose something relatively efficient. One common and often effective choice is to simulate future decisions with one-step lookahead. If we again use off-the-shelf optimization and quadrature routines to traverse the rollout decision tree in Figure 5.6 with this particular choice, the running time of the policy with a horizon of $\tau$ is $\mathcal{O}\left(n^{2} q^{\tau}\right)$, significantly faster 

![](https://cdn.mathpix.com/cropped/2023_09_22_d3cf6bd410ce34714365g-17.jpg?height=226&width=939&top_left_y=498&top_left_x=296)

than the optimal policy. Although there is still exponential growth with respect to $q$, we typically have $q \ll n{ }^{17}$ so we can usually entertain farther horizons with rollout than with limited lookahead with the same amount of computational effort.

Due to the flexibility in the design of the base policy, rollout is a remarkably flexible approximation scheme. For example, we can combine rollout with the idea of limiting the decision horizon to yield approximate policies with tunable running time. In fact, we can interpret $\ell$-step lookahead as a special case of rollout, where the base policy designs the next $\ell-1$ decisions optimally assuming a myopic horizon and then simply terminates early, discarding any remaining budget.

We may also adopt a base policy that designs all remaining observations simultaneously. Ignoring the dependence between these decisions can provide a computational advantage while retaining awareness of the evolving decision horizon, and such batch rollout schemes have proven useful in Bayesian optimization. A batch rollout policy is illustrated as a decision tree in Figure 5.7. Although we account for the entire horizon, the tree depth is reduced dramatically compared to the optimal policy.

\subsection*{COST-AWARE OPTIMIZATION AND TERMINATION AS A DECISION}

Thus far we have only considered the construction of optimization policies under a known budget on the total number of observations. Although this scenario is pervasive, it is not universal. In some situations, we might wish instead to use our evolving beliefs about the objective function to decide dynamically when termination is the best course of action.

Dynamic termination can be especially prudent when we want to reason explicitly about the cost of data acquisition during optimization. For example, if this cost were to vary across the domain, it would not be sensible to define a budget in terms of function evaluations. However, by accounting for observation costs in the utility function, we can reason about cost-benefit tradeoffs during optimization and seek to terminate whenever the expected cost of further observation outweighs any expected benefit it might provide.

\section*{Modeling termination decisions and the optimal policy}

We consider a modification to the sequential decision problem we analyzed in the known-budget case, wherein we now allow ourselves to
Figure 5.7: A batch rollout policy as a decision tree. Given a putative value for the next evaluation $(x, y)$, we design all remaining decisions simultaneously using a batch base policy and take the expectation of the terminal utility with respect to their values.

17 For estimating a one-dimensional expectation we might take $q$ on the order of roughly 10, but for optimizing a nonconvex acquisition function over the domain we might take $n$ on the order of thousands or more.

limited lookahead as rollout

batch rollout action space, $\mathcal{A}$

termination option, $\varnothing$

bound on total number of observations, $\tau_{\max }$

18 It is possible to consider unbounded sequential decision problems, but this is probably not of practical interest in Bayesian optimization:

M. H. DEGROOT (1970). Optimal Statistical Decisions. McGraw-Hill. [§ 12.7]
19 This can be proven through various "information never hurts" (in expectation) results. terminate optimization at any time of our choosing. Suppose we are at an arbitrary point of optimization and have already obtained data $\mathcal{D}$. We face the following decision: should we terminate optimization immediately and return $\mathcal{D}$ ? If not, where should we make our next observation?

We model this scenario as a decision problem under uncertainty with an action space equal to the domain $\mathcal{X}$, representing potential observation locations if we decide to continue, augmented with a special additional action $\varnothing$ representing immediate termination:

$$
\mathcal{A}=\mathcal{X} \cup\{\varnothing\}
$$

For the sake of analysis, after the termination action has been selected, it is convenient to model the decision process as not actually terminating, but rather continuing with the collapsed action space $\mathcal{A}=\{\varnothing\}$ - once you terminate, there's no going back.

As before, we may derive the optimal optimization policy in the adaptive termination case via induction on the decision horizon $\tau$. However, we must address one technical issue: the base case of the induction, which analyzes the "final" decision, breaks down if we allow the possibility of a nonterminating sequence of decisions. To sidestep this issue, we assume there is a fixed and known upper bound $\tau_{\max }$ on the total number of observations we may make, at which point optimization is compelled to terminate regardless of any other concern. This is not an overly restrictive assumption in the context of Bayesian optimization. Because observations are assumed to be expensive, we can adopt some suitably absurd upper bound without issue; for example, $\tau_{\max }=1000000$ would suffice for an overwhelming majority of plausible scenarios. ${ }^{18}$

After assuming the decision process is bounded, our previous inductive argument carries through after we demonstrate how to compute the value of the termination action. Fortunately, this is straightforward: termination does not augment our data, and once this action is taken, no other action will ever again be allowed. Therefore the expected marginal gain from termination is always zero:

$$
\alpha_{\tau}(\varnothing ; \mathcal{D})=0 .
$$

With this, substituting $\mathcal{A}$ for $\mathcal{X}$ in $(5.15-5.17)$ now gives the optimal policy.

Intuitively, the result in (5.20) implies that termination is only the optimal decision if there is no observation offering positive expected gain in utility. For the utility functions described in the next chapter all of which are agnostic to costs and measure optimization progress alone - reaching this state is actually impossible. ${ }^{19}$ However, explicitly accounting for observation costs in addition to optimization progress in the utility function resolves this issue, as we will demonstrate.

\section*{Example: cost-aware optimization}

To illustrate the behavior of a policy allowing early termination, we return to our motivating scenario of accounting for observation costs. - observations _ p posterior mean posterior $95 \%$ credible interval

![](https://cdn.mathpix.com/cropped/2023_09_22_d3cf6bd410ce34714365g-19.jpg?height=665&width=1682&top_left_y=547&top_left_x=267)

Figure 5.8: Illustration of one-step lookahead with the option to terminate. With a linear utility and additive costs, the expected marginal gain $\alpha_{1}$ is the expected marginal gain to the data utility $\alpha_{1}^{\prime}$ adjusted for the cost of acquisition c. For some points, the cost-adjusted expected gain is negative, in which case we would prefer immediate termination to observing there. However, continuing with the chosen point is expected to increase the utility of the current data.

Consider the objective function belief in the top panel of Figure 5.8 (which is identical to that from our running example from Figures 5.1-5.3) and suppose that the cost of observation now depends on location according to a known cost function $c(x),{ }^{20}$ illustrated in the middle panel.

If we wish to reason about observation costs in the optimization policy, we must account for them somehow, and the most natural place to do so is in the utility function. Depending on the situation, there are many ways we could proceed ${ }^{21}$ however, one natural approach is to first select a utility function measuring the quality of a returned dataset alone, ignoring any costs incurred to acquire it. We call this quantity the data utility and notate it with $u^{\prime}(\mathcal{D})$. The data utility is akin to the cost-agnostic utility from the known-budget case, and any one of the options described in the next chapter could reasonably fill this role.

We now adjust the data utility to account for the cost of data acquisition. In many applications, these costs are additive, so that the total cost of gathering a dataset $\mathcal{D}$ is simply

$$
c(\mathcal{D})=\sum_{x \in \mathcal{D}} c(x)
$$

If the acquisition cost can be expressed in the same units as the data utility - for example, if both can be expressed in monetary terms ${ }^{22}-$ then we might reasonably evaluate a dataset $\mathcal{D}$ by the cost-adjusted utility:

$$
u(\mathcal{D})=u^{\prime}(\mathcal{D})-c(\mathcal{D}) .
$$

2o We will consider unknown and stochastic costs in $\S 11.1$, p. 245.

observation cost function, $c(x)$

21 We wish to stress this point - there is considerable flexibility beyond the scheme we describe.

data utility, $u^{\prime}(\mathcal{D})$

Chapter 6: Utility Functions for Optimization, p. 109

observation costs, $c(\mathcal{D})$

22 Some additional discussion on this natural approach can be found in:

H. RAIFFA and R. SCHLAIfER (1961). Applied Statistical Decision Theory. Division of Research, Graduate School of Business Administration, Harvard University. [chapter 4] example and discussion

defining optimization policies via acquisition functions: p. 88

introduction to Bayesian decision theory: $\S 5.1$, p. 89

modeling policy decisions: §5.2, p. 91

\section*{Demonstration: one-step lookahead with cost-aware utility}

Returning to the scenario in Figure 5.8, let us adopt a cost-aware utility function of the above form (5.22) and consider the behavior of a one-step lookahead approximation to the optimal optimization policy.

For these choices, if we were to continue optimization by evaluating at a point $x$, the resulting one-step marginal gain in utility would be:

$$
u\left(\mathcal{D}_{1}\right)-u(\mathcal{D})=\left[u^{\prime}\left(\mathcal{D}_{1}\right)-u^{\prime}(\mathcal{D})\right]-c(x),
$$

the cost-adjusted marginal gain in the data utility alone. Therefore the expected marginal gain in utility is:

$$
\alpha_{1}(x ; \mathcal{D})=\alpha_{1}^{\prime}(x ; \mathcal{D})-c(x)
$$

where $\alpha_{1}^{\prime}$ is the one-step expected gain in the data utility (5.8). That is, we simply adjust what would have been the acquisition function in the cost-agnostic setting by subtracting the cost of data acquisition. To prefer evaluating at $x$ to immediate termination, this quantity must have positive expected value (5.20).

The resulting policy is illustrated in Figure 5.8. The middle panel shows the cost-agnostic acquisition function $\alpha_{1}^{\prime}$ (from Figure 5.1), which is then adjusted for observation cost in the bottom panel. This renders the expected marginal gain negative in some locations, where observations are not expected to be worth their cost. However, in this case there are still regions where observation is favored to termination, and optimization continues at the selected location. Comparing with the cost-agnostic setting in Figure 5.1, the optimal observation has shifted from the righthand side to the left-hand side of the previously best-seen point, as an observation there is more cost-effective.

\subsection*{SUMMARY OF MAJOR IDEAS}

- Optimization policies can be conveniently defined via an acquisition function assigning a score to each potential observation location. We then design observations by maximizing the acquisition function (5.1).

- Bayesian decision theory is a general framework for optimal decision making under uncertainty, through which we can derive optimal optimization policies and stopping rules.

- The key elements of a decision problem under uncertainty are:

- an action space $\mathcal{A}$, from which we must choose an action $a$,

- uncertainty in elements $\psi$ relevant to the decision, represented by a posterior belief $p(\psi \mid \mathcal{D})$, and

- a utility function $u(a, \psi, \mathcal{D})$ quantifying the quality of the action $a$ assuming a given realization of the uncertain elements $\psi$.

Given these, an optimal decision maximizes the expected utility (5.2-5.3).

- Optimization policy decisions may be cast in this framework by defining a utility function for the data returned by an optimizer, then designing each observation location to maximize the expected utility with respect to all future data yet to be obtained (5.5-5.6).

- To ensure the optimality of a sequence of decisions, we must recursively assume the optimality of all future decisions. This is known as BELLMAN's principle of optimality. Under this assumption, the optimal policy can be derived inductively and assumes a simple recursive form (5.15-5.17).

- The cost of computing the optimal policy grows exponentially with the decision horizon, but several techniques under the umbrella approximate dynamic programming provide tractable approximations. Two notable examples are limited lookahead, where the decision horizon is artificially limited, and rollout, where future decisions are simulated suboptimally.

- Through careful accounting, we may explicitly account for the (possibly nonuniform) cost of data acquisition in the utility function. Offering a termination option and computing the resulting optimal policy then allows us to adaptively terminate optimization when continuing optimization becomes a losing battle of cost versus expected gain.

In the next chapter we will discuss several prominent utility functions for measuring the quality of a dataset returned by an optimization procedure. In the following chapter, we will demonstrate how many common acquisition functions for Bayesian optimization may be realized by performing one-step lookahead with these utility functions.
BELLMAN's principle of optimality: $§ 5.2$, p. 99

computational burden and approximation of the optimal policy: §5.3, p. 99

termination as a decision: $§ 5.4$, p. 103

Chapter 7: Common Bayesian Optimization Policies, p. 123 