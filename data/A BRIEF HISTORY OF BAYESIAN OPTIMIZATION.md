\section*{A BRIEF HISTORY OF BAYESIAN OPTIMIZATION}

In this chapter we provide a historical survey of the ideas underpinning Bayesian optimization, including important mathematical precedents. We also document the progression of major ideas in Bayesian optimization, from its first appearance in 1962 to the present day. A major goal will be to identify the point of introduction of prominent Bayesian optimization policies, as well as notable instances of subsequent reintroduction when ideas were forgotten and later rediscovered and refined.

The Bayesian approach to optimization is founded on a simple premise: experiments should be designed with purpose, both guided by our knowledge and aware of our ignorance. The optimization policies built on this principle can be understood as statistical manifestations of rational inquiry, where we design a sequence of experiments to systematically reveal the maximal value attained by the system of interest.

Statistical approaches to experimental design have a long history, with the earliest examples appearing over 200 years ago. ${ }^{1,2}$ A landmark early contribution was SMITH's 1918 dissertation, ${ }^{3}$ which considered experimental design for polynomial regression models to minimize a measure of predictive uncertainty. Shortly thereafter, FISHER published a hugely influential guide to statistical experimental design based on his experience analyzing crop experiments at Rothamsted Experimental Station, ${ }^{4}$ which was instrumental in shaping modern statistical practice. These works served to establish the field of optimal design, an expansive subject which has now enjoyed a century of study. Numerous excellent references are available. ${ }^{5,6}$

Early work in optimal design did not consider the possibility of adaptively designing a sequence of experiments, an essential feature of Bayesian optimization. Instead, the focus was optimizing fixed designs to minimize some measure of uncertainty when performing inference with the resulting data. This paradigm is practical when experiments are extremely time consuming but can easily run in parallel, such as the agricultural experiments studied extensively by FISHER. The most common goals considered in classical optimal design are accurate estimation of model parameters and confident prediction at unseen locations; these goals are usually formulated in terms of optimizing some statistical criterion as a function of the design. ${ }^{7}$ However, in 1941, HOTELLING notably studied experimental designs for estimating the location of the maximum of an unknown function in this nonadaptive setting. ${ }^{8}$ This was perhaps the first rigorous treatment of batch optimization.

\section*{SEQUENTIAL ANALYSIS AND BAYESIAN EXPERIMENTAL DESIGN}

Concentrated study of sequential experiments began during World War II with WALD, who pioneered the field of sequential analysis. The seminal

\section*{2}

1 J. D. GERGONNE (1815). Application de la méthode des moindres quarrés à l'interpolation des suites. Annales de Mathématiques pures et appliquées 6:242-252.

2 C. S. PEIRCE (1876). Note on the Theory of the Economy of Research. In: Report of the Superintendent of the United States Coast Survey Showing the Progress of the Work for the Fiscal Year Ending with June, 1876.

3 к. SмIтн (1918). On the Standard Deviations of Adjusted and Interpolated Values of an Observed Polynomial Function and Its Constants and the Guidance They Give towards a Proper Choice of the Distribution of Observations. Biometrika 12(1-2):1-85.

4 R. A. FISHER (1935). The Design of Experiments. Oliver and Boyd.

5 G. E. P. BOx et al. (2005). Statistics for Experimenters: Design, Innovation, and Discovery. John Wiley \& Sons.

6 D. C. MONTGomery (2019). Design and Analysis of Experiments. John Wiley \& Sons.

7 These criteria often have alphabetic names: A-optimality, D-optimality, v-optimality, etc.

8 H. HOTELLING (1941). Experimental Determination of the Maximum of a Function. The Annals of Mathematical Statistics 12(1):20-45. 9 A. WALD (1945). Sequential Tests of Statistical Hypotheses. The Annals of Mathematical Statistics 16(2):117-186.

10 A. WALD (1947). Sequential Analysis. John Wiley \& Sons.
11 M. FRIEDMAN and L. J. SAVAGE (1947). Planning Experiments Seeking Maxima. In: Selected Techniques of Statistical Analysis for Scientific and Industrial Research, and Production and Management Engineering.

12 H. HOTELLING (1941). Experimental Determination of the Maximum of a Function. The Annals of Mathematical Statistics 12(1):20-45.

13 G. E. P. BOX and K. B. WILson (1951). On the Experimental Attainment of Optimum Conditions. Fournal of the Royal Statistical Society Series B (Methodological) 13(1):1-45.

14 G. E. P. BOX (1954). The Exploration and Exploitation of Response Surfaces: Some General Considerations and Examples. Biometrics 10(1): 16-6o.

15 G. E. P. Box and P. V. Youle (1954). The Exploration and Exploitation of Response Surfaces: An Example of the Link between the Fitted Surface and the Basic Mechanism of the System. Biometrics 11(3):287-323.

16 H. CHERNOFF (1959). Sequential Design of Experiments. The Annals of Mathematical Statistics 30(3):755-770.

17 H. Chernoff (1972). Sequential Analysis and Optimal Design. Society for Industrial and Applied Mathematics.

18 J. BATHER (1996). A Conversation with Herman Chernoff. Statistical Science 11(4):335-350. result was a hypothesis testing procedure for sequentially gathered data that can terminate dynamically once the result is known with sufficient confidence. Such sequential tests can be significantly more data efficient than tests requiring an a priori fixed sample size. The potential benefit of WALD's work to the war effort was immediately recognized, and his research was classified and only published after the war. ${ }^{9,10}$ The introduction of sequential analysis would kick off multiple parallel lines of investigation, leading to both Bayesian sequential experimental design and multi-armed bandits, and eventually all modern approaches to Bayesian optimization.

The success of sequential analysis led numerous researchers to investigate sequential experimental design in the following years. Sequential experimental design for optimization was proposed by FRIEDMAN and SAVAGE as early as $1947 .{ }^{11}$ They argued that nonadaptive optimization procedures, as considered earlier by HOTELLING, ${ }^{12}$ can be wasteful as many experiments may be squandered needlessly exploring suboptimal regions. Instead, FRIEDMAN and SAVAGE suggested sequential optimization could be significantly more efficient, as poor regions of the domain can be quickly discarded while more time is spent exploiting more promising areas. Their proposed algorithm was a simple procedure entailing successive axis-aligned line searches, optimizing each input variable in turn while holding the others fixed - what would now be known as cyclic coordinate descent.

BOx greatly expounded on these ideas, working for years alongside a chemist (WILSON) experimentally optimizing chemical processes as a function of environmental and process parameters, for example, optimizing product yield as a function of reactant concentrations. ${ }^{13,14}$ BOX advocated the method of steepest ascent during early stages of optimization rather than the "one factor at a time" heuristic described by FRIEDMAN and SAVAGE, pointing out that the latter method is prone to becoming stuck in ridge-shaped features of the objective. BOX and YoulE also provided insightful commentary on how the process of optimization, and in particular geometric features of the objective function surface, may lead the experimenter to a greater fundamental understanding of underlying physical processes. ${ }^{15}$

In the following years, researchers developed general methods for sequential experimental design targeting a broad range of experimental goals. An early pioneer was CHERNOFF, a student of wALD's, who extended sequential analysis to the adaptive setting and provided asymptotically optimal procedures for sequential hypothesis testing. ${ }^{16} \mathrm{He}$ also wrote an survey of this early work, ${ }^{17}$ and would eventually remark in an interview: ${ }^{18}$

Although I regard myself as non-Bayesian, I feel in sequential problems it is rather dangerous to play around with non-Bayesian procedures.

Another important contribution around this time was the reintroduction of multi-armed bandits by ROBBINs, which would quickly explode into a massive body of literature. We will return to this line of work momentarily.

The Bayesian approach to sequential experimental design was formalized shortly after CHERNOFF's initial work. Authors such as RAIFFA and SCHLAIFER ${ }^{19}$ and LINDLEY $^{20}$ promoted a general approach based on Bayesian decision theory, wherein the experimenter selects a utility function reflecting their experimental goals and a model for reasoning about experimental outcomes given data, then designs each experiment to maximize the expected utility of the collected data. This is precisely the procedure we outlined in Chapter 5 . As we noted in our presentation, this framework yields theoretically optimal policies, but comes with an unwieldy computational burden.

By the early 196os, the stage was set for Bayesian optimization, which could now be realized by appropriately adapting the now mature field of Bayesian experimental design.

KUSHNER was the first to seize the opportunity with a pair of papers on optimizing a one-dimensional objective observed with noise. ${ }^{21,22}$ All of the major ideas in modern Bayesian optimization were already in place in this initial work, including a Gaussian process model of the objective function and appealing to Bayesian decision theory to derive optimization policies. After dismissing the optimal policy (with respect to the global reward utility (6.5)) as "notoriously difficult"21 and "virtually impossible to compute," ${ }^{22}$ KUSHNER suggests two alternatives "on the basis of heuristic or intuitive considerations": maximizing an upper confidence bound ( $\$ 7.8$ ) and maximizing probability of improvement (§ 7.5), which share credit as the first Bayesian optimization policies to appear in the literature. ${ }^{21}$ The probability of improvement approach seems to have won his favor, and KUSHNER later provided extensive practical advice for realizing this method, including careful discussion of how the improvement threshold could be managed interactively throughout optimization by a human expert in the loop. ${ }^{22}$

A significant body of literature on Bayesian optimization emerged in the Soviet Union following KUSHNER's seminal work. Many of these authors notably explored the promise of one-step lookahead for effective policy design, proposing and studying both expected improvement and the knowledge gradient for the first time. ŠALTENIS ${ }^{23}$ was the first to introduce expected improvement (§ 7.3) in 1971. ${ }^{24}$ This work contains an explicit formula for expected improvement for arbitrary Gaussian process models and the results of an impressive empirical investigation on a Soviet mainframe computer with objective functions in dimensions up to 32 . ŠALTENIS concludes with the following observation:

The relatively large amounts of machine time spent in planning search and the complexity of the algorithm give us grounds to assume that the most effective sphere of application would be mul-
19 H. RAIFFA and R. SCHLAIFER (1961). Applied Statistical Decision Theory. Division of Research, Graduate School of Business Administration, Harvard University.

20 D. V. LINDLEy (1972). Bayesian Statistics, A Review. Society for Industrial and Applied Mathematics.

21 H. J. KUSHNER (1962). A Versatile Stochastic Model of a Function of Unknown and Time Varying Form. Journal of Mathematical Analysis and Applications 5(1):150-167.

22 H. J. KUSHNER (1964). A New Method of Locating the Maximum Point of an Arbitrary Multipeak Curve in the Presence of Noise. Fournal of Basic Engineering 86(1):97-106.

23 Also transliterated SHALTYANIS.

24 V. R. ŠAltenis (1971). One Method of Multiextremum Optimization. Avtomatika i Vychislitel'naya Tekhnika (Automatic Control and Computer Sciences) 5(3):33-38. 25 Often transliterated MOčKus in early work.

26 J. MOckUs (1972). Bayesian Methods of Search for an Extremum. Avtomatika $i$ Vychislitel'naya Tekhnika (Automatic Control and Computer Sciences) 6(3):53-62.

27 J. Mockus (1974). On Bayesian Methods for Seeking the Extremum. Optimization Techniques: IFIP Technical Conference.

28 J. MOckus et al. (1978). The Application of Bayesian Methods for Seeking the Extrememum. In: Towards Global Optimization 2.

29 J. MоскUs (1989). Bayesian Approach to Global Optimization: Theory and Applications. Kluwer Academic Publishers.

30 J. Mockus et al. (2010). Bayesian Heuristic Approach to Discrete and Global Optimization: Algorithms, Visualization, Software, and Applications. Kluwer Academic Publishers.

31 See equation 8 in citation 26 above.
32 J. MоскUS (1972). Bayesian Methods of Search for an Extremum. Avtomatika $i$ Vychislitel'naya Tekhnika (Automatic Control and Computer Sciences) 6(3):53-62. [equations 36-37] tiextremum target functions whose determination involves major computational difficulties.

This remains the target domain of Bayesian optimization today.

Another prominent early contributor was MOCKUs, ${ }^{25}$ who wrote a series of papers on Bayesian optimization in the $1970 \mathrm{Os}^{26,27,28}$ and has written two books on the subject. ${ }^{29,30}$ Like KUSHNER, MOCKUs begins his presentation in these papers by outlining the optimal policy for maximizing the global reward utility (6.5), but rejects it as computationally infeasible. As a practical alternative, MOckus instead promotes what he calls the "one-stage approach," that is, one-step lookahead. ${ }^{26}$ As he had chosen the global reward utility, the resulting optimization policy is to maximize the knowledge gradient (§ 7.4).

This claim may give some readers pause, as MоскUs's work is frequently cited as the origin of expected improvement instead. However, this is inaccurate for multiple reasons. Expected improvement had been introduced by šALTENIS in 1971, one year before MOCKUS's first contribution on Bayesian optimization. Indeed, москUS was aware of and cited šAltenis's work. Further, the acquisition function MOcKus describes is defined with respect to the global reward utility underlying the knowledge gradient, ${ }^{31}$ not the simple reward utility underlying expected improvement.

That said, the situation is slightly more subtle. Mоcкus also discusses two convenient choices of models on the unit interval for which the knowledge gradient happens to equal expected improvement: the Wiener process and the Ornstein-Uhlenbeck (ou) process. Both are rare examples of Gauss-Markov processes, whose Markovian property renders sequential inference particularly convenient, and the early work on Bayesian optimization was dominated by these models due to the extreme computational limitations at the time. москUs also points out these models have a "special propert[y]";2 namely, their Markovian nature ensures that the posterior mean is always maximized at an observed location, and thus the simple and global reward utilities coincide!

\subsection*{LATER REDISCOVERY AND DEVELOPMENT}

The expected improvement and the knowledge gradient acquisition strategies were both reintroduced decades later when computational power had increased to the point that Bayesian optimization could be a practical approach for real problems.

SCHONLAU ${ }^{33}$ and JONEs et al., ${ }^{34}$ working together, proposed maximizing expected improvement for efficient global optimization in the context of the design and analysis of computer experiments (DACE) ${ }^{35}$ Here the objective function represents the output of a computational routine and is assumed to be observed without noise. In addition to promoting expected improvement as a policy, Jones et al. also provided extensive detail for practical implementation, including an insightful discussion on model validation and a branch-and-bound strategy for maximizing the expected improvement acquisition function for a certain class of Gaussian process models. The knowledge gradient was picked up again and further developed by FRAZIER and POWELL, ${ }^{36}$ who coined the name and studied the policy in the discrete and independent (bandit-like) setting, and scOTT et al., ${ }^{37}$ who adopted the policy for continuous optimization.

Three years after reintroducting expected improvement, JONEs wrote an extensive survey of then-current Bayesian optimization policies. ${ }^{38}$ Despite the now-pervasive nature of expected improvement, it is striking that JONES actually promotes maximizing the probability of improvement as the most promising policy for Bayesian optimization throughout this survey. His concern with expected improvement was a potential lack of robustness if the objective function model is misspecified. His proposed alternative was maximizing the probability of improvement over a wide range of improvement targets and evaluating the objective function in parallel ${ }^{39}$ which he regarded as more robust when practical.

The concept of mutual information (§ 7.6) first appeared with SHANNON's introduction of information theory, ${ }^{40}$ where it was called the channel capacity and served as a measure of the amount of information that could be transferred effectively over a noisy communication channel. LINDLEY later reinterpreted mutual information as the expected information gained by a proposed experiment and suggested maximizing this quantity as an effective means of general Bayesian experimental design. ${ }^{41}$

The application of this information-theoretic framework to Bayesian optimization was first proposed by VILLEMONTEIX et al. ${ }^{42}$ and later independently by HENNIG and SCHULER, ${ }^{43}$ the latter of whom coined the now-prominent term entropy search. Both of these initial investigations considered maximizing the mutual information between a measurement and the location of the global optimum $x^{*}(\S 7.6)$, using the formulation in (7.14). HERNÁNDEZ-LOBATO et al. later proposed a different set of approximations based on the equivalent formulation in (7.15) under the name predictive entropy search, as the key quantity is the expected reduction in entropy for the predictive distribution. ${ }^{60}$ Both HOFFMAN and GHAHRAMANI ${ }^{44}$ and WANG and JEGELKA ${ }^{45}$ later pursued maximizing the mutual information with the value of the global optimum $f^{*}(\S 7.6)$ These developments occurred contemporaneously and independently.

Prior to these algorithms designed to reduce the entropy of the value of the global maximum, there were occasional efforts to minimize some other measure of dispersion in this quantity. For example, cAlvin studied an algorithm for optimization on the unit interval $\mathcal{X}=[0,1]$ wherein the variance of $p\left(f^{*} \mid \mathcal{D}\right)$ was greedily minimized via one-step lookahead. ${ }^{46}$ Here the model was again the Wiener process, which has the remarkable property that the distribution of $p\left(f^{*} \mid \mathcal{D}\right)$ (and its variance) is analytically tractable. ${ }^{47}$ The Wiener and ou processes are among the only nontrivial Gaussian processes with this property. ${ }^{48}$

No history of Bayesian optimization would be complete without mentioning the role hyperparameter tuning has played in driving its recent development. With the advent of deep learning, the early 2010 s
36 P. FRAZIER and w. POWELL (2007). The Knowledge Gradient Policy for Offline Learning with Independent Normal Rewards. ADPRL 2007.

37 w. scoтt et al. (2011). The Correlated Knowledge Gradient for Simulation Optimization of Continuous Parameters Using Gaussian Process Regression. SIAM fournal on Optimization 21(3):996-1026.

38 D. R. JONEs (2001). A Taxonomy of Global Optimization Methods Based on Response Surfaces. Fournal of Global Optimization 21(4):345383 .

39 See $\S 7.5$, p. 134 for a discussion of this proposal.

40 C. E. SHANNON (1948). A Mathematical Theory of Communication. The Bell System Technical fournal 27(3):379-423.

This was a central concept in $\S 10.3$, p. 222 as well.

41 D. V. LINDLEy (1956). On a Measure of the Information Provided by an Experiment. The Annals of Mathematical Statistics 27(4):986-1005.

$42 \mathrm{~J}$. Villemonteix et al. (2009). An Informational Approach to the Global Optimization of Expensive-to-evaluate Functions. Fournal of Global Optimization 44(4):509-534.

43 P. HENNIG and C. J. SCHULER (2012). Entropy Search for Information-Efficient Global Optimization. Fournal of Machine Learning Research 13(Jun):1809-1837.

44 M. W. HOFFMAN and z. GHAHRAMANI (2015) Output-Space Predictive Entropy Search for Flexible Global Optimization. Bayesian Optimization Workshop, NeurIPS 2015.

45 z. WANG and s. JEGELKA (2017). Max-value Entropy Search for Efficient Bayesian Optimization. ICML 2017.

46 J. M. CALVIN (1993). Consistency of a Myopic Bayesian Algorithm for One-Dimensional Global Optimization. fournal of Global Optimization 3(2):223-232.

47 In fact, the joint density $p\left(x^{*}, f^{*} \mid \mathcal{D}\right)$ is analytically tractable for the Wiener process with drift, which forms the posterior on each subinterval subdivided by the data:

L. A. SHEPP (1979). The Joint Density of the Maximum and Its Location for a Wiener Process with Drift. Fournal of Applied Probability 16(2): 423-427.

48 R. J. ADLER and J. E. TAYLOR (2007). Random Fields and Geometry. Springer-Verlag. [chapter 4 , footnotes 1-2] 49 J. SNOEK et al. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. NeurIPS 2012. saw the rise of extraordinarily complex learning algorithms trained on extraordinarily large datasets. The great expense of training these models created unprecedented demand for efficient hyperparameter tuning to fuel the rapid development in the area. One could not ask for a moreperfect fit for Bayesian optimization!

Interest in Bayesian optimization for hyperparameter tuning was kicked off in earnest by SNOEK et al., who reported a dramatic improvement in performance when tuning a convolutional neural network via Bayesian optimization, even compared with carefully hand-tuned hyperparameters. ${ }^{49}$ This served as a watershed moment for Bayesian optimization, leading to an explosion of interest from the machine learning community in the following years - although this history began in 1815 , over half of the works cited in this book are from after 2012!

We have now covered the evolution of decision-theoretic Bayesian optimization policies from WALD's introduction of sequential analysis in 1945 to the state-of-the-art. Alongside these developments, a rich and expansive body of literature was forming on the multi-armed problem. A complete survey of this work would be out of this book's scope; however, we can point the interested reader to comprehensive surveys. ${ }^{50,51}$ Both contain excellent bibliographic notes and combined serve as an indispensable guide to the literature on multi-armed bandits ( $\$ 7 \cdot 7)$. Our goal in the following will be to cover developments that directly influenced the evolution of Bayesian optimization.

THOMPSON was the first to seriously study the possibility of sequential experiments in the context of medical treatment; ${ }^{52,53}$ this work predated WALD's work by a decade. THOMPSON considered a model scenario where there are two possible treatments a doctor may prescribe for a disease, but it is not clear which should be preferred. To determine the better treatment, we must undertake a clinical trial, assigning each treatment to several patients and assessing the outcome. Traditionally, a single preliminary experiment would be conducted, after which the apparently better-performing treatment would be adopted and the worseperforming treatment discarded. However, THOMPSON argued that one should never eliminate either treatment at any stage of investigation, even in the face of overwhelming evidence. Rather, he proposed a perpetual clinical trial modeled as what we now call a two-armed bandit: the possible treatments represent alternatives available to the clinician, and patient outcomes determine the rewards. We can now consider assigning treatments for a sequence of patients guided by our evolving knowledge, hoping to efficiently and confidently determine the optimal treatment.

To conduct such a sequential clinical trial effectively, THOMPSON proposed maintaining a belief over which treatment is better in light of available evidence and always selecting the nominally better treatment according to its posterior probability of superiority. This is the eponymous Thompson sampling policy (§ 7.9), which elegantly addresses the exploration-exploitation dilemma. The more evidence we have in favor of one treatment, the more often we prescribe it, exploiting the apparently better choice. Meanwhile, we maintain a diminishing but nonzero probability of selecting the other treatment, forcing continual exploration until the better treatment becomes obvious. In the long term, we will eventually assign the correct treatment to new patients with probability approaching certainty.

THOMPSON's work was in retrospect groundbreaking, but its significance was perhaps not fully realized at the time. However, Thompson sampling for multi-armed bandits has recently enjoyed an explosion of attention due to impressive empirical performance ${ }^{54,55}$ and strong theoretical guarantees (Chapter 10). ${ }^{56,57,58}$ The first direct application of Thompson sampling to Bayesian optimization is due to SHAHRIARI et al., ${ }^{59}$ who adopted an efficient approximation first proposed by HERNÁNDEZLOBATO et al. in the context of entropy search. ${ }^{60}$

Although THOMPSON had introduced bandits in the early 1930s, concerted effort began with ROBBINs's landmark 1952 reintroduction and analysis of the problem. ${ }^{61}$ This work introduced the modern formulation of multi-armed bandits presented in $\S 7.7$, where an arbitrary, unknown reward distribution is associated with each arm and the agent seeks a policy to maximize the expected cumulative reward. ROBBINs explores this problem in the special case of a two-armed bandit with Bernoulli rewards, demonstrating that a simple adaptive policy ("switch on lose, stay on win") can achieve better performance than nonadaptive or random policies. He also presents a family of policies that can achieve asymptotically optimal behavior for any reward distributions. These policies are defined by a simple mechanism that explicitly forces continual but decreasingly frequent exploration of both arms according to a pregenerated schedule, effectively - if crudely - balancing exploration and exploitation.

The simple policies proposed by RoBBINs eventually achieve nearoptimal cumulative reward, but they are not very efficient in the sense of achieving that behavior quickly. RoBBINs returned to this problem three decades later to further address the issue of efficiency. ${ }^{62}$ LAI and ROBBINs introduced policies that dynamically trade off exploration and exploitation by maximizing an upper confidence bound on the reward distributions of the arms ( $\$ 7.8$ ), and demonstrated that these policies are both asymptotically optimal and efficient. AUER et al. later proved that bandit policies based on maximizing upper confidence bounds can provide strong guarantees not only asymptotically but also in the finitebudget case.$^{63}$ Interestingly, KUSHNER had proposed optimization policies based on maximizing an upper confidence bound of the objective function in the continuous case two decades earlier than LAI and ROBBINS's work, ${ }^{21}$ although he did not prove any performance guarantees for this procedure and abandoned the idea in favor of maximizing probability of improvement. This optimization policy would be rediscovered several times, including by cox and JOHN ${ }^{64}$ and by SRINIVAs et al., ${ }^{65}$ who established theoretical guarantees on the rate of convergence of this policy in the Gaussian process case, as discussed in $\S 10.4$.
54 O. CHAPELlE and L. LI (2011). An Empirical Evaluation of Thompson Sampling. NeurIPS 2011.

55 O.-C. GRANmo (2010). Solving Two-Armed Bernoulli Bandit Problems Using a Bayesian Learning Automaton. International fournal of Intelligent Computing and Cybernetics 3(2): 207-234.

56 S. AGRAWAL and N. GOYAL (2012). Analysis of Thompson Sampling for the Multi-Armed Bandit Problem. COLT 2012.

57 D. RUSSO and B. VAN ROY (2014). Learning to Optimize via Posterior Sampling. Mathematics of Operations Research 39(4):1221-1243.

58 D. RUSSO and B. VAN ROY (2016). An Information-Theoretic Analysis of Thompson Sampling. Journal of Machine Learning Research 17(68):1-30.

59 B. SHAHRIARI et al. (2014). An Entropy Search Portfolio for Bayesian Optimization. arXiv: 1406. 4625 [stat.ML].

6o J. M. HERNÁNDEZ-LOBATO et al. (2014). Predictive Entropy Search for Efficient Global Optimization of Black-Box Functions. NeurIPS 2014.

61 H. RobBins (1952). Some Aspects of the Sequential Design of Experiments. Bulletin of the American Mathematical Society 58(5):527535 .

62 T. L. LAI and H. ROBBINS (1985). Asymptotically Efficient Adaptive Allocation Rules. Advances in Applied Mathematics 6(1):4-22.

63 P. AUER et al. (2002). Finite-Time Analysis of the Multiarmed Bandit Problem. Machine Learning 47(2-3):235-256.

64 D. D. Cox and S. JOHN (1992). A Statistical Method for Global Optimization. SMC 1992.

65 N. SRINIVAs et al. (2010). Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design. ICML 2010. Gaussian process modeling in high dimension: $§ 3.5$, p. 61

moving beyond one-step lookahead: § 7.10, p. 150

66 B. SHAHrIARI et al. (2016). Taking the Human out of the Loop: A Review of Bayesian Optimization. Proceedings of the IEEE 104(1):148175

This is an excellent companion survey on Bayesian optimization for readers who have reached this point but still want more.

utility functions: $§ 6$, p. 109

\section*{WHAT'S NEXT?}

The mathematical foundations of Bayesian optimization, as presented in this book and chronicled in this chapter, are by now well established. However, Bayesian optimization continues to deliver impressive performance on a wide variety of problems, and the field continues to develop at a rapid pace in light of this success. What big challenges remain on the horizon? We speculate on some potential opportunities below.

- Effective modeling of objective functions remains a challenge, especially in high dimension. Meanwhile, methods such as stochastic gradient descent routinely (locally) optimize objectives in millions of dimensions, and are "unreasonably effective" at doing so - all with very weak guidance. Can we bridge this gap, either by extending or refining approaches for Gaussian process modeling, or by exploring another model class?

- Nonmyopic policies that reach beyond one-step lookahead have shown impressive empirical performance in initial investigations. However, these policies have not yet been widely adopted, presumably due to their (sometimes significantly) greater computational cost compared to myopic alternatives. The continued development of efficient, yet nonmyopic policies remains a promising avenue of research.

- A guiding philosophy in Bayesian optimization has been to "take the human out of the loop" and hand over complete control of experimental design to an algorithm. ${ }^{66}$ This paradigm has demonstrated remarkable success on "black-box" problems, where the user has little understanding of the system being optimized. However, in settings such as scientific discovery, the user has a deep understanding of and intuition for the mechanisms driving the system of interest, and we should perhaps consider how to "bring them back into the loop." One could imagine an ecosystem of cooperative tools that enable Bayesian optimization algorithms to benefit from user knowledge while facilitating the presentation of experimental progress and evolving model beliefs back to the users.

- In the author's experience, many consumers of Bayesian optimization have experimental goals that are not perfectly captured by any of the common utility functions used in Bayesian optimization - for example, users may want to ensure adequate coverage of the domain or to find a diverse set of locations with high objective values. Although the space of Bayesian optimization policies is fairly crowded, there is still room for innovation. If the ecosytem of cooperative tools envisioned above were realized, we might consider tuning the utility function "on the fly" via interactive preference elicitation.

- Bayesian optimization is not a particularly user-friendly approach, as effective optimization requires careful modeling of the system of interest. However, model construction remains something of a "black art," even in the machine learning community, and thus considerable machine learning expertise can be required to get the most out of this approach. Can we build turnkey Bayesian optimization systems that achieve acceptable performance even in the absence of clear prior beliefs?