\section*{MODEL ASSESSMENT, SELECTION, AND AVERAGING}

The previous chapter offered a glimpse into the flexibility of Gaussian processes, which can evidently model functions with a wide range of behavior. However, a critical question remains: how can we identify which models are most appropriate in a given situation?

The difficulty of this question is compounded by several factors. To begin, the number of possible choices is staggering. Any function can serve as a mean function for a Gaussian process, and we may construct arbitrary complex covariance functions through a variety of mechanisms. Even if we fix the general form of the moment functions, introducing natural parameters such as output and length scales yields an infinite spectrum of possible models.

Further, many systems of interest act as "black boxes," about which we may have little prior knowledge. Before optimization, we may have only a vague notion of which models might be reasonable for a given objective function or how any parameters of these models should be set. We might even be uncertain about aspects of the observation process, such as the nature or precise scale of observation noise. Therefore, we may find ourselves in the unfavorable position of having infinitely many possible models to choose from and no idea how to choose!

Acquiring data, however, provides a way out of this conundrum. After obtaining some observations of the system, we may determine which models are the most compatible with the data and thereby establish preferences over possible choices, a process known as model assessment. Model assessment is a surprisingly complex and nuanced subject - even if we limit the scope to Bayesian methods - and no method can rightfully be called "the" Bayesian approach. ${ }^{1}$ In this chapter we will present one convenient framework for model assessment via Bayesian inference over models, which are evaluated based on their ability to explain observed data and our prior beliefs.

We will begin our presentation by carefully defining the models we will be assessing and discussing how we may build useful spaces of models for consideration. With Gaussian processes, these spaces will most often be built from what we will call model structures, comprising a parametric mean function, covariance function, and observation model; in the context of model assessment, the parameters of these model components are known as hyperparameters. We will then show how to perform Bayesian inference over the hyperparameters of a model structure from observations, resulting in a model posterior enabling model assessment and other tasks. We will later extend this process to multiple model structures and show how we can even automatically search for better model structures.

Central to this approach is a fundamental measure of model fit known as the marginal likelihood of the data or model evidence. Gaussian process models are routinely selected by maximizing this score, which can produce excellent results when sufficient data are available to unambiguously determine the best-fitting model. However, model construction

\section*{4}

prior mean function: $\S 3.1$, p. 46

prior covariance function: $\S 3.2$, p. 49

output and length scales: § 3.4, p. 54

1 The interested reader can find an overview of this rich subject in:

A. VEHTARI and J. OJANEN (2012). A Survey of Bayesian Predictive Methods for Model Assessment, Selection and Comparison. Statistics Surveys 6:142-228.

models and model structures: $§ 4.1$, p. 68

Bayesian inference over parametric model spaces: $§ 4.2$, p. 70

multiple model structures: $§ 4.5$, p. 78

automating model structure search: $§ 4.6$, p. 81

marginal likelihood, model evidence: $§ 4.2$, p. 71

model selection via MAP inference: $\S 4.3$, p. 73 model averaging: § 4.4, p. 74

model, $p(\mathbf{y} \mid \mathbf{x})$

model induced by prior process and observation model

2 Although defining a space of candidate models may seem natural and innocuous, this is actually a major point of contention between different approaches to Bayesian model assessment. If we subscribe to the maxim "all models are wrong," we might conclude that the true model will never be contained in any space we define, no matter how expansive. However, some are likely "more wrong" than others, and we can still reasonably establish preferences over the given space. in the context of Bayesian optimization is unusual as the expense of gathering observations relegates us to the realm of small data. Effective modeling with small datasets requires careful consideration of model uncertainty: models explaining the data equally well may disagree drastically in their predictions, and committing to a single model may yield biased predictions with poorly calibrated uncertainty - and disappointing optimization performance as a result. Model averaging is one solution that has proven effective in Bayesian optimization, where the predictions of multiple models are combined in the interest of robustness.

\subsection*{MODELS AND MODEL STRUCTURES}

In model assessment, we seek to evaluate a space of models according to their ability to explain a set of observations $\mathcal{D}=(\mathbf{x}, \mathbf{y})$. Before taking up this problem in earnest, let us establish exactly what we mean by "model" in this context, which is a model for the given observations, rather than of a latent function alone as was our focus in the previous chapter.

For this discussion we will define a model to be a prior probability distribution over the measured values $y$ that would result from observing at a set of locations $\mathbf{x}: p(\mathbf{y} \mid \mathbf{x})$. In the overarching approach we have adopted for this book, a model is specified indirectly via a prior process on a latent function $f$ and an observation model linking this function to the observed values:

$$
[p(f), p(y \mid x, \phi)] .
$$

Given explicit choices for these components, we may form the desired distribution by marginalizing the latent function values $\boldsymbol{\phi}=f(\mathbf{x})$ through the observation model:

$$
p(\mathbf{y} \mid \mathbf{x})=\int p(\mathbf{y} \mid \mathbf{x}, \boldsymbol{\phi}) p(\boldsymbol{\phi} \mid \mathbf{x}) \mathrm{d} \boldsymbol{\phi} .
$$

All models we consider below will be of this composite form (4.1), but the assessment framework we describe will accommodate arbitrary models.

\section*{Spaces of candidate models}

To proceed, we must establish some space of candidate models we wish to consider as possible explanations of the observed data. ${ }^{2}$ Although this space can in principle be arbitrary, with Gaussian process models it is convenient to consider parametric collections of models defined by parametric forms for the observation model and the prior mean and covariance functions of the latent function. We invested significant effort in the last chapter laying the groundwork to enable this approach: a running theme was the introduction of flexible parametric mean and covariance functions that can assume a wide range of different shapes perfect building blocks for expressive model spaces.

We will call a particular combination of observation model, prior mean function $\mu$, and prior covariance function $K$ a model structure. Corresponding to each model structure is a natural model space formed by exhaustively traversing the joint parameter space:

$$
\mathcal{M}=\{[p(f \mid \boldsymbol{\theta}), p(y \mid x, \phi, \theta)] \mid \boldsymbol{\theta} \in \Theta\},
$$

where

$$
p(f \mid \boldsymbol{\theta})=\mathcal{G} \mathcal{P}\left(f ; \mu(x ; \boldsymbol{\theta}), K\left(x, x^{\prime} ; \boldsymbol{\theta}\right)\right) .
$$

We have indexed the space by a vector $\theta$, the entries of which jointly specify any necessary parameters from their joint range $\Theta$. The entries of $\boldsymbol{\theta}$ are known as hyperparameters of the model structure, as they parameterize the prior distribution for the observations, $p(\mathbf{y} \mid \mathbf{x}, \boldsymbol{\theta})(4.2)$.

In many cases we may be happy with a single suitably flexible model structure for the data, in which case we can proceed with the corresponding space (4.3) as the set of candidate models. We may also consider multiple model structures for the data by taking a discrete union of such spaces, an idea we will return to later in this chapter.

\section*{Example}

Let us momentarily take a step back from abstraction and create an explicit model space for optimization on the interval $\mathcal{X}=[a, b] .{ }^{3}$ Suppose our initial beliefs are that the objective will exhibit stationary behavior with a constant trend near zero, and that our observations will be corrupted by additive noise with unknown signal-to-noise ratio.

For the observation model, we take homoskedastic additive Gaussian noise, a reasonable choice when there is no obvious alternative:

$$
p\left(y \mid \phi, \sigma_{n}\right)=\mathcal{N}\left(y ; \phi, \sigma_{n}^{2}\right),
$$

and leave the scale of the observation noise $\sigma_{n}$ as a hyperparameter. Turning to the prior process, we assume a constant mean function (3.1) with a zero-mean normal prior on the unknown constant:

$$
\mu(x ; c) \equiv c ; \quad p(c)=\mathcal{N}\left(c ; 0, b^{2}\right),
$$

and select the Matérn covariance function with $v=5 / 2$ (3.14) with unknown output scale $\lambda$ (3.20) and unknown length scale $\ell$ (3.22):

$$
K\left(x, x^{\prime} ; \lambda, \ell\right)=\lambda^{2} K_{\mathrm{M}^{5} / 2}(d / \ell) .
$$

Following our discussion in the last chapter, we may eliminate one of the parameters above by marginalizing the unknown constant mean under its assumed prior, ${ }^{4}$ leaving us with the identically zero mean function and an additive contribution to the covariance function (3.3):

$$
\mu(x) \equiv 0 ; \quad K\left(x, x^{\prime} ; \lambda, \ell\right)=b^{2}+\lambda^{2} K_{\mathrm{M} 5 / 2}(d / \ell) .
$$

This, combined with (4.4), completes the specification of a model structure with three hyperparameters: $\theta=\left[\sigma_{n}, \lambda, \ell\right]^{\top}$. Figure 4.1 illustrates model space, $\mathcal{M}$

vector of hyperparameters, $\boldsymbol{\theta}$

range of hyperparameter values, $\Theta$

multiple model structures: $§ 4.5$, p. 78

3 The interval can be arbitrary; our discussion will be purely qualitative.

observation model: additive Gaussian noise with unknown scale

prior mean function: constant mean with unknown value

prior covariance function: Matérn $v=5 / 2$ with unknown output and length scales

eliminating mean parameter via marginalization: §3.1, p. 47

4 We would ideally marginalize the other parameters as well, but it would not result in a Gaussian process, as we will discuss shortly. Figure 4.1: Samples from our example model space for a range of the hyperparameters: $\sigma_{n}$, the observation noise scale, and $\ell$, the characteristic length scale. The output scale $\lambda$ is fixed for each example. Each example demonstrates a sample of the latent function and observations resulting from measurements at a fixed set of 15 locations $\mathrm{x}$. Elements of the model space can model functions with short- or long-scale correlations that are observed with a range of fidelity from virtually exact observation to extreme noise.

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-04.jpg?height=711&width=1028&top_left_y=475&top_left_x=774)

samples from the joint prior over the objective function and the observed values $\mathrm{y}$ that would result from measurements at 15 locations $\mathrm{x}(4.2)$ for a range of these hyperparameters. Even this simple model space is quite flexible, offering degrees of freedom for the variation in the objective function and the precision of our measurements.

\subsection*{BAYESIAN INFERENCE OVER PARAMETRIC MODEL SPACES}

Given a space of candidate models, we now turn to the question of assessing the quality of these models in light of data. There are multiple paths forward, ${ }^{5}$ but Bayesian inference offers one effective solution. By accepting that we can never be absolutely certain regarding which model is the most faithful representation of a given system, we can - as with anything unknown in the Bayesian approach - treat that "best model" as a random variable to be inferred from data and prior beliefs.

We will limit this initial discussion to parametric model spaces built from a single model structure (4.1), which will simplify notation and allow us to conflate models and their corresponding hyperparameters $\boldsymbol{\theta}$ as convenient. We will consider more complex spaces comprising multiple alternative model structures presently.

\section*{Model prior}

We first endow the model space with a prior encoding which models are more plausible a priori, $p(\boldsymbol{\theta}) .{ }^{6}$ For convenience, it is common to design the model hyperparameters such that the uninformative (and possibly improper) "uniform prior"

$$
p(\theta) \propto 1
$$



![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-05.jpg?height=220&width=685&top_left_y=455&top_left_x=274)

may be used, in which case the model prior may not be explicitly acknowledged at all. However, it can be helpful to express at least weakly informative prior beliefs - especially when working with small datasets - as it can offer gentle regularization away from patently absurd choices. This should be possible for most hyperparameters in practice. For example, when modeling a physical system, it would be unlikely that interaction length scales of say one nanometer and one kilometer would be equally plausible a priori; we might capture this intuition with a wide prior on the logarithm of the length scale.

\section*{Model posterior}

Given a set of observations $\mathcal{D}=(\mathbf{x}, \mathbf{y})$, we may appeal to Bayes' theorem to derive the posterior distribution over the candidate models:

$$
p(\boldsymbol{\theta} \mid \mathcal{D}) \propto p(\boldsymbol{\theta}) p(\mathbf{y} \mid \mathbf{x}, \boldsymbol{\theta})
$$

The model posterior provides support to the models most consistent with our prior beliefs and the observed data. Consistency with the data is encapsulated by the $p(\mathbf{y} \mid \mathbf{x}, \boldsymbol{\theta})$ term, the prior PDF over observations evaluated on the actual data. ${ }^{7}$ This value is known as the model evidence or the marginal likelihood of the data, as it serves as a likelihood in Bayes' theorem (4.7) and, in our class of latent function models, is computed by marginalizing the latent function values at the observed locations (4.2).

\section*{Marginal likelihood and Bayesian Occam's razor}

Model assessment becomes trivial in light of the model posterior if we simply establish preferences over models according to their posterior probability. When using the uniform model prior (4.6) (perhaps implicitly), the model posterior is proportional to the marginal likelihood alone, which can be then used directly for model assessment.

It is commonly argued that the model evidence encodes automatic penalization for model complexity, a phenomenon known as Bayesian Occam's razor. ${ }^{8}$ MACKAY outlines a simple argument for this effect by noting that a model $p(\mathbf{y} \mid \mathbf{x})$ must integrate to unity over all possible measurements $\mathbf{y}$. Thus if a "simpler" model wishes to become more "complex" by putting support over a wider range of possible observations, it can only do so by reducing the support for the datasets that are already well explained; see the illustration in the margin.

The marginal likelihood of a given dataset can be conveniently computed in closed form for Gaussian process models with additive Gaussian
Figure 4.2: The dataset for our model assessment example, generated using a hidden model from the space on the facing page.

model posterior, $p(\theta \mid \mathcal{D})$

7 Recall that this distribution is precisely what a model defines: $§ 4.1$, p. 68 .

model evidence, marginal likelihood, $p(\mathbf{y} \mid \mathbf{x}, \theta)$

8 D. J. C. MACKAY (2003). Information Theory, Inference, and Learning Algorithms. Cambridge University Press. [chapter 28]

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-05.jpg?height=262&width=531&top_left_y=1945&top_left_x=1368)

A cartoon of the Bayesian Occam's razor effect due to MACKAY. Interpreting models as PDFs over measurements y, a "simple" model that explains datasets in $\mathcal{S}$ well, but not elsewhere. The "complex" alternative model explains datasets outside $\mathcal{S}$ better, but in $\mathcal{S}$ worse; the probability density must be lower there to explain a broader range of data. Figure 4.3: The posterior distribution over the model space from Figure 4.1 (the range of the axes are compatible with that figure) conditioned on the dataset in Figure 4.2. The output scale is fixed (to its true value) for the purposes of illustration. Significant uncertainty remains in the exact values of the hyperparameters, but the model posterior favors models featuring either short length scales with low noise or long length scales with high noise. The points marked $1-3$ are referenced in Figure 4.4; the point marked $*$ is the MAP (Figure 4.5).

marginal likelihood for Gaussian process models with additive Gaussian noise

interpretation of terms

9 The dataset was realized using a moderate length scale (3o length scales spanning the domain) and a small amount of additive noise, shown below. But this is impossible to know from inspection of the data alone, and many alternative explanations are just as plausible according to the model posterior!

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-06.jpg?height=205&width=529&top_left_y=2282&top_left_x=158)

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-06.jpg?height=711&width=1014&top_left_y=478&top_left_x=772)

noise or exact observation. In this case, we have (2.18):

$$
p(\mathbf{y} \mid \mathbf{x}, \boldsymbol{\theta})=\mathcal{N}(\mathbf{y} ; \boldsymbol{\mu}, \Sigma+\mathrm{N}),
$$

where $\boldsymbol{\mu}$ and $\Sigma$ are the prior mean and covariance of the latent objective function values $\phi(2.3)$, and $\mathrm{N}$ is the observation noise covariance matrix (the zero matrix for exact observation) - all of which may depend on $\boldsymbol{\theta}$. As this value can be exceptionally small and have high dynamic range, the logarithm of the marginal likelihood is usually preferred for computational purposes (A.6-A.7):

$$
\begin{aligned}
\log p(\mathbf{y} \mid \mathbf{x}, \boldsymbol{\theta})= & \\
& -\frac{1}{2}\left[(\mathbf{y}-\boldsymbol{\mu})^{\top}(\Sigma+\mathbf{N})^{-1}(\mathbf{y}-\boldsymbol{\mu})+\log |\Sigma+\mathrm{N}|+n \log 2 \pi\right] .
\end{aligned}
$$

The first term of this expression is the sum of the squared Mahalanobis norms (A.8) of the observations under the prior and represents a measure of data fit. The second term serves as a complexity penalty: the volume of any confidence ellipsoid under the prior is proportional to $|\Sigma+\mathrm{N}|$, and thus this term scales according to the volume of the model's support in observation space. The third term simply ensures normalization.

\section*{Return to example}

Let us return to our example scenario and model space. We invite the reader to consider the hypothetical set of 15 observations in Figure 4.2 from our example system of interest and contemplate which models from our space of candidates in Figure 4.1 might be the most compatible with these observations. ${ }^{9}$

We illustrate the model posterior given this data in Figure 4.3, where, in the interest of visualization, we have fixed the covariance output posterior mean

posterior $95 \%$ credible interval, $y$ posterior $95 \%$ credible interval, $\phi$
1

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-07.jpg?height=214&width=1031&top_left_y=578&top_left_x=273)

2

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-07.jpg?height=260&width=1031&top_left_y=801&top_left_x=273)

3

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-07.jpg?height=234&width=1028&top_left_y=1082&top_left_x=271)

Figure 4.4: Posterior distributions given the observed data corresponding to the three settings of the model hyperparameters marked in Figure 4.3. Although remarkably different in their interpretations, each model represents an equally plausible explanation in the model posterior. Model $1 \mathrm{fa}-$ vors near-exact observations with a short length scale, and models 2-3 favor large observation noise with a range of length scales. scale to its true value and set the range of the axes to be compatible with the samples from Figure 4.1. The model prior was designed to be weakly informative regarding the expected order of magnitude of the hyperparameters by taking independent, wide Gaussian priors on the logarithm of the observation noise and covariance length scale. ${ }^{10}$

The first observation we can make regarding the model posterior is that it is remarkably broad, with many settings of the model hyperparameters remaining plausible after observing the data. However, the model posterior does express a preference for models with either low noise and short length scale or high noise combined with a range of compatible length scales. Figure 4.4 provides examples of objective function and observation posteriors corresponding to the hyperparameters indicated in Figure 4.3. Although each is equally plausible in the posterior, ${ }^{11}$ their explanations of the data are diverse.

\section*{MODEL SELECTION VIA POSTERIOR MAXIMIZATION}

Winnowing down a space of candidate models to a single model for use in inference and prediction is known as model selection. Model selection becomes straightforward if we agree to rank candidates according to the model posterior, as we may then select the maximum a posteriori (MAP) $(4 \cdot 7)$ model: ${ }^{12}$

$$
\hat{\boldsymbol{\theta}}=\underset{\boldsymbol{\theta}}{\arg \max } p(\boldsymbol{\theta}) p(\mathbf{y} \mid \mathbf{x}, \boldsymbol{\theta}) .
$$

When the model prior is flat (4.6), the MAP model corresponds to the maximum likelihood estimate (MLE) of the model hyperparameters. Fig-
12 If we only wish to find the maximum, there is no benefit to normalizing the posterior. Figure 4.5: The predictions of the maximum a posteriori (MAP) model from the example data in Figure 4.2 .

acceleration via gradient-based optimization

gradient of log marginal likelihood with respect to $\theta$ : § C.1, p. 307

model-marginal objective posterior, $p(f \mid \mathcal{D})$ model-marginal predictive distribution, $p(y \mid x, \mathcal{D})$

13 Although it may be unusual to consider the choice of model a "nuisance!"

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-08.jpg?height=234&width=1014&top_left_y=477&top_left_x=772)

ure 4.5 shows the predictions made by the MAP model for our running example; in this case, the MAP hyperparameters are in fact a reasonable match to the parameters used to generate the example dataset.

When the model space is defined over a continuous space of hyperparameters, computation of the MAP model can be significantly accelerated via gradient-based optimization. Here it is advisable to work in the log domain, where the objective becomes the unnormalized log posterior:

$$
\log p(\boldsymbol{\theta})+\log p(\mathbf{y} \mid \mathbf{x}, \boldsymbol{\theta}) .
$$

The log marginal likelihood is given in (4.8), noting that $\boldsymbol{\mu}, \Sigma$, and $\mathrm{N}$ are all implicitly functions of the hyperparameters $\theta$. This objective (4.10) is differentiable with respect to $\theta$ assuming the Gaussian process prior moments, the noise covariance, and the model prior are as well, in which case we may appeal to off-the-shelf gradient methods for solving (4.9). However, a word of warning is in order: the model posterior is not guaranteed to be concave and may have multiple local maxima, so multistart optimization is prudent.

\subsection*{MODEL AVERAGING}

Reliance on a single model is questionable when the model posterior is not well determined by the data. For example, in our running example, a diverse range of models is consistent with the data (Figures $4 \cdot 3-4 \cdot 4$ ). Committing to a single model in this case may systematically bias our predictions and underestimate predictive uncertainty - note how the diversity in predictions from Figure 4.4 is lost in the MAP model (4.5).

An alternative is to marginalize the model with respect to the model posterior, a process known as model averaging:

$$
\begin{aligned}
p(f \mid \mathcal{D}) & =\int p(f \mid \mathcal{D}, \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathcal{D}) \mathrm{d} \boldsymbol{\theta} \\
p(y \mid x, \mathcal{D}) & =\iint p(y \mid x, \phi, \boldsymbol{\theta}) p(\phi \mid x, \mathcal{D}, \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathcal{D}) \mathrm{d} \phi \mathrm{d} \boldsymbol{\theta},
\end{aligned}
$$

where we have marginalized the hyperparameters of both the objective and observation models. Model averaging is more consistent with the ideal Bayesian convention of marginalizing nuisance parameters when possible $^{13}$ and promises robustness to model misspecification, at least over the chosen model space.

Unfortunately, neither of these model-marginal distributions (4.114.12) can be computed exactly for Gaussian process models except in 

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-09.jpg?height=351&width=1630&top_left_y=464&top_left_x=270)

Figure 4.6: A Monte Carlo estimate to the model-marginal predictive distribution (4.11) for our example sceneario using 100 samples drawn from the model posterior in Figure 4.3 (4.14-4.15); see illustration in margin. Samples from the objective function posterior display a variety of behavior due to being associated with different hyperparameters.

some special cases ${ }^{14}$ so we must resort to approximation if we wish to pursue this approach. In fact, maximum a posteriori estimation can be interpreted as one rather crude approximation scheme where the model posterior is replaced by a Dirac delta distribution at the MAP hyperparameters:

$$
p(\boldsymbol{\theta} \mid \mathcal{D}) \approx \delta(\boldsymbol{\theta}-\hat{\boldsymbol{\theta}}) .
$$

This can be defensible when the dataset is large compared to the number of hyperparameters, in which case the model posterior is often unimodal with little residual uncertainty. However, large datasets are the exception rather than the rule in Bayesian optimization, and more sophisticated approximations can pay off when model uncertainty is significant.

\section*{Monte Carlo approximation}

Monte Carlo approximation is one straightforward path forward. Drawing a set of hyperparameter samples from the model posterior,

$$
\left\{\boldsymbol{\theta}_{i}\right\}_{i=1}^{s} \sim p(\boldsymbol{\theta} \mid \mathcal{D}),
$$

yields the following simple Monte Carlo estimates:

$$
\begin{aligned}
p(f \mid \mathcal{D}) & \approx \frac{1}{s} \sum_{i=1}^{s} \mathcal{G} \mathcal{P}\left(f ; \mu_{\mathcal{D}}\left(\boldsymbol{\theta}_{i}\right), K_{\mathcal{D}}\left(\boldsymbol{\theta}_{i}\right)\right) ; \\
p(y \mid x, \mathcal{D}) & \approx \frac{1}{s} \sum_{i=1}^{s} \int p\left(y \mid x, \phi, \boldsymbol{\theta}_{i}\right) p\left(\phi \mid x, \mathcal{D}, \boldsymbol{\theta}_{i}\right) \mathrm{d} \phi .
\end{aligned}
$$

The objective function posterior is approximated by a mixture of Gaussian processes corresponding to the sampled hyperparameters, and the posterior predictive distribution for observations is then derived by integrating a Gaussian mixture (2.36) against the observation model.

Any Markov chain Monte Carlo procedure could be used to generate the hyperparameter samples (4.13); a variation on Hamiltonian Monte
14 A notable example is marginalizing the coefficients of a linear prior mean against a Gaussian prior: $\S 3.1$, p. 47 .

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-09.jpg?height=379&width=531&top_left_y=2015&top_left_x=1368)

The 100 hyperparameter samples used to produce Figure 4.6 . 

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-10.jpg?height=365&width=1631&top_left_y=457&top_left_x=158)

Figure 4.7: An approximation to the model-marginal posterior (4.11) using the central composite design approach proposed by RUE et al. A total of nine hyperparameter samples are used for the approximation, illustrated in the margin below.

15 M. D. HOFFMAN and A. GELMAN (2014). The No-U-turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research 15(4):1593-1623.

Laplace approximation: § B.1, p. 301

16 H. RUE et al. (2009). Approximate Bayesian Inference for Latent Gaussian Models by Using Integrated Nested Laplace Approximations. Journal of the Royal Statistical Society Series B (Methodological) 71(2):319-392.

17 G. E. P. BOX and K. B. WILSON (1951). On the Experimental Attainment of Optimum Conditions. Fournal of the Royal Statistical Society Series B (Methodological) 13(1):1-45.
Carlo (нмC) such as the no U-turn sampler (NUTs) would be a reasonable choice when the gradient of the log posterior (4.10) is available, as it can exploit this information to accelerate mixing. ${ }^{15}$

Figure 4.6 demonstrates a Monte Carlo approximation to the modelmarginal posterior (4.11-4.12) for our running example. Comparing with the MAP approximation in Figure 4.5, the predictive uncertainty of both objective function values and observations has increased considerably due to accounting for model uncertainty in the predictive distributions.

\section*{Deterministic approximation schemes}

The downside of Monte Carlo approximation is relatively inefficient use of the hyperparameter samples - the price of random sampling rather than careful design. This inefficiency in turn leads to an increased computational burden for inference and prediction from having to derive a GP posterior for each sample. Several more efficient (but less accurate) alternative approximations for hyperparameter marginalization have also been proposed. A common simplifying tactic taken by these cheaper procedures is to approximate the hyperparameter posterior with a multivariate normal via a Laplace approximation:

$$
p(\boldsymbol{\theta} \mid \mathcal{D}) \approx \mathcal{N}(\boldsymbol{\theta} ; \hat{\boldsymbol{\theta}}, \mathbf{C}),
$$

where $\hat{\theta}$ is the MAP (4.9). Integrating this approximation into (4.11) gives

$$
p(f \mid \mathcal{D}) \approx \int \mathcal{G P}\left(f ; \mu_{\mathcal{D}}(\boldsymbol{\theta}), K_{\mathcal{D}}(\boldsymbol{\theta})\right) \mathcal{N}(\boldsymbol{\theta} ; \hat{\boldsymbol{\theta}}, \mathrm{C}) \mathrm{d} \boldsymbol{\theta} .
$$

Unfortunately this integral remains intractable due to the nonlinear dependence of the posterior moments on the hyperparameters, but reducing to this common form allows us to derive deterministic approximations against a single assumed posterior.

RUE et al. introduced several approximation schemes representing different tradeoffs between efficiency and fidelity. ${ }^{16}$ Notable among these is a simple, sample-efficient procedure grounded in classical experimental design. Here a central composite design ${ }^{17}$ in hyperparameter 

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-11.jpg?height=448&width=1630&top_left_y=461&top_left_x=270)

Figure 4.8: The approximation to the model-marginal posterior (4.11) for our running example using the approach proposed by OSBORNE et al.

space is transformed to agree with the moments of (4.16), then used as nodes in a numerical quadrature approximation to (4.17). The resulting approximation again takes the form of a (now weighted) mixture of Gaussian processes (4.14): the MAP model augmented by a small number of additional models designed to reflect the important variation in the hyperparameter posterior. The number of hyperparamater samples required by this scheme grows relatively slowly with the dimension of the hyperparameter space: less than 100 for $|\boldsymbol{\theta}| \leq 8$ and less than 1000 for $|\theta| \leq 21 .^{18}$ The nine samples required for our running example are shown in the marginal figure. Figure 4.7 shows the resulting approximate posterior; comparing with the gold-standard Monte Carlo approximation from Figure 4.6, the agreement is excellent.

An even more lightweight approximation was proposed by OSBORNE et al., which despite its crudeness is arguably still preferable to MAP estimation and can be used as a drop-in replacement. ${ }^{19}$ This approach again relies on a Laplace approximation to the hyperparameter posterior (4.16-4.17). The key observation is that under the admittedly strong assumption that the posterior mean were in fact linear in $\boldsymbol{\theta}$ and the posterior covariance independent of $\theta$, we could resolve (4.17) in closed form. We proceed by taking the best linear approximation to the posterior mean around the MAP: ${ }^{20}$

$$
\mu_{\mathcal{D}}(x ; \boldsymbol{\theta}) \approx \mu_{\mathcal{D}}(x ; \hat{\boldsymbol{\theta}})+\mathbf{g}(x)^{\top}(\boldsymbol{\theta}-\hat{\boldsymbol{\theta}}) ; \quad \mathbf{g}(x)=\frac{\partial \mu_{\mathcal{D}}(x ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}(\hat{\boldsymbol{\theta}}),
$$

and assuming the MAP posterior covariance is universal: $K_{\mathcal{D}}(\boldsymbol{\theta}) \approx K_{\mathcal{D}}(\hat{\boldsymbol{\theta}})$. The result is a single Gaussian process approximation to the posterior:

$$
p(f \mid \mathcal{D}) \approx \mathcal{G} \mathcal{P}\left(f ; \hat{\mu}_{\mathcal{D}}, \hat{K}_{\mathcal{D}}\right)
$$

where

$$
\hat{\mu}_{\mathcal{D}}(x)=\mu_{\mathcal{D}}(x ; \hat{\boldsymbol{\theta}}) ; \quad \hat{K}_{\mathcal{D}}\left(x, x^{\prime}\right)=K_{\mathcal{D}}\left(x, x^{\prime} ; \hat{\boldsymbol{\theta}}\right)+\mathbf{g}(x)^{\top} \mathbf{C} \mathbf{g}\left(x^{\prime}\right) .
$$

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-11.jpg?height=374&width=531&top_left_y=1144&top_left_x=1368)

A Laplace approximation to the model posterior (performed in the log domain) and hyperparameter settings corresponding to the central composite design proposed by RUE et al. The samples do a good job covering the support of the true posterior.

18 S. M. SANCHEZ and P. J. SANCHEZ (2005). Very Large Fractional Factorial and Central Composite Designs. ACM Transactions on Modeling and Computer Simulation 15(4):362-377.

19 M. A. OSBORNE et al. (2012). Active Learning of Model Evidence Using Bayesian Quadrature. NeurIPS 2012.

20 This is analogous to the linearization step in the extended Kalman filter, whereas the central composite design approach is closer to the unscented Kalman filter in pushing samples through the nonlinear transformation. uncertainty in additive noise scale $\sigma_{n}$

21 This term vanishes if the hyperparameters are completely determined by the data, in which case the approximation regresses gracefully to the MAP estimate.

22 In general we have

$$
\begin{aligned}
& p(y \mid x, \mathcal{D})= \\
& \quad \iint p\left(y \mid x, \phi, \sigma_{n}\right) p\left(\phi, \sigma_{n} \mid x, \mathcal{D}\right) \mathrm{d} \phi \mathrm{d} \sigma_{n},
\end{aligned}
$$

and we have resolved the integral on $\phi$ using the single-GP approximation. model structure index, $\mathcal{M}$

model structure prior, $\operatorname{Pr}(\mathcal{M})$
This is the MAP model with covariance inflated by a term determined by the dependence of the posterior mean on the hyperparameters, $\mathrm{g}$, and the uncertainty in the hyperparameters, $\mathrm{C}^{21}$

OSBORNE et al. did not address how to account for uncertainty in observation model parameters when approximating $p(y \mid x, \mathcal{D})$, but we can derive a natural approach for independent additive Gaussian noise with unknown scale $\sigma_{n}$. Given $x$, let $p(\phi \mid x, \mathcal{D}) \approx \mathcal{N}\left(\phi ; \mu, \sigma^{2}\right)$ as in (4.18). We must approximate ${ }^{22}$

$$
p(y \mid x, \mathcal{D}) \approx \int \mathcal{N}\left(y ; \mu, \sigma^{2}+\sigma_{n}^{2}\right) p\left(\sigma_{n} \mid x, \mathcal{D}\right) \mathrm{d} \sigma_{n} .
$$

A moment-matched approximation $p(y \mid x, \mathcal{D}) \approx \mathcal{N}\left(y ; m, s^{2}\right)$ is possible by appealing to the law of total variance:

$$
m=\mathbb{E}[y \mid x, \mathcal{D}] \approx \mu ; \quad s^{2}=\operatorname{var}[y \mid x, \mathcal{D}] \approx \sigma^{2}+\mathbb{E}\left[\sigma_{n}^{2} \mid x, \mathcal{D}\right] .
$$

If the noise scale is parameterized by its logarithm, then the Laplace approximation (4.16) in particular yields

$p\left(\log \sigma_{n} \mid x, \mathcal{D}\right) \approx \mathcal{N}\left(\log \sigma_{n} ; \log \hat{\sigma}_{n}, s^{2}\right) ; \quad \mathbb{E}\left[\sigma_{n}^{2} \mid x, \mathcal{D}\right] \approx \hat{\sigma}_{n}^{2} \exp \left(2 s^{2}\right)$.

Thus we predict with the MAP estimate $\hat{\sigma}_{n}$ inflated by a factor commensurate with the residual uncertainty in the noise contribution.

Figure 4.8 shows the resulting approximation for our running example. Although not perfect, the predictive uncertainty in the observations is more faithful than the MAP model from Figure 4.5 , which severely underestimates the scale of observation noise in the posterior.

\subsection*{MULTiple MODEL STRUCTURES}

We have now covered model inference, selection, and averaging with a single parametric model space (4.1). With a bit of extra bookkeeping, we may extend this framework to handle multiple model structures comprising different combinations of parametric prior moments and observation models.

To begin, we may build a space of candidate models by taking a discrete union of parametric spaces as in (4.1), with one built from each desired model structure: $\left\{\mathcal{M}_{i}\right\}$. It is natural to index this space by $(\theta, \mathcal{M})$, where $\boldsymbol{\theta}$ is understood to be a vector of hyperparameters associated with the specified model structure; the size and interpretation of this vector may differ across structures. All that remains is to derive our previous results while managing this compound structure-hyperparameter index.

We may define a model prior over this compound space by combining a prior over the chosen model structures with priors over the hyperparameters of each:

$$
p(\boldsymbol{\theta}, \mathcal{M})=\operatorname{Pr}(\mathcal{M}) p(\boldsymbol{\theta} \mid \mathcal{M}) .
$$

Given data, the model posterior has a similar form as before (4.7):

$$
p(\boldsymbol{\theta}, \mathcal{M} \mid \mathcal{D})=\operatorname{Pr}(\mathcal{M} \mid \mathcal{D}) p(\boldsymbol{\theta} \mid \mathcal{D}, \mathcal{M})
$$



![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-13.jpg?height=203&width=1017&top_left_y=464&top_left_x=268)

The structure-conditional hyperparameter posterior $p(\theta \mid \mathcal{D}, \mathcal{M})$ is as in (4.7) and may be reasoned about following our previous discussion. The model structure posterior is then given by

$$
\begin{aligned}
& \operatorname{Pr}(\mathcal{M} \mid \mathcal{D}) \propto \operatorname{Pr}(\mathcal{M}) p(\mathbf{y} \mid \mathbf{x}, \mathcal{M}) \\
& p(\mathbf{y} \mid \mathbf{x}, \mathcal{M})=\int p(\mathbf{y} \mid \mathbf{x}, \boldsymbol{\theta}, \mathcal{M}) p(\boldsymbol{\theta} \mid \mathcal{M}) \mathrm{d} \boldsymbol{\theta} .
\end{aligned}
$$

The expression in (4.22) is the normalizing constant of the structureconditional hyperparameter posterior (4.7), which we could ignore when there was only a single model structure. This integral is in general intractable, but several approximations are feasible. One effective choice is the Laplace approximation (4.16), which provides an approximation to the integral as a side effect (B.2). The classical Bayesian information criterion (BIC) may be seen as an approximation to this approximation. ${ }^{23}$

Model selection may now be pursued by maximizing the model posterior over the model space as before, although we may no longer appeal to gradient methods as the model space is not continuous with multiple model structures. A simple approach would be to find the MAP hyperparameters for each of the model structures separately, then use these MAP points to approximate (4.22) for each structure via the Laplace approximation or BIC. This would be sufficient to estimate $(4.20-4.21)$ and maximize over the MAP models.

Turning to model averaging, the model-marginal posterior to the objective function is:

$$
p(f \mid \mathcal{D})=\sum_{i} \operatorname{Pr}\left(\mathcal{M}_{i} \mid \mathcal{D}\right) p\left(f \mid \mathcal{D}, \mathcal{M}_{i}\right)
$$

The structure-conditional, hyperparameter-marginal distribution on each space $p(f \mid \mathcal{D}, \mathcal{M})$ is as before (4.11) and may be approximated following our previous discussion. These are now combined in a mixture distribution weighted by the model structure posterior (4.21).

\section*{Multiple structure example}

We now present an example of model inference, selection, and averaging over multiple model structures using the dataset in Figure $4.9{ }^{24}$ The data were sampled from a Gaussian process with linear prior mean (a linear trend with positive slope is evident) and Matern $v=3 / 2$ prior covariance (3.13), with a small amount of additive Gaussian noise. We also show a sample from the objective function posterior corresponding to the true model generating the data for reference.
Figure 4.9: The objective and dataset for our multiple-model example.

model structure posterior, $\operatorname{Pr}(\mathcal{M} \mid \mathcal{D})$

Laplace approximation: § B.1, p. 301

23 s. KONISHI and G. KITAGAWA (2008). Information Criteria and Statistical Modeling. SpringerVerlag. [chapter 9]

model selection

model averaging

24 The data are used as a demo in the code released with:

C. E. RASMUSSEN and C. K. I. WILLIAMS (2006). Gaussian Processes for Machine Learning. MIT Press. 

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-14.jpg?height=208&width=525&top_left_y=453&top_left_x=157)

M5

Figure 4.10: Sample paths from our example model structures.

initial model structure: p. 69

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-14.jpg?height=223&width=531&top_left_y=454&top_left_x=705)

LIN

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-14.jpg?height=222&width=532&top_left_y=760&top_left_x=702)

M5 + LIN

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-14.jpg?height=225&width=528&top_left_y=453&top_left_x=1255)

$\mathrm{LIN} \times \operatorname{LIN}$

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-14.jpg?height=217&width=511&top_left_y=762&top_left_x=1275)

M5 X LIN

We build a model space comprising several model structures by augmenting our previous space with structures incorporating additional covariance functions. The treatment of the prior mean (unknown constant marginalized against a Gaussian prior) and observation model (additive Gaussian noise with unknown scale) will remain the same for all. The model structures reflect a variety of hypotheses positing potential linear or quadratic behavior:

M5: the Matérn $v=5 / 2$ covariance (3.14) from our previous example;

LIN: the linear covariance (3.16), where the the prior on the slope is vague and centered at zero and the prior on the intercept agrees with the M5 model;

LIN $\times$ LIN: the product of two linear covariances designed as above, modeling a latent quadratic function with unknown coefficients;

M5 + LIN: the sum of a Matérn $v=5 / 2$ and linear covariance designed as in the corresponding individual model structures; and

M5 $\times$ LIN: the product of a Matérn $v=5 / 2$ and linear covariance designed as in the corresponding individual model structures.

Objective function samples from models in each of these structures are shown in Figure 4.10. Among these, the model structure closest to the truth is arguably M5 + LIN.

Following the above discussion, we find the MAP hyperparameters for each of these model structures separately and use a Laplace approximation (4.16) to approximate the hyperparameter posterior on each space, along with the normalizing constant (4.22). Normalizing over the structures provides an approximate model structure posterior:

$$
\begin{aligned}
\operatorname{Pr}\left(\mathrm{M}_{5} \mid \mathcal{D}\right) & \approx 10.8 \% ; \\
\operatorname{Pr}(\mathrm{M} 5+\mathrm{LIN} \mid \mathcal{D}) & \approx 71.8 \% ; \\
\operatorname{Pr}\left(\mathrm{M}_{5} \times \operatorname{LIN} \mid \mathcal{D}\right) & \approx 17.0 \%,
\end{aligned}
$$

with the remaining model structures (LIN and LIN $\times$ LIN) sharing the re- 

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-15.jpg?height=534&width=1649&top_left_y=455&top_left_x=252)

Figure 4.11: An approximation to the model-marginal posterior (4.23) for our multiple-model example. The posterior on each model structure is approximated separately as a mixture of Gaussian processes following RUE et al. (see Figure 4.7); these are then combined by weighting by an approximation of the model structure posterior (4.21). We show the result with three superimposed, transparent credible intervals, which are shaded with respect to their weight in contributing to the final approximation.

maining $0.4 \%$. The $\mathrm{M}_{5}+$ LIN model structure is the clear winner, and there is strong evidence that the purely polynomial models are insufficient for explaining the data alone.

Figure 4.11 illustrates an approximation to the model-marginal posterior (4.23), approximated by applying RUE et al.'s central composite design approach to each of the model structures, then combining these into a Gaussian process mixture by weighting by the approximate model structure posterior. The highly asymmetric credible intervals reflect the diversity in explanations for the data offered by the chosen model structures, and the combined model makes reasonable predictions of our example objective function sampled from the true model.

For this example, averaging over the model structure has important implications regarding the behavior of the resulting optimization policy. Figure 4.12 illustrates a common acquisition function ${ }^{25}$ built from the offthe-shelf M5 model, as well as from the structure-marginal model. The former chooses to exploit near what it believes is a local optimum, but the latter has a strong belief in an underlying linear trend and chooses to explore the right-hand side of the domain instead. For our example objective function sample, this would in fact reveal the global optimum with the next observation.

\subsection*{AUTOMATING MODEL STRUCTURE SEARCH}

We now have a comprehensive framework for reasoning about model uncertainty, including methods for model assessment, selection, and averaging across one or multiple model structures. However, it is still not clear how we should determine which model structures to consider for a given system. This is critical as our model inference procedure requires approximation to marginal predictive distribution

averaging over a space of Gaussian processes in policy computation: $\S 8.10$, p. 192

25 to be specific, expected improvement: $§ 7 \cdot 3$, p. 127 
![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-16.jpg?height=272&width=1622&top_left_y=455&top_left_x=160)

acquisition function $\quad \boldsymbol{\nabla}$ next observation location
![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-16.jpg?height=196&width=1624&top_left_y=792&top_left_x=160)

Figure 4.12: Optimization policies built from the MAP M5 model (left) and the structure-marginal posterior (right). The M5 model chooses to exploit near the local optimum, but the structure-marginal model is aware of the underlying linear trend and chooses to explore the right-hand side as a result.

26 D. DUVENAUd et al. (2013). Structure Discovery in Nonparametric Regression through Compositional Kernel Search. ICML 2013.

addition and multiplication of covariance functions: $§ 3.4$, p. 60

base kernels, $\mathcal{B}$ the space of candidate models to be predefined; equivalently, the model prior (4.19) is implicitly set to zero for every model outside this space. Ideally, we would simply enumerate every possible model structure and average over all of them, but even a naïve approximation of this ideal would entail overwhelming computational effort.

However, the set of model structures we consider for a given dataset can be adaptively tailored as we gather data. One powerful idea is to appeal to metaheuristics such as local search: by establishing a suitable space of candidate model structures, we can dynamically explore this space for the best explanations of available data.

\section*{Spaces of candidate model structures}

To enable this approach, we must first establish a sufficiently rich space of candidate model structures. DUVENAUD et al. proposed one convenient mechanism for defining such a space via a simple productive grammar. ${ }^{26}$ The idea is to appeal to the closure of covariance functions under addition and pointwise multiplication to systematically build up families of increasingly complex models from simple components. We begin by choosing a set of so-called base kernels, $\mathcal{B}$, modeling relatively simple behavior, then extend this set to an infinite family of compositions via the following context-free grammar:

$$
\begin{aligned}
& K \rightarrow B \\
& K \rightarrow K+K \\
& K \rightarrow K K \\
& K \rightarrow(K) .
\end{aligned}
$$

The symbol $B$ in the first rule represents any desired base kernel. The five model structures considered in our multiple-structure example above in fact represent five members of the language generated by this grammar with the base kernels $\mathcal{B}=\left\{K_{\mathrm{M} 5 / 2}, K_{\mathrm{LIN}}\right\}$ or simply $\mathcal{B}=\{\mathrm{M} 5, \mathrm{LIN}\}$. The grammar however also generates arbitrarily more complicated expressions such as

$$
\left(\mathrm{M}_{5}+(\mathrm{M} 5+\mathrm{LIN}) \mathrm{M}_{5}\right)\left(\mathrm{M}_{5}+\mathrm{M} 5\right)+\mathrm{LIN} .
$$

We are free to design the base kernels to capture any potential atomic behavior in the objective function. For example, if the domain is high dimensional and we suspect that the objective may depend only on sparse interactions of mostly independent variables, we might design the base kernels to model variations in single variables at a time, then rely on the grammar to generate an array of possible interaction structures.

Other spaces of model structures have also been proposed for automated structure search. With an eye toward high-dimensional domains, GARDNER et al. for example considered spaces of additive model structures indexed by every possible partition of the input variables. ${ }^{27}$ This is an expressive class of model structures, but the number of partitions grows so rapidly that exhaustive search is not feasible.

\section*{Searching over model structures}

Once a space of candidate model structures has been established, we may develop a search procedure seeking the most promising structures to explain a given dataset. Several approaches have been proposed for this search with a range of complexity, all of which frame the problem in terms of optimizing some figure of merit over the space. Although any score could be used in this context, a natural choice is an approximation to the (unnormalized) model structure posterior (4.21) such as the Laplace approximation or the Bayesian information criterion, and every method described below uses one of these two scores.

DUVENAUD et al. suggested traversing their kernel grammar via greedy search - here, we first evaluate the base kernels, then subject the productive rules to the best among them to generate similar structures to search next. ${ }^{26}$ We continue in this manner as desired, alternating between evaluating the newly proposed structures, then using the grammar to expand around the best-seen structure to generate new proposals. This simple procedure is easy to implement and offers a strong baseline.

MALKOMEs et al. refined this approach by replacing greedy search with Bayesian optimization over the space of model structures. ${ }^{28}$ As in the DUVENAUD et al. procedure, the authors pose the problem in terms of maximizing a score over model structures: a Laplace approximation of the (log) unnormalized structure posterior (4.21). This objective function was then modeled using a Gaussian process, which informed a sequential Bayesian optimization procedure seeking to effectively manage the exploration-exploitation tradeoff in the space of candidate structures. The Gaussian process in model space requires a covariance function over model structures, and the authors proposed an exotic "kernel kernel" evaluating the similarity of proposed structures in terms of the overlap

![](https://cdn.mathpix.com/cropped/2023_09_22_234a6a9005560ad66f17g-17.jpg?height=208&width=528&top_left_y=593&top_left_x=1369)

Samples from objective function models incorporating the example covariance structure (4.24).

additive decompositions, $\S 3.5$, p. 61

27 J. R. GARDNER et al. (2017). Discovering and Exploiting Additive Structure for Bayesian Optimization. AISTATS 2017.

28 G. MALKомEs et al. (2016). Bayesian Optimization for Automated Model Selection. NeurIPS 2016. 29 G. MALKOMES and R. GARNETT (2018). Automating Bayesian Optimization with Bayesian Optimization. NeurIPS 2018
30 J. R. GARDNER et al. (2017). Discovering and Exploiting Additive Structure for Bayesian Optimization. AISTATS 2017.
31 J. SNOEK et al. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. NeUrIPS 2012.

models and model structures: § 4.1, p. 68 between their hyperparameter-marginal priors for the given dataset. The resulting optimization procedure was found to rapidly locate promising models across a range of regression tasks.

\section*{End-to-end automation}

Follow-on work demonstrated a completely automated Bayesian optimization system built on this structure search procedure avoiding any manual modeling at all. ${ }^{29}$ The key idea was to dynamically maintain a set of plausible model structures throughout optimization. Predictions are made via model averaging over this set, offering robustness to model misspecification when computing the outer optimization policy. Every time a new observation is obtained, the set of model structures is then updated via a continual Bayesian optimization in model space given the new data. This interleaving of Bayesian optimization in data space and model space offered promising performance.

Finally, GARDNER et al. offered an alternative to optimization over model structures by constructing a Markov chain Monte Carlo routine to sample model structures from their posterior $(4 \cdot 21) \cdot{ }^{30}$ The proposed sampler was a realization of the Metropolis-Hastings algorithm with a custom proposal distribution making minor modifications to the incumbent structure. In the case of the additive decompositions considered in that work, this step consisted of applying random atomic operations such as merging or splitting components of the existing decomposition. Despite the absolutely enormous number of possible additive decompositions, this MCMC routine was able to quickly locate promising structures, and averaging over the sampled structures for prediction resulted in superior optimization performance as well.

\section*{$4 \cdot 7$ SUMMARY OF MAJOR IDEAS}

We have presented a convenient framework for model assessment, selection, and averaging grounded in Bayesian inference; this is the predominant approach with Gaussian process models. In the context of Bayesian optimization, perhaps the most important development was the notion of model averaging, which has proven beneficial to empirical performance ${ }^{31}$ and has become standard practice.

- Model assessment entails deriving preferences over a space of candidate models of a given system in light of available data.

- In its purest form, a model in this context is a prior distribution over observed values $\mathbf{y}$ arising from observations at a given set of locations $\mathbf{x}$, $p(\mathbf{y} \mid \mathbf{x})$. A convenient mechanism for specifying a model is via a prior process for a latent function, $p(f)$, and an observation model conditioned on this function, $p(y \mid x, \phi)(4.1-4.2)$.

- With Gaussian process models, it is convenient to work with combinations of parametric forms for the prior mean function, prior covariance function, and observation model, a construct we call a model structure. A model structure defines a space of corresponding models by traversing its parameter space (4.3), allowing us to build expressive model spaces.

- Once we delineate a space of candidate models, model assessment becomes straightforward if we make the - perhaps dubious but nonetheless practical - assumption that the mechanism generating our data is contained within this space. This allows us to treat that true model as a random variable and proceed via Bayesian inference over the chosen model space.

- This inference proceeds as usual. We first define a model prior capturing any initial beliefs over the model space. Then, given a set of observations $\mathcal{D}=(\mathbf{x}, \mathbf{y})$, the model posterior is proportional to the model prior and a measure of model fit known as the marginal likelihood or model evidence, the probability (density) of the observed data under the model.

- In addition to quantifying fit, the model evidence encodes an automatic penalty for model complexity, an effect known as Bayesian Occam's razor

- Model evidence can be computed in closed form for Gaussian process models with additive Gaussian observation noise (4.8).

- Model inference is especially convenient when the model space is a single model structure, but can be extended to spaces built from multiple model structures with a bit of extra bookkeeping.

- The model posterior provides a simple means of model assessment by establishing preferences according to posterior probability. If we must commit to a single model to explain the data - a task known as model selection - we then select the maximum a posteriori (MAP) model.

- Model selection may not be prudent when the model posterior is very flat, which is common when observations are scarce. In this case many models may be compatible with the data but incompatible in their predictions, which should be accounted for in the interest of robustness. Model averaging is a natural solution, where we marginalize the unknown model when making predictions according to the model posterior.

- Model averaging cannot in general be performed in closed form for Gaussian process models; however, we may proceed via MCMC sampling (4.14-4.15) or by appealing to more lightweight approximation schemes.

- Appealing to metaheuristics allows us to automatically search a space of candidate model structures to explain a given dataset. Once sufficiently mature, such schemes may some day enable fully automated Bayesian optimization pipelines that sidestep explicit modeling altogether.

The next chapter marks a major departure from our discussion thus far, which has focused on modeling and making predictions from data. We will now shift our attention from inference to decision making, with the goal of building effective optimization policies informed by the models we have now fully developed. This endeavor will consume the bulk of the remainder of the book.
Bayesian inference over (parametric) model spaces: $§ 4.2$, p. 70

Bayesian Occam’s razor, § 4.2, p. 71

multiple model structures: $§ 4.5$, p. 78

model selection via MAP inference: $§ 4.3$, p. 73

model averaging: § 4.4, p. 74

approximations to model-marginal posterior: Figures 4.6-4.8 and surrounding text

automating model structure search: $§ 4.6$, p. 81 The first step will be to develop a framework for optimal decision making under uncertainty. Our work to this point will serve an essential component of this framework, as every such decision will be made with reference to a posterior belief about what might happen as a result. In the context of optimization, this belief will take the form of a posterior predictive distribution for proposed observations given data, and our investment in building faithful models will pay off in spades.