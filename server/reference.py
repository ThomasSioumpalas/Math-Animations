"""The reference library: what the teacher model is given before it plans an explanation.

Each card states, in our own words, the standard definition, the key formulas, the conditions under
which they hold, a typical mistake and where the idea is used in machine learning. "read" lists the
textbook chapters where the topic is treated (shown to the user as sources); "d2l" lists sections of
the open textbook "Dive into Deep Learning" (CC BY-SA 4.0) that server/sources.py can quote.

Books (chapter titles as printed):
  MML   Deisenroth, Faisal, Ong: Mathematics for Machine Learning (Cambridge University Press, 2020)
  ILA   Strang: Introduction to Linear Algebra (Wellesley-Cambridge Press)
  VMLS  Boyd, Vandenberghe: Introduction to Applied Linear Algebra (Cambridge University Press, 2018)
  PRML  Bishop: Pattern Recognition and Machine Learning (Springer, 2006)
  ESL   Hastie, Tibshirani, Friedman: The Elements of Statistical Learning (Springer, 2009)
  EIT   Cover, Thomas: Elements of Information Theory (Wiley, 2006)
  CVX   Boyd, Vandenberghe: Convex Optimization (Cambridge University Press, 2004)
  D2L   Zhang, Lipton, Li, Smola: Dive into Deep Learning (d2l.ai)
Adding a card: keep every statement checkable against the chapters it cites.
"""

BOOKS = {
    "MML": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning",
    "ILA": "Strang: Introduction to Linear Algebra",
    "VMLS": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra",
    "PRML": "Bishop: Pattern Recognition and Machine Learning",
    "ESL": "Hastie, Tibshirani, Friedman: The Elements of Statistical Learning",
    "EIT": "Cover, Thomas: Elements of Information Theory",
    "CVX": "Boyd, Vandenberghe: Convex Optimization",
    "D2L": "Zhang, Lipton, Li, Smola: Dive into Deep Learning",
}
_APX = "chapter_appendix-mathematics-for-deep-learning/"

CARDS = [
    {"id": "linear-map", "title": "Matrices as linear transformations",
     "keywords": ["linear transformation", "linear map", "matrix vector", "matrix-vector", "matrix times vector",
                  "basis vector", "column space", "span", "linear combination", "columns of"],
     "facts": "A matrix A ∈ ℝ^{m×n} is the linear map x ↦ Ax; linear means A(αx + βy) = αAx + βAy. "
              "Its columns are the images of the standard basis vectors: Ae_j is column j, so Ax = Σ_j x_j·(column j), "
              "a linear combination of the columns. The set of all outputs is the column space (the span of the "
              "columns); its dimension is the rank. Grid lines stay parallel and evenly spaced and the origin stays "
              "fixed under every linear map. Mistake: reading the rows, not the columns, as where the basis vectors land.",
     "read": [("MML", "Linear Algebra"), ("ILA", "Introduction to Vectors; Solving Linear Equations")],
     "d2l": [_APX + "geometry-linear-algebraic-ops.md", "chapter_preliminaries/linear-algebra.md"]},
    {"id": "matmul", "title": "Matrix multiplication",
     "keywords": ["matrix multiplication", "matrix product", "multiply matrices", "hadamard", "composition", "compose",
                  "ab ≠ ba", "commutative"],
     "facts": "For A ∈ ℝ^{m×n} and B ∈ ℝ^{n×p}, (AB)_{ij} = Σ_k A_{ik}B_{kj}: row i of A dotted with column j of B; "
              "the inner sizes must match and the result is m×p. AB is the composition 'first B, then A' (read right "
              "to left). It is associative and distributive but in general not commutative: AB ≠ BA. (AB)ᵀ = BᵀAᵀ and "
              "(AB)⁻¹ = B⁻¹A⁻¹. The element-wise (Hadamard) product A∘B is a different operation. det(AB) = det(A)·det(B).",
     "read": [("MML", "Linear Algebra"), ("VMLS", "Matrix multiplication")],
     "d2l": ["chapter_preliminaries/linear-algebra.md"]},
    {"id": "determinant", "title": "Determinant",
     "keywords": ["determinant", "det(", "det ", "area scaling", "volume scaling", "singular matrix", "orientation"],
     "facts": "det(A) is the signed factor by which the linear map A scales area (2-D) or volume (n-D): the unit "
              "square/cube is mapped to a parallelogram/parallelepiped of signed volume det(A). For [[a, b], [c, d]] it "
              "is ad − bc. det(A) < 0: orientation is reversed. det(A) = 0 ⇔ the columns are linearly dependent ⇔ space "
              "is flattened onto a lower dimension ⇔ A is not invertible (singular). det(AB) = det(A)det(B), "
              "det(A⁻¹) = 1/det(A), det(Aᵀ) = det(A), and det(A) is the product of the eigenvalues. "
              "Mistake: det is not linear, det(A + B) ≠ det(A) + det(B) and det(cA) = cⁿdet(A).",
     "read": [("MML", "Matrix Decompositions (Determinant and Trace)"), ("ILA", "Determinants")],
     "d2l": [_APX + "geometry-linear-algebraic-ops.md"]},
    {"id": "inverse-rank", "title": "Inverse, rank and solving Ax = b",
     "keywords": ["inverse", "invertible", "rank", "null space", "nullspace", "kernel", "solve ax", "ax = b",
                  "linear system", "linearly dependent", "linearly independent", "full rank"],
     "facts": "A square A is invertible iff det(A) ≠ 0 iff its columns are linearly independent (full rank) iff "
              "Ax = 0 only for x = 0. Then A⁻¹A = AA⁻¹ = I and Ax = b has the unique solution x = A⁻¹b. For 2×2: "
              "A⁻¹ = (1/(ad − bc))·[[d, −b], [−c, a]]. rank(A) = dimension of the column space = number of independent "
              "columns = number of independent rows. Ax = b is solvable iff b lies in the column space; the solutions of "
              "Ax = 0 form the null space, of dimension n − rank(A) (rank–nullity). Geometrically A⁻¹ undoes the "
              "transformation; a flattened space (det = 0) cannot be un-flattened.",
     "read": [("MML", "Linear Algebra"), ("ILA", "Solving Linear Equations; Vector Spaces and Subspaces")],
     "d2l": [_APX + "geometry-linear-algebraic-ops.md"]},
    {"id": "norms", "title": "Norms and the dot product",
     "keywords": ["norm", "l1", "l2", "l∞", "unit ball", "unit circle", "length of a vector", "dot product",
                  "inner product", "angle between", "orthogonal", "perpendicular", "cosine similarity", "frobenius"],
     "facts": "A norm satisfies ‖x‖ ≥ 0 with equality only for x = 0, ‖αx‖ = |α|‖x‖, and the triangle inequality "
              "‖x + y‖ ≤ ‖x‖ + ‖y‖. ‖x‖₁ = Σ|x_i|, ‖x‖₂ = √(Σx_i²) = √(xᵀx), ‖x‖∞ = max|x_i|, and ‖x‖∞ ≤ ‖x‖₂ ≤ ‖x‖₁. "
              "Unit balls in 2-D: a diamond (L1), a circle (L2), an axis-aligned square (L∞). The dot product "
              "xᵀy = ‖x‖₂‖y‖₂cos θ; xᵀy = 0 ⇔ orthogonal. Cauchy–Schwarz: |xᵀy| ≤ ‖x‖₂‖y‖₂. For matrices the Frobenius "
              "norm is ‖A‖_F = √(Σ A_ij²) = √(tr(AᵀA)). In ML: L2 penalties (weight decay) shrink all weights, L1 "
              "penalties favour sparse weights because the L1 ball has corners on the axes.",
     "read": [("MML", "Analytic Geometry"), ("VMLS", "Norm and distance")],
     "d2l": ["chapter_preliminaries/linear-algebra.md", _APX + "geometry-linear-algebraic-ops.md"]},
    {"id": "projection", "title": "Orthogonal projection and least squares",
     "keywords": ["projection", "project onto", "least squares", "normal equation", "pseudoinverse", "pseudo-inverse",
                  "moore-penrose", "overdetermined", "residual"],
     "facts": "The orthogonal projection of b onto the line through a is p = (aᵀb / aᵀa)·a; the error b − p is "
              "perpendicular to a. Onto the column space of A (independent columns): p = A(AᵀA)⁻¹Aᵀb, with projection "
              "matrix P = A(AᵀA)⁻¹Aᵀ, P² = P = Pᵀ. Least squares: when Ax = b has no solution, the x̂ minimising "
              "‖Ax − b‖₂² solves the normal equations AᵀAx̂ = Aᵀb, and Ax̂ is exactly the projection of b onto the "
              "column space; the residual is orthogonal to every column. The pseudoinverse A⁺ = (AᵀA)⁻¹Aᵀ gives "
              "x̂ = A⁺b (in general A⁺ comes from the SVD and returns the minimum-norm solution). Linear regression is "
              "this with A = the data matrix.",
     "read": [("MML", "Analytic Geometry (Orthogonal Projections); Linear Regression"), ("ILA", "Orthogonality"),
              ("VMLS", "Least squares")],
     "d2l": ["chapter_linear-regression/linear-regression.md"]},
    {"id": "eigen", "title": "Eigenvalues, eigenvectors and eigendecomposition",
     "keywords": ["eigenvalue", "eigenvector", "eigen", "eigendecomposition", "diagonaliz", "spectral",
                  "characteristic polynomial", "qλq", "symmetric matrix", "positive definite"],
     "facts": "v ≠ 0 is an eigenvector of square A with eigenvalue λ if Av = λv: A only stretches v by λ (flips it if "
              "λ < 0), it does not turn it. Eigenvalues are the roots of det(A − λI) = 0; their sum is tr(A), their "
              "product det(A). If A has n independent eigenvectors (columns of Q), A = QΛQ⁻¹ with Λ diagonal, and "
              "Aᵏ = QΛᵏQ⁻¹. Not every matrix is diagonalisable (a shear is not; a rotation has no real eigenvectors). "
              "Spectral theorem: a real symmetric matrix has real eigenvalues and orthonormal eigenvectors, so "
              "A = QΛQᵀ with QᵀQ = I: rotate to the eigenbasis, stretch along the axes, rotate back. Symmetric A is "
              "positive definite iff all λ > 0. Mistake: eigenvectors are directions, any nonzero multiple is also one.",
     "read": [("MML", "Matrix Decompositions"), ("ILA", "Eigenvalues and Eigenvectors")],
     "d2l": [_APX + "eigendecomposition.md"]},
    {"id": "svd", "title": "Singular value decomposition",
     "keywords": ["svd", "singular value", "singular vector", "low rank", "low-rank", "rank-k", "eckart"],
     "facts": "Every real m×n matrix factors as A = UΣVᵀ with U (m×m) and V (n×n) orthogonal and Σ diagonal with "
              "singular values σ₁ ≥ σ₂ ≥ … ≥ 0. Geometry: Vᵀ rotates/reflects, Σ stretches along the axes, U "
              "rotates/reflects; the unit circle is mapped to an ellipse with semi-axes σ_i along the columns of U. "
              "σ_i = √λ_i(AᵀA); the columns of V are eigenvectors of AᵀA, those of U of AAᵀ. rank(A) = number of "
              "nonzero σ_i; |det A| = Πσ_i for square A; ‖A‖₂ = σ₁. Unlike the eigendecomposition it exists for every "
              "matrix, square or not. Truncating to the k largest σ_i gives the best rank-k approximation in the "
              "spectral and Frobenius norms (Eckart–Young), the basis of PCA and compression. For symmetric positive "
              "semidefinite A the SVD equals the eigendecomposition. Mistake: singular values are not eigenvalues in general.",
     "read": [("MML", "Matrix Decompositions (Singular Value Decomposition; Matrix Approximation)"),
              ("ILA", "The Singular Value Decomposition (SVD)")],
     "d2l": [_APX + "eigendecomposition.md"]},
    {"id": "pca", "title": "Principal component analysis",
     "keywords": ["pca", "principal component", "dimensionality reduction", "explained variance", "principal axis",
                  "principal axes"],
     "facts": "Centre the data (subtract the mean of each variable). The first principal component is the unit "
              "direction u maximising the variance of the projections uᵀx; it is the eigenvector of the covariance "
              "matrix S = (1/n)XᵀX (X centred) with the largest eigenvalue, and that eigenvalue is the variance along "
              "it. Further components are the next eigenvectors, mutually orthogonal. Projecting onto the top k "
              "components maximises retained variance and, equivalently, minimises the mean squared reconstruction "
              "error; the fraction of variance explained is (λ₁ + … + λ_k)/Σλ_i. Via the SVD X = UΣVᵀ: the components "
              "are the columns of V and λ_i = σ_i²/n. Mistakes: forgetting to centre; PCA depends on the units, so "
              "standardise variables on different scales; components are directions of variance, not of class separation.",
     "read": [("MML", "Dimensionality Reduction with Principal Component Analysis"), ("PRML", "Continuous Latent Variables"),
              ("ESL", "Unsupervised Learning")],
     "d2l": [_APX + "eigendecomposition.md"]},
    {"id": "covariance", "title": "Covariance and correlation",
     "keywords": ["covariance", "correlation", "correlated", "pearson", "covariance matrix", "correlation matrix",
                  "standardiz", "standardis", "z-score", "independent variables"],
     "facts": "cov(X, Y) = E[(X − μ_X)(Y − μ_Y)] = E[XY] − E[X]E[Y]; cov(X, X) = var(X). The correlation "
              "ρ = cov(X, Y)/(σ_Xσ_Y) lies in [−1, 1], is unit-free, and equals ±1 iff Y is an exact linear function of "
              "X. It measures linear association only: independent ⇒ uncorrelated, but uncorrelated does not imply "
              "independent (Y = X² with symmetric X). The covariance matrix Σ_ij = cov(X_i, X_j) is symmetric positive "
              "semidefinite; the correlation matrix is the covariance matrix of the standardised variables and has "
              "ones on the diagonal. Σ is singular (det = 0) iff some variable is an exact linear combination of the "
              "others. Sample versions divide by n − 1 (unbiased) or n. var(aX + bY) = a²var(X) + b²var(Y) + 2ab·cov(X, Y). "
              "Mistake: correlation is not causation, and a single outlier can change it greatly.",
     "read": [("MML", "Probability and Distributions (Summary Statistics and Independence)"),
              ("PRML", "Introduction (Probability Theory)")],
     "d2l": [_APX + "random-variables.md"]},
    {"id": "expectation", "title": "Expectation and variance",
     "keywords": ["expectation", "expected value", "variance", "standard deviation", "mean of", "random variable",
                  "fair die", "dice", "probability mass", "probability density", "pmf", "pdf", "marginal",
                  "conditional probability", "chain rule of probability", "joint distribution", "independence"],
     "facts": "Discrete: E[X] = Σ_x x·P(x); continuous: E[X] = ∫x·p(x)dx. A pmf gives probabilities that sum to 1; a "
              "density p(x) ≥ 0 integrates to 1, can exceed 1, and P(a ≤ X ≤ b) is the area under it. Expectation is "
              "linear: E[aX + bY] = aE[X] + bE[Y], always. var(X) = E[(X − μ)²] = E[X²] − (E[X])²; var(aX + b) = "
              "a²var(X); the standard deviation is √var(X), in the units of X. Variances add only for uncorrelated "
              "variables. Marginal: P(x) = Σ_y P(x, y). Conditional: P(y|x) = P(x, y)/P(x). Chain rule: "
              "P(x, y) = P(y|x)P(x). Independence: P(x, y) = P(x)P(y). A fair die: E[X] = 3.5, var(X) = 35/12. "
              "Mistake: E[X] need not be a possible value, and E[g(X)] ≠ g(E[X]) in general.",
     "read": [("MML", "Probability and Distributions"), ("PRML", "Introduction (Probability Theory)")],
     "d2l": ["chapter_preliminaries/probability.md", _APX + "random-variables.md"]},
    {"id": "bayes", "title": "Bayes' rule",
     "keywords": ["bayes", "posterior", "prior", "likelihood", "medical test", "false positive", "base rate",
                  "sensitivity", "specificity", "map estimat", "maximum a posteriori"],
     "facts": "P(A|B) = P(B|A)P(A)/P(B), with P(B) = P(B|A)P(A) + P(B|¬A)P(¬A) (law of total probability). "
              "posterior ∝ likelihood × prior. Medical test: with prevalence 1%, sensitivity 90% and false-positive "
              "rate 5%, P(disease | positive) = 0.9·0.01/(0.9·0.01 + 0.05·0.99) ≈ 15%: most positives are false "
              "because the healthy group is so much larger (base-rate neglect). Mistake: confusing P(A|B) with P(B|A). "
              "In ML: Bayesian inference updates a prior over parameters, p(θ|D) ∝ p(D|θ)p(θ); the MAP estimate "
              "maximises it, and with a Gaussian prior on the weights MAP is least squares with an L2 penalty.",
     "read": [("MML", "Probability and Distributions (Sum Rule, Product Rule, and Bayes' Theorem)"),
              ("PRML", "Introduction (Probability Theory)")],
     "d2l": ["chapter_preliminaries/probability.md", _APX + "naive-bayes.md"]},
    {"id": "gaussian", "title": "Gaussian distributions",
     "keywords": ["gaussian", "normal distribution", "bell curve", "multivariate normal", "covariance ellipse",
                  "isotropic", "mahalanobis", "central limit"],
     "facts": "N(x; μ, σ²) = (1/√(2πσ²))·exp(−(x − μ)²/(2σ²)); about 68% / 95% / 99.7% of the mass lies within 1 / 2 / 3 "
              "σ of μ. In d dimensions N(x; μ, Σ) ∝ exp(−½(x − μ)ᵀΣ⁻¹(x − μ)) with Σ symmetric positive definite. "
              "Contours of equal density are ellipses centred at μ whose axes are the eigenvectors of Σ, with "
              "half-lengths proportional to √λ_i. Σ = σ²I (isotropic): circles; diagonal Σ: axis-aligned ellipses "
              "(uncorrelated, and for a Gaussian therefore independent, components); full Σ: tilted ellipses. "
              "Marginals and conditionals of a Gaussian are Gaussian; a linear map gives Ax + b ~ N(Aμ + b, AΣAᵀ). "
              "Central limit theorem: suitably scaled sums of many independent finite-variance variables are "
              "approximately Gaussian. Maximising a Gaussian likelihood is minimising squared error.",
     "read": [("MML", "Probability and Distributions (Gaussian Distribution)"), ("PRML", "Probability Distributions")],
     "d2l": [_APX + "distributions.md", _APX + "random-variables.md"]},
    {"id": "mle", "title": "Maximum likelihood estimation",
     "keywords": ["maximum likelihood", "mle", "likelihood function", "log-likelihood", "log likelihood", "estimator",
                  "unbiased", "biased estimator", "n − 1", "n-1", "consistent", "standard error", "confidence interval"],
     "facts": "Given i.i.d. data, the likelihood L(θ) = Π_i p(x_i|θ) is a function of θ (not a distribution over θ). "
              "The MLE θ̂ maximises it; in practice one maximises the log-likelihood Σ_i log p(x_i|θ), which has the same "
              "maximiser. Coin with k heads in n tosses: L(θ) = θᵏ(1 − θ)ⁿ⁻ᵏ and θ̂ = k/n. Gaussian: μ̂ = sample mean and "
              "σ̂² = (1/n)Σ(x_i − x̄)², which is biased (E[σ̂²] = ((n − 1)/n)σ²); dividing by n − 1 removes the bias. "
              "Bias of an estimator = E[θ̂] − θ; consistency = convergence to θ as n grows; the standard error of the "
              "mean is σ/√n. Minimising cross-entropy is maximising the likelihood; with Gaussian noise it is "
              "minimising mean squared error.",
     "read": [("MML", "When Models Meet Data (Parameter Estimation)"), ("PRML", "Introduction; Probability Distributions")],
     "d2l": [_APX + "maximum-likelihood.md", _APX + "statistics.md"]},
    {"id": "information", "title": "Entropy, cross-entropy and KL divergence",
     "keywords": ["entropy", "cross-entropy", "cross entropy", "kl divergence", "kullback", "information theory",
                  "self-information", "surprise", "bits", "nats", "mutual information"],
     "facts": "Self-information of an event of probability p: I = −log p (bits for log₂, nats for ln); rare events "
              "carry more. Entropy H(P) = −Σ_x P(x)log P(x) = expected information; it is 0 for a certain outcome and "
              "maximal (log n) for the uniform distribution on n outcomes; a fair coin has 1 bit. Cross-entropy "
              "H(P, Q) = −Σ_x P(x)log Q(x) is the expected code length when coding P with a code built for Q. "
              "KL(P‖Q) = Σ_x P(x)log(P(x)/Q(x)) = H(P, Q) − H(P) ≥ 0, with equality iff P = Q. KL is not symmetric and "
              "not a distance, and is infinite if Q(x) = 0 where P(x) > 0. Minimising cross-entropy over Q is "
              "minimising KL(P‖Q), which for the empirical P is maximum likelihood: the classification loss.",
     "read": [("EIT", "Entropy, Relative Entropy, and Mutual Information"), ("PRML", "Introduction (Information Theory)")],
     "d2l": [_APX + "information-theory.md"]},
    {"id": "gradient-descent", "title": "Gradient descent",
     "keywords": ["gradient descent", "learning rate", "step size", "sgd", "stochastic gradient", "gradient",
                  "loss surface", "loss landscape", "convex", "minimum of", "momentum", "optimizer", "optimiser"],
     "facts": "The gradient ∇f(x) points in the direction of steepest increase, so x ← x − η∇f(x) with learning rate "
              "η > 0 moves downhill. For f(x) = ½ax² the update is x ← (1 − ηa)x: it converges iff 0 < η < 2/a, "
              "oscillates for η > 1/a and diverges for η > 2/a. In general, for an L-smooth function η < 2/L is "
              "needed (η ≤ 1/L guarantees decrease); too small is slow, too large overshoots. On an elongated bowl "
              "(ill-conditioned: curvatures of very different size) the path zig-zags across the steep direction and "
              "crawls along the flat one; momentum and adaptive methods address this. For convex f every local minimum "
              "is global; for non-convex losses gradient descent finds stationary points. SGD uses the gradient of a "
              "random mini-batch: an unbiased, noisy estimate that is far cheaper per step. Level sets on a contour "
              "map are perpendicular to the gradient.",
     "read": [("MML", "Continuous Optimization"), ("CVX", "Unconstrained minimization")],
     "d2l": ["chapter_optimization/gd.md", "chapter_optimization/sgd.md", "chapter_optimization/momentum.md"]},
    {"id": "regression", "title": "Linear regression",
     "keywords": ["linear regression", "regression line", "best fit", "line of best fit", "mean squared error", "mse",
                  "slope and intercept", "r squared", "r²"],
     "facts": "Model ŷ = wᵀx + b. Least squares minimises Σ(y_i − ŷ_i)², the sum of squared residuals (vertical "
              "distances). One feature: slope = cov(x, y)/var(x), intercept = ȳ − slope·x̄, and the line passes through "
              "(x̄, ȳ). In matrix form ŵ = (XᵀX)⁻¹Xᵀy (normal equations), needing independent columns of X. The "
              "residuals sum to zero and are orthogonal to the features. R² = 1 − SS_res/SS_tot is the fraction of "
              "variance explained; with one feature R² = r². Least squares is the maximum-likelihood estimate under "
              "Gaussian noise. Mistakes: outliers have large influence because errors are squared; a good fit says "
              "nothing about causation or about behaviour outside the data range.",
     "read": [("MML", "Linear Regression"), ("PRML", "Linear Models for Regression"), ("ESL", "Linear Methods for Regression")],
     "d2l": ["chapter_linear-regression/linear-regression.md"]},
    {"id": "generalization", "title": "Overfitting, bias–variance and regularization",
     "keywords": ["overfit", "underfit", "bias-variance", "bias–variance", "bias variance", "generalization",
                  "generalisation", "regulariz", "regularis", "weight decay", "ridge", "lasso", "validation",
                  "cross-validation", "cross validation", "k-fold", "capacity", "early stopping", "test error",
                  "training error"],
     "facts": "Training error measures fit to the data used; generalisation error is the expected error on new data "
              "and is estimated on held-out data (validation for choosing hyper-parameters, test only for the final "
              "estimate; k-fold cross-validation averages over k splits). Underfitting: both errors high (model too "
              "simple). Overfitting: training error low, validation error high (the model fits noise). For squared "
              "loss, expected test error = bias² + variance + irreducible noise: more capacity lowers bias and raises "
              "variance, giving a U-shaped test error. Ridge / weight decay minimises loss + λ‖w‖₂²; "
              "ŵ = (XᵀX + λI)⁻¹Xᵀy shrinks the weights towards 0 (never exactly 0), raising bias and lowering variance: "
              "λ = 0 is plain least squares, λ → ∞ gives w → 0. Lasso (L1) sets some weights exactly to 0. More data "
              "reduces variance. Early stopping halts training when the validation loss stops improving.",
     "read": [("ESL", "Linear Methods for Regression; Model Assessment and Selection"),
              ("PRML", "Linear Models for Regression (The Bias-Variance Decomposition)"), ("MML", "When Models Meet Data")],
     "d2l": ["chapter_linear-regression/generalization.md", "chapter_linear-regression/weight-decay.md"]},
    {"id": "classification", "title": "Logistic regression, softmax and decision boundaries",
     "keywords": ["logistic regression", "sigmoid", "softmax", "softplus", "decision boundary", "classifier",
                  "classification", "logit", "linearly separable", "nearest neighbor", "nearest neighbour", "knn", "k-nn"],
     "facts": "Logistic regression models P(y = 1|x) = σ(wᵀx + b) with σ(z) = 1/(1 + e^{−z}) ∈ (0, 1), σ(0) = ½, "
              "σ(−z) = 1 − σ(z), σ′ = σ(1 − σ). The decision boundary P = ½ is the hyperplane wᵀx + b = 0, perpendicular "
              "to w, so the classifier is linear. It is trained by minimising cross-entropy (the negative "
              "log-likelihood), which is convex; there is no closed form. Softmax generalises to K classes: "
              "p_k = e^{z_k}/Σ_j e^{z_j}, positive and summing to 1; adding a constant to all logits changes nothing. "
              "softplus(z) = log(1 + e^z) is a smooth ReLU whose derivative is σ(z). k-nearest neighbours predicts the "
              "majority label of the k closest training points: small k gives a jagged, high-variance boundary, large "
              "k a smoother, more biased one; it depends on the distance scale of the features.",
     "read": [("PRML", "Linear Models for Classification"), ("ESL", "Prototype Methods and Nearest-Neighbors")],
     "d2l": ["chapter_linear-classification/softmax-regression.md"]},
    {"id": "kmeans", "title": "k-means clustering",
     "keywords": ["k-means", "kmeans", "k means", "clustering", "cluster", "centroid"],
     "facts": "k-means partitions points into k clusters to minimise the within-cluster sum of squared distances "
              "J = Σ_i ‖x_i − μ_{c(i)}‖². Lloyd's algorithm alternates: (1) assign each point to its nearest centre; "
              "(2) move each centre to the mean of its points. Neither step increases J, so it converges in finitely "
              "many steps, but only to a local minimum that depends on the initial centres (hence several restarts or "
              "k-means++ seeding). k is chosen by the user; J always decreases as k grows, so it cannot be chosen by "
              "minimising J. Clusters are separated by straight boundaries (Voronoi cells), so k-means suits compact, "
              "similarly sized, roughly spherical clusters; it is sensitive to feature scale and outliers.",
     "read": [("PRML", "Mixture Models and EM (K-means Clustering)"), ("VMLS", "Clustering"), ("ESL", "Unsupervised Learning")],
     "d2l": []},
    {"id": "neural-net", "title": "Neural networks and backpropagation",
     "keywords": ["neural network", "perceptron", "mlp", "hidden layer", "activation function", "relu", "tanh",
                  "backpropagation", "backprop", "forward pass", "forward propagation", "chain rule", "vanishing gradient"],
     "facts": "A layer computes h = φ(Wx + b): an affine map followed by a nonlinearity φ applied element-wise "
              "(ReLU max(0, z), sigmoid, tanh). Without φ, stacked layers collapse into one linear map, so depth adds "
              "nothing; with it the network can represent nonlinear functions (universal approximation with enough "
              "hidden units). The forward pass evaluates layer by layer and stores the intermediate values; "
              "backpropagation applies the chain rule from the loss backwards, δ_l = (W_{l+1}ᵀδ_{l+1}) ∘ φ′(z_l), "
              "∂L/∂W_l = δ_l·h_{l−1}ᵀ, giving all gradients at about the cost of one forward pass. Sigmoid and tanh "
              "saturate (derivative near 0), which shrinks gradients in deep networks; ReLU has derivative 1 for "
              "positive inputs. Weights must be initialised randomly: equal weights keep the units identical.",
     "read": [("PRML", "Neural Networks"), ("MML", "Vector Calculus (Backpropagation and Automatic Differentiation)")],
     "d2l": ["chapter_multilayer-perceptrons/mlp.md", "chapter_multilayer-perceptrons/backprop.md"]},
    {"id": "attention", "title": "Attention",
     "keywords": ["attention", "transformer", "self-attention", "query", "key and value", "scaled dot"],
     "facts": "Scaled dot-product attention: Attention(Q, K, V) = softmax(QKᵀ/√d_k)·V. Each query is compared with "
              "every key by a dot product; the softmax turns the scores of one query into weights that are positive "
              "and sum to 1; the output is the weighted average of the values. Dividing by √d_k keeps the scores from "
              "growing with the dimension, which would saturate the softmax. In self-attention Q, K and V are linear "
              "maps of the same sequence, so every position can read from every other; it is permutation-equivariant, "
              "hence positional encodings. The matrix of weights (one row per query) is what an attention heatmap shows.",
     "read": [("D2L", "Attention Mechanisms and Transformers")],
     "d2l": ["chapter_attention-mechanisms-and-transformers/attention-scoring-functions.md",
             "chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.md"]},
    {"id": "calculus", "title": "Derivatives, gradients and the chain rule",
     "keywords": ["derivative", "partial derivative", "jacobian", "hessian", "taylor", "tangent line", "slope of",
                  "integral", "area under"],
     "facts": "f′(x) = lim_{h→0} (f(x + h) − f(x))/h is the slope of the tangent; f(x + h) ≈ f(x) + f′(x)h. For "
              "f: ℝⁿ → ℝ the gradient ∇f collects the partial derivatives, points in the direction of steepest ascent "
              "and has length equal to that slope; the directional derivative along a unit vector u is ∇fᵀu. Chain "
              "rule: (f∘g)′(x) = f′(g(x))g′(x); for vector functions the Jacobians multiply. The Hessian (matrix of "
              "second partials) describes curvature: at a stationary point, positive definite ⇒ local minimum, "
              "negative definite ⇒ local maximum, indefinite ⇒ saddle. Second-order Taylor: "
              "f(x + d) ≈ f(x) + ∇fᵀd + ½dᵀHd. The definite integral ∫_a^b f is the signed area under f; "
              "differentiation and integration are inverse operations (fundamental theorem of calculus).",
     "read": [("MML", "Vector Calculus")],
     "d2l": [_APX + "single-variable-calculus.md", _APX + "multivariable-calculus.md", "chapter_preliminaries/calculus.md"]},
]
