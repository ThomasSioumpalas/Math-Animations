// Ready-made prompts shown under the composer, grouped into collections and levels.
// Every prompt is self-contained: it names the data, the steps and the visuals, so the
// result can be checked by hand. Numbers quoted as results were verified with numpy.

export type Level = 'Beginner' | 'Intermediate' | 'Advanced'

export type Example = {
  title: string
  level: Level
  minutes: string
  summary: string
  prompt: string
}

export type ExampleGroup = { name: string; hint: string; items: Example[] }
export type Collection = { id: string; name: string; groups: ExampleGroup[] }

const BEGINNER_NOTE = 'Explain every step for a total beginner, with captions.'

export const COLLECTIONS: Collection[] = [
  // ------------------------------------------------------------------ linear algebra
  {
    id: 'linalg',
    name: 'Linear algebra',
    groups: [
      {
        name: 'Core concepts',
        hint: 'One idea per video, with small numbers you can check by hand.',
        items: [
          { title: 'Matrix × vector = mixing columns', level: 'Beginner', minutes: '~2 min',
            summary: 'Ax computed as row·x dot products and as a combination of columns.',
            prompt: `Explain the matrix-vector product with A = [[2, 1], [1, 3]] and x = [3, -1]. First compute Ax entry by entry as dot products of each row of A with x. Then show the same result geometrically: on a grid, draw column 1 of A scaled by 3 and column 2 scaled by -1, place them tip to tail, and show they land exactly on Ax = [5, 0]. Conclude that Ax is a linear combination of the columns of A, weighted by the entries of x. ${BEGINNER_NOTE}` },
          { title: 'Matrix multiplication, entry by entry', level: 'Beginner', minutes: '~2–3 min',
            summary: 'Shapes, every row·column sum, and why AB ≠ BA.',
            prompt: `Explain matrix multiplication with A = [[1, 2, 0], [0, 1, 3]] (2×3) and B = [[2, 1], [1, 0], [4, 1]] (3×2). Start with the shape rule (m×n times n×p gives m×p) and why the inner sizes must match. Compute every entry of C = AB as the dot product of a row of A with a column of B, showing each product term. Then show that BA has a different shape (3×3), so matrix multiplication is not commutative. Finally contrast it with the element-wise (Hadamard) product on two 2×2 matrices. ${BEGINNER_NOTE}` },
          { title: 'Transpose, shapes and broadcasting', level: 'Beginner', minutes: '~1.5 min',
            summary: 'Mirror across the diagonal; adding a vector to every row.',
            prompt: `Explain the transpose using A = [[1, 2], [3, 4], [5, 6]]: draw its main diagonal and animate every entry (i, j) flying to position (j, i), so the 3×2 matrix becomes 2×3. Show that a column vector becomes a row vector and that (AB)ᵀ = BᵀAᵀ with a small 2×2 example. Then explain broadcasting: add the vector b = [10, 20] to every row of A and show the result. ${BEGINNER_NOTE}` },
          { title: 'Inverse: undoing a transformation', level: 'Beginner', minutes: '~2 min',
            summary: 'Transform a grid with A, undo it with A⁻¹, solve Ax = b.',
            prompt: `Explain the inverse matrix with A = [[2, 1], [1, 1]]. Show a grid with the basis vectors being transformed by A, then transformed back by A⁻¹ = [[1, -1], [-1, 2]] to the original grid. Verify A⁻¹A = I by multiplying the matrices entry by entry. Then solve Ax = b for b = [5, 3] step by step: Ax = b, A⁻¹Ax = A⁻¹b, Ix = A⁻¹b, and compute x = [2, 1]. Check it by computing A·x. ${BEGINNER_NOTE}` },
          { title: 'Determinant = area scaling', level: 'Beginner', minutes: '~1.5 min',
            summary: 'The unit square grows 6×; a singular matrix flattens the plane.',
            prompt: `Explain the determinant as area scaling. Transform the unit square with A = [[3, 1], [0, 2]] and show its area becomes 6, which equals det(A) = 3·2 − 1·0. Then apply the singular matrix [[1, 2], [2, 4]] to a grid and show the whole plane collapses onto a line, so the area is 0 and det = 0. Mention that det = 1 means the transformation preserves area. ${BEGINNER_NOTE}` },
          { title: 'Norms: how long is a vector?', level: 'Intermediate', minutes: '~2–3 min',
            summary: 'L1, L2 and L∞ of the same vector, their unit balls, and the angle formula.',
            prompt: `Explain vector norms with x = [3, -4]. Compute ‖x‖₁ = |3| + |-4| = 7, ‖x‖₂ = √(3² + 4²) = 5 and ‖x‖∞ = max(|3|, |4|) = 4, showing each calculation. Draw the unit "circles" of each norm on the same axes: a diamond for L1, a circle for L2 and a square for L∞. Show the squared L2 norm as xᵀx = 25. Explain the three rules every norm must satisfy (zero only at 0, triangle inequality, scaling). Finally take y = [4, 3], compute xᵀy = 0 and use xᵀy = ‖x‖‖y‖cos θ to show the vectors are perpendicular. ${BEGINNER_NOTE}` },
          { title: 'Eigenvectors: directions that do not turn', level: 'Intermediate', minutes: '~2–3 min',
            summary: 'Most vectors rotate under A, eigenvectors only stretch.',
            prompt: `Explain eigenvectors with A = [[2, 1], [1, 2]]. Draw about eight vectors around a circle, apply A, and show that most of them change direction while the vectors along [1, 1] and [1, -1] stay on their own line. Compute A·[1, 1] = [3, 3] = 3·[1, 1] and A·[1, -1] = [1, -1] = 1·[1, -1], so the eigenvalues are 3 and 1. Explain Av = λv, why any rescaled eigenvector is still an eigenvector (so we use unit length), and that a zero eigenvalue would mean the matrix is singular. ${BEGINNER_NOTE}` },
        ],
      },
      {
        name: 'Connected ideas',
        hint: 'Several related ideas in one video, each building on the previous one.',
        items: [
          { title: 'When can we solve Ax = b?', level: 'Intermediate', minutes: '~3–4 min',
            summary: 'Linear combinations, span, column space, dependence and when an inverse exists.',
            prompt: `Explain when the system Ax = b has a solution, using the idea that Ax is a combination of the columns of A. Case 1: A = [[2, 1], [1, 3]] has independent columns, they span the whole plane, so every b has exactly one solution (show the solution for b = [5, 0]). Case 2: A = [[1, 2], [1, 2]] has identical columns: the span is only a line, so b = [3, 3] has infinitely many solutions while b = [1, 0] has none. Case 3: a 3×2 matrix whose two columns span a plane inside 3-D space: only targets on that plane are reachable. Conclude with the rule: an inverse exists only for a square matrix whose columns are linearly independent (otherwise it is called singular). ${BEGINNER_NOTE}` },
          { title: 'Special matrices toolkit', level: 'Intermediate', minutes: '~3 min',
            summary: 'Diagonal, symmetric and orthogonal matrices, and why each one is cheap.',
            prompt: `Explain the special matrices that make computations cheap. Diagonal: D = diag([2, 0.5]) just scales each coordinate (show a grid stretching), and its inverse is diag([1/2, 2]). Symmetric: a distance matrix between three points A, B, C is equal to its own transpose (show the mirror). Orthonormal vectors: unit length and perpendicular (dot product 0). Orthogonal matrix: a 45° rotation Q, show QᵀQ = I entry by entry, that its inverse is just its transpose, and that it keeps every vector's length (show a vector of length 5 before and after). ${BEGINNER_NOTE}` },
          { title: 'Eigendecomposition: A = QΛQᵀ', level: 'Advanced', minutes: '~3–4 min',
            summary: 'Unit circle → ellipse, as rotate, stretch, rotate back.',
            prompt: `Explain the eigendecomposition of the symmetric matrix A = [[2, 1], [1, 2]]. Find its eigenvalues 3 and 1 with unit eigenvectors along [1, 1] and [1, -1]. Draw the unit circle and show that A turns it into an ellipse whose axes point along the eigenvectors with lengths 3 and 1. Then animate the same result as three moves: Qᵀ rotates the eigenvectors onto the axes, Λ = diag(3, 1) stretches, Q rotates back. Verify A = QΛQᵀ by multiplying the matrices. End with what the eigenvalues tell us: det(A) = 3·1 = 3 (product), trace = 3 + 1 = 4 (sum), and all eigenvalues positive means positive definite, so xᵀAx > 0 for every nonzero x. ${BEGINNER_NOTE}` },
          { title: '3-D transformations and volume', level: 'Advanced', minutes: '~3 min',
            summary: 'A 3×3 matrix moving all of space; the unit cube’s volume is the determinant.',
            prompt: `Explain 3x3 matrices as transformations of 3-D space. Show 3-D space with the basis vectors î, ĵ, k̂ and the unit cube. Apply A = [[2, 0, 0], [0, 1, 1], [0, 0, 1]] and show that each column of A is where one basis vector lands, and that the cube becomes a slanted box whose volume is det(A) = 2. Turn the camera around the result so the depth is clear. Then apply B = [[1, 0, 1], [0, 1, 1], [0, 0, 0]] to a fresh space and show all of space getting squished onto a plane, so det(B) = 0 and B has no inverse. ${BEGINNER_NOTE}` },
          { title: 'SVD: rotate, stretch, rotate', level: 'Advanced', minutes: '~3–4 min',
            summary: 'Any matrix, even non-symmetric, is a rotation-scaling-rotation.',
            prompt: `Explain the singular value decomposition A = UDVᵀ with A = [[3, 0], [4, 5]]. Show the unit circle mapped by A to an ellipse. Decompose the motion into Vᵀ (a rotation), D (stretching by the singular values 3√5 ≈ 6.71 and √5 ≈ 2.24) and U (another rotation), animating each step. Show that the singular values are the square roots of the eigenvalues of AᵀA = [[25, 20], [20, 25]] (eigenvalues 45 and 5). Explain why SVD exists for every matrix, even non-square ones, while eigendecomposition does not. ${BEGINNER_NOTE}` },
          { title: 'Pseudoinverse and least squares', level: 'Advanced', minutes: '~3–4 min',
            summary: 'Fit a line through 3 points that no line passes through exactly.',
            prompt: `Explain how to "solve" a system with no exact solution. Fit a line y = w₀ + w₁x to the points (0, 1), (1, 2), (2, 2): write it as Xw = y with X = [[1, 0], [1, 1], [1, 2]] and y = [1, 2, 2], and show that no line hits all three points. Compute XᵀX and Xᵀy, then w = (XᵀX)⁻¹Xᵀy ≈ [1.17, 0.5], and draw the fitted line with the three vertical error segments. Explain that this w makes ‖Xw − y‖ as small as possible, that it is what the pseudoinverse X⁺ = VD⁺Uᵀ gives, and that for systems with many solutions the pseudoinverse picks the one with the smallest norm. ${BEGINNER_NOTE}` },
          { title: 'Trace and Frobenius norm tricks', level: 'Intermediate', minutes: '~2 min',
            summary: 'Sum of the diagonal, ‖A‖F = √Tr(AAᵀ), and cycling products.',
            prompt: `Explain the trace operator. With A = [[1, 2], [3, 4]] show Tr(A) = 1 + 4 = 5 and Tr(Aᵀ) = Tr(A). Compute the Frobenius norm ‖A‖F = √(1² + 2² + 3² + 4²) = √30 ≈ 5.48 and show it equals √Tr(AAᵀ) by computing AAᵀ. Then take a 2×3 matrix and a 3×2 matrix and show Tr(AB) = Tr(BA) even though AB is 2×2 and BA is 3×3. ${BEGINNER_NOTE}` },
        ],
      },
      {
        name: 'Big picture',
        hint: 'Longer videos that tie the whole topic together.',
        items: [
          { title: 'One matrix, every concept', level: 'Intermediate', minutes: '~4–5 min',
            summary: 'A single 2×2 matrix viewed through every idea of the topic.',
            prompt: `Use the single matrix A = [[2, 1], [1, 2]] to tour all of basic linear algebra, one short step each: (1) as a table of numbers: shape, transpose, it is symmetric; (2) as a transformation of the grid, with Ax as a combination of its columns; (3) its determinant 3 as area scaling of the unit square; (4) its inverse undoing the transformation; (5) the norms of its columns; (6) its eigenvectors [1, 1] and [1, -1] with eigenvalues 3 and 1, and the unit circle becoming an ellipse; (7) A = QΛQᵀ; (8) trace = 4 = sum of eigenvalues, det = 3 = product. Finish with a one-screen summary map connecting the ideas. ${BEGINNER_NOTE}` },
          { title: 'PCA from scratch', level: 'Advanced', minutes: '~4–5 min',
            summary: 'Compress 2-D points to 1-D using the top eigenvector of XᵀX.',
            prompt: `Explain principal components analysis as lossy compression. Use the centered points (-3, -2), (-2, -1), (-1, -1), (1, 0), (2, 2), (3, 2) as rows of X. Goal: store one number c per point instead of two, decode with x ≈ d·c using a unit vector d. Show that the best code is c = dᵀx (projection) and the reconstruction is ddᵀx. Rotate a candidate line through the origin and show the total squared reconstruction error changing, with the error segments drawn. Then show the best direction is the eigenvector of XᵀX with the largest eigenvalue: compute XᵀX, its eigenvalues (≈ 41.25 and 0.75) and the top eigenvector, and show the points projected onto it. ${BEGINNER_NOTE}` },
          { title: 'Tour part 1: objects and operations', level: 'Beginner', minutes: '~4 min',
            summary: 'Scalars to tensors, transpose, broadcasting, products, linear systems.',
            prompt: `First part of a linear algebra tour. Introduce scalars, vectors (points in space), matrices (rows and columns, indexing A₂,₁) and tensors (a stack of matrices), each with a small example. Show the transpose as a mirror across the diagonal, matrix addition, scalar multiplication and broadcasting a vector to every row. Explain the matrix product with a 2×3 times 3×2 example, entry by entry, and the dot product as a special case. End by writing a system of two linear equations as Ax = b and showing each row of A as one equation. ${BEGINNER_NOTE}` },
          { title: 'Tour part 2: solving and measuring', level: 'Intermediate', minutes: '~4–5 min',
            summary: 'Identity, inverse, span, dependence, norms and special matrices.',
            prompt: `Second part of a linear algebra tour. Show the identity matrix leaving vectors unchanged, then the inverse solving Ax = b for A = [[2, 1], [1, 1]] and b = [5, 3]. Explain span and linear dependence with a pair of independent columns versus a pair of identical columns, and why a singular matrix has no inverse. Measure vectors with the L1, L2 and L∞ norms of [3, -4] and draw their unit balls. Finish with diagonal, symmetric and orthogonal matrices and why each is cheap to work with. ${BEGINNER_NOTE}` },
          { title: 'Tour part 3: decompositions', level: 'Advanced', minutes: '~5 min',
            summary: 'Eigendecomposition, SVD, pseudoinverse, trace, determinant.',
            prompt: `Third part of a linear algebra tour, about breaking matrices into parts (like factoring 12 = 2·2·3). Eigendecomposition of [[2, 1], [1, 2]]: eigenvectors, eigenvalues 3 and 1, unit circle to ellipse, A = QΛQᵀ, positive definiteness. SVD of [[3, 0], [4, 5]] as rotate-stretch-rotate, singular values as square roots of the eigenvalues of AᵀA. The pseudoinverse giving the least-squares line through (0, 1), (1, 2), (2, 2). The trace as the sum of eigenvalues and the determinant as their product and as volume scaling. End with a summary of which decomposition to use when. ${BEGINNER_NOTE}` },
        ],
      },
    ],
  },

  // ------------------------------------------------------------------ probability
  {
    id: 'prob',
    name: 'Probability & information',
    groups: [
      {
        name: 'Core concepts',
        hint: 'One idea per video, with small numbers you can check by hand.',
        items: [
          { title: 'Probability mass vs density', level: 'Beginner', minutes: '~2 min',
            summary: 'Bars that sum to 1 vs a curve whose area is 1.',
            prompt: `Explain the difference between a probability mass function and a probability density function. PMF: a fair die, six bars of height 1/6 that add up to 1; show P(x = 3) = 1/6 and that every bar is between 0 and 1. PDF: the uniform density on [0, 2] has height 0.5 and the area under it is 1; the probability of an interval is the area over it, e.g. P(0.5 ≤ x ≤ 1.5) = 0.5. Show that a density can be bigger than 1 (uniform on [0, 0.5] has height 2) and that a single point has probability 0. ${BEGINNER_NOTE}` },
          { title: 'Marginal probability: summing out', level: 'Beginner', minutes: '~2 min',
            summary: 'Row and column sums of a joint table written in the margins.',
            prompt: `Explain marginal probability with a joint table of weather and how a person commutes: P(sunny, walk) = 0.3, P(sunny, bus) = 0.2, P(rainy, walk) = 0.1, P(rainy, bus) = 0.4. Show the table as a 2×2 grid and check it sums to 1. Sum each row and write the result in the margin: P(sunny) = 0.5, P(rainy) = 0.5. Sum each column: P(walk) = 0.4, P(bus) = 0.6. Explain that this is why it is called a "marginal" probability, and that for continuous variables the sum becomes an integral. ${BEGINNER_NOTE}` },
          { title: 'Conditional probability and the chain rule', level: 'Beginner', minutes: '~2–3 min',
            summary: 'Zooming into one row of the table; splitting a joint into a chain.',
            prompt: `Explain conditional probability with the weather/commute table P(sunny, walk) = 0.3, P(sunny, bus) = 0.2, P(rainy, walk) = 0.1, P(rainy, bus) = 0.4. "Given rainy" means zooming into the rainy row and rescaling it so it sums to 1: P(walk | rainy) = 0.1 / 0.5 = 0.2, compared with P(walk | sunny) = 0.6. Show the formula P(y | x) = P(x, y) / P(x). Then explain the chain rule P(a, b, c) = P(a | b, c)·P(b | c)·P(c) with a three-level probability tree. Mention that conditioning is not the same as intervening (speaking German vs being taught German). ${BEGINNER_NOTE}` },
          { title: 'Expectation and variance of a die', level: 'Beginner', minutes: '~2 min',
            summary: 'Weighted average 3.5, squared deviations, standard deviation.',
            prompt: `Explain expectation and variance with a fair die. Compute E[x] = (1 + 2 + 3 + 4 + 5 + 6)·1/6 = 3.5 as a probability-weighted average and show it as the balance point of the six bars. Compute the deviations from 3.5, square them, and average them to get Var(x) = 35/12 ≈ 2.92, then σ ≈ 1.71. Show linearity of expectation: E[2x + 1] = 2·3.5 + 1 = 8. ${BEGINNER_NOTE}` },
          { title: 'Bayes’ rule: a medical test', level: 'Beginner', minutes: '~2–3 min',
            summary: 'Why a positive test can still mean only a 15% chance of disease.',
            prompt: `Explain Bayes' rule with a medical test. 1% of people have a disease; the test detects it 90% of the time; it wrongly says positive for 5% of healthy people. Visualize 10,000 people as one big square split into proportional rectangles (an area diagram): 100 sick (90 test positive) and 9,900 healthy (495 test positive), with each group's count labeled. Of the 585 positives only 90 are sick, so P(sick | positive) = 90 / 585 ≈ 15.4%. Then write the same reasoning as P(x | y) = P(x)·P(y | x) / P(y), with P(y) computed by summing over both groups. ${BEGINNER_NOTE}` },
          { title: 'Sigmoid and softplus', level: 'Intermediate', minutes: '~2 min',
            summary: 'Squashing numbers into probabilities; a smooth ReLU.',
            prompt: `Explain the logistic sigmoid σ(x) = 1 / (1 + e⁻ˣ) and the softplus ζ(x) = log(1 + eˣ). Plot σ from -10 to 10: it maps any number into (0, 1), so it can produce a probability; show σ(0) = 0.5 and the flat saturated regions. Show σ'(x) = σ(x)(1 − σ(x)) is at most 0.25, and 1 − σ(x) = σ(−x). Plot softplus next to max(0, x), showing it is a smoothed version with ζ(0) = log 2 ≈ 0.69, that its derivative is the sigmoid, and that ζ(x) − ζ(−x) = x. ${BEGINNER_NOTE}` },
          { title: 'Information and entropy', level: 'Intermediate', minutes: '~2–3 min',
            summary: 'Surprising events carry more information; entropy is the average.',
            prompt: `Explain self-information I(x) = −log P(x): a sunrise (probability ≈ 1) carries almost no information while an eclipse (rare) carries a lot, and two independent coin flips carry twice the information of one. Then explain Shannon entropy as the average information. For a coin with P(heads) = p, plot H(p) = −p log p − (1 − p) log(1 − p) for p from 0 to 1: it is 0 for a certain outcome and largest at p = 0.5 (log 2 ≈ 0.69 nats = 1 bit). Compute it for p = 0.9 as a worked example. ${BEGINNER_NOTE}` },
        ],
      },
      {
        name: 'Connected ideas',
        hint: 'Several related ideas in one video, each building on the previous one.',
        items: [
          { title: 'From a joint table to Bayes', level: 'Intermediate', minutes: '~3–4 min',
            summary: 'Marginals, conditionals, chain rule, independence and Bayes on one table.',
            prompt: `Use one joint table to connect the core rules of probability: P(sunny, walk) = 0.3, P(sunny, bus) = 0.2, P(rainy, walk) = 0.1, P(rainy, bus) = 0.4. (1) Marginals by summing rows and columns. (2) Conditionals by rescaling a row: P(walk | rainy) = 0.2. (3) The product rule: P(rainy, walk) = P(walk | rainy)·P(rainy). (4) Independence test: compare P(sunny, walk) = 0.3 with P(sunny)·P(walk) = 0.2, so the variables are dependent. (5) Bayes' rule: compute P(rainy | walk) = 0.1 / 0.4 = 0.25 from the other direction. ${BEGINNER_NOTE}` },
          { title: 'Covariance, correlation and independence', level: 'Intermediate', minutes: '~3 min',
            summary: 'Linked variables, the covariance matrix, and dependence with zero covariance.',
            prompt: `Explain covariance with three small scatter plots: points rising together (positive), one rising while the other falls (negative), and no pattern (near zero). Compute Cov(x, y) for 4 points of your choice step by step as the average product of deviations from the means. Show the 2×2 covariance matrix, with variances on the diagonal, and correlation as covariance rescaled to [-1, 1]. Then show that zero covariance does not mean independence: take x uniform on [-1, 1] and y = s·x where s is randomly +1 or −1; the points form an X shape, y clearly depends on x, but Cov(x, y) = 0. ${BEGINNER_NOTE}` },
          { title: 'The distribution zoo', level: 'Intermediate', minutes: '~4 min',
            summary: 'Bernoulli, categorical, Gaussian, Laplace, Dirac, empirical, mixtures.',
            prompt: `Tour the most common probability distributions, each with its formula and a plot. Bernoulli with φ = 0.3: P(x) = φˣ(1 − φ)¹⁻ˣ, mean φ, variance φ(1 − φ) = 0.21. Categorical over 4 classes. Gaussian N(μ, σ²): bell curve with peak at μ and inflection points at μ ± σ; vary μ and σ to show their effect; mention the precision β = 1/σ². Exponential and Laplace: sharp peaks at a point. Dirac delta and the empirical distribution that puts mass 1/m on each of m data points. Finally a mixture of two Gaussians, choosing a component first and then sampling from it. ${BEGINNER_NOTE}` },
          { title: 'Gaussians in 2-D and covariance shapes', level: 'Advanced', minutes: '~3–4 min',
            summary: 'Isotropic, diagonal and full covariance as circles and ellipses.',
            prompt: `Explain the multivariate Gaussian in 2-D. Sample and plot points from three Gaussians: isotropic covariance [[1, 0], [0, 1]] (round cloud), diagonal covariance [[1, 0], [0, 3]] (stretched along one axis) and full covariance [[2, 1.5], [1.5, 2]] (tilted ellipse). For the full one, compute the eigenvectors and eigenvalues of the covariance matrix and show they are exactly the directions and squared lengths of the ellipse axes. Finish by showing the three clouds together as a Gaussian mixture model, and the idea of a hidden (latent) component label. ${BEGINNER_NOTE}` },
          { title: 'Entropy, cross-entropy and KL divergence', level: 'Advanced', minutes: '~4 min',
            summary: 'How many extra bits a wrong distribution costs; why KL is asymmetric.',
            prompt: `Explain how to compare two distributions over 3 outcomes: P = [0.7, 0.2, 0.1] (the truth) and Q = [1/3, 1/3, 1/3] (a model). Using log base 2 (bits), compute the entropy H(P), the cross-entropy H(P, Q) = −Σ P log Q and the KL divergence D(P‖Q) = H(P, Q) − H(P), showing every term with bar charts. Then compute D(Q‖P) and show it differs, so KL is not a true distance. Finish with the bimodal picture: fitting one Gaussian to a mixture of two Gaussians spreads over both peaks when minimizing D(p‖q), but picks a single peak when minimizing D(q‖p). ${BEGINNER_NOTE}` },
          { title: 'Change of variables: densities stretch', level: 'Advanced', minutes: '~3 min',
            summary: 'Why p_y(y) needs the |dx/dy| factor, and the Jacobian determinant.',
            prompt: `Explain the change-of-variables rule for densities. Let x be uniform on [0, 1] and y = x / 2. Show the tempting wrong answer p_y(y) = p_x(2y): a rectangle of height 1 on [0, 0.5] with area only 0.5, which is not a valid density. Fix it with p_y(y) = p_x(g⁻¹(y))·|dx/dy| = 2: squeezing the rectangle horizontally must make it taller so the area stays 1. Then generalize to 2-D: a transformation scales small areas by |det J|, the Jacobian determinant, which connects this rule to the determinant as area scaling. ${BEGINNER_NOTE}` },
          { title: 'Graphs of probabilistic models', level: 'Advanced', minutes: '~3–4 min',
            summary: 'Directed and undirected graphs as factorizations of a distribution.',
            prompt: `Explain structured probabilistic models. Draw a directed graph with nodes a, b, c, d, e and edges a→b, a→c, b→c, b→d, c→e, and build its factorization p(a)p(b | a)p(c | a, b)p(d | b)p(e | c) one factor per node. For binary variables, count parameters: the full joint table needs 2⁵ − 1 = 31 numbers, the factorized model needs 1 + 2 + 4 + 2 + 2 = 11. Then draw an undirected graph with cliques {a, b, c}, {b, d}, {c, e}, write p = (1/Z)·φ₁(a, b, c)·φ₂(b, d)·φ₃(c, e) and explain the normalizing constant Z. ${BEGINNER_NOTE}` },
        ],
      },
      {
        name: 'Big picture',
        hint: 'Longer videos that tie the whole topic together.',
        items: [
          { title: 'One coin, all of probability', level: 'Intermediate', minutes: '~5 min',
            summary: 'A biased coin carried through distributions, information and Bayesian updating.',
            prompt: `Use a single biased coin with P(heads) = 0.7 to tour probability and information theory, one short step each: (1) as a Bernoulli distribution with mean 0.7 and variance 0.21; (2) the frequentist view (long-run frequency of many simulated flips) vs the Bayesian view (degree of belief); (3) a sigmoid turning a score into the probability 0.7, and the logit log(0.7/0.3) ≈ 0.85 turning it back; (4) the information of seeing heads vs tails and the entropy of the coin; (5) the cross-entropy and KL divergence when a model wrongly assumes 0.5; (6) Bayes' rule: start with a uniform belief over P(heads) ∈ {0.1, 0.2, …, 0.9}, observe heads, heads, tails, and update the bar chart of beliefs after each flip. End with a summary map. ${BEGINNER_NOTE}` },
          { title: 'Tour part 1: describing uncertainty', level: 'Beginner', minutes: '~4–5 min',
            summary: 'Why probability, random variables, PMF/PDF, marginal, conditional, independence.',
            prompt: `First part of a probability tour. Start with the three sources of uncertainty (randomness, incomplete observation, incomplete modeling) and frequentist vs Bayesian probability. Introduce random variables, a PMF (a die) and a PDF (a uniform density, area = 1). Using the table P(sunny, walk) = 0.3, P(sunny, bus) = 0.2, P(rainy, walk) = 0.1, P(rainy, bus) = 0.4, show marginal probabilities, conditional probabilities, the chain rule and an independence check. ${BEGINNER_NOTE}` },
          { title: 'Tour part 2: summarizing and modeling', level: 'Intermediate', minutes: '~5 min',
            summary: 'Expectation, variance, covariance, common distributions, mixtures.',
            prompt: `Second part of a probability tour. Expectation and variance of a fair die (3.5 and ≈ 2.92), linearity of expectation, covariance and correlation with a small scatter plot and the covariance matrix. Then the common distributions: Bernoulli, categorical, Gaussian (μ and σ, inflection points at μ ± σ), exponential and Laplace, Dirac and empirical, and a Gaussian mixture with a hidden component label. Finish with the sigmoid and softplus as functions that produce valid parameters for these distributions. ${BEGINNER_NOTE}` },
          { title: 'Tour part 3: reasoning and information', level: 'Advanced', minutes: '~5 min',
            summary: 'Bayes, change of variables, entropy, KL, cross-entropy, graphical models.',
            prompt: `Third part of a probability tour. Bayes' rule with a medical test (1% prevalence, 90% detection, 5% false positives → 15.4%). The change-of-variables rule for densities (y = x/2 doubles the density). Self-information, entropy of a coin (maximal at p = 0.5), cross-entropy and KL divergence between P = [0.7, 0.2, 0.1] and Q = [1/3, 1/3, 1/3], and why KL is asymmetric. End with a directed graphical model that factorizes a joint distribution and saves parameters (31 numbers down to 11). ${BEGINNER_NOTE}` },
        ],
      },
    ],
  },

  // ------------------------------------------------------------------ statistics
  {
    id: 'stats',
    name: 'Statistics & learning',
    groups: [
      {
        name: 'Core concepts',
        hint: 'One idea per video, with small numbers you can check by hand.',
        items: [
          { title: 'Estimators: guessing from data', level: 'Beginner', minutes: '~2 min',
            summary: 'An estimate is a random variable: new data, new estimate.',
            prompt: `Explain point estimation. A coin has a true but unknown P(heads) = θ = 0.6. Simulate a dataset of 10 flips and estimate θ̂ as the fraction of heads. Repeat with 20 new datasets and collect the estimates as dots on a number line around the true value 0.6, building up a histogram. Explain that θ̂ is a function of random data, so it is itself a random variable, while the true θ is fixed. ${BEGINNER_NOTE}` },
          { title: 'Bias: why divide by m − 1?', level: 'Intermediate', minutes: '~3 min',
            summary: 'The sample variance underestimates; Bessel’s correction fixes it.',
            prompt: `Explain the bias of an estimator with the variance of {2, 4, 9}. Mean = 5, deviations −3, −1, 4, squares 9, 1, 16, sum 26. Dividing by m = 3 gives 8.67, dividing by m − 1 = 2 gives 13. Then simulate many samples of size 3 from a Gaussian with true variance 4, average both estimators over the samples, and show the /m version settles near (m − 1)/m · 4 ≈ 2.67 (bias −σ²/m) while the /(m − 1) version settles near 4. Define bias = E[θ̂] − θ and unbiased estimators. ${BEGINNER_NOTE}` },
          { title: 'Standard error and confidence intervals', level: 'Intermediate', minutes: '~3 min',
            summary: 'How sure is a test-set error? SE = σ/√m and the 95% interval.',
            prompt: `Explain the standard error of the mean, SE = σ/√m. A model makes errors on 12 of 100 test examples: mean error 0.12, σ = √(0.12·0.88) ≈ 0.325, SE ≈ 0.0325, so the 95% confidence interval is 0.12 ± 1.96·0.0325 ≈ [0.056, 0.184]. Show the interval on a number line, then show it shrinking by half with 400 test examples. Finish by comparing two models' intervals: overlapping intervals mean we cannot claim one is better. ${BEGINNER_NOTE}` },
          { title: 'Consistency: more data, closer to the truth', level: 'Intermediate', minutes: '~2 min',
            summary: 'The running mean converges; the "first sample" estimator never does.',
            prompt: `Explain consistency with die rolls. Plot the running mean of 200 simulated rolls converging to the true mean 3.5, and shade a ±0.2 band that the estimate eventually stays inside. Contrast it with the silly estimator "use only the first roll": it is unbiased (its expectation is 3.5) but it never gets closer as more data arrives, so it is not consistent. ${BEGINNER_NOTE}` },
          { title: 'Maximum likelihood for a coin', level: 'Intermediate', minutes: '~3 min',
            summary: 'Plot the likelihood, take logs, find the peak at 5/7.',
            prompt: `Explain maximum likelihood estimation with the flips H, H, T, H, T, H, H (5 heads, 2 tails). Write the likelihood L(θ) = θ⁵(1 − θ)² and plot it for θ from 0 to 1, evaluating it at a few points. Explain why we take the log (products of many small numbers underflow; logs turn products into sums), plot the log-likelihood, and show both peak at the same θ = 5/7 ≈ 0.714, the fraction of heads. ${BEGINNER_NOTE}` },
          { title: 'Train, validation, test and k-fold', level: 'Beginner', minutes: '~2–3 min',
            summary: 'Why we hold data out, and how cross-validation reuses it.',
            prompt: `Explain how we evaluate models honestly. Show a dataset of 20 examples as squares, split into training and test sets, then split training again 80/20 into training and validation for choosing settings like the polynomial degree. Explain why the test set must never be used for choices. Then animate 5-fold cross-validation: the data split into 5 folds, each fold taking a turn as the test set, and the final error as the average of the 5 results. ${BEGINNER_NOTE}` },
        ],
      },
      {
        name: 'Connected ideas',
        hint: 'Several related ideas in one video, each building on the previous one.',
        items: [
          { title: 'Underfitting, overfitting and capacity', level: 'Intermediate', minutes: '~3–4 min',
            summary: 'Degree 1, 2 and 9 polynomials on data from a quadratic; the U-curve.',
            prompt: `Explain underfitting and overfitting. Generate 8 noisy points from a quadratic. Fit a straight line (underfits: misses the curvature), a quadratic (fits well) and a degree-9 polynomial (passes through every point but wiggles wildly between them). Show training error and test error for each, then draw the classic picture: training error falls as capacity grows while test error makes a U shape, with an optimal capacity in between. Explain the generalization gap. ${BEGINNER_NOTE}` },
          { title: 'Bias–variance trade-off', level: 'Intermediate', minutes: '~3–4 min',
            summary: 'MSE = bias² + variance, with dartboards and fitted curves.',
            prompt: `Explain the bias-variance trade-off. Start with four dartboards (low/high bias × low/high variance). Then fit a straight line and a degree-9 polynomial to 10 different noisy datasets from the same quadratic and overlay the fitted curves: the lines are consistently wrong (high bias), the polynomials disagree wildly (high variance). Show MSE = bias² + variance and plot bias, variance and total error against model capacity, with the optimum at the bottom of the U. ${BEGINNER_NOTE}` },
          { title: 'Regularization and weight decay', level: 'Intermediate', minutes: '~3 min',
            summary: 'J(w) = MSE + λ‖w‖²: large, medium and zero λ on a degree-9 fit.',
            prompt: `Explain regularization with weight decay. Fit a degree-9 polynomial to 8 noisy points from a quadratic by minimizing J(w) = MSE + λ·wᵀw. Show three fits: very large λ (almost a flat line, underfits), a medium λ (recovers the right curved shape), and λ near 0 (wild overfitting). Show the size of the weights in each case as bars. Explain that regularization expresses a preference for simpler solutions and aims to reduce test error, not training error. ${BEGINNER_NOTE}` },
          { title: 'Maximum likelihood = cross-entropy = MSE', level: 'Advanced', minutes: '~4 min',
            summary: 'Why least squares is maximum likelihood under Gaussian noise.',
            prompt: `Explain how maximum likelihood connects to the losses used in machine learning. Show the empirical distribution of a small dataset as spikes and a model distribution as a curve; maximizing likelihood equals minimizing the KL divergence (and the cross-entropy) between them. Then do linear regression with Gaussian noise: write log p(y | x) for 5 points, show it equals −m·log σ − (m/2)·log 2π − Σ(ŷ − y)²/(2σ²), and plot both the log-likelihood and the mean squared error as functions of the slope w: they peak and bottom out at exactly the same w. ${BEGINNER_NOTE}` },
          { title: 'Frequentist vs Bayesian estimation', level: 'Advanced', minutes: '~4 min',
            summary: 'One point estimate vs a whole distribution of beliefs.',
            prompt: `Compare frequentist and Bayesian estimation for a coin, after observing H, H, T, H, T, H, H. Frequentist: one point estimate θ̂ = 5/7 ≈ 0.714. Bayesian: start with a uniform prior over θ on a grid of values, multiply by the likelihood after each flip, normalize, and animate the posterior bar chart sharpening. Predict the next flip by averaging over the posterior (≈ 0.667 with a uniform prior) instead of using one value. Show the MAP estimate as the peak of the posterior and how a prior that favors θ = 0.5 pulls it toward 0.5. ${BEGINNER_NOTE}` },
          { title: 'MAP estimation is weight decay', level: 'Advanced', minutes: '~3 min',
            summary: 'A Gaussian prior on the weight shrinks it, exactly like λ‖w‖².',
            prompt: `Explain why MAP estimation with a Gaussian prior equals weight decay. Fit y = w·x through the points (1, 2), (2, 3), (3, 7). Maximum likelihood (least squares) gives w = Σxy / Σx² = 29/14 ≈ 2.07. Add a Gaussian prior centered at 0: log posterior = log likelihood + log prior, and the log prior is a −λw² penalty, so w_MAP = Σxy / (Σx² + λ). Show the fitted line shrinking toward a flat line as λ goes from 0 to 2 (w ≈ 1.81) to 7 (w ≈ 1.38), with the prior and likelihood curves multiplied into the posterior. ${BEGINNER_NOTE}` },
        ],
      },
      {
        name: 'Big picture',
        hint: 'Longer videos that tie the whole topic together.',
        items: [
          { title: 'From data to model, end to end', level: 'Intermediate', minutes: '~5 min',
            summary: 'One small dataset through splitting, fitting, overfitting, regularizing, evaluating.',
            prompt: `Follow one small regression dataset (8 noisy points from a quadratic) through the whole learning pipeline, one short step each: (1) split into training and test points; (2) fit a straight line by least squares, written as Xw = y and solved with (XᵀX)⁻¹Xᵀy; (3) explain it as maximum likelihood with Gaussian noise; (4) increase capacity to a quadratic and then a degree-7 polynomial and watch test error rise again (overfitting); (5) add weight decay λ‖w‖² and explain it as a Gaussian prior (MAP); (6) report the test error with a 95% confidence interval. End with a summary map of the pipeline. ${BEGINNER_NOTE}` },
          { title: 'Tour part 1: estimating things', level: 'Intermediate', minutes: '~5 min',
            summary: 'Point estimates, bias, variance, standard error, MSE, consistency.',
            prompt: `First part of a statistics tour. Point estimators as functions of random data (coin with θ = 0.6, many simulated datasets). Bias with the sample variance of {2, 4, 9} and the m − 1 correction. Variance of an estimator and the standard error σ/√m with a 95% confidence interval for a test error of 0.12 on 100 examples. MSE = bias² + variance. Consistency with the running mean of die rolls versus the first-roll estimator. ${BEGINNER_NOTE}` },
          { title: 'Tour part 2: learning principles', level: 'Advanced', minutes: '~5 min',
            summary: 'Maximum likelihood, KL and MSE, Bayesian posteriors, MAP, capacity.',
            prompt: `Second part of a statistics tour. Maximum likelihood for the coin flips H, H, T, H, T, H, H (peak at 5/7), the log trick, and why maximizing likelihood equals minimizing cross-entropy and, under Gaussian noise, mean squared error. Bayesian updating of a belief over θ and predicting by averaging over the posterior. MAP estimation as likelihood plus prior, and a Gaussian prior as weight decay. Finish with capacity, underfitting and overfitting (the U-shaped test error) and validation sets for choosing hyperparameters. ${BEGINNER_NOTE}` },
        ],
      },
    ],
  },
]
