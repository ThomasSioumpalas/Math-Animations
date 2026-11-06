# Math Animations

Manim animations that explain linear algebra and how it is used in machine learning,
following chapter 2 of *Deep Learning* (Goodfellow, Bengio, Courville).
No LaTeX needed: all captions use plain text.

## Setup (Windows / PowerShell, with uv)

```powershell
uv python install 3.12
uv sync
uv run manim checkhealth
```

## Run

```powershell
uv run manim -pql -a column_space.py   # all scenes in a file (-pql = preview, low quality)
uv run manim -pqh inverse.py UndoA     # one scene in HD
```

Videos are written to `media/videos/`.

## Scenes

| File | Scenes | Topic |
|---|---|---|
| `column_space.py` | Combination, FullRank, Singular, ThreeByTwo | Column space, span, linear dependence (§2.4) |
| `inverse.py` | UndoA, SolveAxB, InverseColumns, AreaAndDet | How the inverse A⁻¹ works, determinant (§2.3) |
| `norms.py` | LpNorms, UnitBalls, NormRules, SquaredVsL1, CountingIsNotNorm, Frobenius, DotProductAngle | Norms and the dot product (§2.5) |
| `ml_students.py` | S1_StudentAsVector … S5_TrainingIsANorm | Why ML needs all of this: predicting student grades with X w = y and least squares |
