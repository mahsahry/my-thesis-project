
This is the correct way to render algorithms on GitHub.

---

# ✅ **Do you also want the SSSTR algorithm?**
Here is the GitHub-ready version (optional):

```markdown
## Algorithm 2 — Semi-Supervised Self-Training (SSSTR)

```text
Input:
    L — labeled dataset
    U — unlabeled dataset
    C — base classifier
    h — number of high-confidence samples to add each iteration

Output:
    Expanded labeled dataset L

Procedure:
    1. Repeat until U is empty:
           a. Train classifier C on L
           b. Predict labels for all samples in U
           c. Rank U by descending prediction confidence
           d. Select the top h samples
           e. Move selected samples from U to L
           f. Optionally apply resampling (SMOTE / CBUTE) to L
    2. Return L
