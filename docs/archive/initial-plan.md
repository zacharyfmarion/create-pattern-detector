# Initial Plan

I’d treat this as: “design a model that outputs a *valid origami graph*, not just a bunch of lines.”

Since you’re restricting to:

* perfectly square sheets
* crease patterns that are *already* globally flat-foldable
* and satisfy local theorems (Kawasaki, Maekawa, etc.)([Wikipedia][1])

you can bake in *a lot* of structure that generic LSD/wireframe methods ignore.

Below is how I’d design a custom system end-to-end.

---

## 0. Target representation

Let’s make the thing we want to predict explicit.

For a given crease pattern, represent it as:

* A square domain: corners at ((0,0), (1,0), (1,1), (0,1)) after rectification.
* A **planar straight-line graph**:

  * Vertices (V = {v_i}_{i=1}^N), (v_i \in [0,1]^2)
  * Undirected edges (E \subseteq V \times V) (creases)
* Optional: mountain / valley label per edge (c_e \in {\text{M},\text{V}})

Local flat-foldability at each interior vertex means:

* Even degree (number of incident creases is even)
* Kawasaki: alternating sum of sector angles around the vertex is 0; equivalently, sum of alternating angles = 180°.([Wikipedia][1])
* Maekawa: (|#M - #V| = 2) (if you care about M/V).([Wikipedia][2])

The network’s job: from an RGB (or grayscale) image of the CP, output this graph.

---

## 1. Data & augmentation (where the priors really pay off)

You can generate *unlimited* synthetic training data:

1. **Procedural CP generator**

   * Randomly sample flat-foldable one-vertex patterns (enforce Kawasaki/Maekawa at the vertex).([origametry.net][3])
   * Stitch them together into multi-vertex patterns while maintaining consistency (or borrow existing CP generators / libraries and adapt).
   * Enforce square boundary and avoid self-intersections.

2. **Vector → raster**

   * Render the graph as high-res line art (e.g., 2048×2048, 1–2 px lines).
   * Keep the *vector* graph as ground truth (G^* = (V^*, E^*)).

3. **Domain-specific augmentations**

   * Random homographies (to simulate off-angle photos).
   * Vary stroke width, contrast, paper texture, wrinkles, light gradients.
   * Add small print artifacts: incomplete ink, smudges, scanner noise.

You can later fine-tune on a smaller set of real scanned CPs with manual labels.

---

## 2. Overall architecture: “image → graph with origami constraints”

I’d use a **two-head architecture**:

1. A high-res **pixel head** to capture fine geometry.
2. A **graph head** that works with candidate vertices & edges and enforces origami structure.

Think of it as:
**Backbone → Pixel features → (A) line field + (B) vertex proposals → Graph network → constrained crease pattern.**

### 2.1 Backbone

* A high-resolution backbone like HRNet or a light ViT/SegFormer variant, configured to keep stride small (e.g. final stride 4 or 2) so you don’t get HAWP’s 128×128 problem.
* Optionally include **coordinate channels** ((x, y)) and a “distance to nearest side of square” channel so the model knows about the square domain explicitely.

### 2.2 Pixel head: crease field + orientation + junction hints

From backbone features, predict:

1. **Crease segmentation**

   * 3-class per pixel: {background, crease, boundary}
   * Trained with cross-entropy + a mild class-balancing (creases are sparse).

2. **Orientation field**

   * For pixels predicted as crease: regress ((\cos\theta, \sin\theta)) of the local crease direction (L2 loss on the unit vector).
   * This makes post-hoc grouping & skeletonization easier (you know local directions).

3. **Junction heatmap (full resolution)**

   * A single-channel heatmap with Gaussian bumps centered at true vertices (V^*).
   * Use a CenterNet-style loss: focal loss + optionally a small local offset vector (dx, dy) from pixel to vertex center for sub-pixel accuracy.

This gives you a dense, fine-grained geometric description *without* ever doing coarse heatmaps like 128×128.

---

## 3. From pixels to a candidate graph (non-learned but geometry-aware)

At inference (and for some losses during training):

1. **Skeletonize the crease mask**

   * Threshold the crease prob map → thin to 1-px skeleton.
   * Assign each skeleton pixel the orientation from the orientation head.

2. **Junction detection**

   * Local maxima in the junction heatmap + offset refinement → vertex candidates (\tilde{V} = {\tilde{v}_i}).
   * Also add skeleton crossing points (pixels with >2 neighbors in skeleton) as candidates.

3. **Edge tracing**

   * Follow skeleton pixels from vertex to vertex along consistent orientation to form **candidate edges** (\tilde{E}).
   * Merge nearly collinear edge fragments.

You now have a “raw” graph (\tilde{G} = (\tilde{V}, \tilde{E})). It will be noisy: extra vertices, missing edges, small gaps, etc. That’s fine—that’s what the next stage is for.

---

## 4. Graph head: refine to a valid, flat-foldable crease pattern

Now put a **graph neural network / transformer** on top of (\tilde{G}):

### 4.1 Node & edge features

For each candidate vertex (i):

* Position: ((x_i, y_i))
* Local image embedding: sample backbone features at that location (bilinear sampling).
* Degree estimate, local skeleton junction type (T, X, …) if you want.

For each candidate edge (i!-!j):

* Geometric features: length, direction, midpoint coord, distance to square boundary, etc.
* Line evidence: integrate crease probabilities/orientation along the candidate edge.

### 4.2 GNN / transformer

Run a few message-passing layers (or a transformer over the set of vertices + edges):

* Node updates aggregate from neighboring edges.
* Edge updates aggregate from endpoints and maybe nearby edges crossing at junctions.

### 4.3 Graph predictions

The graph head outputs:

* A **keep/drop probability for each edge** (p_e).
* (Optionally) a **correction vector** (\Delta v_i) to refine vertex positions.
* (Optionally) an M/V class for each kept edge.

Then you:

* Keep edges with (p_e > \tau).
* Update vertex coords via (v_i = \tilde{v}_i + \Delta v_i).
* Drop isolated vertices (degree 0).

Now you’ve got a **clean predicted graph** (G = (V, E)).

---

## 5. Origami-aware losses (where Kawasaki & Maekawa enter)

Here’s where you leverage “valid, flat-foldable”:

### 5.1 Standard supervision losses

On synthetic data you *know* the ground truth graph:

* Edge classification: binary cross-entropy between (p_e) and whether that candidate edge matches a ground-truth edge.
* Vertex regression: L2 / smooth-L1 on vertex coordinates.
* M/V classification (if labels are available in your CP generator).

Plus the usual pixel-level losses from section 2.2.

### 5.2 Geometric consistency losses

For each predicted vertex (v) and its incident edges ({e_k}):

1. **Kawasaki loss** (angles only)

   * Compute sector angles ({\alpha_1,\ldots,\alpha_{2n}}) by sorting incident edges by angle around the vertex.
   * Define:
     [
     L_{\text{Kawasaki}}(v) = \left( \sum_{odd} \alpha_{2k-1} - \sum_{even} \alpha_{2k} \right)^2
     ]
   * Normalize by (\pi^2) to be scale-invariant. For a perfect flat-foldable one-vertex CP, this difference is 0.([Wikipedia][1])

2. **Even-degree loss**

   * Encourage even degree: e.g. penalize (\sigma(\text{degree}(v) \bmod 2)) (use a differentiable relaxation).

3. **Maekawa loss** (if you model M/V)

   * Let (M_v, V_v) be the predicted counts of mountain vs valley creases incident to v.
   * Add ((|M_v - V_v| - 2)^2) as a penalty.([Wikipedia][2])

Because your synthetic data already *satisfies* these theorems, these losses mainly act as **regularizers** that push the network toward origami-consistent solutions, especially in ambiguous / noisy regions.

You can weight them lightly at first and increase once the basic detection is working.

---

## 6. Inference pipeline on real images

Putting it all together:

1. **Rectify the square**

   * Detect the sheet corners (can be another small network) and warp to a canonical square.

2. **Run backbone + pixel head**

   * Get crease mask, orientation, and junction heatmap at full (or near-full) resolution.

3. **Build candidate graph (\tilde{G})**

   * Skeletonize, find junctions, trace edges.

4. **Run graph head**

   * Get refined graph (G) that (usually) satisfies Kawasaki & Maekawa automatically because of training + regularization.

5. **Post-fix small violations**

   * If some vertices slightly violate angle sums due to noise, you can do a small geometric optimization (e.g., adjust nearby vertex positions/edge angles) with hard constraints for Kawasaki at each vertex.

---

## 7. Why this beats generic LSD/wireframe for CPs

Compared to something like DeepLSD or HAWP:

* **High-res + skeletonization** avoids your current “merged junctions at 128×128” problem.
* The **graph head sees the whole structure**, not just local pixels:

  * It can learn that 3-degree vertices are rare / impossible in interior,
  * That certain angle combinations are unlikely,
  * That short “dangling” segments should be dropped.
* **Origami-theorem losses** inject domain knowledge that generic LSD cannot know:

  * If the network is unsure about a tiny missing edge, it can infer “I need one more crease here to satisfy Kawasaki/Maekawa,” and keep that edge candidate.

You still get all the advantages of modern deep line detection (robustness to noise, camera perspective, shading), but the final output is much closer to a mathematically valid CP.

---

If you want, next step we can zoom in on any one part (e.g. “how exactly do I define candidates and labels for the GNN?” or “how do I procedurally generate random flat-foldable CPs?”) and flesh it out with concrete data structures or pseudo-code.

[1]: https://en.wikipedia.org/wiki/Kawasaki%27s_theorem?utm_source=chatgpt.com "Kawasaki's theorem"
[2]: https://en.wikipedia.org/wiki/Maekawa%27s_theorem?utm_source=chatgpt.com "Maekawa's theorem"
[3]: https://origametry.net/combgeom/flatfold/flat.html?utm_source=chatgpt.com "Notes on Flat Folding - Tom's Combinatorial Geometry Class"
