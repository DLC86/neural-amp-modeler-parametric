# Knobs and switches in the capture app

The **Controls** tab replaces **Knobs**. Choose **Add control**, then **Knob** or
**Switch**. A knob keeps its range, step, Avoid zero and Gain/Drive settings. For a
switch, enter its name, number of positions (at least two), and a distinct name
for every position, in physical switch order. Use its **positions — edit** button
to change these later. Positions are categorical: intermediate positions are
independent states, not interpolated knob values.

## Capture counts

Let K be the number of knobs and S the product of all switch position counts.
For example, a three-position switch and a two-position switch give S = 6.

* Recommended training LHS captures: **15 × K × S**.
* Recommended validation captures: **min(3 × K, 10) × S**.
* Training must cover every switch combination: at least **S** captures. If
  validation is enabled, it must also have at least **S** captures.
* Corner captures repeat the existing continuous-knob corner design for **every**
  switch combination, including intermediate positions. Duplicates of existing
  training captures are skipped. The app displays the corner count before planning.

The recommendations are starting heuristics, not a guarantee of model accuracy.
The minimum counts ensure switch-state coverage only. Complex rigs may need more
captures or active learning. Counts in the Plan tab are totals, not per-state
counts. Uneven totals are distributed as evenly as possible across states.
Each state gets a separate knob sampling plan; validation settings remain disjoint
from training settings and reserved corners. A coarse knob grid can run out of
distinct settings; the planner reports the affected switch combination.

For two knobs and one three-position switch, the initial recommendation is 90
training LHS captures and 18 validation captures, plus up to 12 corner captures
(less any duplicates already in training). Without switches, counts and planning
retain the previous behavior.

With **switches only**, capture every joint state once (S training captures) and
set validation to zero: there are no remaining unseen settings for a held-out
settings split. Corner captures then add nothing because all states are covered.

The initial plan groups captures by switch state in declared position order to
reduce switching, then sweeps the knobs within each group. Existing generated
plans keep their stored counts and progress. After changing control names, types
or switch positions, regenerate the plan before adding corners.

## Project and model files

Existing knob-only projects still load. The project retains its historical `knobs`
JSON key for compatibility, now containing both control types; switches carry
`type: "switch"`, `enum_names` and a default position index. Capture parameters
store the exact position names, including numeric-looking names such as `"1"`.

ConcatWaveNet, HyperWaveNet, HyperA2 and active-learning model configurations retain
switch types, all labels and the default index. Switch labels containing characters
that Windows disallows in filenames are escaped in filenames; the original labels
remain unchanged in project, dataset and model metadata. Existing input/output level
calibration metadata is preserved.
