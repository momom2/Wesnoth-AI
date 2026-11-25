"""Do weights that start at zero receive less gradient?
(docs/audit_fixes_prereg_20261125.md, the user's question of 2026-11-25.)

Reads the simple loop's Adam state at a round, changes nothing. Adam keeps,
per weight, a running average of its squared gradient (`exp_avg_sq`, here
corrected for the parameter's own number of steps). Its square root is the
gradient's typical size at that weight over the recent rounds. Printed for
the projection of the unit row's added columns, which started at zero at
round 544, beside the old unit projection and the whole network.

    python -I -X utf8 gradient_sizes.py round_0576/adam.pt
"""
import sys
from pathlib import Path

import torch

OLD, NEW = "encoder.unit_feat_proj.weight", "encoder.unit_feat_proj2.weight"


def main() -> None:
    torch.set_num_threads(2)
    for path in sys.argv[1:]:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        names = list(state["names"])
        moments = state["optimizer"]["state"]
        beta2 = float(state["optimizer"]["param_groups"][0]["betas"][1])

        def size(i: int):
            m = moments[i]
            steps = int(m["step"])
            v = m["exp_avg_sq"].double() / (1 - beta2 ** steps)
            return steps, float(v.mean().sqrt()), float(v.sqrt().median())

        total_sq, total_n = 0.0, 0
        for i, name in enumerate(names):
            if i in moments and name != NEW:
                m = moments[i]
                v = m["exp_avg_sq"].double() / (1 - beta2 ** int(m["step"]))
                total_sq += float(v.sum())
                total_n += v.numel()
        print(f"{path}: Adam's step {int(state['step'])}")
        print(f"  whole network but the new projection: gradient per weight, rms {(total_sq / total_n) ** 0.5:.3g}")
        for name in (OLD, NEW):
            if name in names and names.index(name) in moments:
                steps, rms, median = size(names.index(name))
                print(f"  {name}: {steps} steps, gradient per weight rms {rms:.3g}, median {median:.3g}")


if __name__ == "__main__":
    main()
