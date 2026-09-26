"""The az loop's training recipe: the trainer settings tools/az_loop.py
steps with.

`configure_az_trainer` sets the loss the loop trains and its learning
rate. The loop, its training-path benchmark (tools/bench_train_step.py)
and the tests that reproduce the loop's step configure the trainer
through it, so they train the same loss. The batch size and the bf16
switch are performance settings the benchmark varies: az_loop sets them
itself, after this.
"""


def configure_az_trainer(trainer, *, lr: float = 1e-4, value_coef: float = 1.0) -> None:
    """The loss the az loop trains: squared error on the value mean, no
    auxiliary terms, clip 1; the learning rate goes to the config and
    to every optimizer group."""
    cfg = trainer.config
    cfg.value_loss_form = "mse_mean"
    cfg.value_coef = float(value_coef)
    cfg.learning_rate = float(lr)
    for g in trainer.optimizer.param_groups:
        g["lr"] = float(lr)
    cfg.aux_coef = 0.0
    cfg.gbc_coef = 0.0
    cfg.moves_left_coef = 0.0
    cfg.value_label_smoothing = 0.0
    cfg.trust_lambda = 0.0
    cfg.grad_clip = 1.0
