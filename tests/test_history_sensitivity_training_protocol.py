from sensitivity.train_history_sensitivity import build_parser


def test_history_sensitivity_uses_manuscript_optimization_defaults():
    args = build_parser().parse_args(
        [
            "--history-minutes",
            "120",
            "--train-parquet",
            "train.parquet",
            "--val-parquet",
            "val.parquet",
        ]
    )

    assert args.epochs == 100
    assert args.batch_size == 8192
    assert args.learning_rate == 3e-4
    assert args.weight_decay == 1e-2
    assert args.scheduler_patience == 8
    assert args.scheduler_factor == 0.5
    assert args.min_learning_rate == 1e-6
    assert args.max_grad_norm == 5.0
    assert args.early_stop_patience == 50
    assert args.checkpoint_interval == 5
    assert args.seed == 42
