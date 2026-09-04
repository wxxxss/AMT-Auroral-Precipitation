from training.train_amt import build_parser


def test_manuscript_training_defaults_to_fixed_100_epochs_without_early_stopping():
    args = build_parser().parse_args(
        ["--train-parquet", "train.parquet", "--val-parquet", "val.parquet"]
    )

    assert args.epochs == 100
    assert not hasattr(args, "early_stop_patience")
    assert args.scheduler_patience == 8
