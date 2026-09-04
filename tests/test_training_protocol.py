from training.train_amt import build_parser


def test_manuscript_training_defaults_match_table_6():
    args = build_parser().parse_args(
        ["--train-parquet", "train.parquet", "--val-parquet", "val.parquet"]
    )

    assert args.epochs == 100
    assert args.batch_size == 8192
    assert args.learning_rate == 3e-4
    assert args.weight_decay == 1e-2
    assert args.early_stop_patience == 50
    assert args.scheduler_patience == 8
    assert args.scheduler_factor == 0.5
    assert args.min_learning_rate == 1e-6
    assert args.max_grad_norm == 5.0
    assert args.checkpoint_interval == 5
