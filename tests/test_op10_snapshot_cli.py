from evaluation.evaluate_boundary_statistics import build_parser as build_boundary_parser
from evaluation.evaluate_spatial_mlt_mlat import build_parser as build_spatial_parser


def test_boundary_cli_accepts_explicit_op10_snapshot_root():
    args = build_boundary_parser().parse_args(
        [
            "--ealb-txt",
            "EALB.txt",
            "--palb-txt",
            "PALB.txt",
            "--omni-parquet",
            "omni.parquet",
            "--model-path",
            "model.pth",
            "--scaler-path",
            "scaler.pkl",
            "--snapshot-root",
            "op10_work",
        ]
    )
    assert args.snapshot_root == "op10_work"


def test_spatial_cli_accepts_explicit_op10_snapshot_root():
    args = build_spatial_parser().parse_args(
        [
            "--test-data",
            "test.parquet",
            "--ovation-omni",
            "omni.parquet",
            "--model-path",
            "model.pth",
            "--scaler-path",
            "scaler.pkl",
            "--snapshot-root",
            "op10_work",
        ]
    )
    assert args.snapshot_root == "op10_work"
