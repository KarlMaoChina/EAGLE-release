from __future__ import annotations

import argparse
from pathlib import Path

from eagle.clinical import apply_scaler
from eagle.data import FILE_NAMES, load_case_table, load_spacing, load_volume, save_nifti
from eagle.infer import FreezePackage, fit_freeze_stats, load_ensemble, predict_volume, prepare_clinical_row
from eagle.io import read_table, write_json
from eagle.preprocess import prepare_dual_view
from eagle.runtime import resolve_device, seed_everything
from eagle.segmentation import postprocess_segmentation
from eagle.spec import DEPLOYMENT_THRESHOLD, SELECTED_RADIOMICS_FEATURES
from eagle.train import TrainConfig, train_dvse_stream, train_fusion, train_tabular


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eagle",
        description="EAGLE v1.0: dual-view CT + clinical/radiomic fusion for gallbladder lesion risk scores.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    preprocess = sub.add_parser("preprocess", help="Build dual-view 96x112x80 patches from a volume and mask.")
    preprocess.add_argument("--image", required=True)
    preprocess.add_argument("--mask", required=True)
    preprocess.add_argument("--output-dir", required=True)
    preprocess.add_argument("--case-id", required=True)
    preprocess.add_argument("--no-refine-mask", action="store_true")

    post = sub.add_parser("postprocess-mask", help="Apply anatomy-aware connected-component cleanup to a mask.")
    post.add_argument("--mask", required=True)
    post.add_argument("--output", required=True)
    post.add_argument("--spacing", nargs=3, type=float, default=None)

    stats = sub.add_parser("fit-stats", help="Fit imputation and z-score stats on a development table.")
    stats.add_argument("--table", required=True)
    stats.add_argument("--output-dir", required=True)

    infer = sub.add_parser("infer", help="Run five-fold ensemble inference at the frozen threshold T=0.5.")
    infer.add_argument("--image", required=True)
    infer.add_argument("--mask", required=True)
    infer.add_argument("--clinical-table", required=True)
    infer.add_argument("--case-id", required=True)
    infer.add_argument("--weights", required=True, help="Directory containing biamf/fold_*.pt and preprocess_stats.json")
    infer.add_argument("--radiomics-table", default=None)
    infer.add_argument("--output", default=None)
    infer.add_argument("--device", default=None)

    train_dvse = sub.add_parser("train-dvse", help="Train one DVSE stream (standard or enlarged).")
    _add_common(train_dvse)
    train_dvse.add_argument("--table", required=True)
    train_dvse.add_argument("--image-root", required=True)
    train_dvse.add_argument("--output-dir", required=True)
    train_dvse.add_argument("--stream", choices=("standard", "enlarged"), required=True)
    train_dvse.add_argument("--num-workers", type=int, default=0)

    train_tab = sub.add_parser("train-tabular", help="Train the clinical+radiomics encoder.")
    _add_common(train_tab)
    train_tab.add_argument("--table", required=True)
    train_tab.add_argument("--image-root", required=True)
    train_tab.add_argument("--output-dir", required=True)
    train_tab.add_argument("--num-workers", type=int, default=0)

    train_f = sub.add_parser("train-fusion", help="Train BiAMF with frozen extractors.")
    _add_common(train_f)
    train_f.add_argument("--table", required=True)
    train_f.add_argument("--image-root", required=True)
    train_f.add_argument("--output-dir", required=True)
    train_f.add_argument("--extractor-root", default=None)
    train_f.add_argument("--num-workers", type=int, default=0)

    return parser


def _train_config(args: argparse.Namespace) -> TrainConfig:
    seed_everything(args.seed)
    return TrainConfig(
        table_path=Path(args.table),
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "postprocess-mask":
        mask = load_volume(args.mask)
        spacing = tuple(args.spacing) if args.spacing else load_spacing(args.mask)
        cleaned, info = postprocess_segmentation(mask, spacing)
        save_nifti(cleaned, args.output, spacing)
        print(info)
        return 0

    if args.command == "preprocess":
        image = load_volume(args.image)
        mask = load_volume(args.mask)
        spacing = load_spacing(args.image)
        if not args.no_refine_mask:
            mask, _ = postprocess_segmentation(mask, spacing)
        prepared = prepare_dual_view(image, mask, spacing)
        if prepared is None:
            raise SystemExit("Preprocessing failed: mask empty after resampling.")
        out = Path(args.output_dir) / args.case_id
        save_nifti(prepared.standard_image, out / FILE_NAMES["standard_image"], spacing)
        save_nifti(prepared.standard_mask, out / FILE_NAMES["standard_mask"], spacing)
        save_nifti(prepared.enlarged_image, out / FILE_NAMES["enlarged_image"], spacing)
        save_nifti(prepared.enlarged_mask, out / FILE_NAMES["enlarged_mask"], spacing)
        write_json(out / "preprocess.json", {"case_id": args.case_id, "source_spacing": list(spacing)})
        return 0

    if args.command == "fit-stats":
        fit_freeze_stats(args.table, args.output_dir)
        return 0

    if args.command == "infer":
        device = resolve_device(args.device)
        package = FreezePackage.from_dir(args.weights)
        models = load_ensemble(package, device)
        clinical_table = load_case_table(args.clinical_table)
        row = clinical_table.loc[clinical_table["case_id"] == args.case_id]
        if row.empty:
            raise SystemExit(f"case_id {args.case_id} was not found in the clinical table.")
        clinical = prepare_clinical_row(row.iloc[0], package)
        if args.radiomics_table:
            ra_table = read_table(args.radiomics_table)
            id_col = "case_id" if "case_id" in ra_table.columns else ra_table.columns[0]
            ra_row = ra_table.loc[ra_table[id_col].astype(str) == args.case_id]
            if ra_row.empty:
                raise SystemExit(f"case_id {args.case_id} was not found in the radiomics table.")
            radiomics = apply_scaler(ra_row, package.radiomics_scaler)[list(SELECTED_RADIOMICS_FEATURES)].to_numpy(
                dtype="float32"
            )[0]
        else:
            radiomics = apply_scaler(row, package.radiomics_scaler)[list(SELECTED_RADIOMICS_FEATURES)].to_numpy(
                dtype="float32"
            )[0]
        image, mask, spacing = load_volume(args.image), load_volume(args.mask), load_spacing(args.image)
        result = predict_volume(
            image,
            mask,
            spacing,
            clinical,
            radiomics,
            models,
            device,
            case_id=args.case_id,
        )
        if args.output:
            write_json(args.output, result)
        print(
            {
                "case_id": result["case_id"],
                "probability": result["probability"],
                "positive": result["positive"],
                "threshold": DEPLOYMENT_THRESHOLD,
            }
        )
        return 0

    if args.command == "train-dvse":
        train_dvse_stream(_train_config(args), stream=args.stream)
        return 0
    if args.command == "train-tabular":
        train_tabular(_train_config(args))
        return 0
    if args.command == "train-fusion":
        train_fusion(_train_config(args), extractor_root=args.extractor_root)
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
