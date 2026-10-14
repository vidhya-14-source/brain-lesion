"""Generate a reproducible software QA contact sheet and numerical report."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from processing.pipeline import load_models, analyze_image
from utils.synthetic import synthetic_sample
from utils.visualization import overlay


def main():
    output = Path("qa_artifacts")
    output.mkdir(exist_ok=True)
    bundle = load_models(output / "intentionally_missing_checkpoint.pth")
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), facecolor="#0b1220")
    records = []
    for row, sample in enumerate("ABC"):
        image, injection = synthetic_sample(sample)
        result = analyze_image(image, bundle)
        records.append({"sample": sample, "candidate_count": len(result["regions"]), "score": result["score"],
                        "threshold": result["threshold"], "injected_pixels": int(injection.sum()),
                        "covered_injected_pixels": int((result["mask"] & injection).sum())})
        panels = [(image, "Synthetic " + sample), (result["reconstruction"], "Demo reference"),
                  (result["heat"], "Anomaly intensity [0, 1]"),
                  (overlay(result["original"], result["heat"], result["mask"], result["regions"]), f'{len(result["regions"])} candidates')]
        for col, (array, title) in enumerate(panels):
            axes[row, col].imshow(array, cmap="inferno" if col == 2 else "gray", vmin=0, vmax=1)
            axes[row, col].set_title(title, color="#dce8f6", fontsize=10)
            axes[row, col].axis("off")
    fig.suptitle("Synthetic software tests · DEMO MODE · not medical images", color="#55dfbf")
    fig.tight_layout()
    fig.savefig(output / "synthetic_qa.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    (output / "synthetic_qa.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
