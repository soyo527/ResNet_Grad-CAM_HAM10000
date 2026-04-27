import torch


def apply_priority_thresholds(
    probabilities: torch.Tensor,
    class_names: list[str],
    priority_thresholds: list[tuple[str, float]],
    nv_suppression_threshold: float = 0.5,
) -> torch.Tensor:
    _, predictions = torch.max(probabilities, dim=1)
    predictions = predictions.clone()

    try:
        nv_idx = class_names.index("nv")
    except ValueError:
        nv_idx = -1

    configs = []
    for class_name, threshold in priority_thresholds:
        if class_name in class_names:
            configs.append((class_names.index(class_name), class_name, threshold))

    for target_idx, target_name, threshold in reversed(configs):
        target_probs = probabilities[:, target_idx]
        if target_name == "mel" and nv_idx != -1:
            mask = (target_probs > threshold) & (probabilities[:, nv_idx] < nv_suppression_threshold)
        else:
            mask = target_probs > threshold
        predictions[mask] = target_idx

    return predictions
