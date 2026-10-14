import cv2
import numpy as np


def heatmap_rgb(heat):
    return cv2.cvtColor(cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def overlay(image, heat, mask, regions):
    base = np.repeat(np.clip(image[..., None], 0, 1), 3, axis=2)
    color = heatmap_rgb(heat).astype(np.float32) / 255
    alpha = (0.65 * np.clip(heat / 0.5, 0, 1))[..., None]
    result = np.ascontiguousarray(((1 - alpha) * base * 255 + alpha * color * 255).astype(np.uint8))
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (85, 255, 190), 1)
    for i, region in enumerate(regions, 1):
        x0, y0, x1, y1 = region["bbox_xyxy"]
        cv2.rectangle(result, (x0, y0), (x1 - 1, y1 - 1), (85, 255, 190), 1)
        cv2.putText(result, str(i), (x0, max(9, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return result
