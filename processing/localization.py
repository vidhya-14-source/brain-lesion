from skimage.measure import label, regionprops


def localize(mask, heat):
    regions = []
    for i, prop in enumerate(regionprops(label(mask, connectivity=2), intensity_image=heat), 1):
        y0, x0, y1, x1 = prop.bbox
        cy, cx = prop.centroid
        regions.append({"region": f"Suspicious Region {i}", "area_pixels": int(prop.area),
                        "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                        "centroid_xy": [float(cx), float(cy)],
                        "mean_anomaly": float(prop.intensity_mean),
                        "max_anomaly": float(prop.intensity_max)})
    return regions
