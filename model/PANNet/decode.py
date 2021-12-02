    def get_boxes(
        self, pred: np.ndarray, fx: float = 1, fy: float = 1, min_area: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        text_region = pred[0] > self.binary_threshold
        kernel = (pred[1] > self.binary_threshold) * text_region  # kernel

        num_labels, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)

        bboxes = []
        for label_id in range(1, num_labels):
            points = np.array(np.where(label == label_id)).transpose((1, 0))[:, ::-1]
            if points.shape[0] < min_area:
                continue

            rect = cv2.minAreaRect(points)
            poly = cv2.boxPoints(rect).astype(np.int)

            d_i = cv2.contourArea(poly) * 1.5 / cv2.arcLength(poly, True)
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            shrinked_poly = np.array(pco.Execute(d_i))
            if shrinked_poly.size == 0:
                continue

            rect = cv2.minAreaRect(shrinked_poly)
            shrinked_poly = cv2.boxPoints(rect).astype(np.int)
            # if cv2.contourArea(shrinked_poly) < 800 / (fx * fy):
            #     continue

            bboxes.append(
                [
                    shrinked_poly[1] / fx,
                    shrinked_poly[2] / fy,
                    shrinked_poly[3] / fx,
                    shrinked_poly[0] / fy,
                ]
            )

        return bboxes
