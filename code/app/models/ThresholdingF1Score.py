import tensorflow as tf


class ThresholdingF1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(ThresholdingF1Score, self).__init__(name=name, **kwargs)
        self.true_positives = tf.keras.metrics.TruePositives()
        self.false_positives = tf.keras.metrics.FalsePositives()
        self.false_negatives = tf.keras.metrics.FalseNegatives()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # unpack semi-supervised format if given
        if type(y_true) is tuple:
            (y_true, _) = y_true
            (y_pred, _) = y_pred
        
        # round both y values to binarize the classification task
        y_true = tf.math.round(y_true)
        y_pred = tf.math.round(y_pred)

        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.false_positives.update_state(y_true, y_pred, sample_weight)
        self.false_negatives.update_state(y_true, y_pred, sample_weight)

    def result(self):
        tp = self.true_positives.result()
        fp = self.false_positives.result()
        fn = self.false_negatives.result()
        return tp / (tp + 0.5 * (fp + fn))

    def reset_state(self):
        self.true_positives.reset_state()
        self.false_positives.reset_state()
        self.false_negatives.reset_state()
