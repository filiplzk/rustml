use crate::*;

pub fn cross_entropy_loss(x: &Matrix, tgt: usize) -> (Value, Matrix) {
        assert!(x.shape.0 == 1);

        let mut max_logit = Value::new(f32::NEG_INFINITY);
        for c in 0..x.shape.1 {
            let val = x[(0, c)].clone();

            if val.data().val > max_logit.data().val {
                max_logit = val;
            }
        }
        
        let x_shifted = x + Matrix::fill_value(x.shape, &-max_logit);
        let log_sum_exp = x_shifted.exp_each().sum().log();

        let log_probs = x_shifted - Matrix::fill_value(x.shape, &log_sum_exp);

        let nll = -log_probs[(0, tgt)].clone();
        (nll, log_probs.exp_each())
}

pub fn mse(x: &Matrix, y: &Matrix) -> Value {
    let diff = x - y;
    diff.hadamard_product(&diff).average()
}