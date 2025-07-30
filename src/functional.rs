use crate::*;


/// Calculates the mean squared error between 2 tensors
pub fn mse<T: AnyFloat>(x: &Tensor<T>, y: &Tensor<T>) -> Tensor<T> {
    let diff = &(x - y);
    (diff * diff).sum_all() / Tensor::fill([1], T::from(diff.size()).unwrap())
}

/// Calculates the sigmoid function
pub fn sigmoid<T: AnyFloat>(tensor: &Tensor<T>) -> Tensor<T> {
    let ones = Tensor::ones_like(tensor);
    let exp_mt = &(-tensor).exp();

    Tensor::ones_like(tensor) / (ones + exp_mt)
}

/// Performs softmax along the last dimension
pub fn softmax<T: AnyFloat>(tensor: &Tensor<T>) -> Tensor<T> {
    let rdim = tensor.shape()[tensor.dim() - 1];
    
    let tensor = tensor - tensor.max([tensor.dim()-1]).stack_new_dim(tensor.dim() - 1, rdim);

    let exp = tensor.exp();
    let exp_sum = exp.sum([tensor.dim()-1]).stack_new_dim(exp.dim()-1, rdim);

    exp / exp_sum
}

/// Calculates the cross-entropy loss of 2 tensors
/// The tensors should hold probabilities, so every value must belong to the interval [0, 1]
pub fn cross_entropy_loss<T: AnyFloat>(pred: &Tensor<T>, tgt: &Tensor<T>) -> Tensor<T> {
    assert!(pred.dim() >= 2, "cross_entropy_loss(): Expected a Tensor of dim >=2");
    assert!(*pred.shape() == *tgt.shape(), "cross_entropy_loss(): Got Tensors with different shapes");

    let eps = Tensor::fill_like(pred, T::from(1e-30).unwrap());
    let pred = pred + eps;

    (-pred.log() * tgt).sum([pred.dim()-1])
}

