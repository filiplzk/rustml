use crate::*;

pub fn softmax<T: AnyFloat>(tensor: &Tensor<T>) -> Tensor<T> {
    let rdim = tensor.shape()[tensor.dim() - 1];
    
    // let shifted= tensor - tensor.max([tensor.dim()-1]).right_broadcast([rdim]);
    let shifted = tensor;  // TODO add shifting for numerical stability

    let exp = shifted.exp();
    let exp_sum = exp.sum([shifted.dim()-1]).stack_new_dim(exp.dim()-1, rdim);

    exp / exp_sum
}

pub fn cross_entropy_loss<T: AnyFloat>(pred: &Tensor<T>, tgt: &Tensor<T>) -> Tensor<T> {
    assert!(pred.dim() >= 2, "cross_entropy_loss(): Expected a Tensor of dim >=2");
    assert!(*pred.shape() == *tgt.shape(), "cross_entropy_loss(): Got Tensors with different shapes");

    (-pred.log() * tgt).sum([pred.dim()-1])
}

pub fn mse<T: AnyFloat>(x: &Tensor<T>, y: &Tensor<T>) -> Tensor<T> {
    let diff = &(x - y);
    (diff * diff).sum_all() / Tensor::fill([1], T::from(diff.size()).unwrap())
}