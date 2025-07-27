use std::iter::zip;
use crate::*;
use num_traits::Float;


pub struct SGD<T: Float> {
    params: Vec<Tensor<T>>,
    grads: Vec<Tensor<T>>,
    lr: T,
    momentum: T
}

impl<T: Float> SGD<T> {
    pub fn new(params: Vec<Tensor<T>>, lr: T, momentum: T) -> Self {
        let grads: Vec<Tensor<T>> = params.iter()
            .map(|t| Tensor::zeros_like(t))
            .collect();

        Self {
            params,
            grads,
            lr,
            momentum
        }
    }

    pub fn step(&mut self) {
        for (t, g ) in zip(&self.params, &mut self.grads) {
            *g = Tensor::fill_like(t, self.momentum) * &*g + Tensor::fill_like(t, self.lr) * t.grad_tensor();
            t.set(t - &*g);
        }
    }
}