use std::iter::zip;

use crate::*;

pub struct SGD {
    params: Vec<Value>,
    grads: Vec<f32>,
    lr: f32,
    momentum: f32
}

impl SGD {
    pub fn new(params: Vec<Value>, lr: f32, momentum: f32) -> Self {
        let param_count = params.len();
        Self {
            params,
            grads: vec![0.0_f32; param_count],
            lr,
            momentum
        }
    }

    pub fn step(&mut self) {
        for (val, g )in zip(&mut self.params, &mut self.grads) {
            *g = self.momentum * *g - self.lr * val.data().grad;
            val.data_mut().val += *g;
        }
    }
}