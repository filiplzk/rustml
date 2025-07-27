// use crate::*;

// use std::iter::zip;

// use num_traits::Float;

// pub struct SGD<T: Float> {
//     params: Vec<Scalar<T>>,
//     grads: Vec<T>,
//     lr: T,
//     momentum: T
// }

// impl<T: Float> SGD<T> {
//     pub fn new(params: Vec<Scalar<T>>, lr: T, momentum: T) -> Self {
//         let param_count = params.len();
//         Self {
//             params,
//             grads: vec![T::zero(); param_count],
//             lr,
//             momentum
//         }
//     }

//     pub fn step(&mut self) {
//         for (val, g )in zip(&mut self.params, &mut self.grads) {
//             *g = self.momentum * *g - self.lr * val.grad();
//             val.set(val.val() + *g);
//         }
//     }
// }