use std::{cell::Ref, ops::{Index, IndexMut}};

use crate::tensor;

use super::*;

use num_traits::{Float, Num, NumAssign};

// TODO add gradient flow
impl<T: Num + Copy> Tensor<T> {
    // pub fn squeeze(&self) -> Self {
    //     let mut out = self.clone();
    //     out.shape.retain(|&x| x != 1);
    //     out.init()
    // }

    // pub fn unsqueeze(&self, dim: usize) -> Self {
    //     let mut out = self.clone();
    //     out.shape.insert(dim, 1);
    //     out.init()
    // }

    pub fn transpose(&self, dim1: usize, dim2: usize) -> Self {
        let mut new_shape = self.shape().clone();
        new_shape.swap(dim1, dim2);
        
        let out = Tensor::zeros(new_shape);
        for i in 0..self.size() {
            let val = self.flat()[i];
            let mut new_ndidx = self.get_ndidx(i);
            new_ndidx.swap(dim1, dim2);
            *out.get_mut(new_ndidx) = val;
        }
        out
    }

    // pub fn left_broadcast<S: AsRef<[usize]>>(&self, shape: S) -> Tensor<T> {
    //     let shape = shape.as_ref().to_vec();
    //     let new_shape = [&shape[..], &self.shape[..]].concat();
    //     let repeat_count = shape.iter().product();

    //     let mut data = Vec::with_capacity(repeat_count * self.size());
    //     for _ in 0..repeat_count {
    //         data.extend_from_slice(&self.data);
    //     }

    //     Self::from_shape_data(new_shape, data)
    // }

    // pub fn right_broadcast<S: AsRef<[usize]>>(&self, shape: S) -> Tensor<T> {
    //     let shape = shape.as_ref().to_vec();
    //     let new_shape = [&self.shape[..], &shape[..]].concat();
    //     let repeat_count = shape.iter().product();

    //     let mut data = Vec::with_capacity(repeat_count * self.size());
    //     for scalar in &self.data {
    //         for _ in 0..repeat_count {
    //             data.push(scalar.clone());
    //         }
    //     }

    //     Self::from_shape_data(new_shape, data)
    // }
}
