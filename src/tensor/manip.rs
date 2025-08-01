use std::ops::{Bound, Range, RangeBounds};

use super::*;
use num_traits::{Num, NumCast};
use plotters::prelude::Ranged;

// TODO add gradient flow
impl<T: AnyNumber> Tensor<T> {
    /// Returns a new tensor of the given numeric type with the same data
    pub fn cast<U: AnyNumber>(&self) -> Tensor<U> {
        let data = self.flat()
            .iter()
            .map(|&x| U::from(x).unwrap())
            .collect();
        
        Tensor::from_shape_data(self.shape().clone(), data)
    }

    /// Returns a new tensor with a new given dimension stacked along it a given number of times
    pub fn stack_new_dim(&self, dim: usize, count: usize) -> Tensor<T> {
        let mut new_shape = self.shape().clone();
        new_shape.insert(dim, count);
    
        let out = Tensor::zeros(new_shape);
        for idx in 0..self.size() {
            let val = self.flat()[idx];
            let mut ndidx = self.get_ndidx(idx);
            ndidx.insert(dim, 0);
            for j in 0..count {
                ndidx[dim] = j;
                *out.get_mut(&ndidx) = val;
            }
        }

        if self.grad_enabled() {
            out.handle_mut().has_grad = true;
            out.handle_mut().grad_enabled = true;
            out.handle_mut().children = Children::NewDim(self.clone(), dim);
        }
        out
    }

    /// Returns a new tensor with all dimensions of size 1 removed
    pub fn squeeze(&self) -> Self {
        let mut new_shape = self.shape().clone();        
        new_shape.retain(|&x| x != 1);
        if new_shape.is_empty() {
            new_shape = vec![1];
        }
        
        self.reshape(new_shape)
    }

    /// Returns a new tensor with a given dimension of size 1 removed
    pub fn squeeze_at(&self, dim: usize) -> Self {
        let mut new_shape = self.shape().clone();
        new_shape.remove(dim);
        if new_shape.is_empty() {
            new_shape = vec![1];
        }
        
        self.reshape(new_shape)
    }

    /// Returns a new tensor with a new given dimension of size 1
    pub fn unsqueeze(&self, dim: usize) -> Self {
        self.stack_new_dim(dim, 1)
    }

    /// Returns a new tensor with 2 given dimensions transposed
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

    /// Returns a new tensor with the same data but some other shape
    pub fn reshape<S: AsRef<[usize]>>(&self, shape: S) -> Tensor<T> {
        let shape: Vec<usize> = shape.as_ref().to_vec();
        assert!(shape.iter().product::<usize>() == self.size(), "When reshaping, new shape must have the same total size");

        let out = Tensor::from_shape_data(shape,self.flat().clone());
        if self.grad_enabled() {
            out.handle_mut().has_grad = true;
            out.handle_mut().grad_enabled = true;
            out.handle_mut().children = Children::Reshape(self.clone());
        }
        out
    }

    ///  Returns a new tensor sliced along a given dimension
    pub(super) fn slice_1dim(&self, dim: usize, range: Range<usize>) -> Tensor<T> {        
        let new_dim_cnt = range.len();
        let mut new_shape = self.shape().clone();
        new_shape[dim] = new_dim_cnt;
        
        let out = Tensor::zeros(new_shape);
        for i in 0..self.size() {
            let val = self.flat()[i];
            let mut ndidx = self.get_ndidx(i);
            if range.contains(&ndidx[dim]) {
                ndidx[dim] -= range.start;
                *out.get_mut(ndidx) = val;
            }
        }

        if self.grad_enabled() {
            out.handle_mut().has_grad = true;
            out.handle_mut().grad_enabled = true;
            out.handle_mut().children = Children::Slice(self.clone(), dim, range);
        }
        out
    }

    /// Returns a new tensor where each dimension is sliced in a given way
    pub fn slice<S: AsRef<[Range<usize>]>>(&self, shape: S) -> Tensor<T> {
        let shape = shape.as_ref().to_vec();

        let mut out = self.clone();
        for (dim, range) in shape.iter().enumerate() {
            out = out.slice_1dim(dim, range.clone());
        }

        out
    }
}



impl Tensor<usize> {
    /// Returns a new tensor where each value is converted into a one-hot vector
    pub fn one_hot(&self, dim_size: usize) -> Tensor<usize> {
        let mut new_shape = self.shape().clone();
        new_shape.push(dim_size);

        let out = Tensor::zeros(new_shape);
        for idx in 0..self.size() {
            let val = self.flat()[idx];
            let mut ndidx = self.get_ndidx(idx);
            ndidx.push(val);
            *out.get_mut(ndidx) = 1;
        }
        out
    }
}
