use super::*;
use num_traits::{Num, NumCast};

// TODO add gradient flow
impl<T: Num + Copy> Tensor<T> {
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
        out
    }

    pub fn squeeze(&self) -> Self {
        let out = self.clone();
        out.handle_mut().shape.retain(|&x| x != 1);
        if out.handle().shape.is_empty() {
            out.handle_mut().shape = vec![1];
        }
        out.init()
    }

    pub fn unsqueeze(&self, dim: usize) -> Self {
        let out = self.clone();
        out.handle_mut().shape.insert(dim, 1);
        out.init()
    }

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

    pub fn left_broadcast<S: AsRef<[usize]>>(&self, shape: S) -> Tensor<T> {
        let mut shape_rev = shape.as_ref().to_vec();
        shape_rev.reverse();

        let mut out = self.clone();
        for cnt in shape_rev {
            out = out.stack_new_dim(0, cnt);
        }
        out
    }

    pub fn right_broadcast<S: AsRef<[usize]>>(&self, shape: S) -> Tensor<T> {
        let shape = shape.as_ref().to_vec();

        let mut out = self.clone();
        for cnt in shape {
            out = out.stack_new_dim(out.dim(), cnt);
        }
        out
    }
}


impl<T: Num + Copy + NumCast> Tensor<T> {
    pub fn cast<U: Num + Copy + NumCast>(&self) -> Tensor<U> {
        let data = self.flat()
            .iter()
            .map(|&x| U::from(x).unwrap())
            .collect();
        
        Tensor::from_shape_data(self.shape().clone(), data)
    }
}


impl Tensor<usize> {
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