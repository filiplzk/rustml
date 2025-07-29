use super::*;
use num_traits::{Num, NumCast};

// TODO add gradient flow
impl<T: AnyNumber> Tensor<T> {
    pub fn cast<U: AnyNumber>(&self) -> Tensor<U> {
        let data = self.flat()
            .iter()
            .map(|&x| U::from(x).unwrap())
            .collect();
        
        Tensor::from_shape_data(self.shape().clone(), data)
    }
    
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

    pub fn squeeze(&self) -> Self {
        let mut new_shape = self.shape().clone();        
        new_shape.retain(|&x| x != 1);
        
        self.reshape(new_shape)
    }

    pub fn unsqueeze(&self, dim: usize) -> Self {
        self.stack_new_dim(dim, 1)
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