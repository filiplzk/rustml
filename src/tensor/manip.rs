use super::*;
use num_traits::Num;

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
