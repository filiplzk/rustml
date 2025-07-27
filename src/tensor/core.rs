use std::{cell::{Ref, RefCell, RefMut}, rc::Rc};

use crate::tensor;

use super::*;

use num_traits::{Num, Float};

#[derive(Clone)]
pub struct Tensor<T: Num + Copy> {
    pub(super) inner: Rc<RefCell<TensorCore<T>>>
}

pub struct TensorCore<T: Num + Copy> {
    pub(super) shape: Vec<usize>,
    pub(super) stride: Vec<usize>,
    pub(super) data: Vec<T>,

    pub(super) has_grad: bool,
    pub(super) grad_enabled: bool,
    pub(super) grad: Vec<T>,
    pub(super) children: Children<T>,

    pub(super) matrix_count: usize,
    pub(super) matrix_stride: usize,
}

impl<T: Num + Copy> Tensor<T> {
    pub(super) fn init(&self) -> Self {
        let mut tensor = self.inner.borrow_mut();
        tensor.stride = vec![0; tensor.shape.len()];
        let mut p = 1;
        for i in (0..tensor.shape.len()).rev() {
            tensor.stride[i] = p;
            p *= tensor.shape[i];
        }

        self.clone()
    }

    pub(super) fn handle(&self) -> Ref<TensorCore<T>> {
        self.inner.borrow()
    }

    pub(super) fn handle_mut(&self) -> RefMut<TensorCore<T>> {
        self.inner.borrow_mut()
    }

    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.inner) as usize
    }

    pub fn dim(&self) -> usize {
        self.handle().shape.len()
    }

    pub fn shape(&self) -> Ref<Vec<usize>> {
        Ref::map(self.handle(), |t| &t.shape)
    }

    pub fn size(&self) -> usize {
        self.handle().data.len()
    }
    
    pub fn flat(&self) -> Ref<Vec<T>> {
        Ref::map(self.handle(), |t| &t.data)
    }

    pub fn flat_mut(&self) -> RefMut<Vec<T>> {
        RefMut::map(self.handle_mut(), |t| &mut t.data)
    }

    pub fn grad(&self) -> Ref<Vec<T>> {
        Ref::map(self.handle(), |t| &t.grad)
    }

    pub(super) fn grad_mut(&self) -> RefMut<Vec<T>> {
        RefMut::map(self.handle_mut(), |t| &mut t.grad)
    }

    pub fn grad_enabled(&self) -> bool {
        self.handle().grad_enabled
    }
    
    pub fn offset<S: AsRef<[usize]>>(&self, index: S) -> usize {
        index.as_ref()
        .iter()
        .zip(&self.handle().stride)
        .map(|(&i, &s)| i * s)
        .sum()
    }

    pub fn get_ndidx(&self, idx: usize) -> Vec<usize> {
        let mut out = vec![0; self.dim()];
        for i in 0..self.dim() {
            let d = self.shape()[i];
            let s = self.handle().stride[i];
            out[i] = (idx / s) % d;
        }
        out
    }

    pub fn get<S: AsRef<[usize]>>(&self, idx: S) -> T {
        let flat_idx = self.offset(idx.as_ref().to_vec());
        self.flat()[flat_idx]
    }

    pub fn get_mut<S: AsRef<[usize]>>(&self, idx: S) -> RefMut<T> {
        let flat_idx = self.offset(idx.as_ref().to_vec());
        RefMut::map(self.handle_mut(), |t| &mut t.data[flat_idx])
    }

    pub fn set(&self, t: Tensor<T>) {
        *self.flat_mut() = t.flat().clone();
    }
}


impl<T: Float> Tensor<T> {
    pub fn with_grad(&self) -> Self {
        {
            let mut tensor: RefMut<'_, TensorCore<T>> = self.handle_mut();
            tensor.has_grad = true;
            tensor.grad_enabled = true;
            tensor.grad = vec![T::zero(); tensor.data.len()];
        }
        self.clone()
    }

    pub fn no_grad(&self) -> Self {
        {
            let mut tensor = self.handle_mut();
            tensor.has_grad = false;
            tensor.grad_enabled = false;
            tensor.children = Children::None;
        }
        self.clone()
    }

    pub fn enable_grad(&self) {
        let mut tensor: RefMut<'_, TensorCore<T>> = self.handle_mut();
        assert!(tensor.has_grad, "Can't toggle gradient on a detached tensor");
        tensor.grad_enabled = true;
    }

    pub fn disable_grad(&self) {
        let mut tensor: RefMut<'_, TensorCore<T>> = self.handle_mut();
        assert!(tensor.has_grad, "Can't toggle gradient on a detached tensor");
        tensor.grad_enabled = false;
    }

    pub fn grad_tensor(&self) -> Tensor<T> {
        Tensor::from_shape_data(self.shape().clone(), self.grad().clone())
    }
}