use std::{cell::{Ref, RefCell, RefMut}, rc::Rc};
use super::*;
use num_traits::{Num, Float};


/// A tensor of any numeric type
#[derive(Clone)]
pub struct Tensor<T: AnyNumber> {
    pub(super) inner: Rc<RefCell<TensorCore<T>>>
}

/// Internal tensor's data
pub(super) struct TensorCore<T: AnyNumber> {
    pub(super) shape: Vec<usize>,
    pub(super) stride: Vec<usize>,
    pub(super) data: Vec<T>,

    pub(super) has_grad: bool,
    pub(super) grad_enabled: bool,
    pub(super) grad: Vec<T>,
    pub(super) children: Children<T>,
}

impl<T: AnyNumber> Tensor<T> {
    /// Calculates the tensor's strides and returns itself 
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

    /// Returns an immutable handle to the data of the tensor 
    pub(super) fn handle(&self) -> Ref<TensorCore<T>> {
        self.inner.borrow()
    }

    /// Returns a mutable handle to the data of the tensor 
    pub(super) fn handle_mut(&self) -> RefMut<TensorCore<T>> {
        self.inner.borrow_mut()
    }

    /// Returns a unique id of tensor's data
    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.inner) as usize
    }

    // Returns the tensor's number of dimensions
    pub fn dim(&self) -> usize {
        self.handle().shape.len()
    }

    /// Returns the tensor's shape as an immutable reference
    pub fn shape(&self) -> Ref<Vec<usize>> {
        Ref::map(self.handle(), |t| &t.shape)
    }

    /// Returns the total number of values the tenor holds
    pub fn size(&self) -> usize {
        self.handle().data.len()
    }
    
    /// Returs the tensor's data, flattened, as an immutable reference 
    pub fn flat(&self) -> Ref<Vec<T>> {
        Ref::map(self.handle(), |t| &t.data)
    }

    /// Returs the tensor's data, flattened, as an immutable reference 
    pub fn flat_mut(&self) -> RefMut<Vec<T>> {
        RefMut::map(self.handle_mut(), |t| &mut t.data)
    }

    /// Returs the tensor's gradient (flat) vector as an immutable reference 
    pub fn grad(&self) -> Ref<Vec<T>> {
        Ref::map(self.handle(), |t| &t.grad)
    }

    /// Returs the tensor's gradient (flat) vector as a mutable reference 
    pub(super) fn grad_mut(&self) -> RefMut<Vec<T>> {
        RefMut::map(self.handle_mut(), |t| &mut t.grad)
    }

    /// Returns a boolean saying whether the tensor has enabled gradients
    pub fn grad_enabled(&self) -> bool {
        self.handle().grad_enabled
    }
    
    /// Calculates the tensor's flat index from an n-dimensional index
    pub fn offset<S: AsRef<[usize]>>(&self, index: S) -> usize {
        index.as_ref()
        .iter()
        .zip(&self.handle().stride)
        .map(|(&i, &s)| i * s)
        .sum()
    }

    /// Calculates the tensor's n-dimensional index from a flat index
    pub fn get_ndidx(&self, idx: usize) -> Vec<usize> {
        let mut out = vec![0; self.dim()];
        for i in 0..self.dim() {
            let d = self.shape()[i];
            let s = self.handle().stride[i];
            out[i] = (idx / s) % d;
        }
        out
    }

    /// Returns a value at a given location
    pub fn get<S: AsRef<[usize]>>(&self, idx: S) -> T {
        let flat_idx = self.offset(idx.as_ref().to_vec());
        self.flat()[flat_idx]
    }

    /// Returns a value at a given location as a mutable reference
    pub fn get_mut<S: AsRef<[usize]>>(&self, idx: S) -> RefMut<T> {
        let flat_idx = self.offset(idx.as_ref().to_vec());
        RefMut::map(self.handle_mut(), |t| &mut t.data[flat_idx])
    }

    /// Copies one tenor's data into own data
    /// Equivalent to *self.flat_mut() = t.flat().clone()
    pub fn set(&self, t: Tensor<T>) {
        *self.flat_mut() = t.flat().clone();
    }

    /// Returns the only value in the tensor
    pub fn item(&self) -> T {
        assert!(self.size() == 1, "item() works only for tensors with 1 element");

        self.flat()[0]
    }
}


impl<T: AnyFloat> Tensor<T> {
    /// Permanently allows for gradient calculation for the tensor and returns itself
    /// Should only be called at most once per tensor. For a temporary change, use Tensor::enable_grad()
    pub fn with_grad(&self) -> Self {
        {
            let mut tensor= self.handle_mut();
            tensor.has_grad = true;
            tensor.grad_enabled = true;
            tensor.grad = vec![T::zero(); tensor.data.len()];
        }
        self.clone()
    }

    /// Permanently disallows gradient calculation for the tensor and returns itself
    /// Should only be called at most once per tensor. For a temporary change, use Tensor::disable_grad()
    pub fn no_grad(&self) -> Self {
        {
            let mut tensor = self.handle_mut();
            tensor.has_grad = false;
            tensor.grad_enabled = false;
            tensor.children = Children::None;
        }
        self.clone()
    }

    /// For a tensor with allowed gradient calculation, enables gradients
    pub fn enable_grad(&self) {
        let mut tensor: RefMut<'_, TensorCore<T>> = self.handle_mut();
        assert!(tensor.has_grad, "Can't toggle gradient on a detached tensor");
        tensor.grad_enabled = true;
    }

    /// For a tensor with allowed gradient calculation, disables gradients
    pub fn disable_grad(&self) {
        let mut tensor: RefMut<'_, TensorCore<T>> = self.handle_mut();
        assert!(tensor.has_grad, "Can't toggle gradient on a detached tensor");
        tensor.grad_enabled = false;
    }

    /// Returns the tensor's gradients as a tensor
    pub fn grad_tensor(&self) -> Tensor<T> {
        Tensor::from_shape_data(self.shape().clone(), self.grad().clone())
    }
}