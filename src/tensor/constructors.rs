use std::{cell::RefCell, f32::consts::PI, rc::Rc};
use super::*;
use rand::{distr::{uniform::{SampleRange, SampleUniform}, Distribution, StandardUniform}, Rng};
use num_traits::{Float, Num};


impl<T: AnyNumber> Tensor<T> {
    /// Constructs a tensor with given shape and flat data (vector)
    pub(super) fn from_shape_data(shape: Vec<usize>, data: Vec<T>) -> Self {
        let len = data.len();
        let tensor = TensorCore {
            shape,
            stride: vec![],
            data,
            has_grad: false,
            grad_enabled: false,
            grad: vec![T::zero(); len],
            children: Children::None,
        };
        let inner = Rc::new(RefCell::new(tensor));

        Self { inner }.init()
    }

    /// Constructs a tensor with given shape and flat data, additionally with given children if some condition is met
    pub(crate) fn from_op(shape: Vec<usize>, data: Vec<T>, grad_criterion: bool, maybe_children: Children<T>) -> Self {
        let len = data.len();
        let tensor = TensorCore {
            shape,
            stride: vec![],
            data,
            has_grad: grad_criterion,
            grad_enabled: grad_criterion,
            grad: vec![T::zero(); len],
            children: if grad_criterion { maybe_children } else { Children::None },
        };
        let inner = Rc::new(RefCell::new(tensor));

        Self { inner }.init()
    }

    /// Constructs a "null" tensor, that is, a tensor with no values in it
    /// Used as a placeholder
    pub fn null() -> Self {
        Self::from_shape_data(vec![], vec![])
    }

    /// Constructs a tensor with given shape and flat data
    pub fn from_flat<S: AsRef<[usize]>, V: AsRef<[T]>>(shape: S, vals: V) -> Self {
        let shape = shape.as_ref().to_vec();
        let vals = vals.as_ref().to_vec();
        let data = (0..vals.len())
            .map(|i: usize| vals[i])
            .collect();
    
        Self::from_shape_data(shape, data)
    }

    /// Constructs a tensor with given shape filled with some value
    pub fn fill<S: AsRef<[usize]>>(shape: S, val: T) -> Self {
        let shape = shape.as_ref().to_vec();
        let data = vec![val; shape.iter().product()];

        Self::from_shape_data(shape, data)
    }

    /// Constructs a tensor with given shape filled with zeros
    pub fn zeros<S: AsRef<[usize]>>(shape: S) -> Self {
        Self::fill(shape, T::zero())
    }
    
    /// Constructs a tensor with given shape filled with ones
    pub fn ones<S: AsRef<[usize]>>(shape: S) -> Self {
        Self::fill(shape, T::one())
    }

    /// Constructs a tensor with the same shape as given tensor, filled with some value
    pub fn fill_like<U: AnyNumber>(t: &Tensor<U>, val: T) -> Self {
        let data = vec![val; t.size()];
        Self::from_shape_data(t.shape().clone(), data)
    }

    /// Constructs a tensor with the same shape as given tensor, filled with zeros
    pub fn zeros_like<U: AnyNumber>(t: &Tensor<U>) -> Self {
        Self::zeros(t.shape().clone())
    }

    /// Constructs a tensor with the same shape as given tensor, filled with ones
    pub fn ones_like<U: AnyNumber>(t: &Tensor<U>) -> Self {
        Self::ones(t.shape().clone())
    }

    /// Constructs a tensor with the same shape as given tensor and given flat data
    pub fn from_flat_like<U: AnyNumber, V: AsRef<[T]>>(t: &Tensor<U>, vals: V) -> Self {
        Self::from_flat(t.shape().clone(), vals)
    }
}

impl<T: AnyNumber + SampleUniform> Tensor<T> {
    /// Constructs a tensor with given shape, where each value is sampled uniformly from a given range
    pub fn new_uniform<S: AsRef<[usize]>>(rng: &mut impl Rng, shape: S, range: impl SampleRange<T> + Clone) -> Self {
        let shape = shape.as_ref().to_vec();
        let data = (0..shape.iter().product())
            .map(|_| rng.random_range(range.clone()))
            .collect();
    
        Self::from_shape_data(shape, data)
    }
}

impl<T: AnyFloat> Tensor<T>
where 
    StandardUniform: Distribution<T>
{
    /// Constructs a tensor with given shape, where each value is sampled from normal distribution with given mean and standard deviation
    /// Sampling is done using the Box-Muller transform
    pub fn new_normal<S: AsRef<[usize]>>(rng: &mut impl Rng, shape: S, mean: T, std: T) -> Self {
        let shape = shape.as_ref().to_vec();
        let data = (0..shape.iter().product())
            .map(|_| {
                let u1 = rng.random();
                let u2 = rng.random();
                let sampled = (T::from(-2.0).unwrap() * u1.ln()).sqrt() * (T::from(2.0*PI).unwrap() * u2).cos();
                mean + sampled * std
            })
            .collect();
    
        Self::from_shape_data(shape, data)
    }
}