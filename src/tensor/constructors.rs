use std::{cell::RefCell, f32::consts::PI, rc::Rc};
use super::*;
use rand::{distr::{uniform::{SampleRange, SampleUniform}, Distribution, StandardUniform}, Rng};
use num_traits::{Float, Num};


impl<T: AnyNumber> Tensor<T> {
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

    pub(super) fn from_op(shape: Vec<usize>, data: Vec<T>, grad_criterion: bool, maybe_children: Children<T>) -> Self {
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


    pub fn null() -> Self {
        Self::from_shape_data(vec![], vec![])
    }

    pub fn from_flat<S: AsRef<[usize]>, V: AsRef<[T]>>(shape: S, vals: V) -> Self {
        let shape = shape.as_ref().to_vec();
        let vals = vals.as_ref().to_vec();
        let data = (0..vals.len())
            .map(|i: usize| vals[i])
            .collect();
    
        Self::from_shape_data(shape, data)
    }

    pub fn fill<S: AsRef<[usize]>>(shape: S, val: T) -> Self {
        let shape = shape.as_ref().to_vec();
        let data = vec![val; shape.iter().product()];

        Self::from_shape_data(shape, data)
    }

    pub fn zeros<S: AsRef<[usize]>>(shape: S) -> Self {
        Self::fill(shape, T::zero())
    }
    
    pub fn ones<S: AsRef<[usize]>>(shape: S) -> Self {
        Self::fill(shape, T::one())
    }

    pub fn fill_like<U: AnyNumber>(t: &Tensor<U>, val: T) -> Self {
        let data = vec![val; t.size()];
        Self::from_shape_data(t.shape().clone(), data)
    }

    pub fn zeros_like<U: AnyNumber>(t: &Tensor<U>) -> Self {
        Self::zeros(t.shape().clone())
    }

    pub fn ones_like<U: AnyNumber>(t: &Tensor<U>) -> Self {
        Self::ones(t.shape().clone())
    }

    pub fn from_flat_like<U: AnyNumber, V: AsRef<[T]>>(t: &Tensor<U>, vals: V) -> Self {
        Self::from_flat(t.shape().clone(), vals)
    }
}

impl<T: AnyNumber + SampleUniform> Tensor<T> {
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