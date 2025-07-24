use crate::*;

use rand::{distr::{uniform::{SampleRange, SampleUniform}, Distribution, StandardUniform}, Rng};

use num_traits::{Float, Num};



pub trait Module<T: Num + Copy> {
    fn forward(&self, x: &Tensor<T>) -> Tensor<T>;
    fn params(&mut self) -> Vec<Scalar<T>>;
}


pub struct Linear<T: Num + Copy> {
    weights: Tensor<T>,
    biases: Tensor<T>,
}

impl<T: Num + Copy> Linear<T> {
    pub fn zeros(input: usize, output: usize) -> Self {
        Self {
            weights: Tensor::zeros([input, output]),
            biases: Tensor::zeros([1, output])
        }
    }
}

impl<T: Num + Copy + SampleUniform> Linear<T> {
    pub fn new_uniform<R: SampleRange<T> + Clone>(r: &mut impl Rng, input: usize, output: usize, w_range: R, b_range: R) -> Self {
        Self {
            weights: Tensor::new_uniform(r, [input, output], w_range),
            biases: Tensor::new_uniform(r, [1, output], b_range)
        }
    }
}

impl<T: Float> Linear<T> {
    pub fn with_grad(mut self) -> Self {
        self.weights = self.weights.with_grad();
        self.biases = self.biases.with_grad();
        self
    }
}

impl<T: Float> Linear<T>
where
    StandardUniform: Distribution<T>
{
    pub fn new_normal(
        r: &mut impl Rng,
        input: usize, output: usize,
        w_mean: T, w_std: T,
        b_mean: T, b_std: T) -> Self 
    {
        Self {
            weights: Tensor::new_normal(r, [input, output], w_mean, w_std),
            biases: Tensor::new_normal(r, [1, output], b_mean, b_std)
        }
    }

    pub fn new_he(r: &mut impl Rng, input: usize, output: usize) -> Self {
        let w_std = (T::from(2.0).unwrap() / T::from(input).unwrap()).sqrt();
        Self {
            weights: Tensor::new_normal(r, [input, output], T::zero(), w_std),
            biases: Tensor::zeros([1, output])
        }
    }
}

impl<T: Num + Copy> Module<T> for Linear<T> {
    fn forward(&self, x: &Tensor<T>) -> Tensor<T> {
        assert!(x.dim() >= 2, "Linear::forward(): Expected tensor of dim >= 2, got less");
        
        let batch_dims = &x.shape[0..x.dim()-2];
        x.mat_mul(&self.weights.left_broadcast(batch_dims)) + &self.biases.left_broadcast(batch_dims)
    }

    fn params(&mut self) -> Vec<Scalar<T>> {
        [&self.weights.flat()[..], &self.biases.flat()[..]].concat()
    }
}

pub struct Tanh { }
impl Tanh { pub fn new() -> Self { Tanh { } } }

impl<T: Float> Module<T> for Tanh {
    fn forward(&self, x: &Tensor<T>) -> Tensor<T> {
        let ex = x.exp();
        let emx = (-x).exp(); // e^{-x}
        let numerator = &ex - &emx;
        let denominator = &ex + &emx;

        numerator / denominator
    }

    fn params(&mut self) -> Vec<Scalar<T>> {
        vec![]
    }
}

pub struct ReLU { }
impl ReLU { pub fn new() -> Self { ReLU { } } }

impl<T: Num + Copy + PartialOrd> Module<T> for ReLU {
    fn forward(&self, x: &Tensor<T>) -> Tensor<T> {
        x.max_with(&Tensor::zeros(x.shape.clone()))
    }

    fn params(&mut self) -> Vec<Scalar<T>> {
        vec![]
    }
}

pub struct Sequential<T: Num + Copy> {
    pub layers: Vec<Box<dyn Module<T>>>
}

impl<T: Num + Copy> Sequential<T> {
    pub fn new() -> Self {
        Self { layers: vec![] }
    }

    pub fn add<M: Module<T> + 'static>(&mut self, layer: M) {
        self.layers.push(Box::new(layer));
    }
}

impl<T: Num + Copy> Module<T> for Sequential<T> {
    fn forward(&self, x: &Tensor<T>) -> Tensor<T> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out);
        };

        out
    }

    fn params(&mut self) -> Vec<Scalar<T>> {
        let mut data = Vec::new();
        for layer in &mut self.layers {
            for param in layer.params() {
                data.push(param);
            }
        };

        data
    }
}
