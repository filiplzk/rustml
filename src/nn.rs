use crate::*;
use rand::{distr::{uniform::{SampleRange, SampleUniform}, Distribution, StandardUniform}, Rng};
use num_traits::{Float, NumAssignOps};


pub trait Module<T: Float> {
    fn forward(&self, x: &Tensor<T>) -> Tensor<T>;
    fn params(&self) -> Vec<Tensor<T>>;

    fn train(&self) {
        for t in &self.params() {
            t.disable_grad();
        }
    }

    fn eval(&self) {
        for t in &self.params() {
            t.enable_grad();
        }
    }
}


pub struct Linear<T: Float> {
    weights: Tensor<T>,
    biases: Tensor<T>,
}

impl<T: Float + NumAssignOps> Module<T> for Linear<T> {
    fn forward(&self, x: &Tensor<T>) -> Tensor<T> {
        assert!(x.dim() >= 2, "Linear::forward(): Expected tensor of dim >= 2, got less");
        
        let batch_dims = &x.shape()[0..x.dim()-2];
        x.matmul(&self.weights.left_broadcast(batch_dims)) + &self.biases.left_broadcast(batch_dims)
    }

    fn params(&self) -> Vec<Tensor<T>> {
        vec![self.weights.clone(), self.biases.clone()]
    }
}

impl<T: Float> Linear<T> {
    pub fn zeros(input: usize, output: usize) -> Self {
        Self {
            weights: Tensor::zeros([input, output]),
            biases: Tensor::zeros([1, output])
        }
    }
}

impl<T: Float + SampleUniform> Linear<T> {
    pub fn new_uniform<R: SampleRange<T> + Clone>(r: &mut impl Rng, input: usize, output: usize, w_range: R, b_range: R) -> Self {
        Self {
            weights: Tensor::new_uniform(r, [input, output], w_range).with_grad(),
            biases: Tensor::new_uniform(r, [1, output], b_range).with_grad()
        }
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
            weights: Tensor::new_normal(r, [input, output], w_mean, w_std).with_grad(),
            biases: Tensor::new_normal(r, [1, output], b_mean, b_std).with_grad()
        }
    }

    pub fn new_he(r: &mut impl Rng, input: usize, output: usize) -> Self {
        let w_std = (T::from(2.0).unwrap() / T::from(input).unwrap()).sqrt();
        Self {
            weights: Tensor::new_normal(r, [input, output], T::zero(), w_std).with_grad(),
            biases: Tensor::zeros([1, output]).with_grad()
        }
    }
}


pub struct Sequential<T: Float> {
    pub layers: Vec<Box<dyn Module<T>>>
}

impl<T: Float> Module<T> for Sequential<T> {
    fn forward(&self, x: &Tensor<T>) -> Tensor<T> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out);
        };

        out
    }

    fn params(&self) -> Vec<Tensor<T>> {
        let mut data = Vec::new();
        for layer in &self.layers {
            for param in layer.params() {
                data.push(param);
            }
        };

        data
    }
}

impl<T: Float> Sequential<T> {
    pub fn new() -> Self {
        Self { layers: vec![] }
    }

    pub fn add<M: Module<T> + 'static>(&mut self, layer: M) {
        self.layers.push(Box::new(layer));
    }
}


pub struct Tanh;
impl Tanh { pub fn new() -> Self { Tanh } }

impl<T: Float> Module<T> for Tanh {
    fn forward(&self, x: &Tensor<T>) -> Tensor<T> {
        let ex = x.exp();
        let emx = (-x).exp(); // e^{-x}
        let numerator = &ex - &emx;
        let denominator = &ex + &emx;

        numerator / denominator
    }

    fn params(&self) -> Vec<Tensor<T>> {
        vec![]
    }
}


pub struct ReLU;
impl ReLU { pub fn new() -> Self { ReLU } }

impl<T: Float> Module<T> for ReLU {
    fn forward(&self, x: &Tensor<T>) -> Tensor<T> {
        x.max(&Tensor::zeros_like(x))
    }

    fn params(&self) -> Vec<Tensor<T>> {
        vec![]
    }
}

pub struct Softmax;
impl Softmax { pub fn new() -> Self { Softmax } }

impl<T: Float + NumAssignOps> Module<T> for Softmax {
    fn forward(&self, x: &Tensor<T>) -> Tensor<T> {
        functional::softmax(x)
    }

    fn params(&self) -> Vec<Tensor<T>> {
        vec![]
    }
}