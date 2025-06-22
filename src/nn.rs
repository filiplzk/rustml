use rand::rngs::ThreadRng;

use crate::*;

pub trait Module {
    fn forward(&self, x: &Matrix) -> Matrix;
    fn params(&mut self) -> Vec<Value>;
}


pub struct Linear {
    weights: Matrix,
    biases: Matrix,
}

impl Linear {
    pub fn new(r: &mut ThreadRng, input: usize, output: usize) -> Self {
        Self {
            weights: Matrix::new_he(r, (input, output)),
            biases: Matrix::fill((1, output), 0.1)
        }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Matrix) -> Matrix {
        x * &self.weights + &self.biases
    }

    fn params(&mut self) -> Vec<Value> {
        [&self.weights.params()[..], &self.biases.params()[..]].concat()
    }
}

pub struct Tanh { }
impl Tanh { pub fn new() -> Self { Tanh { } } }

impl Module for Tanh {
    fn forward(&self, x: &Matrix) -> Matrix {
        let ex = x.exp_each();
        let emx = (-x).exp_each(); // e^{-x}
        let numerator = &ex - &emx;
        let denominator = &ex + &emx;

        numerator.hadamard_product(&denominator.inv_each())
    }

    fn params(&mut self) -> Vec<Value> {
        vec![]
    }
}

pub struct ReLU { }
impl ReLU { pub fn new() -> Self { ReLU { } } }

impl Module for ReLU {
    fn forward(&self, x: &Matrix) -> Matrix {
        x.max(&Matrix::fill(x.shape, 0.0))
    }

    fn params(&mut self) -> Vec<Value> {
        vec![]
    }
}

pub struct Sequential {
    layers: Vec<Box<dyn Module>>
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: vec![] }
    }

    pub fn add<M: Module + 'static>(&mut self, layer: M) {
        self.layers.push(Box::new(layer));
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Matrix) -> Matrix {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out);
        };

        out
    }

    fn params(&mut self) -> Vec<Value> {
        let mut data = Vec::new();
        for layer in &mut self.layers {
            for param in layer.params() {
                data.push(param);
            }
        };

        data
    }
}
