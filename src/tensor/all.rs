use super::backprop;

use std::ops::{Add, Neg, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign};
use std::ops::{Index, IndexMut};
use std::fmt;

use rand::distr::uniform::{SampleRange, SampleUniform};
use rand::distr::{Distribution, StandardUniform};
use rand::Rng;

use num_traits::{Float, Num, NumAssign};



#[derive(Clone)]
pub struct Tensor<T: Num + Copy> {
    pub shape: Vec<usize>,
    data: Vec<T>,
    grad: Option<Vec<T>>,
    stride: Vec<usize>,
    matrix_count: usize,
    matrix_stride: usize
}

impl<T: Num + Copy> Tensor<T> {
    pub fn mat_mul(&self, rhs: &Tensor<T>) -> Tensor<T> {
        assert!(self.dim() >= 2 && rhs.dim() >= 2, "mat_mul(): Tensors of dimension >=2 expected, got less");
        assert!(self.shape[0..self.dim()-2] == rhs.shape[0..rhs.dim()-2], "mat_mul(): Tensor batch dimensions not equal");
        
        let (x, y) = self.mat_shape();
        let (yy, z) = rhs.mat_shape();
        assert!(y == yy, "mat_mul(): Tensor matrix dimensions not matching");

        let rhs_mt = rhs.mat_transpose();
        let new_shape = [&self.shape[..self.dim()-2], &[x, z]].concat();
        let mut out = Tensor::zeros(new_shape);

        for mat in 0..self.matrix_count {
            for r in 0..x {
                for c in 0..z {
                    for i in 0..y {
                        *out.mat_idx_mut((mat, r, c)) += self.mat_idx((mat, r, i)) * rhs_mt.mat_idx((mat, c, i));
                    }
                }
            }
        }

        out
    }

    pub fn mat_transpose(&self) -> Tensor<T> {
        assert!(self.dim() >= 2, "mat_transpose(): Tensor of dimension >=2 expected, got less");

        let (mr, mc) = self.mat_shape();
        let mut new_shape = self.shape.clone();
        new_shape.swap(self.shape.len() - 2, self.shape.len() - 1);
        let mut out = Tensor::zeros(new_shape);

        for mat in 0..self.matrix_count {
            for r in 0..mr {
                for c in 0..mc {
                    *out.mat_idx_mut((mat, c, r)) = self.mat_idx((mat, r, c)).clone();
                }
            }
        }
        out
    }
}

impl<T: Num + Copy + NumAssignOps> Tensor<T> {
    fn sum_1dim(&self, dim: usize) -> Tensor<T> {
        let mut new_shape = self.shape.clone();
        new_shape.remove(dim);
    
        let mut out = Tensor::zeros(new_shape);
    
        for idx in 0..self.size() {
            let dim_cnt = (idx / self.stride[dim]) % self.shape[dim];
            let dim_idx = idx % self.stride[dim];
            
            // index math magic
            let out_idx = (idx - dim_cnt*self.stride[dim] - dim_idx) / self.shape[dim] + dim_idx;
            out.data[out_idx] += &self.data[idx];
        }
    
        out
    }
}


    pub fn sum<S: AsRef<[usize]>>(&self, dims: S) -> Tensor<T> {
        let mut dims = dims.as_ref().to_vec();
        dims.reverse();

        let mut out = self.clone();
        for dim in dims {
            out = out.sum_1dim(dim);
        }
        out
    }

    pub fn sum_all(&self) -> Scalar<T> {
        let mut out = Scalar::new(T::zero());
        for scalar in &self.data {
            out += scalar;
        }
        out
    }

    fn product_1dim(&self, dim: usize) -> Tensor<T> {
        let mut new_shape = self.shape.clone();
        new_shape.remove(dim);

        let mut out = Tensor::ones(new_shape);

        for idx in 0..self.size() {
            let dim_cnt = (idx / self.stride[dim]) % self.shape[dim];
            let dim_idx = idx % self.stride[dim];
            
            // index math magic
            let out_idx = (idx - dim_cnt*self.stride[dim] - dim_idx) / self.shape[dim] + dim_idx;
            out.data[out_idx] *= &self.data[idx];
        }

        out
    }

    pub fn product<S: AsRef<[usize]>>(&self, dims: S) -> Tensor<T> {
        let mut dims = dims.as_ref().to_vec();
        dims.reverse();

        let mut out = self.clone();
        for dim in dims {
            out = out.product_1dim(dim);
        }
        out
    }

    pub fn product_all(&self) -> Scalar<T> {
        let mut out = Scalar::new(T::one());
        for scalar in &self.data {
            out *= scalar;
        }
        out
    }


impl<T: Num + Copy + PartialOrd> Tensor<T> {
    fn max_1dim(&self, dim: usize, start_val: T) -> Tensor<T> {
        let mut new_shape = self.shape.clone();
        new_shape.remove(dim);

        let mut out = Tensor::fill(new_shape, start_val);

        for idx in 0..self.size() {
            let dim_cnt = (idx / self.stride[dim]) % self.shape[dim];
            let dim_idx = idx % self.stride[dim];
            
            // index math magic
            let out_idx = (idx - dim_cnt*self.stride[dim] - dim_idx) / self.shape[dim] + dim_idx;
            if self.data[idx].val() > out.data[out_idx].val() {
                out.data[out_idx] = self.data[idx].clone();    
            }
        }

        out
    }

    pub fn max<S: AsRef<[usize]>>(&self, dims: S) -> Tensor<T> {
        let mut dims = dims.as_ref().to_vec();
        dims.reverse();
        
        let start_val = (-(-self).max_all()).val();  // TODO fix ts
        let mut out = self.clone();
        for dim in dims {
            out = out.max_1dim(dim, start_val);
        }
        out
    }

    pub fn max_all(&self) -> Scalar<T> {
        let mut out = self.data[0].clone();
        for scalar in &self.data {
            if scalar.val() > out.val() {
                out = scalar.clone();
            }
        }
        out
    }
}


impl<T: Num + Copy + SampleUniform> Tensor<T> {
    pub fn new_uniform<S: AsRef<[usize]>>(r: &mut impl Rng, shape: S, range: impl SampleRange<T> + Clone) -> Self {
        let shape = shape.as_ref().to_vec();
        let data = (0..shape.iter().product())
            .map(|_| Scalar::new_uniform(r, range.clone()))
            .collect();

        Self::from_shape_data(shape, data)
    }
}

impl<T: Num + Copy + PartialOrd> Tensor<T> {
    pub fn max_with(&self, rhs: &Tensor<T>) -> Tensor<T> {
        assert!(self.shape == rhs.shape, "max_with(): Tensor shapes not equal");

        let data = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(x, y)| x.max(y))
            .collect();

        Self::from_shape_data(self.shape.clone(), data)
    }
}

impl<T: Float> Tensor<T> {
    pub fn with_grad(mut self) -> Tensor<T> {
        for scalar in &mut self.data {
            scalar.handle().grad = Some(T::zero());
        }
        self
    }

    pub fn exp(&self) -> Tensor<T> {
        let data = self.data.iter()
            .map(|x| x.exp())
            .collect();

        Self::from_shape_data(self.shape.clone(), data)
    }

    pub fn log(&self) -> Tensor<T> {
        let data = self.data.iter()
            .map(|x| x.log())
            .collect();

        Self::from_shape_data(self.shape.clone(), data)
    }
}


impl<T: Float> Tensor<T>
where
    StandardUniform: Distribution<T>
{
    pub fn new_normal<S: AsRef<[usize]>>(r: &mut impl Rng, shape: S, mean: T, std: T) -> Self {
        let shape = shape.as_ref().to_vec();
        let data = (0..shape.iter().product())
            .map(|_| Scalar::new_normal(r, mean, std))
            .collect();
    
        Self::from_shape_data(shape, data).init()
    }
    
    pub fn new_he<S: AsRef<[usize]>>(r: &mut impl Rng, shape: S) -> Self {
        let shape = shape.as_ref().to_vec();
        assert!(shape.len() >= 2, "Tensor of dimension >=2 expected, got less");
    
        let input_size = shape[shape.len() - 2];
        Self::new_normal(r, shape, T::zero(), (T::from(2.0).unwrap() / T::from(input_size).unwrap()).sqrt())
    }
}

impl<T: Num + Copy> Add<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;
    fn add(self, rhs: &Tensor<T>) -> Tensor<T> {
        assert!(self.shape == rhs.shape, "ops::Add(): Tensor shapes not equal");

        let data: Vec<Scalar<T>> = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(x, y)| x + y)
            .collect();

        Tensor::from_shape_data(self.shape.clone(), data)
    }
}
impl<T: Num + Copy> Add<&Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn add(self, rhs: &Tensor<T>) -> Tensor<T> { &self + rhs } }
impl<T: Num + Copy> Add<Tensor<T>> for &Tensor<T> { type Output = Tensor<T>; fn add(self, rhs: Tensor<T>) -> Tensor<T> { self + &rhs } }
impl<T: Num + Copy> Add<Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn add(self, rhs: Tensor<T>) -> Tensor<T> { &self + &rhs } }
impl<T: Num + Copy> AddAssign<&Tensor<T>> for Tensor<T> { fn add_assign(&mut self, rhs: &Tensor<T>) { *self = &*self + rhs; } }
impl<T: Num + Copy> AddAssign<Tensor<T>> for Tensor<T> { fn add_assign(&mut self, rhs: Tensor<T>) { *self = &*self + &rhs; } }

impl<T: Num + Copy> Neg for &Tensor<T> {
    type Output = Tensor<T>;
    fn neg(self) -> Tensor<T> {
        let data: Vec<Scalar<T>> = self.data.iter()
            .map(|x| -x)
            .collect();

        Tensor::from_shape_data(self.shape.clone(), data)
    }
}
impl<T: Num + Copy> Neg for Tensor<T> { type Output = Tensor<T>; fn neg(self) -> Tensor<T> { -&self } }

impl<T: Num + Copy> Sub<&Tensor<T>> for &Tensor<T> { type Output = Tensor<T>; fn sub(self, rhs: &Tensor<T>) -> Tensor<T> { self + (-rhs) } }
impl<T: Num + Copy> Sub<&Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn sub(self, rhs: &Tensor<T>) -> Tensor<T> { self + (-rhs) } }
impl<T: Num + Copy> Sub<Tensor<T>> for &Tensor<T> { type Output = Tensor<T>; fn sub(self, rhs: Tensor<T>) -> Tensor<T> { self + (-rhs) } }
impl<T: Num + Copy> Sub<Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn sub(self, rhs: Tensor<T>) -> Tensor<T> { self + (-rhs) } }
impl<T: Num + Copy> SubAssign<&Tensor<T>> for Tensor<T> { fn sub_assign(&mut self, rhs: &Tensor<T>) { *self = &*self - rhs; } }
impl<T: Num + Copy> SubAssign<Tensor<T>> for Tensor<T> { fn sub_assign(&mut self, rhs: Tensor<T>) { *self = &*self - &rhs; } }

impl<T: Num + Copy> Mul<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;
    fn mul(self, rhs: &Tensor<T>) -> Tensor<T> {
        assert!(self.shape == rhs.shape, "ops::Mul(): Tensor shapes not equal");

        let data: Vec<Scalar<T>> = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(x, y)| x * y)
            .collect();

        Tensor::from_shape_data(self.shape.clone(), data)
    }
}
impl<T: Num + Copy> Mul<&Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn mul(self, other: &Tensor<T>) -> Tensor<T> { &self * other } }
impl<T: Num + Copy> Mul<Tensor<T>> for &Tensor<T> { type Output = Tensor<T>; fn mul(self, other: Tensor<T>) -> Tensor<T> { self * &other } }
impl<T: Num + Copy> Mul<Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn mul(self, other: Tensor<T>) -> Tensor<T> { &self * &other } }
impl<T: Num + Copy> MulAssign<&Tensor<T>> for Tensor<T> { fn mul_assign(&mut self, rhs: &Tensor<T>) { *self = &*self * rhs; } }
impl<T: Num + Copy> MulAssign<Tensor<T>> for Tensor<T> { fn mul_assign(&mut self, rhs: Tensor<T>) { *self = &*self * &rhs; } }

impl<T: Num + Copy> Div<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;
    fn div(self, rhs: &Tensor<T>) -> Tensor<T> {
        assert!(self.shape == rhs.shape, "ops::Div(): Tensor shapes not equal");

        let data: Vec<Scalar<T>> = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(x, y)| x / y)
            .collect();

        Tensor::from_shape_data(self.shape.clone(), data)
    }
}
impl<T: Num + Copy> Div<&Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn div(self, rhs: &Tensor<T>) -> Tensor<T> { &self / rhs } }
impl<T: Num + Copy> Div<Tensor<T>> for &Tensor<T> { type Output = Tensor<T>; fn div(self, rhs: Tensor<T>) -> Tensor<T> { self / &rhs } }
impl<T: Num + Copy> Div<Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn div(self, rhs: Tensor<T>) -> Tensor<T> { &self / &rhs } }
impl<T: Num + Copy> DivAssign<&Tensor<T>> for Tensor<T> { fn div_assign(&mut self, rhs: &Tensor<T>) { *self = &*self / rhs; } }
impl<T: Num + Copy> DivAssign<Tensor<T>> for Tensor<T> { fn div_assign(&mut self, rhs: Tensor<T>) { *self = &*self / &rhs; } }