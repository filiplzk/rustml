use std::{cell::Ref};
use std::ops::{Add, AddAssign, Deref, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};


use crate::tensor;

use super::*;

use num_traits::{Float, Num, NumAssign, NumAssignOps};

impl<T: Num + Copy> Tensor<T> {
    pub fn duplicate(&self) -> Tensor<T> {
        let data = self.flat().clone();
        let children = Children::Id(self.clone());
        Tensor::from_op(self.shape().clone(), data, self.grad_enabled(), children)
    }
}

impl<T: Num + Copy + NumAssignOps> Tensor<T> {
    pub fn matmul(&self, rhs: &Tensor<T>) -> Tensor<T> {
        assert!(self.dim() >= 2 && rhs.dim() >= 2, "Matmul can be done on tensors with dim >= 2");
        
        let b = &self.shape()[..self.dim()-2];
        let b_rhs = &rhs.shape()[..self.dim()-2];
        assert!(b == b_rhs, "Matmul requires the same batch dimentions");
        
        let batch_cnt = b.iter().product();
        let (r1, c1) = (self.shape()[self.dim()-2], self.shape()[self.dim()-1]);
        let (r2, c2) = (rhs.shape()[rhs.dim()-2], rhs.shape()[rhs.dim()-1]);
        assert!(c1 == r2, "Matmul: matrix shapes do not match");

        let out_shape = [b, &[r1], &[c2]].concat();
        let out = Tensor::zeros(out_shape);
        for batch in 0..batch_cnt {
            let b_1_off = batch * r1 * c1;
            let b_2_off = batch * r2 * c2;
            let b_out_off = batch * r1 * c2;
            for r in 0..r1 {
                for c in 0..c2 {
                    let mut tot = T::zero();
                    for i in 0..c1 {
                        let v1 = self.flat()[b_1_off + r * c1 + i];
                        let v2 = rhs.flat()[b_2_off + i * c2 + c];
                        tot += v1 * v2;
                    }
                    out.flat_mut()[b_out_off + r * c2 + c] = tot;
                }
            }
        }

        if self.grad_enabled() || rhs.grad_enabled() {
            out.handle_mut().has_grad = true;
            out.handle_mut().grad_enabled = true;
            out.handle_mut().children = Children::Matmul(self.clone(), rhs.clone());
        }
        out
    }
}

impl<T: Num + Copy + PartialOrd> Tensor<T> {
    pub fn min(&self, rhs: &Tensor<T>) -> Tensor<T> {
        let data = self.flat()
            .iter()
            .zip(rhs.flat().iter())
            .map(|(&x, &y)| if x < y { x } else { y })
            .collect();
        let children = Children::Min(self.clone(), rhs.clone());
        Tensor::from_op(self.shape().clone(), data, self.grad_enabled() || rhs.grad_enabled(), children)
    }

    pub fn max(&self, rhs: &Tensor<T>) -> Tensor<T> {
        let data = self.flat()
            .iter()
            .zip(rhs.flat().iter())
            .map(|(&x, &y)| if x < y { x } else { y })
            .collect();
        let children = Children::Max(self.clone(), rhs.clone());
        Tensor::from_op(self.shape().clone(), data, self.grad_enabled() || rhs.grad_enabled(), children)
    }
}

impl<T: Num + Copy> Neg for &Tensor<T> {
    type Output = Tensor<T>;
    fn neg(self) -> Tensor<T> {
        let data = self.flat()
            .iter()
            .map(|&x| T::zero()-x)
            .collect();
        let children = Children::Neg(self.clone());
        Tensor::from_op(self.shape().clone(), data, self.grad_enabled(), children)
    }
}
impl<T: Num + Copy> Neg for Tensor<T> { type Output = Tensor<T>; fn neg(self) -> Tensor<T> { -&self } }


impl<T: Num + Copy> Add<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;
    fn add(self, rhs: &Tensor<T>) -> Tensor<T> {
        let data = self.flat()
            .iter()
            .zip(rhs.flat().iter())
            .map(|(&x, &y)| x + y)
            .collect();
        let children = Children::Add(self.clone(), rhs.clone());
        Tensor::from_op(self.shape().clone(), data, self.grad_enabled() || rhs.grad_enabled(), children)
    }
}
impl<T: Num + Copy> Add<&Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn add(self, rhs: &Tensor<T>) -> Tensor<T> { &self + rhs } }
impl<T: Num + Copy> Add<Tensor<T>> for &Tensor<T> { type Output = Tensor<T>; fn add(self, rhs: Tensor<T>) -> Tensor<T> { self + &rhs } }
impl<T: Num + Copy> Add<Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn add(self, rhs: Tensor<T>) -> Tensor<T> { &self + &rhs } }
impl<T: Num + Copy> AddAssign<&Tensor<T>> for Tensor<T> { fn add_assign(&mut self, rhs: &Tensor<T>) { *self = &*self + rhs; } }
impl<T: Num + Copy> AddAssign<Tensor<T>> for Tensor<T> { fn add_assign(&mut self, rhs: Tensor<T>) { *self = &*self + &rhs; } }


impl<T: Num + Copy> Sub<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;
    fn sub(self, rhs: &Tensor<T>) -> Tensor<T> {
        let data = self.flat()
            .iter()
            .zip(rhs.flat().iter())
            .map(|(&x, &y)| x - y)
            .collect();
        let children = Children::Sub(self.clone(), rhs.clone());
        Tensor::from_op(self.shape().clone(), data, self.grad_enabled() || rhs.grad_enabled(), children)
    }
}
impl<T: Num + Copy>  Sub<&Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn sub(self, rhs: &Tensor<T>) -> Tensor<T> { self + (-rhs) } }
impl<T: Num + Copy>  Sub<Tensor<T>> for &Tensor<T> { type Output = Tensor<T>; fn sub(self, rhs: Tensor<T>) -> Tensor<T> { self + (-rhs) } }
impl<T: Num + Copy>  Sub<Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn sub(self, rhs: Tensor<T>) -> Tensor<T> { self + (-rhs) } }
impl<T: Num + Copy> SubAssign<&Tensor<T>> for Tensor<T> { fn sub_assign(&mut self, rhs: &Tensor<T>) { *self = &*self - rhs; } }
impl<T: Num + Copy> SubAssign<Tensor<T>> for Tensor<T> { fn sub_assign(&mut self, rhs: Tensor<T>) { *self = &*self - &rhs; } }


impl<T: Num + Copy> Mul<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;
    fn mul(self, rhs: &Tensor<T>) -> Tensor<T> {
        let data = self.flat()
            .iter()
            .zip(rhs.flat().iter())
            .map(|(&x, &y)| x * y)
            .collect();
        let children = Children::Mul(self.clone(), rhs.clone());
        Tensor::from_op(self.shape().clone(), data, self.grad_enabled() || rhs.grad_enabled(), children)
    }
}
impl<T: Num + Copy> Mul<&Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn mul(self, rhs: &Tensor<T>) -> Tensor<T> { &self * rhs } }
impl<T: Num + Copy> Mul<Tensor<T>> for &Tensor<T> { type Output = Tensor<T>; fn mul(self, rhs: Tensor<T>) -> Tensor<T> { self * &rhs } }
impl<T: Num + Copy> Mul<Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn mul(self, rhs: Tensor<T>) -> Tensor<T> { &self * &rhs } }
impl<T: Num + Copy> MulAssign<&Tensor<T>> for Tensor<T> { fn mul_assign(&mut self, rhs: &Tensor<T>) { *self = &*self * rhs; } }
impl<T: Num + Copy> MulAssign<Tensor<T>> for Tensor<T> { fn mul_assign(&mut self, rhs: Tensor<T>) { *self = &*self * &rhs; } }


impl<T: Num + Copy> Div<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;
    fn div(self, rhs: &Tensor<T>) -> Tensor<T> {
        let data = self.flat()
            .iter()
            .zip(rhs.flat().iter())
            .map(|(&x, &y)| x / y)
            .collect();
        let children = Children::Div(self.clone(), rhs.clone());
        Tensor::from_op(self.shape().clone(), data, self.grad_enabled() || rhs.grad_enabled(), children)
    }
}
impl<T: Num + Copy> Div<&Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn div(self, rhs: &Tensor<T>) -> Tensor<T> { &self / rhs } }
impl<T: Num + Copy> Div<Tensor<T>> for &Tensor<T> { type Output = Tensor<T>; fn div(self, rhs: Tensor<T>) -> Tensor<T> { self / &rhs } }
impl<T: Num + Copy> Div<Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn div(self, rhs: Tensor<T>) -> Tensor<T> { &self / &rhs } }
impl<T: Num + Copy> DivAssign<&Tensor<T>> for Tensor<T> { fn div_assign(&mut self, rhs: &Tensor<T>) { *self = &*self / rhs; } }
impl<T: Num + Copy> DivAssign<Tensor<T>> for Tensor<T> { fn div_assign(&mut self, rhs: Tensor<T>) { *self = &*self / &rhs; } }


// floating point operations
impl<T: Float> Tensor<T> {
    pub fn exp(&self) -> Tensor<T> {
        let data = self.flat()
            .iter()
            .map(|&x| x.exp())
            .collect();
        let children = Children::Exp(self.clone());
        Tensor::from_op(self.shape().clone(), data, self.grad_enabled(), children)
    }

    pub fn log(&self) -> Tensor<T> {
        let data = self.flat()
            .iter()
            .map(|&x| x.ln())
            .collect();
        let children = Children::Log(self.clone());
        Tensor::from_op(self.shape().clone(), data, self.grad_enabled(), children)
    }

    pub fn pow(&self, rhs: &Tensor<T>) -> Tensor<T> {
        let data = self.flat()
            .iter()
            .zip(rhs.flat().iter())
            .map(|(&x, &y)| x.powf(y))
            .collect();
        let children = Children::Pow(self.clone(), rhs.clone());
        Tensor::from_op(self.shape().clone(), data, self.grad_enabled(), children)
    }

    pub fn sqrt(&self) -> Tensor<T> {
        self.pow(&Tensor::fill_like(self, T::from(0.5).unwrap()))
    }
}
