use std::{cell::Ref};
use std::ops::{Add, AddAssign, Deref, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};


use super::*;

use num_traits::{Float, Num, NumAssign};

impl<T: Num + Copy> Tensor<T> {
    pub fn duplicate(&self) -> Tensor<T> {
        let data = self.flat().clone();
        let children = Children::Id(self.clone());
        Tensor::from_op(self.shape().clone(), data, self.grad_enabled(), children)
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
}
