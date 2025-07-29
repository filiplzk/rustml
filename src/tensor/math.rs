use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use super::*;
use num_traits::{Float, Num, NumAssignOps};


impl<T: AnyNumber> Tensor<T> {
    pub fn duplicate(&self) -> Tensor<T> {
        let data = self.flat().clone();
        let children = Children::Id(self.clone());
        Tensor::from_op(self.shape().clone(), data, self.grad_enabled(), children)
    }
}

impl<T: AnyNumber> Tensor<T> {
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
                        tot = tot + v1 * v2;
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

    pub fn matmul_at(&self, rhs: &Tensor<T>) -> Tensor<T> {
        assert!(self.dim() >= 2 && rhs.dim() >= 2, "Matmul can be done on tensors with dim >= 2");
        
        let b = &self.shape()[..self.dim()-2];
        let b_rhs = &rhs.shape()[..self.dim()-2];
        assert!(b == b_rhs, "Matmul requires the same batch dimentions");
        
        let batch_cnt = b.iter().product();
        let (c1, r1) = (self.shape()[self.dim()-2], self.shape()[self.dim()-1]);
        let (r2, c2) = (rhs.shape()[rhs.dim()-2], rhs.shape()[rhs.dim()-1]);
        assert!(c1 == r2, "Matmul_at: matrix shapes do not match");

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
                        let v1 = self.flat()[b_1_off + i * r1 + r];
                        let v2 = rhs.flat()[b_2_off + i * c2 + c];
                        tot = tot + v1 * v2;
                    }
                    out.flat_mut()[b_out_off + r * c2 + c] = tot;
                }
            }
        }

        if self.grad_enabled() || rhs.grad_enabled() {
            out.handle_mut().has_grad = true;
            out.handle_mut().grad_enabled = true;
            out.handle_mut().children = Children::MatmulAT(self.clone(), rhs.clone());
        }
        out
    }

    pub fn matmul_bt(&self, rhs: &Tensor<T>) -> Tensor<T> {
        assert!(self.dim() >= 2 && rhs.dim() >= 2, "Matmul can be done on tensors with dim >= 2");
        
        let b = &self.shape()[..self.dim()-2];
        let b_rhs = &rhs.shape()[..self.dim()-2];
        assert!(b == b_rhs, "Matmul requires the same batch dimentions");
        
        let batch_cnt = b.iter().product();
        let (r1, c1) = (self.shape()[self.dim()-2], self.shape()[self.dim()-1]);
        let (c2, r2) = (rhs.shape()[rhs.dim()-2], rhs.shape()[rhs.dim()-1]);
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
                        let v2 = rhs.flat()[b_2_off + c * r2 + i];
                        tot = tot + v1 * v2;
                    }
                    out.flat_mut()[b_out_off + r * c2 + c] = tot;
                }
            }
        }

        if self.grad_enabled() || rhs.grad_enabled() {
            out.handle_mut().has_grad = true;
            out.handle_mut().grad_enabled = true;
            out.handle_mut().children = Children::MatmulBT(self.clone(), rhs.clone());
        }
        out
    }

    fn sum_1dim(&self, dim: usize) -> Tensor<T> {
        let mut new_shape = self.shape().clone();
        new_shape.remove(dim);
    
        let out = Tensor::<T>::zeros(new_shape);
    
        for idx in 0..self.size() {
            let dim_cnt = (idx / self.handle().stride[dim]) % self.shape()[dim];
            let dim_idx = idx % self.handle().stride[dim];
            
            // index math magic
            let out_idx = (idx - dim_cnt*self.handle().stride[dim] - dim_idx) / self.shape()[dim] + dim_idx;
            out.flat_mut()[out_idx] += self.flat()[idx];
        }

        if self.grad_enabled() {
            out.handle_mut().has_grad = true;
            out.handle_mut().grad_enabled = true;
            out.handle_mut().children = Children::DimSum(self.clone(), dim);
        }
        out
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

    pub fn sum_all(&self) -> Tensor<T> {
        let dims: Vec<usize> = (1..=self.dim()).collect();
        self.unsqueeze(0).sum(dims).squeeze()
    }

    fn prod_1dim(&self, dim: usize) -> Tensor<T> {
        let mut new_shape = self.shape().clone();
        new_shape.remove(dim);
    
        let out = Tensor::<T>::ones(new_shape);
    
        for idx in 0..self.size() {
            let dim_cnt = (idx / self.handle().stride[dim]) % self.shape()[dim];
            let dim_idx = idx % self.handle().stride[dim];
            
            // index math magic
            let out_idx = (idx - dim_cnt*self.handle().stride[dim] - dim_idx) / self.shape()[dim] + dim_idx;
            out.flat_mut()[out_idx] *= self.flat()[idx];
        }

        if self.grad_enabled() {
            out.handle_mut().has_grad = true;
            out.handle_mut().grad_enabled = true;
            out.handle_mut().children = Children::DimProd(self.clone(), dim);
        }
        out
    }

    pub fn prod<S: AsRef<[usize]>>(&self, dims: S) -> Tensor<T> {
        let mut dims = dims.as_ref().to_vec();
        dims.reverse();

        let mut out = self.clone();
        for dim in dims {
            out = out.prod_1dim(dim);
        }
        out
    }

    pub fn prod_all(&self) -> Tensor<T> {
        let dims: Vec<usize> = (1..=self.dim()).collect();
        self.unsqueeze(0).prod(dims).squeeze()
    }

    fn min_1dim(&self, dim: usize) -> Tensor<T> {
        let out = self.slice_1dim(dim, 0..1).squeeze_at(dim);
    
        for idx in 0..self.size() {
            let dim_cnt = (idx / self.handle().stride[dim]) % self.shape()[dim];
            let dim_idx = idx % self.handle().stride[dim];
            
            // index math magic
            let out_idx = (idx - dim_cnt*self.handle().stride[dim] - dim_idx) / self.shape()[dim] + dim_idx;
            if self.flat()[idx] < out.flat()[out_idx] {
                out.flat_mut()[out_idx] = self.flat()[idx];
            }
        }

        if self.grad_enabled() {
            out.handle_mut().has_grad = true;
            out.handle_mut().grad_enabled = true;
            out.handle_mut().children = Children::DimMin(self.clone(), dim);
        }
        out
    }

    pub fn min<S: AsRef<[usize]>>(&self, dims: S) -> Tensor<T> {
        let mut dims = dims.as_ref().to_vec();
        dims.reverse();

        let mut out = self.clone();
        for dim in dims {
            out = out.min_1dim(dim);
        }
        out
    }

    pub fn min_all(&self) -> Tensor<T> {
        let dims: Vec<usize> = (1..=self.dim()).collect();
        self.unsqueeze(0).min(dims).squeeze()
    }

    fn max_1dim(&self, dim: usize) -> Tensor<T> {
        let out = self.slice_1dim(dim, 0..1).squeeze_at(dim);
    
        for idx in 0..self.size() {
            let dim_cnt = (idx / self.handle().stride[dim]) % self.shape()[dim];
            let dim_idx = idx % self.handle().stride[dim];
            
            // index math magic
            let out_idx = (idx - dim_cnt*self.handle().stride[dim] - dim_idx) / self.shape()[dim] + dim_idx;
            if self.flat()[idx] > out.flat()[out_idx] {
                out.flat_mut()[out_idx] = self.flat()[idx];
            }
        }

        if self.grad_enabled() {
            out.handle_mut().has_grad = true;
            out.handle_mut().grad_enabled = true;
            out.handle_mut().children = Children::DimMax(self.clone(), dim);
        }
        out
    }

    pub fn max<S: AsRef<[usize]>>(&self, dims: S) -> Tensor<T> {
        let mut dims = dims.as_ref().to_vec();
        dims.reverse();

        let mut out = self.clone();
        for dim in dims {
            out = out.max_1dim(dim);
        }
        out
    }

    pub fn max_all(&self) -> Tensor<T> {
        let dims: Vec<usize> = (1..=self.dim()).collect();
        self.unsqueeze(0).max(dims).squeeze()
    }
}

impl<T: AnyNumber> Tensor<T> {
    pub fn min_with(&self, rhs: &Tensor<T>) -> Tensor<T> {
        let data = self.flat()
            .iter()
            .zip(rhs.flat().iter())
            .map(|(&x, &y)| if x < y { x } else { y })
            .collect();
        let children = Children::Min(self.clone(), rhs.clone());
        Tensor::from_op(self.shape().clone(), data, self.grad_enabled() || rhs.grad_enabled(), children)
    }

    pub fn max_with(&self, rhs: &Tensor<T>) -> Tensor<T> {
        let data = self.flat()
            .iter()
            .zip(rhs.flat().iter())
            .map(|(&x, &y)| if x > y { x } else { y })
            .collect();
        let children = Children::Max(self.clone(), rhs.clone());
        Tensor::from_op(self.shape().clone(), data, self.grad_enabled() || rhs.grad_enabled(), children)
    }
}

impl<T: AnyNumber> Neg for &Tensor<T> {
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
impl<T: AnyNumber> Neg for Tensor<T> { type Output = Tensor<T>; fn neg(self) -> Tensor<T> { -&self } }


impl<T: AnyNumber> Add<&Tensor<T>> for &Tensor<T> {
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
impl<T: AnyNumber> Add<&Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn add(self, rhs: &Tensor<T>) -> Tensor<T> { &self + rhs } }
impl<T: AnyNumber> Add<Tensor<T>> for &Tensor<T> { type Output = Tensor<T>; fn add(self, rhs: Tensor<T>) -> Tensor<T> { self + &rhs } }
impl<T: AnyNumber> Add<Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn add(self, rhs: Tensor<T>) -> Tensor<T> { &self + &rhs } }
impl<T: AnyNumber> AddAssign<&Tensor<T>> for Tensor<T> { fn add_assign(&mut self, rhs: &Tensor<T>) { *self = &*self + rhs; } }
impl<T: AnyNumber> AddAssign<Tensor<T>> for Tensor<T> { fn add_assign(&mut self, rhs: Tensor<T>) { *self = &*self + &rhs; } }


impl<T: AnyNumber> Sub<&Tensor<T>> for &Tensor<T> {
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
impl<T: AnyNumber>  Sub<&Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn sub(self, rhs: &Tensor<T>) -> Tensor<T> { self + (-rhs) } }
impl<T: AnyNumber>  Sub<Tensor<T>> for &Tensor<T> { type Output = Tensor<T>; fn sub(self, rhs: Tensor<T>) -> Tensor<T> { self + (-rhs) } }
impl<T: AnyNumber>  Sub<Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn sub(self, rhs: Tensor<T>) -> Tensor<T> { self + (-rhs) } }
impl<T: AnyNumber> SubAssign<&Tensor<T>> for Tensor<T> { fn sub_assign(&mut self, rhs: &Tensor<T>) { *self = &*self - rhs; } }
impl<T: AnyNumber> SubAssign<Tensor<T>> for Tensor<T> { fn sub_assign(&mut self, rhs: Tensor<T>) { *self = &*self - &rhs; } }


impl<T: AnyNumber> Mul<&Tensor<T>> for &Tensor<T> {
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
impl<T: AnyNumber> Mul<&Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn mul(self, rhs: &Tensor<T>) -> Tensor<T> { &self * rhs } }
impl<T: AnyNumber> Mul<Tensor<T>> for &Tensor<T> { type Output = Tensor<T>; fn mul(self, rhs: Tensor<T>) -> Tensor<T> { self * &rhs } }
impl<T: AnyNumber> Mul<Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn mul(self, rhs: Tensor<T>) -> Tensor<T> { &self * &rhs } }
impl<T: AnyNumber> MulAssign<&Tensor<T>> for Tensor<T> { fn mul_assign(&mut self, rhs: &Tensor<T>) { *self = &*self * rhs; } }
impl<T: AnyNumber> MulAssign<Tensor<T>> for Tensor<T> { fn mul_assign(&mut self, rhs: Tensor<T>) { *self = &*self * &rhs; } }


impl<T: AnyNumber> Div<&Tensor<T>> for &Tensor<T> {
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
impl<T: AnyNumber> Div<&Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn div(self, rhs: &Tensor<T>) -> Tensor<T> { &self / rhs } }
impl<T: AnyNumber> Div<Tensor<T>> for &Tensor<T> { type Output = Tensor<T>; fn div(self, rhs: Tensor<T>) -> Tensor<T> { self / &rhs } }
impl<T: AnyNumber> Div<Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn div(self, rhs: Tensor<T>) -> Tensor<T> { &self / &rhs } }
impl<T: AnyNumber> DivAssign<&Tensor<T>> for Tensor<T> { fn div_assign(&mut self, rhs: &Tensor<T>) { *self = &*self / rhs; } }
impl<T: AnyNumber> DivAssign<Tensor<T>> for Tensor<T> { fn div_assign(&mut self, rhs: Tensor<T>) { *self = &*self / &rhs; } }


// floating point operations
impl<T: AnyFloat> Tensor<T> {
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
