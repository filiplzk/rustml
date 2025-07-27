use std::f32::consts::PI;
use std::rc::Rc;
use std::cell::{RefCell, RefMut};
use std::fmt;
use std::ops::{Add, Neg, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign};

use rand::distr::uniform::{SampleRange, SampleUniform};
use rand::distr::{Distribution, StandardUniform};
use rand::Rng;

use num_traits::{Float, Num, NumAssignOps};

#[derive(Clone)]
pub(super) struct ScalarData<T: Num + Copy> {
    pub val: T,
    pub grad: Option<T>,
    pub children: Children<T>,
    pub locked: bool
}

#[derive(Clone)]
pub(super) enum Children<T: Num + Copy> {
    None,
    Neg(Scalar<T>),
    Exp(Scalar<T>),
    Log(Scalar<T>),
    Max(Scalar<T>, T),  // max(variable, const)
    Add(Scalar<T>, Scalar<T>),
    Mul(Scalar<T>, Scalar<T>),
    Div(Scalar<T>, Scalar<T>)
}

impl<T: Num + Copy> Children<T> {
    fn children_vec(&self) -> Vec<Scalar<T>> {
        match self {
            Children::None => {
                vec![]
            }
            
            Children::Neg(v1) |
            Children::Exp(v1) |
            Children::Log(v1) |
            Children::Max(v1, _) => {
                vec![v1.clone()]
            }

            Children::Add(v1, v2) |
            Children::Mul(v1, v2) | 
            Children::Div(v1, v2) => {
                vec![v1.clone(), v2.clone()]
            }
        }
    }
}

impl<T: Float + NumAssignOps> Children<T> {
    fn update_grads(&mut self, cur_grad: T) {
        match &self {
            Children::None => (),
            
            Children::Neg(v1) => {
                if let Some(grad) = &mut v1.handle().grad {
                    *grad += T::from(-1.0).unwrap() * cur_grad;
                }
            }
            Children::Exp(v1) => {
                let val = v1.val();
                if let Some(grad) = &mut v1.handle().grad {
                    *grad += val.exp() * cur_grad;
                }
            }
            Children::Log(v1) => {
                let val = v1.val();
                if let Some(grad) = &mut v1.handle().grad {
                    *grad += T::from(1.0).unwrap() / val * cur_grad;
                }
            }
            Children::Max(v1, c) => {
                let val = v1.val();
                if let Some(grad) = &mut v1.handle().grad {
                    if val > *c {
                        *grad += T::from(1.0).unwrap() * cur_grad;
                    }
                }
            }
    
            Children::Add(v1, v2) => {
                if let Some(grad) = &mut v1.handle().grad {
                    *grad += T::from(1.0).unwrap() * cur_grad;
                }
                if let Some(grad) = &mut v2.handle().grad {
                    *grad += T::from(1.0).unwrap() * cur_grad;
                }
            }
            Children::Mul(v1, v2) => {
                let v1val = v1.val();
                let v2val = v2.val();
                if let Some(grad) = &mut v1.handle().grad {
                    *grad += v2val * cur_grad;
                }
                if let Some(grad) = &mut v2.handle().grad {
                    *grad += v1val * cur_grad;
                }
            }
            Children::Div(v1, v2) => {
                let v1val = v1.val();
                let v2val = v2.val();
                if let Some(grad) = &mut v1.handle().grad {
                    *grad += T::one() / v2val * cur_grad;
                }
                if let Some(grad) = &mut v2.handle().grad {
                    *grad += -v1val / (v2val * v2val) * cur_grad;
                }
            }

        };
    }
}

#[derive(Clone)]
pub struct Scalar<T: Num + Copy> {
    data: Rc<RefCell<ScalarData<T>>>
}

impl<T: Num + Copy> Scalar<T> {
    pub fn new(val: T) -> Self {
        Self {
            data: Rc::new(RefCell::new(
                ScalarData {
                    val,
                    grad: None,
                    children: Children::None,
                    locked: false
                }
            ))
        }
    }

    pub fn val(&self) -> T {
        self.data.borrow().val
    }
    
    pub(super) fn handle(&self) -> RefMut<ScalarData<T>> {
        self.data.borrow_mut()
    }

    pub fn has_grad(&self) -> bool {
        self.handle().grad.is_some()
    }

    pub fn set(&self, val: T) {
        assert!(self.handle().locked || !self.has_grad(), "set(): A Scalar with a gradient can only be modified after backpropagation");
        self.handle().val = val;
    }

    pub fn assign(&mut self, val: Scalar<T>) {
        self.data = val.data;
    }

    fn _with_grad(self) -> Self {
        self.handle().grad = Some(T::zero());
        self
    }
}

impl<T: Num + Copy + SampleUniform> Scalar<T> {
    pub fn new_uniform(r: &mut impl Rng, range: impl SampleRange<T>) -> Self {
        Self::new(r.random_range(range))
    }
}

impl<T: Num + Copy + PartialOrd> Scalar<T> {
    pub fn max(&self, rhs: &Scalar<T>) -> Scalar<T> {
        let max_val = if self.val() > rhs.val() { self.val() } else { rhs.val() };

        let mut out = Scalar::new(max_val);
        if self.has_grad() {
            out = out._with_grad();
            out.handle().children = Children::Max(self.clone(), rhs.val());
        }
        out
    }
}



impl<T: Float + NumAssignOps> Scalar<T> {
    fn rev_toposort(&self, vec: &mut Vec<Scalar<T>>, lock: bool) {
        self.handle().locked = lock;
    
        for child in &self.handle().children.children_vec() {
            if child.handle().locked != lock {
                child.rev_toposort(vec, lock);
            }
        }
    
        if self.has_grad() {
            vec.push(self.clone());
        }
    }

    pub fn backward(&self) -> Vec<Scalar<T>> {
        assert!(self.handle().grad.is_some(), "backward(): Can't start backpropagation on a detached Scalar");
    
        self.handle().grad = Some(T::one());
    
        let mut topo = Vec::new();
        self.rev_toposort(&mut topo, true);
        topo.reverse();
    
        for val in &topo {
            let grad = val.grad();
            val.handle().children.update_grads(grad);
        }

        topo
    }
    
    pub fn backward_reset(&self, topo: Vec<Scalar<T>>) {
        for val in &topo {
            val.handle().grad = Some(T::zero());
            val.handle().locked = false;
        }
    }
}


impl<T: Float> Scalar<T> {
    pub fn grad(&self) -> T {
        self.data.borrow().grad.expect("grad(): Float Scalar has no grad")
    }
    
    pub fn with_grad(self) -> Self {
        self._with_grad()
    }
    
    pub fn exp(&self) -> Scalar<T> {
        let mut out = Scalar::new(self.val().exp());
        if self.has_grad() {
            out = out._with_grad();
            out.handle().children = Children::Exp(self.clone());
        }
        out
    }
    
    pub fn log(&self) -> Scalar<T> {
        let mut out = Scalar::new(self.val().ln());
        if self.has_grad() {
            out = out._with_grad();
            out.handle().children = Children::Log(self.clone());
        }
        out
    }
}

impl<T: Float> Scalar<T>
where 
    StandardUniform: Distribution<T>
{
    pub fn new_normal(r: &mut impl Rng, mean: T, std: T) -> Self {
        let u1: T = r.random();
        let u2: T = r.random();
        let sampled = (T::from(-2.0).unwrap() * u1.ln()).sqrt() * (T::from(2.0*PI).unwrap() * u2).cos();
        Self::new(mean + sampled * std)
    }
}


impl<T: Num + Copy> Add<&Scalar<T>> for &Scalar<T> {
    type Output = Scalar<T>;
    fn add(self, rhs: &Scalar<T>) -> Scalar<T> {
        let mut out = Scalar::new(self.val() + rhs.val());
        if self.has_grad() || rhs.has_grad() {
            out = out._with_grad();
            out.handle().children = Children::Add(self.clone(), rhs.clone());
        }
        out
    }
}
impl<T: Num + Copy> Add<&Scalar<T>> for Scalar<T> { type Output = Scalar<T>; fn add(self, rhs: &Scalar<T>) -> Scalar<T> { &self + rhs } }
impl<T: Num + Copy> Add<Scalar<T>> for &Scalar<T> { type Output = Scalar<T>; fn add(self, rhs: Scalar<T>) -> Scalar<T> { self + &rhs } }
impl<T: Num + Copy> Add<Scalar<T>> for Scalar<T> { type Output = Scalar<T>; fn add(self, rhs: Scalar<T>) -> Scalar<T> { &self + &rhs } }
impl<T: Num + Copy> AddAssign<&Scalar<T>> for Scalar<T> { fn add_assign(&mut self, rhs: &Scalar<T>) { *self = &*self + rhs; } }
impl<T: Num + Copy> AddAssign<Scalar<T>> for Scalar<T> { fn add_assign(&mut self, rhs: Scalar<T>) { *self = &*self + &rhs; } }


impl<T: Num + Copy> Neg for &Scalar<T> {
    type Output = Scalar<T>;
    fn neg(self) -> Scalar<T> {
        let mut out = Scalar::new(T::zero() - self.val());
        if self.has_grad() {
            out = out._with_grad();
            out.handle().children = Children::Neg(self.clone());
        }
        out
    }
}
impl<T: Num + Copy> Neg for Scalar<T> { type Output = Scalar<T>; fn neg(self) -> Scalar<T> { -&self } }

impl<T: Num + Copy>  Sub<&Scalar<T>> for &Scalar<T> { type Output = Scalar<T>; fn sub(self, rhs: &Scalar<T>) -> Scalar<T> { self + (-rhs) } }
impl<T: Num + Copy>  Sub<&Scalar<T>> for Scalar<T> { type Output = Scalar<T>; fn sub(self, rhs: &Scalar<T>) -> Scalar<T> { self + (-rhs) } }
impl<T: Num + Copy>  Sub<Scalar<T>> for &Scalar<T> { type Output = Scalar<T>; fn sub(self, rhs: Scalar<T>) -> Scalar<T> { self + (-rhs) } }
impl<T: Num + Copy>  Sub<Scalar<T>> for Scalar<T> { type Output = Scalar<T>; fn sub(self, rhs: Scalar<T>) -> Scalar<T> { self + (-rhs) } }
impl<T: Num + Copy> SubAssign<&Scalar<T>> for Scalar<T> { fn sub_assign(&mut self, rhs: &Scalar<T>) { *self = &*self - rhs; } }
impl<T: Num + Copy> SubAssign<Scalar<T>> for Scalar<T> { fn sub_assign(&mut self, rhs: Scalar<T>) { *self = &*self - &rhs; } }

impl<T: Num + Copy>  Mul for &Scalar<T> {
    type Output = Scalar<T>;
    fn mul(self, rhs: &Scalar<T>) -> Scalar<T> {
        let mut out = Scalar::new(self.val() * rhs.val());
        if self.has_grad() || rhs.has_grad() {
            out = out._with_grad();
            out.handle().children = Children::Mul(self.clone(), rhs.clone());
        }
        out
    }
}
impl<T: Num + Copy> Mul<&Scalar<T>> for Scalar<T> { type Output = Scalar<T>; fn mul(self, rhs: &Scalar<T>) -> Scalar<T> { &self * rhs } }
impl<T: Num + Copy> Mul<Scalar<T>> for &Scalar<T> { type Output = Scalar<T>; fn mul(self, rhs: Scalar<T>) -> Scalar<T> { self * &rhs } }
impl<T: Num + Copy> Mul<Scalar<T>> for Scalar<T> { type Output = Scalar<T>; fn mul(self, rhs: Scalar<T>) -> Scalar<T> { &self * &rhs } }
impl<T: Num + Copy> MulAssign<&Scalar<T>> for Scalar<T> { fn mul_assign(&mut self, rhs: &Scalar<T>) { *self = &*self * rhs; } }
impl<T: Num + Copy> MulAssign<Scalar<T>> for Scalar<T> { fn mul_assign(&mut self, rhs: Scalar<T>) { *self = &*self * &rhs; } }

impl<T: Num + Copy> Div for &Scalar<T> {
    type Output = Scalar<T>;
    fn div(self, rhs: &Scalar<T>) -> Scalar<T> {
        let mut out = Scalar::new(self.val() / rhs.val());
        if self.has_grad() || rhs.has_grad() {
            out = out._with_grad();
            out.handle().children = Children::Div(self.clone(), rhs.clone());
        }
        out
    }
}
impl<T: Num + Copy> Div<&Scalar<T>> for Scalar<T> { type Output = Scalar<T>; fn div(self, rhs: &Scalar<T>) -> Scalar<T> { &self / rhs } }
impl<T: Num + Copy> Div<Scalar<T>> for &Scalar<T> { type Output = Scalar<T>; fn div(self, rhs: Scalar<T>) -> Scalar<T> { self / &rhs } }
impl<T: Num + Copy> Div<Scalar<T>> for Scalar<T> { type Output = Scalar<T>; fn div(self, rhs: Scalar<T>) -> Scalar<T> { &self / &rhs } }
impl<T: Num + Copy> DivAssign<&Scalar<T>> for Scalar<T> { fn div_assign(&mut self, rhs: &Scalar<T>) { *self = &*self / rhs; } }
impl<T: Num + Copy> DivAssign<Scalar<T>> for Scalar<T> { fn div_assign(&mut self, rhs: Scalar<T>) { *self = &*self / &rhs; } }

// ---- debugging stuff

impl<T: Num + Copy + fmt::Debug> fmt::Debug for Scalar<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let grad_str: String = match self.handle().grad {
            None => "None".to_string(),
            Some(g) => format!("{:?}", g)
        };
        write!(f, "Scalar(val={:?}, grad={})", self.val(), grad_str)
    }
}

impl<T: Num + Copy + fmt::Display> fmt::Display for Scalar<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.5}", self.val())
    }
}


// fn op_label(c: &Children) -> String {
//     match c {
//         Children::None => "Leaf".to_string(),
//         Children::Add(_, _) => "Sum of:".to_string(),
//         Children::Mul(_, _) => "Product of: ".to_string(),
//         Children::Inv(_) => "Inverse of: ".to_string(),
//         Children::Exp(_) => "Exp of: ".to_string(),
//         Children::Max(_, _) => "Max of: ".to_string(),
//         Children::Log(_) => "Log of: ".to_string(),
//     }
// }

// pub fn print_backward(val: &Scalar, depth: usize) {
//     if depth > 0 {
//         let pipes: String = std::iter::repeat("| ").take(depth).collect();
//         print!("{}", pipes);
//     }
//     println!("{}, {}", val, op_label(&val.handle().children.clone()));
//     for child in &val.handle().children.children_vec() {
//         print_backward(&child, depth + 1);
//     }
// }
