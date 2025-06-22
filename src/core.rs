use std::f32::consts::PI;
use std::fmt;
use std::rc::Rc;
use std::cell::{Ref, RefCell, RefMut};

use rand::rngs::ThreadRng;
use rand::Rng;

#[derive(Debug)]
pub struct ValueData {
    pub val: f32,
    pub grad: f32,
    pub children: Children,
    pub vis: bool
}

#[derive(Debug, Clone)]
pub enum Children {
    None,
    Add(Value, Value),
    Mul(Value, Value),
    Inv(Value),
    Exp(Value),
    Max(Value, Value),  // max(variable, const)
    Log(Value),
}

impl Children {
    fn children_vec(&self) -> Vec<Value> {
        match self {
            Children::None => vec![],
            Children::Add(v1, v2) => vec![v1.clone(), v2.clone()],
            Children::Mul(v1, v2) => vec![v1.clone(), v2.clone()],
            Children::Inv(v1) => vec![v1.clone()],
            Children::Exp(v1) => vec![v1.clone()],
            Children::Max(v1, _) => vec![v1.clone()],
            Children::Log(v1) => vec![v1.clone()]
        }
    }
    
    fn update_grads(&mut self, cur_grad: f32) {
        match &self {
            Children::None => (),
            
            Children::Add(v1, v2) => {
                v1.data_mut().grad += 1.0 * cur_grad;
                v2.data_mut().grad += 1.0 * cur_grad;
            }
            
            Children::Mul(v1, v2) => {
                let v1val = v1.data().val;
                let v2val = v2.data().val;
                v1.data_mut().grad += v2val * cur_grad;
                v2.data_mut().grad += v1val * cur_grad;
            }

            Children::Inv(v1) => {
                let val = v1.data().val;
                v1.data_mut().grad += -1.0 / (val * val) * cur_grad;
            }

            Children::Exp(v1) => {
                let val = v1.data().val;
                v1.data_mut().grad += val.exp() * cur_grad;
            }

            Children::Max(v1, c) => {
                if v1.data().val > c.data().val {
                    v1.data_mut().grad += 1.0 * cur_grad;
                }
            }

            Children::Log(v1) => {
                let val = v1.data().val;
                v1.data_mut().grad += 1.0 / val * cur_grad;
            }
        };
    }
}

pub trait ParamGroup {
    fn params(&self) -> Vec<Value>;
}

#[derive(Debug, Clone)]
pub struct Value {
    data: Rc<RefCell<ValueData>>    
}

impl ParamGroup for Value {
    fn params(&self) -> Vec<Value> {
        vec![self.clone()]
    }
}

impl Value {
    pub fn new(val: f32) -> Self {
        Self {
            data: Rc::new(RefCell::new(
                ValueData {
                    val,
                    grad: 0.0,
                    children: Children::None,
                    vis: false
                }
            ))
        }
    }

    pub fn new_uniform(r: &mut ThreadRng, low: f32, high: f32) -> Self {
        let val: f32 = r.r#gen();
        Self::new(low + (high-low) * val)
    }

    pub fn new_normal(r: &mut ThreadRng, mean: f32, std: f32) -> Self {
        let u1: f32 = r.r#gen();
        let u2: f32 = r.r#gen();
        let sampled = (-2.0 * u1.ln()).sqrt() * (2.0*PI*u2).cos();
        Self::new(mean + sampled * std)
    }

    pub fn data(&self) -> Ref<ValueData> {
        self.data.borrow()
    }

    pub fn data_mut(&self) -> RefMut<ValueData> {
        self.data.borrow_mut()
    }

    fn rev_toposort(&self, vec: &mut Vec<Value>) {
        self.data_mut().vis = true;

        for child in self.data().children.children_vec() {
            if !child.data().vis {
                child.rev_toposort(vec);
            }
        }

        vec.push(self.clone());
    }

    pub fn backward(&self) {
        self.data_mut().grad = 1.0;

        let mut topo = Vec::new();
        self.rev_toposort(&mut topo);
        topo.reverse();

        for val in topo {
            let grad = val.data().grad;
            val.data_mut().children.update_grads(grad);
        }
    }

    pub fn reset_graph(&self) {
        self.data_mut().grad = 0.0;
        self.data_mut().vis = false;
        for child in self.data().children.children_vec() {
            if child.data().vis {
                child.reset_graph();
            }
        }

    }

    pub fn inv(&self) -> Value {
        let out = Value::new(1.0 / self.data().val);
        out.data_mut().children = Children::Inv(self.clone());
        out
    }

    pub fn exp(&self) -> Value {
        let out = Value::new(self.data().val.exp());
        out.data_mut().children = Children::Exp(self.clone());
        out
    }

    pub fn max(&self, rhs: &Value) -> Value {
        let out = Value::new(self.data().val.max(rhs.data().val));
        out.data_mut().children = Children::Max(self.clone(), rhs.clone());
        out
    }

    pub fn log(&self) -> Value {
        let out = Value::new(self.data().val.ln());
        out.data_mut().children = Children::Log(self.clone());
        out
    }

}

impl std::ops::Add<&Value> for &Value {
    type Output = Value;
    fn add(self, rhs: &Value) -> Value {
        let out = Value::new(self.data().val + rhs.data().val);
        out.data_mut().children = Children::Add(self.clone(), rhs.clone());
        out
    }
}
impl std::ops::Add<&Value> for Value { type Output = Value; fn add(self, rhs: &Value) -> Value { &self + rhs } }
impl std::ops::Add<Value> for &Value { type Output = Value; fn add(self, rhs: Value) -> Value { self + &rhs } }
impl std::ops::Add<Value> for Value { type Output = Value; fn add(self, rhs: Value) -> Value { &self + &rhs } }
impl std::ops::AddAssign<&Value> for Value { fn add_assign(&mut self, rhs: &Value) { *self = &*self + rhs; } }
impl std::ops::AddAssign<Value> for Value { fn add_assign(&mut self, rhs: Value) { *self = &*self + &rhs; } }


impl std::ops::Neg for &Value {
    type Output = Value;
    fn neg(self) -> Value {
        self * Value::new(-1.0)
    }
}
impl std::ops::Neg for Value { type Output = Value; fn neg(self) -> Value { -&self } }

impl std::ops::Sub<&Value> for &Value { type Output = Value; fn sub(self, rhs: &Value) -> Value { self + (-rhs) } }
impl std::ops::Sub<&Value> for Value { type Output = Value; fn sub(self, rhs: &Value) -> Value { self + (-rhs) } }
impl std::ops::Sub<Value> for &Value { type Output = Value; fn sub(self, rhs: Value) -> Value { self + (-rhs) } }
impl std::ops::Sub<Value> for Value { type Output = Value; fn sub(self, rhs: Value) -> Value { self + (-rhs) } }
impl std::ops::SubAssign<&Value> for Value { fn sub_assign(&mut self, rhs: &Value) { *self = &*self - rhs; } }
impl std::ops::SubAssign<Value> for Value { fn sub_assign(&mut self, rhs: Value) { *self = &*self - &rhs; } }

impl std::ops::Mul for &Value {
    type Output = Value;
    fn mul(self, other: &Value) -> Value {
        let out = Value::new(self.data().val * other.data().val);
        out.data_mut().children = Children::Mul(self.clone(), other.clone());
        out
    }
}
impl std::ops::Mul<&Value> for Value { type Output = Value; fn mul(self, other: &Value) -> Value { &self * other } }
impl std::ops::Mul<Value> for &Value { type Output = Value; fn mul(self, other: Value) -> Value { self * &other } }
impl std::ops::Mul<Value> for Value { type Output = Value; fn mul(self, other: Value) -> Value { &self * &other } }
impl std::ops::MulAssign<&Value> for Value { fn mul_assign(&mut self, rhs: &Value) { *self = &*self * rhs; } }
impl std::ops::MulAssign<Value> for Value { fn mul_assign(&mut self, rhs: Value) { *self = &*self * &rhs; } }

impl std::ops::Div for &Value {
    type Output = Value;
    fn div(self, other: &Value) -> Value {
        let out = self * other.inv();
        out
    }
}
impl std::ops::Div<&Value> for Value { type Output = Value; fn div(self, other: &Value) -> Value { &self / other } }
impl std::ops::Div<Value> for &Value { type Output = Value; fn div(self, other: Value) -> Value { self / &other } }
impl std::ops::Div<Value> for Value { type Output = Value; fn div(self, other: Value) -> Value { &self / &other } }
impl std::ops::DivAssign<&Value> for Value { fn div_assign(&mut self, rhs: &Value) { *self = &*self / rhs; } }
impl std::ops::DivAssign<Value> for Value { fn div_assign(&mut self, rhs: Value) { *self = &*self / &rhs; } }

// ---- debugging stuff

fn op_label(c: Children) -> String {
    match c {
        Children::None => "Leaf".to_string(),
        Children::Add(_, _) => "Sum of:".to_string(),
        Children::Mul(_, _) => "Product of: ".to_string(),
        Children::Inv(_) => "Inverse of: ".to_string(),
        Children::Exp(_) => "Exp of: ".to_string(),
        Children::Max(_, _) => "Max of: ".to_string(),
        Children::Log(_) => "Log of: ".to_string(),
    }
}

impl fmt::Display for Value { // For user-facing output with `{}`
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(val={}, grad={})", self.data().val, self.data().grad)
    }
}

pub fn pretty_print(val: &Value, depth: usize) {
    if depth > 0 {
        let pipes: String = std::iter::repeat("| ").take(depth).collect();
        print!("{}", pipes);
    }
    println!("{}, {}", val, op_label(val.data().children.clone()));
    for child in val.data().children.children_vec() {
        pretty_print(&child, depth + 1);
    }

}


// vector and matrix

#[derive(Clone)]
pub struct Matrix {
    data: Vec<Value>,
    pub shape: (usize, usize)
}

impl ParamGroup for Matrix {
    fn params(&self) -> Vec<Value> {
        let mut out: Vec<Value> = Vec::new();
        
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                out.push(self[(r, c)].clone());
            }
        }

        out
    }
}

impl Matrix {
    pub fn fill(shape: (usize, usize), val: f32) -> Self {
        let mut data = vec![Value::new(0.0); shape.0 * shape.1];
        for i in 0..shape.0 * shape.1 {
            data[i] = Value::new(val);
        }

        Self { data, shape }
    }

    pub fn fill_value(shape: (usize, usize), val: &Value) -> Self {
        let mut data = vec![Value::new(0.0); shape.0 * shape.1];
        for i in 0..shape.0 * shape.1 {
            data[i] = val.clone();
        }

        Self { data, shape }
    }
    
    pub fn new(nums: &[&[f32]]) -> Self {
        let shape = (nums.len(), nums[0].len());
        let mut out: Matrix = Matrix::fill(shape, 0.0);
        for r in 0..shape.0 {
            for c in 0..shape.1 {
                out[(r, c)] = Value::new(nums[r][c]);
            }
        }

        out
    }

    pub fn new_uniform(r: &mut ThreadRng, shape: (usize, usize), low: f32, high: f32) -> Self {
        let data = (0..shape.0 * shape.1)
            .map(|_| Value::new_uniform(r, low, high))
            .collect();

        Self { data, shape }
    }

    pub fn new_normal(r: &mut ThreadRng, shape: (usize, usize), mean: f32, std: f32) -> Self {
        let data = (0..shape.0 * shape.1)
            .map(|_| Value::new_normal(r, mean, std))
            .collect();

        Self { data, shape }
    }

    pub fn new_he(r: &mut ThreadRng, shape: (usize, usize)) -> Self {
        Self::new_normal(r, shape, 0.0, (2.0 / shape.0 as f32).sqrt())
    }

    pub fn one_hot(size: usize, idx: usize) -> Self {
        let mut out = Self::fill((1, size), 0.0);
        out[(0, idx)] = Value::new(1.0);
        out
    }

    pub fn row(&self, index: usize) -> Matrix {
        let mut out = Matrix::fill((1, self.shape.1), 0.0);
        for c in 0..self.shape.1 {
            out[(index, c)] = self[(index, c)].clone()
        }

        out
    }

    pub fn sum(&self) -> Value {
        let mut cur = Value::new(0.0);
        for val in &self.data {
            cur += val;
        }
        cur
    }

    pub fn product(&self) -> Value {
        let mut cur = Value::new(1.0);
        for val in &self.data {
            cur *= val
        }
        cur
    }

    pub fn transpose(&self) -> Matrix {
        let (r, c) = self.shape;
        let mut out = Matrix::fill((c, r), 0.0);
        for y in 0..r {
            for x in 0..c {
                out[(x, y)] = self[(y, x)].clone();
            }
        }
        out
    }

    pub fn hadamard_product(&self, rhs: &Matrix) -> Matrix {
        assert!(self.shape == rhs.shape);

        let mut out = Matrix::fill(self.shape, 0.0);

        for y in 0..self.shape.0 {
            for x in 0..self.shape.1 {
                out[(y, x)] = &self[(y, x)] * &rhs[(y, x)];
            }
        }

        out
    }

    pub fn inv_each(&self) -> Matrix {
        let data = (0..self.shape.0 * self.shape.1)
            .map(|i: usize| self.data[i].inv())
            .collect();

        Matrix { data, shape: self.shape }
    }

    pub fn exp_each(&self) -> Matrix {
        let data = (0..self.shape.0 * self.shape.1)
            .map(|i: usize| self.data[i].exp())
            .collect();

        Matrix { data, shape: self.shape }
    }

    pub fn max(&self, rhs: &Matrix) -> Matrix {
        let data = (0..self.shape.0 * self.shape.1)
            .map(|i: usize| self.data[i].max(&rhs.data[i]))
            .collect();

        Matrix { data, shape: self.shape }
    }

    pub fn average(&self) -> Value {
        self.sum() / Value::new((self.shape.0 * self.shape.1) as f32)
    }
}

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = Value;
    fn index(&self, index: (usize, usize)) -> &Value {
        &self.data[self.shape.1 * index.0 + index.1]
    }
}
impl std::ops::IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Value {
        &mut self.data[self.shape.1 * index.0 + index.1]
    }
}

impl std::ops::Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Matrix {
        assert!(self.shape == rhs.shape);

        let data: Vec<Value> = (0..self.shape.0 * self.shape.1)
            .map(|i: usize| &self.data[i] + &rhs.data[i])
            .collect();

        Matrix { data, shape: self.shape }
    }
}
impl std::ops::Add<&Matrix> for Matrix { type Output = Matrix; fn add(self, rhs: &Matrix) -> Matrix { &self + rhs } }
impl std::ops::Add<Matrix> for &Matrix { type Output = Matrix; fn add(self, rhs: Matrix) -> Matrix { self + &rhs } }
impl std::ops::Add<Matrix> for Matrix { type Output = Matrix; fn add(self, rhs: Matrix) -> Matrix { &self + &rhs } }
impl std::ops::AddAssign<&Matrix> for Matrix { fn add_assign(&mut self, rhs: &Matrix) { *self = &*self + rhs; } }
impl std::ops::AddAssign<Matrix> for Matrix { fn add_assign(&mut self, rhs: Matrix) { *self = &*self + &rhs; } }

impl std::ops::Neg for &Matrix {
    type Output = Matrix;
    fn neg(self) -> Matrix {
        let data: Vec<Value> = (0..self.shape.0 * self.shape.1)
            .map(|i: usize| -self.data[i].clone())
            .collect();

        Matrix { data, shape: self.shape }
    }
}
impl std::ops::Neg for Matrix { type Output = Matrix; fn neg(self) -> Matrix { -&self } }

impl std::ops::Sub<&Matrix> for &Matrix { type Output = Matrix; fn sub(self, rhs: &Matrix) -> Matrix { self + (-rhs) } }
impl std::ops::Sub<&Matrix> for Matrix { type Output = Matrix; fn sub(self, rhs: &Matrix) -> Matrix { self + (-rhs) } }
impl std::ops::Sub<Matrix> for &Matrix { type Output = Matrix; fn sub(self, rhs: Matrix) -> Matrix { self + (-rhs) } }
impl std::ops::Sub<Matrix> for Matrix { type Output = Matrix; fn sub(self, rhs: Matrix) -> Matrix { self + (-rhs) } }
impl std::ops::SubAssign<&Matrix> for Matrix { fn sub_assign(&mut self, rhs: &Matrix) { *self = &*self - rhs; } }
impl std::ops::SubAssign<Matrix> for Matrix { fn sub_assign(&mut self, rhs: Matrix) { *self = &*self - &rhs; } }

impl std::ops::Mul for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Matrix {
        assert!(self.shape.1 == rhs.shape.0);

        let mut out = Matrix::fill((self.shape.0, rhs.shape.1), 0.0);

        let rhs_t = rhs.transpose();

        for y in 0..out.shape.0 {
            for x in 0..out.shape.1 {
                for i in 0..self.shape.1 {
                    out[(y, x)] += &self[(y, i)] * &rhs_t[(x, i)];
                }
            }
        }

        out
    }
}
impl std::ops::Mul<&Matrix> for Matrix { type Output = Matrix; fn mul(self, other: &Matrix) -> Matrix { &self * other } }
impl std::ops::Mul<Matrix> for &Matrix { type Output = Matrix; fn mul(self, other: Matrix) -> Matrix { self * &other } }
impl std::ops::Mul<Matrix> for Matrix { type Output = Matrix; fn mul(self, other: Matrix) -> Matrix { &self * &other } }
impl std::ops::MulAssign<&Matrix> for Matrix { fn mul_assign(&mut self, rhs: &Matrix) { *self = &*self * rhs; } }
impl std::ops::MulAssign<Matrix> for Matrix { fn mul_assign(&mut self, rhs: Matrix) { *self = &*self * &rhs; } }

impl std::ops::Div<&Matrix> for &Matrix {
    type Output = Matrix;

    fn div(self, rhs: &Matrix) -> Matrix {
        assert!(self.shape == rhs.shape);

        self.hadamard_product(&rhs.inv_each())
    }
}
impl std::ops::Div<&Matrix> for Matrix { type Output = Matrix; fn div(self, rhs: &Matrix) -> Matrix { &self / rhs } }
impl std::ops::Div<Matrix> for &Matrix { type Output = Matrix; fn div(self, rhs: Matrix) -> Matrix { self / &rhs } }
impl std::ops::Div<Matrix> for Matrix { type Output = Matrix; fn div(self, rhs: Matrix) -> Matrix { &self / &rhs } }
impl std::ops::DivAssign<&Matrix> for Matrix { fn div_assign(&mut self, rhs: &Matrix) { *self = &*self / rhs; } }
impl std::ops::DivAssign<Matrix> for Matrix { fn div_assign(&mut self, rhs: Matrix) { *self = &*self / &rhs; } }


// debugging

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = String::new();

        for r in 0..self.shape.0 {
            out.push_str("{ ");
            for c in 0..self.shape.1 {
                out.push_str(format!("{} ", self[(r, c)].data().val).as_str());
            }
            out.push_str("}\n");
        }
        out.pop();

        write!(f, "{}", out)
    }
}