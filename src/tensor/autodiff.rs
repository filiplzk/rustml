use std::{cell::{Ref, RefCell, RefMut}, collections::HashSet, fmt, rc::Rc};
use super::*;
use num_traits::{Float, Num, NumAssign};

pub(super) enum Children<T: Num + Copy> {
    None,
    
    Id(Tensor<T>),
    Neg(Tensor<T>),
    Exp(Tensor<T>),
    Log(Tensor<T>),
    
    Add(Tensor<T>, Tensor<T>),
    Sub(Tensor<T>, Tensor<T>),
    Mul(Tensor<T>, Tensor<T>),
    Div(Tensor<T>, Tensor<T>),
    Min(Tensor<T>, Tensor<T>),
    Max(Tensor<T>, Tensor<T>)
}

impl<T: Float + fmt::Display> Children<T> {
    fn children_vec(&self) -> Vec<Tensor<T>> {
        match &self {
            Children::None => {
                vec![]
            }

            Children::Id(x) |
            Children::Neg(x) |
            Children::Exp(x) |
            Children::Log(x) => {
                vec![x.clone()]
            }

            Children::Add(x, y) |
            Children::Sub(x, y) |
            Children::Mul(x, y) |
            Children::Div(x, y) |
            Children::Min(x, y) |
            Children::Max(x, y) => {
                vec![x.clone(), y.clone()]
            }
        }
    }

    fn update_grads(&self, cur_grad: &Tensor<T>) {
        let mut tensors = vec![];
        let mut grads = vec![];
        match self {
            Children::None => (),

            Children::Id(t) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    grads.push(Tensor::ones_like(t))
                }
            }
            Children::Neg(t) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    grads.push(-Tensor::ones_like(t))
                }
            }
            Children::Exp(t) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    grads.push(t.exp());
                }
            }
            Children::Log(t) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    grads.push(Tensor::ones_like(t) / t);
                }
            }
            
            Children::Add(t1, t2) => {
                if t1.grad_enabled() {
                    tensors.push(t1);
                    grads.push(Tensor::ones_like(t1));
                }
                if t2.grad_enabled() {
                    tensors.push(t2);
                    grads.push(Tensor::ones_like(t2));
                }
            }
            Children::Sub(t1, t2) => {
                if t1.grad_enabled() {
                    tensors.push(t1);
                    grads.push(Tensor::ones_like(t1));
                }
                if t2.grad_enabled() {
                    tensors.push(t2);
                    grads.push(-Tensor::ones_like(t2));
                }
            }
            Children::Mul(t1, t2) => {
                if t1.grad_enabled() {
                    tensors.push(t1);
                    grads.push(t2.clone());
                }
                if t2.grad_enabled() {
                    tensors.push(t2);
                    grads.push(t1.clone());
                }
            }
            Children::Div(t1, t2) => {
                if t1.grad_enabled() {
                    tensors.push(t1);
                    grads.push(Tensor::ones_like(t2) / t2);
                }
                if t2.grad_enabled() {
                    tensors.push(t2);
                    grads.push(-t1 / (t2 * t2));
                }
            }
            Children::Min(t1, t2) => {
                if t1.grad_enabled() {
                    tensors.push(t1);
                    grads.push(Tensor::from_flat_like(t1, 
                    t1.flat()
                            .iter()
                            .zip(t2.grad().iter())
                            .map(|(&x, &y)| if x > y { T::one() } else { T::zero() })
                            .collect::<Vec<T>>()
                    ));
                }
                if t2.grad_enabled() {
                    tensors.push(t2);
                    grads.push(Tensor::from_flat_like(t2, 
                    t2.flat()
                            .iter()
                            .zip(t1.grad().iter())
                            .map(|(&y, &x)| if y > x { T::one() } else { T::zero() })
                            .collect::<Vec<T>>()
                    ));
                }
            }
            Children::Max(t1, t2) => {
                if t1.grad_enabled() {
                    tensors.push(t1);
                    grads.push(Tensor::from_flat_like(t1, 
                    t1.flat()
                            .iter()
                            .zip(t2.grad().iter())
                            .map(|(&x, &y)| if x < y { T::one() } else { T::zero() })
                            .collect::<Vec<T>>()
                    ));
                }
                if t2.grad_enabled() {
                    tensors.push(t2);
                    grads.push(Tensor::from_flat_like(t2, 
                    t2.flat()
                            .iter()
                            .zip(t1.grad().iter())
                            .map(|(&y, &x)| if y < x { T::one() } else { T::zero() })
                            .collect::<Vec<T>>()
                    ));
                }
            }
        }

        for (&t, g) in tensors.iter().zip(grads.iter()) {
            *t.grad_mut() = (t.grad_tensor() + g * cur_grad).flat().clone();
        }

    }
}

impl<T: Float + fmt::Display> Tensor<T> {
    fn rev_toposort(&self, vec: &mut Vec<Tensor<T>>, seen: &mut HashSet<usize>) {
        seen.insert(self.id());
        
        for child in self.handle().children.children_vec() {
            if !seen.contains(&child.id()) {
                child.rev_toposort(vec, seen);
            }
        }
    
        if self.grad_enabled() {
            vec.push(self.clone());
        }
    }

    fn toposort(&self) -> Vec<Tensor<T>> {
        let mut topo = Vec::new();
        self.rev_toposort(&mut topo, &mut HashSet::new());
        topo.reverse();
        topo
    }

    pub fn backward(&self) {
        // assert!(self.size() == 1, "Can't backpropagate on a tensor holding more than 1 value");
        assert!(self.grad_enabled(), "Can't backpropagate on a tensor with disabled gradients");

        self.handle_mut().grad = vec![T::one(); self.size()];
    
        for t in self.toposort() {
            let cur_grad = &t.grad_tensor();
            t.handle_mut().children.update_grads(cur_grad);
        }
    }

    pub fn zero_grad(&self) {
        assert!(self.size() == 1, "Can't reset on a tensor holding more than 1 value");
        assert!(self.grad_enabled(), "Can't reset on a tensor with disabled gradients");

        for t in self.toposort() {
            t.handle_mut().grad = vec![T::zero(); t.size()];
            t.handle_mut().children = Children::None;
        }
    }
}