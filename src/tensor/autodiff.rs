use std::{collections::HashSet, ops::Range, time::Instant};
use crate::functional;

use super::*;
use num_traits::{Float, Num, NumAssignOps};

/// An object containing all information on how the tensor was created
/// Connects "tensor nodes" in a "computation graph" and makes the gradient flow during backpropagation
#[derive(Clone)]
pub(crate) enum Children<T: AnyNumber> {
    None,
    
    Id(Tensor<T>),
    Neg(Tensor<T>),
    Exp(Tensor<T>),
    Log(Tensor<T>),
    Sigmoid(Tensor<T>),
    DimSum(Tensor<T>, usize),
    DimProd(Tensor<T>, usize),
    DimMin(Tensor<T>, usize),
    DimMax(Tensor<T>, usize),
    NewDim(Tensor<T>, usize),
    Reshape(Tensor<T>),
    Slice(Tensor<T>, usize, Range<usize>),

    Add(Tensor<T>, Tensor<T>),
    Sub(Tensor<T>, Tensor<T>),
    Mul(Tensor<T>, Tensor<T>),
    Div(Tensor<T>, Tensor<T>),
    Min(Tensor<T>, Tensor<T>),
    Max(Tensor<T>, Tensor<T>),
    Pow(Tensor<T>, Tensor<T>),
    Matmul(Tensor<T>, Tensor<T>),
    MatmulAT(Tensor<T>, Tensor<T>),
    MatmulBT(Tensor<T>, Tensor<T>),
}

impl<T: AnyFloat> Children<T> {
    /// Returns a vector of child tensors
    fn children_vec(&self) -> Vec<Tensor<T>> {
        match &self {
            Children::None => {
                vec![]
            }

            Children::Id(x) |
            Children::Neg(x) |
            Children::Exp(x) |
            Children::Log(x) |
            Children::Sigmoid(x) |
            Children::DimSum(x, _) |
            Children::DimProd(x, _) |
            Children::DimMin(x, _) |
            Children::DimMax(x, _) |
            Children::NewDim(x, _) |
            Children::Reshape(x) |
            Children::Slice(x, _, _) => {
                vec![x.clone()]
            }

            Children::Add(x, y) |
            Children::Sub(x, y) |
            Children::Mul(x, y) |
            Children::Div(x, y) |
            Children::Min(x, y) |
            Children::Max(x, y) |
            Children::Pow(x, y) |
            Children::Matmul(x, y) |
            Children::MatmulAT(x, y) |
            Children::MatmulBT(x, y) => {
                vec![x.clone(), y.clone()]
            }
        }
    }

    /// Calculates gradients of the children using the chain rule
    fn update_grads(&self, parent: &Tensor<T>, cur_grad: &Tensor<T>) {
        let mut tensors = vec![];
        let mut grads = vec![];
        match self {
            Children::None => (),

            Children::Id(t) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    grads.push(Tensor::ones_like(t) * cur_grad)
                }
            }
            Children::Neg(t) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    grads.push(-Tensor::ones_like(t) * cur_grad)
                }
            }
            Children::Exp(t) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    grads.push(t.exp() * cur_grad);
                }
            }
            Children::Log(t) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    grads.push(Tensor::ones_like(t) / t * cur_grad);
                }
            }
            Children::Sigmoid(t) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    grads.push(functional::sigmoid(t) * functional::sigmoid(&(Tensor::ones_like(t) - t)) * cur_grad);
                }
            }
            Children::DimSum(t, dim) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    grads.push(cur_grad.stack_new_dim(*dim, t.shape()[*dim]))
                }
            }
            Children::DimProd(t, dim) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    grads.push((parent * cur_grad).stack_new_dim(*dim, t.shape()[*dim]) / t);
                }
            }
            Children::DimMin(t, dim) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    let parent_stack = &parent.stack_new_dim(*dim, t.shape()[*dim]);
                    let cur_grad_stack = &cur_grad.stack_new_dim(*dim, t.shape()[*dim]);
                    let data = t.flat().iter()
                        .zip(parent_stack.flat().iter())
                        .zip((cur_grad_stack).flat().iter())
                        .map(|((&x, &p), &g)| if x == p { g } else { T::zero() })
                        .collect();
                    
                    grads.push(Tensor::from_shape_data(t.shape().clone(), data));
                }
            }
            Children::DimMax(t, dim) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    let parent_stack = &parent.stack_new_dim(*dim, t.shape()[*dim]);
                    let cur_grad_stack = &cur_grad.stack_new_dim(*dim, t.shape()[*dim]);
                    let data = t.flat().iter()
                        .zip(parent_stack.flat().iter())
                        .zip((cur_grad_stack).flat().iter())
                        .map(|((&x, &p), &g)| if x == p { g } else { T::zero() })
                        .collect();
                    
                    grads.push(Tensor::from_shape_data(t.shape().clone(), data));
                }
            }
            Children::NewDim(t, dim) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    grads.push(cur_grad.sum([*dim]))
                }
            }
            Children::Reshape(t) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    grads.push(cur_grad.reshape(t.shape().clone()))
                }
            }
            Children::Slice(t, dim, range) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    let new_grad = Tensor::<T>::zeros_like(t);
                    for (idx, &val) in cur_grad.flat().iter().enumerate() {
                        let mut ndidx: Vec<usize> = cur_grad.get_ndidx(idx);
                        ndidx[*dim] += range.start;
                        *new_grad.get_mut(ndidx) = val;
                    }
                    grads.push(new_grad);
                }
            }

            Children::Add(t1, t2) => {
                if t1.grad_enabled() {
                    tensors.push(t1);
                    grads.push(Tensor::ones_like(t1) * cur_grad);
                }
                if t2.grad_enabled() {
                    tensors.push(t2);
                    grads.push(Tensor::ones_like(t2) * cur_grad);
                }
            }
            Children::Sub(t1, t2) => {
                if t1.grad_enabled() {
                    tensors.push(t1);
                    grads.push(Tensor::ones_like(t1) * cur_grad);
                }
                if t2.grad_enabled() {
                    tensors.push(t2);
                    grads.push(-Tensor::ones_like(t2) * cur_grad);
                }
            }
            Children::Mul(t1, t2) => {
                if t1.grad_enabled() {
                    tensors.push(t1);
                    grads.push(t2.clone() * cur_grad);
                }
                if t2.grad_enabled() {
                    tensors.push(t2);
                    grads.push(t1.clone() * cur_grad);
                }
            }
            Children::Div(t1, t2) => {
                if t1.grad_enabled() {
                    tensors.push(t1);
                    grads.push(Tensor::ones_like(t2) / t2 * cur_grad);
                }
                if t2.grad_enabled() {
                    tensors.push(t2);
                    grads.push(-t1 / (t2 * t2) * cur_grad);
                }
            }
            Children::Min(t1, t2) => {
                if t1.grad_enabled() {
                    tensors.push(t1);
                    grads.push(Tensor::from_flat_like(t1, 
                    t1.flat()
                            .iter()
                            .zip(t2.grad().iter())
                            .map(|(&x, &y)| if x < y { T::one() } else { T::zero() })
                            .collect::<Vec<T>>()
                    ) * cur_grad);
                }
                if t2.grad_enabled() {
                    tensors.push(t2);
                    grads.push(Tensor::from_flat_like(t2, 
                    t2.flat()
                            .iter()
                            .zip(t1.grad().iter())
                            .map(|(&y, &x)| if y < x { T::one() } else { T::zero() })
                            .collect::<Vec<T>>()
                    ) * cur_grad);
                }
            }
            Children::Max(t1, t2) => {
                if t1.grad_enabled() {
                    tensors.push(t1);
                    grads.push(Tensor::from_flat_like(t1, 
                    t1.flat()
                            .iter()
                            .zip(t2.grad().iter())
                            .map(|(&x, &y)| if x > y { T::one() } else { T::zero() })
                            .collect::<Vec<T>>()
                    ) * cur_grad);
                }
                if t2.grad_enabled() {
                    tensors.push(t2);
                    grads.push(Tensor::from_flat_like(t2, 
                    t2.flat()
                            .iter()
                            .zip(t1.grad().iter())
                            .map(|(&y, &x)| if y > x { T::one() } else { T::zero() })
                            .collect::<Vec<T>>()
                    ) * cur_grad);
                }
            }
            Children::Pow(t1, t2) => {
                if t1.grad_enabled() {
                    tensors.push(t1);
                    grads.push(Tensor::from_flat_like(t1, 
                    t1.flat()
                            .iter()
                            .zip(t2.grad().iter())
                            .map(|(&x, &y)| y * x.powf(y - T::one()) ) // TODO
                            .collect::<Vec<T>>()
                    ) * cur_grad);
                }
                if t2.grad_enabled() {
                    tensors.push(t2);
                    grads.push(Tensor::from_flat_like(t2, 
                    t2.flat()
                            .iter()
                            .zip(t1.grad().iter())
                            .map(|(&y, &x)| x.powf(y) * x.ln() )
                            .collect::<Vec<T>>()
                    ) * cur_grad);
                }
            }
            Children::Matmul(t1, t2) => {
                if t1.grad_enabled() {
                    let grad = cur_grad.matmul_bt(t2);
                    tensors.push(t1);
                    grads.push(grad);
                }
                if t2.grad_enabled() {
                    let grad = t1.matmul_at(cur_grad);
                    tensors.push(t2);
                    grads.push(grad);
                }
            }
            Children::MatmulAT(t1, t2) => {
                if t1.grad_enabled() {
                    let grad = cur_grad.matmul_bt(t2);
                    tensors.push(t1);
                    grads.push(grad);
                }
                if t2.grad_enabled() {
                    let grad = t1.matmul(cur_grad);
                    tensors.push(t2);
                    grads.push(grad);
                }
            }
            Children::MatmulBT(t1, t2) => {
                if t1.grad_enabled() {
                    let grad = cur_grad.matmul(t2);
                    tensors.push(t1);
                    grads.push(grad);
                }
                if t2.grad_enabled() {
                    let grad = cur_grad.matmul_at(t1);
                    tensors.push(t2);
                    grads.push(grad);
                }
            }
        }

        for (&t, g) in tensors.iter().zip(grads.iter()) {
            *t.grad_mut() = (t.grad_tensor() + g).flat().clone();
        }

    }
}

impl<T: AnyFloat> Tensor<T> {
    /// Returns a reverse topological ordering of the computational graph
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

    /// Returns a topological ordering of the computational graph
    fn toposort(&self) -> Vec<Tensor<T>> {
        let mut topo = Vec::new();
        self.rev_toposort(&mut topo, &mut HashSet::new());
        topo.reverse();
        topo
    }

    /// Performs a backpropagation algorithm from a unit tensor
    pub fn backward(&self) {
        assert!(self.size() == 1, "Can't backpropagate on a tensor holding more than 1 value");
        assert!(self.grad_enabled(), "Can't backpropagate on a tensor with disabled gradients");
        self.handle_mut().grad = vec![T::one(); self.size()];

        for t in &self.toposort() {
            let cur_grad = &t.grad_tensor();
            let children = t.handle().children.clone();
            children.update_grads(&t, cur_grad);
        }
    }

    /// Zeroes out all gradients from the computational graph starting from the given tensor
    /// Should be called after each backprogation, otherwise gradients will accumulate
    pub fn zero_grad(&self) {
        assert!(self.size() == 1, "Can't reset on a tensor holding more than 1 value");
        assert!(self.grad_enabled(), "Can't reset on a tensor with disabled gradients");

        for t in self.toposort() {
            t.handle_mut().grad = vec![T::zero(); t.size()];
            t.handle_mut().children = Children::None;
        }
    }
}