use std::{collections::HashSet, time::Instant};
use super::*;
use num_traits::{Float, Num, NumAssignOps};

pub(super) enum Children<T: Num + Copy> {
    None,
    
    Id(Tensor<T>),
    Neg(Tensor<T>),
    Exp(Tensor<T>),
    Log(Tensor<T>),
    DimSum(Tensor<T>, usize),
    NewDim(Tensor<T>, usize),
    BCLeft(Tensor<T>, usize),
    
    Add(Tensor<T>, Tensor<T>),
    Sub(Tensor<T>, Tensor<T>),
    Mul(Tensor<T>, Tensor<T>),
    Div(Tensor<T>, Tensor<T>),
    Min(Tensor<T>, Tensor<T>),
    Max(Tensor<T>, Tensor<T>),
    Pow(Tensor<T>, Tensor<T>),
    Matmul(Tensor<T>, Tensor<T>),
    Matmul_at(Tensor<T>, Tensor<T>),
    Matmul_bt(Tensor<T>, Tensor<T>),
}

impl<T: Float + NumAssignOps> Children<T> {
    fn children_vec(&self) -> Vec<Tensor<T>> {
        match &self {
            Children::None => {
                vec![]
            }

            Children::Id(x) |
            Children::Neg(x) |
            Children::Exp(x) |
            Children::Log(x) |
            Children::DimSum(x, _) |
            Children::NewDim(x, _) |
            Children::BCLeft(x, _) => {
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
            Children::Matmul_at(x, y) |
            Children::Matmul_bt(x, y) => {
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
            Children::DimSum(t, dim) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    grads.push(cur_grad.stack_new_dim(*dim, t.shape()[*dim]))
                }
            }
            Children::NewDim(t, dim) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    grads.push(cur_grad.sum([*dim]))
                }
            }
            Children::BCLeft(t, dim_cnt) => {
                if t.grad_enabled() {
                    tensors.push(t);
                    // ugly code but speeds up left-broadcasting (massive speed boost for linear networks)
                    let b_cnt: usize = cur_grad.shape()[0..*dim_cnt].iter().product();
                    let b_stride = cur_grad.size() / b_cnt;

                    let mut grad_data = vec![T::zero(); b_stride];
                    let mut idx = 0;
                    for &val in cur_grad.flat().iter() {
                        grad_data[idx] += val;
                        idx += 1;

                        // avoiding modulos
                        if idx == b_stride {
                            idx = 0;
                        }
                    }

                    let new_shape = cur_grad.shape()[*dim_cnt..].to_vec();
                    grads.push(Tensor::from_shape_data(new_shape, grad_data));
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
            Children::Max(t1, t2) => {
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
            Children::Matmul_at(t1, t2) => {
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
            Children::Matmul_bt(t1, t2) => {
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

impl<T: Float + NumAssignOps> Tensor<T> {
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
        assert!(self.size() == 1, "Can't backpropagate on a tensor holding more than 1 value");
        assert!(self.grad_enabled(), "Can't backpropagate on a tensor with disabled gradients");
        self.handle_mut().grad = vec![T::one(); self.size()];

        for t in &self.toposort() {
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