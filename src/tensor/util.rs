use super::*;

use std::fmt;

use num_traits::{Float, Num, NumAssign};

pub fn make_string<T: Num + Copy + fmt::Display>(tensor: &Tensor<T>, idx: &mut Vec<usize>, layer: usize, val: bool) -> String {
    let next_dim_size = tensor.shape()[idx.len()];

    let single = idx.len() == tensor.dim() - 1;
    let tabs = "    ".repeat(layer);
    let mut out = if single { format!("{tabs}[") } else { format!("{tabs}[\n") };

    for i in 0..next_dim_size {
        idx.push(i);
        if single {
            let val_str = if val {
                tensor.get(&idx).to_string()
            } else {
                format!("{}", tensor.grad()[tensor.offset(&idx)])
            };

            if i == next_dim_size - 1 {
                out.push_str(&format!("{val_str:.7}"));
            } else {
                out.push_str(&format!("{val_str:.7}, "));
            }
        } else {
            if i == next_dim_size - 1 {
                out.push_str(&format!("{}", &make_string(tensor, idx, layer + 1, val)));
            } else {
                out.push_str(&format!("{},\n", &make_string(tensor, idx, layer + 1, val)));
            }
        }
        idx.pop();
    }

    if single {
        out.push(']');
    } else {
        out.push_str(&format!("\n{tabs}]"));
    }
    
    out
}

impl<T: Num + Copy + fmt::Display> std::fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let out = format!("Tensor(Values={}, Grads={})", 
            make_string(&self, &mut vec![], 0, true),
            make_string(&self, &mut vec![], 0, false),
        );

        write!(f, "{}", out)
    }
}

impl<T: Num + Copy + fmt::Display> std::fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let out = format!("{}", make_string(&self, &mut vec![], 0, true));

        write!(f, "{}", out)
    }
}