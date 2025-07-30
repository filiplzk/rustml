// use rustml::*;

fn main() { }
//     let mut rng = rand::rng();

//     // parameters
//     let batch_size = 2;
//     let range = 1.0_f32;
//     let n = 2;
//     let iters = 10000;
//     let lr = 1e-2_f32;
//     let print_step = 10;
//     // ----------

//     let m1 = Tensor::new_uniform(&mut rng, [batch_size, n, n], -range..range);

//     let mut tgt = Tensor::fill([batch_size, n, n], 0.0);
//     for b in 0..batch_size {
//         for i in 0..n {
//             tgt[[b, i, i]] = Scalar::new(1.0);
//         }
//     }

//     let mut m2 = Tensor::fill([batch_size, n, n], 1.0).with_grad();

//     for i in 0..iters {
//         let prod = m1.mat_mul(&m2);
//         // let loss = functional::mse(&prod, &tgt);
//         let loss = Scalar::new(1.0).with_grad();
//         loss.backward();
        
//         for scalar in m2.flat_mut() {
//             scalar.set(scalar.val() - scalar.grad() * lr);
//         }

//         loss.backward_reset();
        
//         if i % print_step == 0 {
//             println!("iteration {}: loss={}", i, loss.val());   
//         }
//     }
    
//     println!("{}", &m1.mat_mul(&m2));
// }
