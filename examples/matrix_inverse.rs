use std::time::Instant;
use rustml::*;

fn main() {
    let start: Instant = Instant::now();
    let mut rng = rand::rng();

    // parameters
    let batch_size = 1;
    let range: f32 = 1.0;
    let n = 32;
    let iters = 50000;
    let lr: f32 = 1e0;
    let print_step = 1000;
    // ----------

    let m1 = Tensor::new_uniform(&mut rng, [batch_size, n, n], -range..range);

    let mut tgt = Tensor::fill([batch_size, n, n], 0.0);
    for b in 0..batch_size {
        for i in 0..n {
            tgt[[b, i, i]] = Scalar::new(1.0);
        }
    }

    let mut m2 = Tensor::fill([batch_size, n, n], 1.0).with_grad();

    for i in 0..iters {
        let prod = m1.mat_mul(&m2);
        let loss = functional::mse(&prod, &tgt);

        let topo = loss.backward();
        
        for scalar in m2.flat_mut() {
            scalar.set(scalar.val() - scalar.grad() * lr);
        }

        loss.backward_reset(topo);
        
        if i % print_step == 0 {
            println!("iteration {}: loss={}", i, loss.val());   
        }
    }
    
    println!("{}", &m1.mat_mul(&m2));

    let elapsed_time_s = start.elapsed().as_millis() as f32 / 1000.0;
    let op_count = batch_size * iters * n * n * n;
    let op_count_per_sec = op_count as f32 / elapsed_time_s;
    println!("Execution time: {:.2}s", elapsed_time_s);
    println!("Estimated number of arithmetic operations: {:.2e}", op_count);
    println!("Estimated FLOPS: {:.2e}", op_count_per_sec);
}
