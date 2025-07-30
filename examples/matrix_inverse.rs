use std::time::Instant;
use rustml::*;

type FT = f64;


fn main() {
    let mut rng = rand::rng();
    
    // parameters
    let batch_size = 1;
    let range: FT = 1.0;
    let n = 24;
    let iters = 600000;
    let lr: FT = 1e0;
    let print_step = 1000;
    // ----------

    let tensor = &Tensor::<FT>::new_uniform(&mut rng, [batch_size, n, n], -range..range);

    let tgt = &Tensor::<FT>::fill([batch_size, n, n], 0.0);
    for b in 0..batch_size {
        for i in 0..n {
            *tgt.get_mut([b, i, i]) = 1.0;
        }
    }
    
    let start: Instant = Instant::now();

    let inverse = &Tensor::<FT>::fill([batch_size, n, n], 1.0).with_grad();
    let mut loss_sum: FT = 0.0;
    for i in 1..=iters {
        let prod = &tensor.matmul(inverse);
        let loss = &functional::mse(prod, tgt);

        loss.backward();
        inverse.set(inverse - inverse.grad_tensor() * Tensor::fill_like(inverse, lr));
        loss.zero_grad();
        
        loss_sum += loss.item();
        if i % print_step == 0 {
            println!("iterations ({}-{})/{}  ->  avg_loss={}", i-print_step+1, i, iters, loss_sum / print_step as FT);   
            loss_sum = 0.0;
        }
    }

    println!("{}", &tensor.matmul(inverse));

    let elapsed_time_s = start.elapsed().as_millis() as f32 / 1000.0;
    let op_count = batch_size * iters * n * n * n;
    let op_count_per_sec = op_count as f32 / elapsed_time_s;
    println!("Execution time: {:.2}s", elapsed_time_s);
    println!("Estimated number of arithmetic operations: {:.2e}", op_count);
    println!("Estimated FLOPS: {:.2e}", op_count_per_sec);
}
