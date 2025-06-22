use rustml::core as eng;

fn main() {
    let mut rng = rand::thread_rng();

    // parameters
    let range = 1.0_f32;
    let n = 20;
    let iters = 100000;
    let lr = 1e-3;
    let log_step = 10;
    // ----------

    let m1 = eng::Matrix::new_uniform(&mut rng, (n, n), -range, range);

    let mut tgt = eng::Matrix::fill((n, n), 0.0);
    for i in 0..n {
        tgt[(i, i)] = eng::Value::new(1.0);
    }

    let m2 = eng::Matrix::fill(m1.shape, 1.0);

    for i in 0..iters {
        let prod = &m1 * &m2;

        let diff = &prod - &tgt;
        let loss = diff.hadamard_product(&diff).sum();

        loss.backward();
        // eng::pretty_print(&loss, 0);
        
        for y in 0..m2.shape.0 {
            for x in 0..m2.shape.1 {
                let grad = m2[(y, x)].data().grad;
                m2[(y, x)].data_mut().val -= grad * lr;
            }
        }

        loss.reset_graph();
        
        if i % log_step == 0 {
            println!("iteration {}: loss={}", i, loss.data().val);   
        }
    }

    println!("{}", &m2);
    println!("{}", &m1 * &m2);

}
