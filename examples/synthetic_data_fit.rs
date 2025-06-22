
use rand::{rngs::ThreadRng, Rng};
use rustml::*;


fn gen_dataset(r: &mut ThreadRng, sample_len: usize, num_classes: usize, dataset_size: usize) -> Vec<(Matrix, usize)> {
    let mut dataset = Vec::new();
    for _ in 0..dataset_size {
        let sample = Matrix::new_normal(r, (1, sample_len), 0.0, 1.0);
        let tgt = r.gen_range(0..num_classes);
        dataset.push((sample, tgt));
    }
    dataset
}


fn main() {
    let mut rng = rand::thread_rng();

    let sample_len = 32;
    let num_classes = 10;
    let dataset_size = 100;
    let dataset: Vec<(Matrix, usize)> = gen_dataset(&mut rng, sample_len, num_classes, dataset_size);
    
    let mut net = nn::Sequential::new();
    net.add(nn::Linear::new(&mut rng, sample_len, 32));
    net.add(nn::Tanh::new());
    net.add(nn::Linear::new(&mut rng, 32, num_classes));

    let epochs = 100000;
    let mut optimizer = optim::SGD::new(net.params(), 1e-4, 0.9);


    for epoch in 0..epochs {
        let mut correct = 0;
        for (data, tgt) in &dataset {
            let out: Matrix = net.forward(data);
            let (loss, _) = functional::cross_entropy_loss(&out, *tgt);
        
            loss.backward();
            optimizer.step();
            loss.reset_graph();
            let mut mx = f32::MIN;
            let mut best_i = 0;
            for i in 0..num_classes {
                if out[(0, i)].data().val > mx {
                    mx = out[(0, i)].data().val;
                    best_i = i;
                }
            }

            if best_i == *tgt {
                correct += 1;
            }
        }

        println!("Epoch {epoch}, accuracy: {:.2}%", correct as f32 / dataset_size as f32 * 100.0);
    }

}