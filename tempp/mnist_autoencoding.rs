use csv::ReaderBuilder;

use rustml::*;


fn main() {
    let mut rng = rand::thread_rng();

    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path("examples/data/MNIST_CSV/mnist_train.csv").unwrap();

    let mut all_rows: Vec<Vec<f32>> = Vec::new();

    for result in rdr.records() {
        // A record is a list of strings.
        let string_record = result.unwrap();

        // Use an iterator to parse each string field into an i32.
        // The `collect()` method can gather items into a `Result<Collection, Error>`.
        // This is a very efficient and idiomatic way to handle parsing.
        let int_vec_result: Result<Vec<f32>, _> = string_record
            .iter()
            .map(|field| field.trim().parse::<f32>()) // .trim() is good practice
            .collect();

        // Check if parsing the entire row was successful.
        match int_vec_result {
            Ok(int_vec) => all_rows.push(int_vec),
            Err(_) => ()
        }
    }

    let mut dataset: Vec<(Matrix, usize)> = Vec::new();

    for row in &all_rows {
        let mut pixels: Vec<f32> = row.clone();
        let label = pixels.remove(0) as usize;
        pixels = pixels.iter().map(|x| x / 256.0).collect();
        dataset.push((Matrix::new(&[&pixels]), label));
    }

    println!("Dataset loaded, {} images", dataset.len());

    
    let mut net = nn::Sequential::new();

    let mut encoder = nn::Sequential::new();
    encoder.add(nn::Linear::new(&mut rng, 28*28, 64));
    encoder.add(nn::ReLU::new());
    encoder.add(nn::Linear::new(&mut rng, 64, 32));
    encoder.add(nn::ReLU::new());
    encoder.add(nn::Linear::new(&mut rng, 32, 16));

    let mut decoder = nn::Sequential::new();
    decoder.add(nn::Linear::new(&mut rng, 16, 32));
    decoder.add(nn::ReLU::new());
    decoder.add(nn::Linear::new(&mut rng, 32, 64));
    decoder.add(nn::ReLU::new());
    decoder.add(nn::Linear::new(&mut rng, 64, 28*28));

    net.add(encoder);
    net.add(decoder);
    
    let mut optimizer = optim::SGD::new(net.params(), 1e-4, 0.9);

    let group_size = 50;

    let mut idx = 0;
    for (data, _) in dataset {
        let reconstruction = net.forward(&data);
        let loss = functional::mse(&data, &reconstruction);

        loss.backward();
        optimizer.step();
        loss.reset_graph();
        
        if idx % group_size == 0 {
            println!("i={idx}, loss={:.3}", loss.data().val);
        }

        idx += 1;
    }

}
