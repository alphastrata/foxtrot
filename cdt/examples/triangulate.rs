use rand::{Rng, SeedableRng};
use std::iter::repeat_with;

use clap::Parser;
use itertools::Itertools;

const N: usize = 1_000_000;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Number of points
    #[clap(short, long, default_value_t = N)]
    num: usize,
    
    /// SVG file to output
    #[clap(short, long = "out")]
    output: Option<String>,
    
    /// Check invariants after each step (slow)
    #[clap(short, long)]
    check: bool,
    
    /// Seed for RNG
    #[clap(short, long)]
    seed: Option<u64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let num = args.num;
    let seed: u64 = args.seed.unwrap_or_else(rand::random);

    // Use a ChaCha RNG to be reproducible across platforms
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

    // Sample as f32 to match the behavior in fuzz.rs
    // (to increase likelihood of collisions)
    let points: Vec<_> = repeat_with(|| rng.random_range(0.0..1.0))
        .tuple_windows()
        .map(|(a, b): (f32, f32)| (a as f64, b as f64))
        .take(num)
        .collect();

    eprintln!("Running with seed {}", seed);
    let now = std::time::Instant::now();
    let mut t = cdt::Triangulation::new(&points)?;
    while !t.done() {
        t.step()?;
        if args.check {
            t.check();
        }
    }
    let result = t.triangles().collect::<Vec<_>>();
    let elapsed = now.elapsed();

    eprintln!(
        "    Triangulated {} points in {}.{}s.\n    Generated {} triangles.",
        num,
        elapsed.as_secs(),
        elapsed.subsec_millis(),
        result.len(),
    );

    if let Some(output) = &args.output {
        eprintln!("    Saving {}", output);
        t.save_debug_svg(output).expect("Could not save SVG");
    }
    Ok(())
}
