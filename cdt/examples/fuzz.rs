use rand::{Rng, SeedableRng};
use std::iter::repeat_with;

use clap::Parser;
use itertools::Itertools;

const N: usize = 64;

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
    
    /// Lock three edges to test constrained triangulation
    #[clap(short, long)]
    lock: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let num = args.num;

    let mut i = 0;
    loop {
        if i % 1000 == 0 {
            eprintln!("{}", i);
        }
        i += 1;

        let seed = rand::random();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        // We generate random points as f32, to make it more likely that
        // some will line up exactly on one axis or another, which can trigger
        // interesting edge cases.  Experimentally, we have X or Y collisions
        // at a rate of about one per 4K fuzzed samples.
        let points: Vec<_> = repeat_with(|| rng.random_range(0.0..1.0))
            .tuple_windows()
            .map(|(a, b): (f32, f32)| (a as f64, b as f64))
            .take(num)
            .collect();

        // Generator to build the triangulation
        let r#gen = || {
            if args.lock {
                cdt::Triangulation::new_with_edges(&points, &[(0, 1), (1, 2), (2, 0)])
            } else {
                cdt::Triangulation::new(&points)
            }
        };

        let mut t = r#gen()?;
        t.check();
        let result = std::panic::catch_unwind(move || {
            while !t.done() {
                t.step().expect("Could not triangulate");
                if args.check {
                    t.check();
                }
            }
        });

        // Count how many steps we can do before failure
        if result.is_err() {
            let mut safe_steps = 0;
            for i in 0..points.len() {
                let mut t = r#gen()?;
                let result = std::panic::catch_unwind(move || {
                    for _ in 0..i {
                        t.step().expect("oh no");
                        if args.check {
                            t.check();
                        }
                    }
                });
                if result.is_ok() {
                    safe_steps = i;
                } else {
                    break;
                }
            }

            let mut t = r#gen()?;
            for _ in 0..safe_steps {
                t.step().expect("Failed too early");
            }

            if let Some(output) = &args.output {
                eprintln!("    Saving {}", output);
                t.save_debug_svg(output).expect("Could not save SVG");
            } else {
                println!("{}", t.to_svg(true));
            }
            eprintln!("Crashed with seed: {}", seed);
            break Ok(());
        }
    }
}
