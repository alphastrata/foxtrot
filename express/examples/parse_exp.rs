use std::fs::File;
use std::io::Read;
use std::time::SystemTime;

use clap::Parser;
use express::parse::{parse, strip_comments_and_lower};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Input EXPRESS file to parse
    input: String,
    
    /// Disable output
    #[clap(short, long)]
    quiet: bool,
    
    /// Output file (optional)
    output: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let input = &args.input;

    let mut f = File::open(input).expect("file opens");
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer).expect("read ok");

    let start = SystemTime::now();
    let s = strip_comments_and_lower(&buffer);
    let mut parsed = parse(&s);

    let end = SystemTime::now();
    let since_the_epoch = end.duration_since(start).expect("Time went backwards");
    eprintln!("time {:?}", since_the_epoch);

    match parsed {
        Err(e) => eprintln!("Got err {:?}", e),
        Ok((_, ref mut p)) => match &args.output {
            Some(output) => std::fs::write(output, format!("Parse tree:\n{:#?}", p))?,
            _ => {
                if !args.quiet {
                    println!("Parse tree:\n{:#?}", parsed);
                }
            }
        },
    };
    Ok(())
}
