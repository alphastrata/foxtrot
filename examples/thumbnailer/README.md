# `step_thumbnailer`

A custom thumbnailer for STEP files on Linux, designed to integrate with file managers that adhere to the freedesktop.org thumbnailing standard. 


### Prerequisites

- **Rust**: If you don't have Rust installed, you can install it using `rustup` from the official website: [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)

### Installation Steps

1.  **Install `step_thumbnailer`**: Open your terminal and run the following command:

    ```bash
    git clone https://www.github.com/alphastrata/foxtrot
    cd foxtrot/examples/thumbnailer
    cargo install step_thumbnailer --path .
    # wait...
    step_thumbnailer --version
    ```

### Options

You can customise the behavior of `step_thumbnailer` by modifying the `Exec` line in the `.thumbnailer` file. The available options are detailed below, based on the `clap::Parser` definition provided.

| Argument                  | `clap` flag                 | Description                                                                          |
| ------------------------- | --------------------------- | ------------------------------------------------------------------------------------ |
| Input File                | `-i`, `--input`             | Input STEP file to render (handled by `%i`).                                         |
| Output File               | `-o`, `--output`            | Output PNG file path (handled by `%o`).                                              |
| Thumbnail Size            | `-s`, `--thumbnail-size`    | Thumbnail size in pixels (handled by `%s`). Default is `512`.                        |
| Transparent Background    | `--transparent`             | Render with a transparent background.                                                |
| Auto-crop                 | `--crop`                    | Auto-crop the output image.                                                          |
| Render Mode               | `--mode`                    | `iso` (single isometric) or `composite` (4-view). Default is `iso`.                  |
| Camera View               | `--view`                    | `isometric`, `top`, `bottom`, `left`, `right`, `front`, `back`. Default is `isometric`. |
| Triangulation Engine      | `--engine`                  | `foxtrot` or `occt` (if compiled with `occt` feature). Default is `foxtrot`.         |
| Decimation Ratio          | `--decimation-ratio`        | Target reduction ratio for mesh decimation (0.0 - 1.0). Default is `0.25`.           |
| Decimation Error          | `--decimation-error`        | Target error tolerance for mesh decimation. Default is `0.02`.                       |
| OCCT Linear Deflection    | `--occt-linear-deflection`  | Linear deflection for OCCT triangulation. Default is `0.01`.                         |
| OCCT Angular Deflection   | `--occt-angular-deflection` | Angular deflection for OCCT triangulation. Default is `0.5`.                         |
| Shaded Rendering          | `--shaded`                  | Use shaded rendering instead of wireframe.                                           |


## Registering the STEP File MIME Type on Linux

Desktop environments on Linux use (always?) a shared MIME database to identify file types and associate them with applications and, in our case, thumbnailers. 

### 1. Create the MIME Type Definition File

First, you need to create a directory in your home folder where user-specific MIME definitions are stored.

```bash
mkdir -p ~/.local/share/mime/packages/
```

Next, create a new XML file in this directory. We'll name it `application-step.xml`.

```bash
nano ~/.local/share/mime/packages/application-step.xml
```

Paste the following content into this file. This defines a new MIME type, `application/step`, and associates the file extensions `.step`, `.stp`, `.STEP`, and `.STP` with it.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns='http://www.freedesktop.org/standards/shared-mime-info'>
  <mime-type type="application/step">
    <comment>STEP 3D model</comment>
    <glob pattern="*.step"/>
    <glob pattern="*.stp"/>
    <glob pattern="*.STEP"/>
    <glob pattern="*.STP"/>
  </mime-type>
</mime-info>
```

Save it!

### 2. Update the MIME Database

```bash
update-mime-database ~/.local/share/mime
```

### 3. Verify the MIME Type Registration

1.  **Find a STEP file** on your system (or create a dummy one: `touch my_model.step`).
2.  **Use the `xdg-mime` command** to check its MIME type:

    ```bash
    xdg-mime query filetype my_model.step
    ```

    The expected output should be:

    ```
    application/step
    ```
