# `step_thumbnailer`

A custom thumbnailer for STEP files on Linux, designed to integrate with file managers that adhere to the freedesktop.org thumbnailing standard. 

## 1. Installation

`step_thumbnailer` is a Rust binary that can be easily installed using `cargo`, the Rust package manager.

### Prerequisites

- **Rust**: If you don't have Rust installed, you can install it using `rustup` from the official website: [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)

### Installation Steps

1.  **Install `step_thumbnailer`**: Open your terminal and run the following command:

    ```bash
    git clone https://www.github.com/alphastrata/foxtrot
    cd foxtrot/thumbnailer
    cargo install step_thumbnailer --path .
    # wait...
    step_thumbnailer --version
    ```

## 2. Configuration

To make your file manager aware of `step_thumbnailer`, you need to create a `.thumbnailer` file. This file tells the system which MIME types this thumbnailer can handle and what command to execute to generate the thumbnail.

### Creating the `.thumbnailer` file

1.  **Create the directory**: Thumbnailer configuration files are located in `/usr/share/thumbnailers/` for system-wide use or `~/.local/share/thumbnailers/` for user-specific use. We'll use the user-specific directory for this guide. If it doesn't exist, create it:

    ```bash
    mkdir -p ~/.local/share/thumbnailers
    ```

2.  **Create the `step_thumbnailer.thumbnailer` file**: Use a text editor to create a new file named `step_thumbnailer.thumbnailer` in the directory you just created:

    ```bash
    sudoedit ~/.local/share/thumbnailers/step_thumbnailer.thumbnailer
    ```

3.  **Add the following content**: Paste the following configuration into the file:

    ```ini
    [Thumbnailer Entry]
    Exec=step_thumbnailer -i %i -o %o -s %s
    MimeType=application/step;application/STEP;application/x-step;
    ```

    -   **`Exec`**: This line specifies the command to be executed.
        -   `%i` is the placeholder for the input file path.
        -   `%o` is the placeholder for the output PNG file path.
        -   `%s` is the placeholder for the desired thumbnail size (width and height in pixels).
    -   **`MimeType`**: This specifies the MIME types for which this thumbnailer should be used. We've included common variations for STEP files.

### Customizing the Thumbnailer (Optional)

You can customize the behavior of `step_thumbnailer` by modifying the `Exec` line in the `.thumbnailer` file. The available options are detailed below, based on the `clap::Parser` definition provided.

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

## 3. Verification

To see your new thumbnailer in action, you need to clear your existing thumbnail cache and have your file manager regenerate them.

1.  **Clear the thumbnail cache**:

    ```bash
    rm -r ~/.cache/thumbnails/*
    ```

3.  **Navigate to a folder with STEP files**: Open your file manager and go to a directory containing STEP files. You should now see the newly generated thumbnails.

## 4. Troubleshooting

If thumbnails are not appearing, here are a few things to check:

-   **`step_thumbnailer` in `PATH`**: Ensure that the `step_thumbnailer` binary is in a directory that is part of your `PATH`.
-   **Executable Permissions**: Make sure the `step_thumbnailer` binary has execute permissions.
-   **`.thumbnailer` file location and content**: Double-check that the `step_thumbnailer.thumbnailer` file is in the correct directory and that its contents are correct.
-   **MIME Type**: Verify the MIME type of your STEP files. You can use the `file` command for this: `file --mime-type your_file.step`. Ensure this MIME type is listed in your `.thumbnailer` file.
-   **File Manager Settings**: Some file managers have settings that control thumbnail generation, such as a maximum file size for which to generate thumbnails. Check your file manager's preferences.
- use something like `btop`, `htop` or `top` etc to see if `step_thumbnailer` is even being run in the bg by your OS' filemanager.