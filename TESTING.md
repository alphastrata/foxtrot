# Testing

It can be helpful to have a bunch of quality `.step` files for a project like this.

# Autodesk's BRepNet dataset:

Some examples are available on their repo:

1. With `svn`:

    ```sh
    svn export https://github.com/AutodeskAILab/BRepNet/trunk/example_files/step_examples
    ```

______________________________________________________________________

2. With `git`:

    - Create a new directory for your project and navigate into it.

    - `mkdir my-brepnet-project cd my-brepnet-project `

    - Initialize an empty Git repository and add the remote.

    - `git init git remote add origin https://github.com/AutodeskAILab/BRepNet.git `

    - Enable sparse checkout. The --cone option is a performance-focused mode that makes it easier to specify top-level directories : `git config core.sparseCheckout true git sparse-checkout init --cone `

    - Tell Git which directory you want : `git sparse-checkout set example_files/step_examples `

    - Pull the files from the main branch: `git pull origin master`

    - Move them into the `foxtrot` repo.

    ______________________________________________________________________

3. Or, you could download them file-by-file.
