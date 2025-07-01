create poetry environment:
poetry init
poetry env info
poetry add
package-mode = false
poetry install --no-root   #these 5th and 6th point should be used if we are not creating any packages.


Optional: If you're using Jupyter Notebooks inside VS Code
After installing ipykernel, run:

poetry run python -m ipykernel install --user --name=tutorial_langgraph


poetry show   # This command shows all the dependencies installed in the poetry environment.