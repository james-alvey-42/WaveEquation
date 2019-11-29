# The Wave Equation

This is a simple implementation of the wave equation in (1 + 1) dimensions. To run the code with different parameters, edit the `params.json` file, or for more precision calculations use `precision-params.json`. The solver works by iterating forward initial data. With a sinusoidal initial condition on the field profile and derivative we find the behaviour shown below:

![gif](movie.gif)

This is made using the `make_video()` method in `wave.py`.
