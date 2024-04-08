# ODE Solver

This project provides implementations of numerical methods to solve ordinary differential equations (ODEs). It includes several solvers such as the fourth-order Runge-Kutta method and integrates with the `scipy` library's `odeint` function.

## Installation

1. **Clone the repository to your local machine:**
    ```bash
    git clone <repository-url>
    ```

2. **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv env
    ```

3. **Activate the virtual environment:**
    - On Windows:
    ```bash
    .\env\Scripts\activate
    ```
    - On macOS and Linux:
    ```bash
    source env/bin/activate
    ```

4. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project offers several functionalities that can be accessed through the provided Python script. Here are the available options:

- **Compare ODE integration methods with scipy library**
- **Plot motion of a pendulum with different time intervals**
- **Compare different Runge-Kutta methods**
- **Benchmark Methods**

To run the script, execute the following command:
```bash
python Ranga_Kutta_4.py
