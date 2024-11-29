Supplementary Material for a poster titled "Optimizing Model Performance and Fairness Through Evolved Sample Weights" submitted to PSB2025.

The code compares the performance of three methods for calculating sample weights: (1) equal weights, (2) deterministic weights calculated using the data characteristics, and (3) weights evolved using a Genetic Algorithm (GA).

Steps to reproduce the results for each conditions given in the Poster.
1. Exceute main() in main.py appropriate number of time. For results in the poster, excute the function 480 times for each condition.
2. Run the functions in analysis.ipynb to collate the results in hv_results (one folder for each condition).
3. Run simple_plts.R to generate the plots.